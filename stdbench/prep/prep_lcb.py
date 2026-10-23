# LiveCodeBench (code generation) data prep. Login node (internet), run with the vllm-rocm.sif python3
# (has huggingface_hub):
#   singularity exec -B /work $P/vllm-rocm.sif python3 prep/prep_lcb.py
# Source: HF livecodebench/code_generation_lite, pinned revision (newest official release = release_v6,
# last updated 2025-06-05; contest dates up to 2025-04). Window: contest_date in [START, END], same
# filter semantics as lcb_runner (start <= date <= end).
# Output:
#   data/lcb.jsonl            one item per problem (prompt fields + metadata, no tests)
#   data/lcb_tests/<qid>.json  public + private tests in lcb_runner order (decoded base64+zlib+pickle),
#                              bound read-only into the grading sandbox at /lcbdata
#   data/SOURCES_lcb.json     revision, file sha256, window, counts
import argparse, base64, collections, hashlib, io, json, os, pickle, zlib
from datetime import datetime
from huggingface_hub import hf_hub_download

REPO = "livecodebench/code_generation_lite"
REV = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
# release files are appended in contest-date order: test5 starts 2024-09, test6 ends 2025-04
FILES = ["test5.jsonl", "test6.jsonl"]
LCB_GITHUB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw", "lcb")
DATA = os.path.join(HERE, "data")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class _NoGlobals(pickle.Unpickler):
    # the official encoding pickles a plain JSON string; refuse any class/global so a tampered
    # dataset file cannot execute code at prep time
    def find_class(self, module, name):
        raise pickle.UnpicklingError("global %s.%s refused" % (module, name))


def decode_private(s):
    # same decoding as lcb_runner/benchmarks/code_generation.py CodeGenerationProblem.__post_init__
    # (json, else base64 -> zlib -> pickle -> json), with a restricted unpickler
    try:
        return json.loads(s)
    except Exception:
        raw = zlib.decompress(base64.b64decode(s.encode("utf-8")))
        return json.loads(_NoGlobals(io.BytesIO(raw)).load())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-05-01")
    a = ap.parse_args()
    t0 = datetime.strptime(a.start, "%Y-%m-%d")
    t1 = datetime.strptime(a.end, "%Y-%m-%d")
    os.makedirs(os.path.join(DATA, "lcb_tests"), exist_ok=True)
    files = {}
    items = []
    for fn in FILES:
        p = hf_hub_download(REPO, fn, repo_type="dataset", revision=REV, local_dir=RAW)
        files[fn] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                d = datetime.fromisoformat(r["contest_date"])
                if not (t0 <= d <= t1):
                    continue
                pub = json.loads(r["public_test_cases"])
                priv = decode_private(r["private_test_cases"])
                meta = json.loads(r["metadata"])
                tests = pub + priv
                types = sorted(set(t["testtype"] for t in tests))
                fn_name = meta.get("func_name", None)
                qid = r["question_id"]
                with open(os.path.join(DATA, "lcb_tests", "%s.json" % qid), "w") as g:
                    json.dump({"inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests],
                               "fn_name": fn_name}, g)
                items.append({"id": "lcb/%s" % qid, "qid": qid, "group": r["difficulty"],
                              "platform": r["platform"], "difficulty": r["difficulty"],
                              "contest_id": r["contest_id"], "contest_date": r["contest_date"],
                              "question_title": r["question_title"], "question_content": r["question_content"],
                              "starter_code": r["starter_code"], "fn_name": fn_name, "testtypes": types,
                              "n_public": len(pub), "n_tests": len(tests)})
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    items.sort(key=lambda it: (it["contest_date"], it["id"]))
    with open(os.path.join(DATA, "lcb.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    by = collections.Counter()
    month = collections.Counter()
    for it in items:
        by["%s/%s" % (it["platform"], it["difficulty"])] += 1
        month[it["contest_date"][:7]] += 1
    post = sum(1 for it in items if it["contest_date"] >= "2026-08-25")
    src = {"dataset": REPO, "revision": REV, "release": "release_v6 (newest official; HF last modified 2025-06-05)",
           "files": files, "window": [a.start, a.end], "n": len(items),
           "by_platform_difficulty": dict(by), "by_month": dict(month), "post_2026_08_25": post,
           "lcb_runner_commit": LCB_GITHUB_COMMIT,
           "vendored": {k: sha256(os.path.join(HERE, "tools", "lcb", "upstream", k))
                        for k in sorted(os.listdir(os.path.join(HERE, "tools", "lcb", "upstream")))}}
    with open(os.path.join(DATA, "SOURCES_lcb.json"), "w") as f:
        json.dump(src, f, indent=1)
    print(json.dumps({k: v for k, v in src.items() if k != "vendored"}, indent=1))


if __name__ == "__main__":
    main()
