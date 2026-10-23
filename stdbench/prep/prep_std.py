# Downloads + builds the item files for math500, aime25, gpqa, mmlupro, ifeval (login node, internet).
# Run with the vllm-rocm.sif python3 (huggingface_hub + pyarrow):
#   singularity exec -B /work $SIF python3 prep/prep_std.py
# Raw files land in raw/<repo>/ at pinned revisions; data/<bench>.jsonl; data/SOURCES_std.json with
# revisions, URLs and sha256 of every raw file.
import csv, hashlib, io, json, os, random, sys, urllib.request
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw")
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
import core

api = HfApi()
SRC = {"hf": {}, "url": {}, "github": {
    "openai/simple-evals": "652c89d0ca9df547706735883097e9537d40dc47",
    "TIGER-AI-Lab/MMLU-Pro": "f418b116db00b065c2aea046518d8fcf74d39872",
    "google-research/google-research (instruction_following_eval)": "3ef52111c1fabc301880818f5a547570e926a76b",
    "QwenLM/Qwen2.5-Math (evaluation)": "a45202bd16f1ec06f433442dc1152d0074773465"}}


def hf(repo, fname):
    if repo not in SRC["hf"]:
        SRC["hf"][repo] = {"revision": api.dataset_info(repo).sha, "files": {}}
    rev = SRC["hf"][repo]["revision"]
    p = hf_hub_download(repo, fname, repo_type="dataset", revision=rev, local_dir=os.path.join(RAW, repo.replace("/", "__")))
    SRC["hf"][repo]["files"][fname] = core.sha256_file(p)
    return p


def url(u, name):
    p = os.path.join(RAW, "url", name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(urllib.request.urlopen(u, timeout=120).read())
    SRC["url"][u] = core.sha256_file(p)
    return p


def jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def dump(name, rows):
    with open(os.path.join(DATA, name + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("%s: %d items" % (name, len(rows)))


# ---------------------------------------------------------------- MATH-500
rows = jsonl(hf("HuggingFaceH4/MATH-500", "test.jsonl"))
dump("math500", [{"id": "math500/" + r["unique_id"], "problem": r["problem"], "answer": r["answer"],
                  "solution": r["solution"], "group": "%s/L%s" % (r["subject"], r["level"])} for r in rows])

# ---------------------------------------------------------------- AIME 2025 (I + II, 30 problems)
ma = pq.read_table(hf("MathArena/aime_2025", "data/train-00000-of-00001.parquet")).to_pylist()
oc = jsonl(hf("opencompass/AIME2025", "aime2025-I.jsonl")) + jsonl(hf("opencompass/AIME2025", "aime2025-II.jsonl"))
assert len(ma) == 30 and len(oc) == 30, (len(ma), len(oc))
ma.sort(key=lambda r: int(r["problem_idx"]))
mism = [i for i, (a, b) in enumerate(zip(ma, oc))
        if str(int(a["answer"])) != str(b["answer"]).strip().replace("^\\circ", "")]  # OC writes 336^\circ
print("AIME answers MathArena vs OpenCompass mismatches (index):", mism)
assert not mism
dump("aime25", [{"id": "aime25/%02d" % int(r["problem_idx"]), "problem": r["problem"], "answer": str(int(r["answer"])),
                 "group": "I" if int(r["problem_idx"]) <= 15 else "II"} for r in ma])

# ---------------------------------------------------------------- GPQA Diamond (simple-evals csv + permutations)
p = url("https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv", "gpqa_diamond.csv")
import pandas  # simple-evals reads it with pandas.read_csv; do the same so row parsing is identical
df = pandas.read_csv(p)
examples = [row.to_dict() for _, row in df.iterrows()]
# simple-evals GPQAEval(n_repeats=4): rng = random.Random(0); examples * 4; permutation = rng.sample(range(4), 4)
# per (repeat, example) in that order. Sample s of item i uses the permutation of repeat s.
rng = random.Random(0)
N_REP = 4
perms = [rng.sample(range(4), 4) for _ in range(len(examples) * N_REP)]
# Idavidrein/gpqa (HF) is gated for this account; the simple-evals CSV above is what simple-evals uses.
out = []
for i, ex in enumerate(examples):
    out.append({"id": "gpqa/%s" % ex["Record ID"], "question": ex["Question"], "correct": ex["Correct Answer"],
                "incorrect": [ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]],
                "perms": [perms[r * len(examples) + i] for r in range(N_REP)],
                "group": ex.get("High-level domain")})
dump("gpqa", out)

# ---------------------------------------------------------------- MMLU-Pro (stratified 1000, seed 0)
test = pq.read_table(hf("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet")).to_pylist()
val = pq.read_table(hf("TIGER-Lab/MMLU-Pro", "data/validation-00000-of-00001.parquet")).to_pylist()


def prep_opts(r):  # official preprocess(): drop "N/A" options
    return [o for o in r["options"] if o != "N/A"]


shots = {}
for r in val:
    shots.setdefault(r["category"], []).append({"question": r["question"], "options": prep_opts(r),
                                                "cot_content": r["cot_content"]})
items = [{"id": "mmlupro/%d" % r["question_id"], "question": r["question"], "options": prep_opts(r),
          "answer": r["answer"], "answer_index": r["answer_index"], "category": r["category"],
          "src": r["src"], "group": r["category"]} for r in test]
for it in items:  # answer letter must still match after dropping N/A
    assert "ABCDEFGHIJ"[it["answer_index"]] == it["answer"] and it["answer_index"] < len(it["options"]), it["id"]
sub = core.stratified_sample(items, lambda it: it["category"], 1000, seed=0)
for it in sub:
    it["shots"] = shots[it["category"]]
dump("mmlupro", sub)
print("mmlupro full test=%d, per-category subset:" % len(items),
      json.dumps({c: sum(1 for it in sub if it["category"] == c) for c in sorted(shots)}))

# ---------------------------------------------------------------- IFEval (541)
rows = jsonl(hf("google/IFEval", "ifeval_input_data.jsonl"))
gh = jsonl(os.path.join(RAW, "gh", "ifeval", "data", "input_data.jsonl"))
# The official checker's own input_data.jsonl (google-research, pinned) is used. The HF copy differs in
# exactly one prompt (key 2785: "at least one placeholder" while kwargs say num_placeholders=3; the
# google-research file says "at least 3 placeholders", consistent with the checker).
diff = [r["key"] for r, g in zip(rows, gh) if r["key"] != g["key"] or r["prompt"] != g["prompt"]]
print("IFEval HF vs google-research prompt differences (keys):", diff)
assert len(diff) <= 1
rows = gh
dump("ifeval", [{"id": "ifeval/%d" % r["key"], "key": r["key"], "prompt": r["prompt"],
                 "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"],
                 "group": r["instruction_id_list"][0].split(":")[0]} for r in rows])

for root, _, files in os.walk(os.path.join(RAW, "gh")):
    for f in files:
        p = os.path.join(root, f)
        SRC.setdefault("github_files_sha256", {})[os.path.relpath(p, RAW)] = core.sha256_file(p)
with open(os.path.join(DATA, "SOURCES_std.json"), "w") as f:
    json.dump(SRC, f, indent=1, sort_keys=True)
print("wrote data/SOURCES_std.json")
