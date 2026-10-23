#!/usr/bin/env python3
"""mix_v3 chat sessions: more math, general and thin-language code, so the calibration pool can reach ~1500 search +
~256 held-out windows (45/25/15/15) with >= ~30 windows for each of the 21 code languages. Same sources, filters and
session format as mix_v1 (collect_prompts.py) and mix_v2 (collect_prompts_v2.py), imported unchanged; every prompt
already used in mix_v1 or mix_v2 (by source id) is excluded. All sessions are split "pool": build_mix3.py picks the
held-out sessions (task-disjoint) itself.

    singularity exec -B /work --env PYTHONPATH=$P/pylib,HF_HOME=$P/hfcache $P/vllm-rocm.sif \
        python3 collect_prompts_v3.py --out $P/calib53/mix_v3/prompts --seed 3
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, "/work/glm53/calib53/mix_v1")
sys.path.insert(0, "/work/glm53/calib53/mix_v2")
import collect_prompts as v1  # noqa: E402
import collect_prompts_v2 as v2  # noqa: E402

C = "/work/glm53/calib53"
# (bucket, source, sessions, prompts per session); targets as before (5200 tokens)
PLAN = [("math", "gsm8k", 18, 14), ("math", "math", 18, 8), ("general", "dolly", 9, 14), ("general", "fineweb_edu", 9, 8)]
# code: languages under ~30 windows after mix_v1 + mix_v2 (measured by build_mix2 --dry-run on the real shards)
CODE = {"c": 3, "kotlin": 3, "lua": 3, "r": 3, "python": 2, "config": 2, "csharp": 2, "dart": 2, "html_css": 2,
        "php": 2, "ruby": 2, "scala": 2, "swift": 2}
PER_CODE = 6
TARGET = 5200


def used_srcs():
    s = set()
    for d in ("mix_v1/prompts", "mix_v2/prompts"):
        for f in ("sessions_search.jsonl", "sessions_heldout.jsonl"):
            for l in open(os.path.join(C, d, f)):
                s |= {p["src"] for p in json.loads(l)["prompts"]}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--cache", default=os.environ.get("HF_HOME", "/work/glm53/hfcache") + "/hub")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "sessions_pool.jsonl")):
        raise SystemExit(f"{a.out}/sessions_pool.jsonl exists: never overwritten")
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(a.seed)
    used = used_srcs()
    fw_used = {u.split("#")[1] for u in used if u.startswith("HuggingFaceFW")}
    sh, names = v1.contamination_index(a.cache)
    gsm, mth = v1.math_prompts(a.cache)
    dol = v1.dolly(a.cache)
    need_fw = sum(n * p for b, s, n, p in PLAN if s == "fineweb_edu")
    fw = [r for r in v1.fineweb_rows(rnd, need_fw * 2) if r["src"].split("#")[1] not in fw_used]
    pools = {"gsm8k": gsm, "math": mth, "dolly": dol, "fineweb_edu": fw}
    sessions = []
    for b, s, n, pp in PLAN:
        pool = [r for r in pools[s] if r["src"] not in used]
        rnd.shuffle(pool)
        if len(pool) < n * pp:
            raise SystemExit(f"{s}: {len(pool)} prompts, need {n * pp}")
        for k in range(n):
            ps = []
            for j, r in enumerate(pool[k * pp:(k + 1) * pp]):
                text = rnd.choice(v1.FW_TEMPLATES).format(x=r["text"]) if s == "fineweb_edu" else r["text"]
                ps.append({"pid": f"v3{s}-p{k:02d}-{j:02d}", "text": text, "src": r["src"]})
            sessions.append({"sid": f"v3-{b}-{s}-pool-{k:02d}", "split": "pool", "bucket": b,
                             "lang": b, "source": s, "target_tokens": TARGET, "prompts": ps})
    # code
    ev2, drop2 = v2.evol(a.cache, sh, names)                      # the 14 mix_v2 groups, all v2 filters
    py, drop1 = v1.code_prompts(a.cache, sh, names)                 # v1 languages (python), all v1 filters
    code_pools = {}
    for g, n in CODE.items():
        need = n * PER_CODE
        if g == "python":
            # the mix_v2 leak filters too (seen in the first v3 draft: an OCR-garbled how_many_times, a JS x_or_y port)
            e2 = [dict(r, kind="evol") for r in py["python"] if r["src"] not in used
                  and not v2.re.search(v2.PORT2, v2.re.sub(r"https?://\S+", " ", r["text"].lower()))
                  and not names.search(v2.snake(r["text"]))]
            ed = []
        else:
            prio, dirs, _, path_rx = v2.GROUPS[g]
            ed = [r for r in v2.edits(a.cache, dirs, path_rx, rnd, need + 400) if r["src"] not in used]
            e2 = [] if g in v2.EVOL_OFF else [r for r in ev2[g] if r["src"] not in used]
        rnd.shuffle(e2)
        code_pools[g] = {"edit": len(ed), "evol": len(e2)}
        mixed, i, j = [], 0, 0
        while len(mixed) < need and (i < len(ed) or j < len(e2)):
            if (len(mixed) % 2 == 0 and i < len(ed)) or j >= len(e2):
                mixed.append(ed[i]); i += 1
            else:
                mixed.append(e2[j]); j += 1
        if len(mixed) < need:
            raise SystemExit(f"{g}: only {len(mixed)} prompts, need {need}")
        for k in range(n):
            ps = [{"pid": f"v3{g}-p{k:02d}-{t:02d}", "text": r["text"], "src": r["src"], "kind": r.get("kind", "evol")}
                  for t, r in enumerate(mixed[k * PER_CODE:(k + 1) * PER_CODE])]
            sessions.append({"sid": f"v3-code-{g}-pool-{k:02d}", "split": "pool", "bucket": "code", "lang": g,
                             "source": g, "target_tokens": TARGET, "system": v2.SYS, "prompts": ps})
    srcs = [p["src"] for x in sessions for p in x["prompts"]]
    assert len(srcs) == len(set(srcs)) and not set(srcs) & used, "duplicate or reused prompt"
    with open(os.path.join(a.out, "sessions_pool.jsonl"), "w") as f:
        for x in sessions:
            f.write(json.dumps(x) + "\n")
    meta = {"seed": a.seed, "plan": PLAN, "code": CODE, "per_code_session": PER_CODE, "target": TARGET,
            "excluded_used_prompts": len(used), "code_pools": code_pools,
            "sessions": dict(Counter(x["bucket"] + "/" + x["source"] for x in sessions)),
            "n_sessions": len(sessions), "n_prompts": len(srcs), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(a.out, "sources.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
