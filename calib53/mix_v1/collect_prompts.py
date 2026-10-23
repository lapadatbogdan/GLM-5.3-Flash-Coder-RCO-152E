#!/usr/bin/env python3
"""Prompt collection for the MIXED calibration set (calib53/mix_v1). Login node, CPU, internet; run in the torch
container (python3.12, pylib for huggingface_hub):

    singularity exec -B /work --env PYTHONPATH=$P/pylib,HF_HOME=$P/hfcache $P/vllm-rocm.sif \
        python3 collect_prompts.py --out $P/calib53/mix_v1/prompts --seed 0

New generations cover three buckets (the agentic 45% and the CommitPackFT half of code reuse calib53/traj_*):
  code     evol-codealpaca-v1 instructions filtered to ONE language (python, go, javascript, typescript, rust, java,
           cpp), decontaminated against HumanEval + MBPP (8-gram word overlap or function names): those are our
           regression tests and must never be in calibration
  math     GSM8K main/train + Hendrycks MATH train (all 7 subjects, all levels); test splits never touched
  general  Dolly-15k open_qa/general_qa questions (short, no context) + FineWeb-Edu sample-10BT excerpts (short,
           int_score >= 3) with an "explain this" instruction

Prompts are split into search / held-out pools FIRST (disjoint), then grouped into multi-turn sessions of one
bucket + language/source: build_calib windows are 1024 tokens cut from token 0 with the tail dropped, so a single
short Q/A gives zero windows; a session keeps asking until its context reaches `target` tokens.
Writes sessions_search.jsonl, sessions_heldout.jsonl, sources.json (dataset commit shas, seed, filters, counts).
"""
import argparse
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

LANGS = {  # language -> regex on the instruction; a prompt must match exactly one
    "python": r"\bpython\b",
    "go": r"\bgolang\b|\bin go\b|\bgo (?:language|program|code|function|routine|module|struct)|\bgo's\b",
    "javascript": r"\bjavascript\b|\bnode\.?js\b",
    "typescript": r"\btypescript\b",
    "rust": r"\brust\b",
    "java": r"\bjava\b(?!\s*script)",
    "cpp": r"c\+\+",
}
# "port this code to X" prompts: the source snippets are very often HumanEval / HumanEval-X / MultiPL-E functions
# (seen: decodeAdvancedShift, isPrime, advanced_sort, five_nine_twelve), so every port request is dropped
PORT = (r"\b(translat\w*|port(ed|ing)? (this|the|it)|convert\w* (this|the|it|following)|rewrite (this|the|it|following)|"
        r"adapt (this|the)|version of (this|the|following)|equivalent (of|in|for)|implemented in|look like in|in \w+ form|"
        r"need this in|see this in|same logic|counterpart|transcribe|transpose this|"
        r"refactor (this|the) \w+ (code )?(in|to|into))\b")
OTHER_LANG = r"\bc#|\bphp\b|\bruby\b|\bswift\b|\bkotlin\b|\bperl\b|\bscala\b|\bhaskell\b|\bjulia\b|\b r\b|\bmatlab\b|\bsql\b|\bhtml\b|\bcss\b"

# sessions: (bucket, sub, n_search, n_heldout, prompts per session, target tokens search, target heldout)
PLAN = [("code", lang, 3, 1, 8, 5200, 2100) for lang in LANGS] + [
    ("math", "gsm8k", 12, 2, 14, 5200, 5200),
    ("math", "math", 12, 1, 8, 5200, 5200),
    ("general", "dolly", 12, 2, 14, 5200, 5200),
    ("general", "fineweb_edu", 12, 1, 8, 5200, 5200),
]
FW_TEMPLATES = [
    "Here is an excerpt from an educational text:\n\n{x}\n\nExplain the main idea to a curious high-school student, "
    "adding the background knowledge needed to understand it and one concrete example.",
    "{x}\n\nWhat is this passage about? Summarize it, then explain the most important concept in more depth.",
    "Read this:\n\n{x}\n\nWhat would someone need to know to fully understand this text? Explain those points clearly.",
    "Excerpt:\n\n{x}\n\nWrite a short, accurate explanation of the topic of this excerpt for a general audience, and "
    "mention one common misconception about it.",
]


def words(s):
    return re.findall(r"[a-z0-9_]+", s.lower())


def shingles(s, n=8):
    w = words(s)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def contamination_index(cache):
    """8-gram shingles and function names of HumanEval + MBPP (all splits)."""
    sh, names = set(), set()
    he = pd.read_parquet(hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet",
                                         repo_type="dataset", cache_dir=cache))
    for _, r in he.iterrows():
        sh |= shingles(r["prompt"])
        names.add(r["entry_point"].lower())
    api = HfApi()
    for f in api.list_repo_files("google-research-datasets/mbpp", repo_type="dataset"):
        if f.endswith(".parquet"):
            df = pd.read_parquet(hf_hub_download("google-research-datasets/mbpp", f, repo_type="dataset", cache_dir=cache))
            col = "text" if "text" in df.columns else "prompt"
            for _, r in df.iterrows():
                sh |= shingles(r[col])
                names |= {m.lower() for m in re.findall(r"def\s+([A-Za-z_]\w*)", r["code"])}
    names = {n for n in names if len(n) >= 6}  # short names (e.g. "add") are too common to mean a leak
    return sh, re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, names))) + r")\b")


def contaminated(text, sh, names):
    if shingles(text) & sh:
        return True
    return bool(names.search(text.lower()))


def code_prompts(cache, sh, names):
    p = hf_hub_download("theblackcat102/evol-codealpaca-v1", "train.jsonl", repo_type="dataset", cache_dir=cache)
    by = {l: [] for l in LANGS}
    drop = Counter()
    for i, line in enumerate(open(p)):
        d = json.loads(line)
        ins = (d.get("instruction") or "").strip()
        if not 80 <= len(ins) <= 1500:
            continue
        low = ins.lower()
        hit = [l for l, rx in LANGS.items() if re.search(rx, low)]
        if len(hit) != 1 or re.search(OTHER_LANG, low):
            continue
        # "port this Python function to Go/Rust/..." prompts are mostly HumanEval/MBPP-style functions in disguise;
        # doctest prompts (>>>) are the HumanEval format itself: both dropped
        if ">>>" in ins or re.search(PORT, low) or (hit[0] != "python" and re.search(r"\bdef\s+\w+\s*\(", ins)):
            drop["translation_or_doctest_" + hit[0]] += 1
            continue
        if contaminated(ins, sh, names):
            drop[hit[0]] += 1
            continue
        by[hit[0]].append({"text": ins, "src": f"theblackcat102/evol-codealpaca-v1:train.jsonl#L{i}"})
    return by, drop


def math_prompts(cache):
    g = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset",
                                        cache_dir=cache))
    gsm = [{"text": r["question"].strip(), "src": f"openai/gsm8k:main/train#{i}"} for i, r in g.iterrows()]
    mth = []
    api = HfApi()
    for f in sorted(api.list_repo_files("EleutherAI/hendrycks_math", repo_type="dataset")):
        if f.endswith(".parquet") and "/train-" in f:
            df = pd.read_parquet(hf_hub_download("EleutherAI/hendrycks_math", f, repo_type="dataset", cache_dir=cache))
            for i, r in df.iterrows():
                if len(r["problem"]) <= 1200:
                    mth.append({"text": r["problem"].strip(), "src": f"EleutherAI/hendrycks_math:{f}#{i}",
                                "level": r.get("level"), "type": r.get("type")})
    return gsm, mth


def fineweb_rows(rnd, n_want):
    """FineWeb-Edu sample-10BT through the datasets-server rows API (no multi-GB parquet download)."""
    out, seen = [], set()
    base = "https://datasets-server.huggingface.co/rows?"
    while len(out) < n_want:
        off = rnd.randrange(0, 9_000_000)
        q = urllib.parse.urlencode({"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
                                    "split": "train", "offset": off, "length": 100})
        for k in range(5):
            try:
                rows = json.load(urllib.request.urlopen(base + q, timeout=60))["rows"]
                break
            except Exception:
                time.sleep(5 * (k + 1))
        else:
            continue
        for r in rows:
            d = r["row"]
            t = d["text"].strip()
            if d.get("int_score", 0) < 3 or not 600 <= len(t) <= 20000 or d["id"] in seen:
                continue
            cut = t[:900]
            cut = cut[:cut.rfind(". ") + 1] if ". " in cut[300:] else cut  # end on a sentence if possible
            seen.add(d["id"])
            out.append({"text": cut, "src": f"HuggingFaceFW/fineweb-edu:sample-10BT/train#{r['row_idx']}",
                        "id": d["id"], "url": d.get("url")})
            if len(out) >= n_want:
                break
    return out


def dolly(cache):
    p = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset",
                        cache_dir=cache)
    out = []
    for i, line in enumerate(open(p)):
        d = json.loads(line)
        if d["category"] in ("open_qa", "general_qa") and not d["context"].strip() and 20 <= len(d["instruction"]) <= 300:
            out.append({"text": d["instruction"].strip(), "src": f"databricks/databricks-dolly-15k#L{i}",
                        "category": d["category"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=os.environ.get("HF_HOME", "/work/glm53/hfcache") + "/hub")
    a = ap.parse_args()
    for f in ("sessions_search.jsonl", "sessions_heldout.jsonl"):
        if os.path.exists(os.path.join(a.out, f)):
            raise SystemExit(f"{a.out}/{f} exists: prompt files are never overwritten (generation may depend on them)")
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(a.seed)
    api = HfApi()
    shas = {r: api.dataset_info(r).sha for r in ("theblackcat102/evol-codealpaca-v1", "openai/gsm8k",
                                                   "EleutherAI/hendrycks_math", "databricks/databricks-dolly-15k",
                                                   "HuggingFaceFW/fineweb-edu", "openai/openai_humaneval",
                                                   "google-research-datasets/mbpp")}
    sh, names = contamination_index(a.cache)
    print(f"decontamination index: {len(sh)} 8-grams, {names.pattern.count('|') + 1} function names", flush=True)
    code, dropped = code_prompts(a.cache, sh, names)
    print("code prompts per language:", {l: len(v) for l, v in code.items()}, "dropped as contaminated:", dict(dropped))
    gsm, mth = math_prompts(a.cache)
    dol = dolly(a.cache)
    need_fw = sum(ns * pp + nh * pp for b, s, ns, nh, pp, _, _ in PLAN if s == "fineweb_edu")
    fw = fineweb_rows(rnd, need_fw)
    pools = {**{("code", l): v for l, v in code.items()}, ("math", "gsm8k"): gsm, ("math", "math"): mth,
             ("general", "dolly"): dol, ("general", "fineweb_edu"): fw}
    print("pools:", {f"{b}/{s}": len(v) for (b, s), v in pools.items()}, flush=True)

    sessions = {"search": [], "heldout": []}
    for b, s, ns, nh, pp, tgt_s, tgt_h in PLAN:
        pool = list(pools[(b, s)])
        rnd.shuffle(pool)
        need = (ns + nh) * pp
        if len(pool) < need:
            raise SystemExit(f"{b}/{s}: only {len(pool)} prompts, need {need}")
        held, srch = pool[:nh * pp], pool[nh * pp:need]  # disjoint pools first, then sessions
        for split, rows, n, tgt in (("heldout", held, nh, tgt_h), ("search", srch, ns, tgt_s)):
            for k in range(n):
                chunk = rows[k * pp:(k + 1) * pp]
                ps = []
                for j, r in enumerate(chunk):
                    text = r["text"]
                    if s == "fineweb_edu":
                        text = rnd.choice(FW_TEMPLATES).format(x=r["text"])
                    ps.append({"pid": f"{s}-{split[0]}{k:02d}-{j:02d}", "text": text, "src": r["src"]})
                sessions[split].append({"sid": f"{b}-{s}-{split}-{k:02d}", "split": split, "bucket": b,
                                        "lang": s if b == "code" else b, "source": s, "target_tokens": tgt,
                                        "prompts": ps})
    for split in ("search", "heldout"):
        allp = [p["src"] for x in sessions[split] for p in x["prompts"]]
        assert len(allp) == len(set(allp))
    hs = {p["src"] for x in sessions["heldout"] for p in x["prompts"]}
    assert not hs & {p["src"] for x in sessions["search"] for p in x["prompts"]}, "search/heldout overlap"
    for split in ("search", "heldout"):
        with open(os.path.join(a.out, f"sessions_{split}.jsonl"), "w") as f:
            for x in sessions[split]:
                f.write(json.dumps(x) + "\n")
    meta = {"seed": a.seed, "dataset_shas": shas, "plan": PLAN, "lang_regex": LANGS, "other_lang_regex": OTHER_LANG,
            "port_regex": PORT,
            "decontamination": "drop evol-codealpaca prompts asking to port/translate code, with doctests (>>>) or, for non-Python languages, a Python def "
                               "to port; then drop if it shares an 8-word shingle with any HumanEval prompt or "
                               "MBPP text, or names a HumanEval entry_point / MBPP function (>= 6 chars)",
            "dropped_contaminated": dict(dropped),
            "pool_sizes": {f"{b}/{s}": len(v) for (b, s), v in pools.items()},
            "sessions": {k: Counter(x["bucket"] + "/" + x["source"] for x in v) for k, v in sessions.items()},
            "prompts": {k: sum(len(x["prompts"]) for x in v) for k, v in sessions.items()},
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(a.out, "sources.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("sessions", "prompts")}, indent=1))


if __name__ == "__main__":
    main()
