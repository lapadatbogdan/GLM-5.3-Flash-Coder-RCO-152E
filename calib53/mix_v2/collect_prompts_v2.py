#!/usr/bin/env python3
"""Extra code languages for the mixed calibration (calib53/mix_v2): sessions in the same format as mix_v1
(collect_prompts.py), for 14 language groups the first mix did not cover. Login node, CPU, internet; torch container:

    singularity exec -B /work --env PYTHONPATH=$P/pylib,HF_HOME=$P/hfcache $P/vllm-rocm.sif \
        python3 collect_prompts_v2.py --out $P/calib53/mix_v2/prompts --seed 0

Two sources per language, alternated inside a session:
  edit   a real file from CommitPackFT (bigcode/commitpackft, old file + commit message -> updated file), the same
         prompt as calib53/gen_poly.py (traj_poly); at most 2 commits per repository
  evol   evol-codealpaca-v1 instructions that name exactly this language (and no other known one), with every
         mix_v1 filter: no port/translate requests, no doctests, no Python def in other languages, no 8-gram or
         function-name overlap with HumanEval / MBPP (our regression tests)
High priority groups get 3 search sessions, medium 2; each group 1 held-out session (prompts disjoint).
Writes sessions_search.jsonl, sessions_heldout.jsonl, sources.json; never overwrites.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, "/work/glm53/calib53/mix_v1")
import collect_prompts as v1  # noqa: E402  (filters + decontamination index of mix_v1, unchanged)
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

SYS = "You are an expert software engineer. You edit code precisely and keep everything that does not need to change."
# group -> (priority, commitpackft dirs, evol regex, optional path filter for edits)
GROUPS = {
    "csharp": ("high", ["c#"], r"\bc#|\basp\.net\b|\.net core\b|\bblazor\b|\bunity3d\b", None),
    "php": ("high", ["php"], r"\bphp\b|\blaravel\b|\bsymfony\b|\bwordpress plugin\b", None),
    "html_css": ("high", ["html", "css", "scss", "vue"], r"\bhtml\b|\bcss\b|\bscss\b|\btailwind\b|\bflexbox\b", None),
    "c": ("high", ["c"], r"\bin c\b|\bc program|\bc language\b|\bansi c\b|\bc code\b|\bc function\b", None),
    "sql": ("high", ["sql"], r"\bsql\b|\bpostgres(?:ql)?\b|\bmysql\b|\bsqlite\b|\bt-sql\b|\bpl/sql\b|\boracle database\b", None),
    "shell": ("high", ["shell"], r"\bbash\b|\bshell script|\bzsh\b|\bposix shell\b", None),
    "config": ("high", ["yaml", "dockerfile", "hcl", "json"],
               r"\bdockerfile\b|\bdocker-compose\b|\bterraform\b|\bgithub actions\b|\bkubernetes\b|\bhelm chart\b|"
               r"\bansible playbook\b|\bci/cd pipeline\b|\bgitlab ci\b",
               r"(\.github/workflows/|docker-compose|dockerfile|\.tf$|k8s|kubernetes|helm|deploy|\.gitlab-ci|"
               r"ansible|playbook|package\.json$|tsconfig|composer\.json$|appsettings|\.eslintrc|config)"),
    "kotlin": ("medium", ["kotlin"], r"\bkotlin\b", None),
    "swift": ("medium", ["swift"], r"\bswift\b|\bswiftui\b", None),
    "ruby": ("medium", ["ruby"], r"\bruby\b|\brails\b", None),
    "scala": ("medium", ["scala"], r"\bscala\b", None),
    "lua": ("medium", ["lua"], r"\blua\b", None),
    "r": ("medium", ["r"], r"\bin r\b|\br programming\b|\br language\b|\bggplot2?\b|\bdplyr\b|\br script\b|\br code\b|\btidyverse\b", None),
    "dart": ("medium", ["dart"], r"\bdart\b|\bflutter\b", None),
}
# extra port patterns seen while reading the first draft (Lua: "How would you express this in Lua? script const isPrime")
PORT2 = (r"(express|say|write|do|redo|implement|code|rewrite|render) (this|it|that|the same)( code| snippet| function| logic)? in\b|"
         r"script const |=> \{|\bocr\b|what('s| is) the \w* ?(equivalent|version)|how would (this|it) (look|be)|"
         r"\bthis (code|snippet|function) in \w+|re-?write the following|\bthe following in \w+|looks? in \w+|"
         r"#include\s*<|\bpublic (static )?(int|void|string|boolean)\b|\bequivalent\b|this would be in|"
         r"how this would|change this code")
EVOL_OFF = {"lua"}  # every Lua evol prompt left after filtering was still a disguised Python port (below_zero)


def snake(text):
    """camelCase identifiers to snake_case, so chooseNum is caught as HumanEval's choose_num."""
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text).lower()
# path / content checks for edits: commitpackft 'c' holds C++ headers, 'r' holds REBOL (.r) files
EDIT_OK = {"c": lambda path, old: path.endswith(".c"),
           "r": lambda path, old: bool(re.search(r"<-|library\(|function\(", old))}
N_SEARCH = {"high": 3, "medium": 2}
N_HELD = 1
PER_SESSION = 6          # prompts per session (alternating edit / evol); sessions stop at their token target
TARGET = (5200, 2100)    # search, held-out (as mix_v1 code)
FENCE = {"c#": "csharp", "shell": "bash", "hcl": "hcl", "dockerfile": "dockerfile"}


def edits(cache, dirs, path_rx, rnd, need):
    rows = []
    for d in dirs:
        p = hf_hub_download("bigcode/commitpackft", f"data/{d}/data.jsonl", repo_type="dataset", cache_dir=cache)
        for i, line in enumerate(open(p)):
            r = json.loads(line)
            old, msg = r.get("old_contents") or "", (r.get("message") or r.get("subject") or "").strip()
            lo = 800 if d in ("r", "dockerfile", "hcl", "lua", "dart", "sql") else 1500
            if not (lo <= len(old) <= 12000 and len(msg) >= 15 and old != r.get("new_contents")):
                continue
            if path_rx and not re.search(path_rx, (r.get("old_file") or "").lower()):
                continue
            if d in EDIT_OK and not EDIT_OK[d]((r.get("old_file") or "").lower(), old):
                continue
            rows.append((d, i, r))
    rnd.shuffle(rows)
    per_repo, out = Counter(), []
    for d, i, r in rows:
        repo = r.get("repos", r.get("old_file"))
        if per_repo[repo] >= 2:
            continue
        per_repo[repo] += 1
        user = (f"File `{r.get('old_file')}` from the repository `{r.get('repos', '')}`:\n"
                f"```{FENCE.get(d, d)}\n{r['old_contents']}\n```\n\n"
                f"Task (from the commit message): {r.get('message', '').strip()}\n\n"
                "Apply this change. Return the complete updated file in one code block, then one line saying what changed.")
        out.append({"text": user, "src": f"bigcode/commitpackft:data/{d}/data.jsonl#L{i}", "kind": "edit"})
        if len(out) >= need:
            break
    return out


def evol(cache, sh, names):
    p = hf_hub_download("theblackcat102/evol-codealpaca-v1", "train.jsonl", repo_type="dataset", cache_dir=cache)
    allrx = {**{g: v[2] for g, v in GROUPS.items()}, **{"v1_" + k: v for k, v in v1.LANGS.items()}}
    by, drop, seen_keys = {g: [] for g in GROUPS}, Counter(), set()
    for i, line in enumerate(open(p)):
        ins = (json.loads(line).get("instruction") or "").strip()
        if not 80 <= len(ins) <= 1500:
            continue
        low = re.sub(r"https?://\S+", " ", ins.lower())  # a language name inside a URL does not count
        hit = [g for g, rx in allrx.items() if re.search(rx, low)]
        if len(hit) != 1 or hit[0] not in GROUPS:
            continue
        g = hit[0]
        if (">>>" in ins or re.search(v1.PORT, low) or re.search(PORT2, low)
                or re.search(r"\bdef\s+\w+\s*\(", ins)):
            drop["port_doctest_def_" + g] += 1
            continue
        if v1.contaminated(ins, sh, names) or names.search(snake(ins)):
            drop["humaneval_mbpp_" + g] += 1
            continue
        key = " ".join(low.split()[:8])  # evol near-duplicates ("Design a Microservice, using either Go or Scala")
        if key in seen_keys:
            drop["near_duplicate_" + g] += 1
            continue
        seen_keys.add(key)
        by[g].append({"text": ins, "src": f"theblackcat102/evol-codealpaca-v1:train.jsonl#L{i}", "kind": "evol"})
    return by, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=os.environ.get("HF_HOME", "/work/glm53/hfcache") + "/hub")
    a = ap.parse_args()
    for f in ("sessions_search.jsonl", "sessions_heldout.jsonl"):
        if os.path.exists(os.path.join(a.out, f)):
            raise SystemExit(f"{a.out}/{f} exists: never overwritten")
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(a.seed)
    sh, names = v1.contamination_index(a.cache)
    ev, dropped = evol(a.cache, sh, names)
    sessions = {"search": [], "heldout": []}
    pools = {}
    for g, (prio, dirs, _, path_rx) in GROUPS.items():
        ns = N_SEARCH[prio]
        need = (ns + N_HELD) * PER_SESSION
        ed = edits(a.cache, dirs, path_rx, rnd, need)
        e2 = [] if g in EVOL_OFF else list(ev[g])
        rnd.shuffle(e2)
        pools[g] = {"edit": len(ed), "evol": len(e2)}
        # alternate edit / evol; when one source runs out the other fills in
        mixed, i, j = [], 0, 0
        while len(mixed) < need and (i < len(ed) or j < len(e2)):
            if (len(mixed) % 2 == 0 and i < len(ed)) or j >= len(e2):
                mixed.append(ed[i]); i += 1
            else:
                mixed.append(e2[j]); j += 1
        if len(mixed) < need:
            raise SystemExit(f"{g}: only {len(mixed)} prompts, need {need}")
        held, srch = mixed[:N_HELD * PER_SESSION], mixed[N_HELD * PER_SESSION:]
        for split, rows, n, tgt in (("heldout", held, N_HELD, TARGET[1]), ("search", srch, ns, TARGET[0])):
            for k in range(n):
                ps = [{"pid": f"{g}-{split[0]}{k:02d}-{t:02d}", "text": r["text"], "src": r["src"], "kind": r["kind"]}
                      for t, r in enumerate(rows[k * PER_SESSION:(k + 1) * PER_SESSION])]
                sessions[split].append({"sid": f"code2-{g}-{split}-{k:02d}", "split": split, "bucket": "code",
                                        "lang": g, "source": g, "target_tokens": tgt, "system": SYS, "prompts": ps})
    hs = {p["src"] for x in sessions["heldout"] for p in x["prompts"]}
    ss = [p["src"] for x in sessions["search"] for p in x["prompts"]]
    assert len(ss) == len(set(ss)) and not hs & set(ss), "duplicate or search/heldout overlap"
    for split in ("search", "heldout"):
        with open(os.path.join(a.out, f"sessions_{split}.jsonl"), "w") as f:
            for x in sessions[split]:
                f.write(json.dumps(x) + "\n")
    api = HfApi()
    meta = {"seed": a.seed, "groups": GROUPS, "n_search": N_SEARCH, "n_heldout": N_HELD, "per_session": PER_SESSION,
            "targets": TARGET, "port2_regex": PORT2, "evol_off": sorted(EVOL_OFF), "pools": pools, "dropped_evol": dict(dropped),
            "dataset_shas": {r: api.dataset_info(r).sha for r in ("bigcode/commitpackft",
                                                                   "theblackcat102/evol-codealpaca-v1")},
            "sessions": {k: len(v) for k, v in sessions.items()},
            "prompts": {k: sum(len(x["prompts"]) for x in v) for k, v in sessions.items()},
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(a.out, "sources.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("pools", "sessions", "prompts", "dropped_evol")}, indent=1))


if __name__ == "__main__":
    main()
