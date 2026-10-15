#!/usr/bin/env python3
"""Final check (12 Oct): moves aside the unresolved attempts affected by the old harness limits and writes
the rerun lists. Affected = (a) context exhausted (~>105K tokens with reasoning, under 200 turns) or (b) empty patch,
no modified tests, under 200 turns, finished before the git add -N fix (12 Oct 18:10)."""
import json, glob, os, sys, time
E = "/work/glm53/calib53/eval"
FIX = time.mktime(time.strptime("2026-10-12 18:15", "%Y-%m-%d %H:%M"))
rows = [l for l in open(E + "/eval_tasks_py.txt") if l.strip()]
for d in sys.argv[1:]:
    hit = []
    for f in glob.glob(f"{E}/{d}/*.json"):
        x = json.load(open(f))
        if x["resolved"] or x["turns"] >= 200 or x["tampered_tests"]:
            continue
        ch = sum(len(str(m["content"])) + len(m.get("reasoning_content") or "") for m in x["messages"]) / 3.5
        old_empty = (not x["model_patch"]) and os.path.getmtime(f) < FIX
        if ch > 105000 or old_empty:
            os.rename(f, f + ".fin"); hit.append(x["task"])
    open(f"{E}/tasks_fin_{d}.txt", "w").writelines([r for r in rows if any(t in r for t in hit)])
    print(d, len(hit), flush=True)
