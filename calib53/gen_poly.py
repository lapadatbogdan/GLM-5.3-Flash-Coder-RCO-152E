#!/usr/bin/env python3
"""Multi-language calibration sessions: GLM-5.3-Flash edits real files from CommitPackFT commits (old file +
commit message -> updated file), reasoning on. Covers the languages our agent harness cannot run offline
(Go module caches are gone, most NodeBB tasks fail to prepare).

    python gen_poly.py --data calib53/commitpackft/data --out calib53/traj_poly --url http://127.0.0.1:8000/v1/chat/completions
One JSON per task (same shape as the agent trajectories: messages with reasoning_content); resumable.
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PER_LANG = {"go": 50, "javascript": 40, "typescript": 40, "rust": 40, "java": 30, "c++": 30, "sql": 25, "shell": 25}
PER_REPO = 2  # commits per repository (1 starved the smaller languages: go 10, rust 16, sql 11)
FENCE = {"c++": "cpp", "shell": "bash", "javascript": "javascript", "typescript": "typescript"}
SYS = "You are an expert software engineer. You edit code precisely and keep everything that does not need to change."


def pick(data_dir, seed=0):
    rnd = random.Random(seed)
    out = []
    for lang, n in PER_LANG.items():
        rows = []
        for line in open(os.path.join(data_dir, lang, "data.jsonl")):
            d = json.loads(line)
            old, msg = d.get("old_contents") or "", (d.get("message") or d.get("subject") or "").strip()
            if 1500 <= len(old) <= 12000 and len(msg) >= 15 and old != d.get("new_contents"):
                rows.append(d)
        rnd.shuffle(rows)
        per_repo = {}
        got = []
        for d in rows:  # at most PER_REPO commits per repo, so no single project dominates
            r = d.get("repos", d.get("old_file"))
            if per_repo.get(r, 0) >= PER_REPO:
                continue
            per_repo[r] = per_repo.get(r, 0) + 1
            got.append((lang, d))
            if len(got) >= n:
                break
        out.append(got)
    # interleave the languages: a run stopped early still covers all of them (the first run did go, then js only)
    mixed = []
    for i in range(max(len(g) for g in out)):
        mixed.extend(g[i] for g in out if i < len(g))
    return mixed


def call(url, messages, max_tokens, temp):
    body = json.dumps({"model": "glm", "messages": messages, "max_tokens": max_tokens, "temperature": temp}).encode()
    for attempt in range(8):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}),
                                                 timeout=3600))
            return r["choices"][0]["message"], r.get("usage", {})
        except Exception as e:  # server busy or restarting
            last = e
            time.sleep(20)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--shard", default="0/1", help="i/n: this job takes every n-th task starting at i")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tasks = pick(args.data)
    si, sn = (int(x) for x in args.shard.split("/"))
    tasks = tasks[si::sn]
    print(f"{len(tasks)} tasks: " + ", ".join(f"{l} {sum(1 for x, _ in tasks if x == l)}" for l in PER_LANG), flush=True)
    lock = threading.Lock()
    done = [0]

    def one(item):
        lang, d = item
        tid = f"{lang}_{d['commit'][:12]}"
        path = os.path.join(args.out, tid + ".json")
        if os.path.exists(path):
            return
        user = (f"File `{d.get('old_file')}` from the repository `{d.get('repos', '')}`:\n"
                f"```{FENCE.get(lang, lang)}\n{d['old_contents']}\n```\n\n"
                f"Task (from the commit message): {d.get('message', '').strip()}\n\n"
                "Apply this change. Return the complete updated file in one code block, then one line saying what changed.")
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        try:
            m, usage = call(args.url, msgs, args.max_tokens, args.temp)
        except Exception as e:
            print(f"FAIL {tid}: {e}", flush=True)
            return
        msgs.append({"role": "assistant", "content": m.get("content") or "",
                     "reasoning_content": m.get("reasoning_content") or ""})
        rec = {"task": tid, "lang": lang, "repo": d.get("repos", ""), "resolved": None, "messages": msgs,
               "usage": usage, "model": "GLM-5.3-Flash Q8_0 (full)", "source": "commitpackft"}
        tmp = path + ".tmp"
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, path)
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(tasks)}] {tid} reasoning {len(m.get('reasoning_content') or '')} ch, "
                  f"answer {len(m.get('content') or '')} ch, {usage.get('completion_tokens')} tok", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, tasks))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
