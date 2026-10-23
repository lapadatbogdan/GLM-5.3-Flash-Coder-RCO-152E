#!/usr/bin/env python3
"""Generate the mixed-calibration answers with the FULL model (llama-server, --jinja, thinking on as always).
Python 3.6 compatible, stdlib only (runs with the host /usr/bin/python3, 3.6.15).

Each session (from collect_prompts.py) is a multi-turn conversation of one bucket/language: the user asks the next
prompt, the model answers with the whole history (reasoning_content passed back, like the agent harness), until the
context (prompt_tokens + completion_tokens of the last turn) reaches the session's target_tokens, the prompts run out,
or an answer is cut by max_tokens (the cut turn is kept, the session stops).

Sampling matches the agent sessions (calib53/run_py.sh + pro_gen_py.py and gen_poly.py): temperature 1.0 only,
every other sampler at llama-server defaults.

Resumable append-only JSONL (--out): one line per finished turn; a line with "end" closes its session. On restart the
history of every session is rebuilt from the file and generation continues at the next turn.

    python3 gen_mix.py --sessions prompts/sessions_search.jsonl prompts/sessions_heldout.jsonl --out gen/turns.jsonl
"""
import argparse
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def call(url, messages, max_tokens, temp, tries=8):
    body = json.dumps({"model": "glm", "messages": messages, "max_tokens": max_tokens,
                       "temperature": temp}).encode()
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=7200).read().decode())
            ch = r["choices"][0]
            return ch["message"], ch.get("finish_reason"), r.get("usage", {})
        except Exception as e:  # server busy / restarting
            last = e
            time.sleep(min(300, 20 * (k + 1)))
    raise last


def load_done(path):
    """sid -> list of turn records, and the set of closed sessions."""
    turns, closed = {}, set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:  # a torn last line after a kill: ignored, that turn is redone
                    continue
                turns.setdefault(r["sid"], []).append(r)
                if r.get("end"):
                    closed.add(r["sid"])
    for sid in turns:
        turns[sid].sort(key=lambda r: r["turn"])
    return turns, closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)  # 3072 cut 3/4 code turns inside the reasoning (smoke test)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--limit-sessions", type=int, default=0, help="smoke test: only the first N sessions")
    ap.add_argument("--shard", type=int, default=0, help="multi-node: this job takes sessions shard::nshards")
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    sessions = []
    for p in a.sessions:
        with open(p) as f:
            sessions.extend(json.loads(l) for l in f if l.strip())
    # interleave buckets so a run stopped early still covers all of them
    by = {}
    for s in sessions:
        by.setdefault((s["split"], s["bucket"], s["source"]), []).append(s)
    order, i = [], 0
    while any(by.values()):
        for k in sorted(by):
            if by[k]:
                order.append(by[k].pop(0))
    if a.limit_sessions:
        order = order[:a.limit_sessions]
    # deterministic disjoint shards over the interleaved order (each shard gets every bucket); one output per shard
    assert 0 <= a.shard < a.nshards, (a.shard, a.nshards)
    order = order[a.shard::a.nshards]
    d = os.path.dirname(os.path.abspath(a.out))
    if not os.path.isdir(d):
        os.makedirs(d)
    if os.path.exists(a.out) and os.path.getsize(a.out):
        with open(a.out, "rb") as f:
            f.seek(-1, 2)
            torn = f.read(1) != b"\n"
        if torn:  # isolate a line torn by a kill so the next append starts on a fresh line (load_done skips it)
            with open(a.out, "a") as f:
                f.write("\n")
    turns, closed = load_done(a.out)
    todo = [s for s in order if s["sid"] not in closed]
    print("shard %d/%d: %d sessions, %d closed, %d to do; max_tokens %d, temp %s, workers %d"
          % (a.shard, a.nshards, len(order), len(order) - len(todo), len(todo), a.max_tokens, a.temp, a.workers), flush=True)
    lock = threading.Lock()
    stats = {"turns": 0, "tok": 0, "t0": time.time()}

    def write(rec):
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def one(s):
        msgs = [{"role": "system", "content": s["system"]}] if s.get("system") else []  # mix_v2 code sessions
        done = turns.get(s["sid"], [])
        for r in done:
            msgs.append({"role": "user", "content": r["user"]})
            msgs.append({"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]})
        for t in range(len(done), len(s["prompts"])):
            p = s["prompts"][t]
            msgs.append({"role": "user", "content": p["text"]})
            try:
                m, fin, usage = call(a.url, msgs, a.max_tokens, a.temp)
            except Exception as e:
                print("FAIL %s turn %d: %r" % (s["sid"], t, e), flush=True)
                return
            content, reasoning = m.get("content") or "", m.get("reasoning_content") or ""
            msgs.append({"role": "assistant", "content": content, "reasoning_content": reasoning})
            ctx = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
            end = None
            if fin == "length":
                end = "cut"
            elif ctx >= s["target_tokens"]:
                end = "target"
            elif t == len(s["prompts"]) - 1:
                end = "exhausted"
            write({"sid": s["sid"], "turn": t, "pid": p["pid"], "src": p["src"], "split": s["split"],
                   "bucket": s["bucket"], "lang": s["lang"], "source": s["source"], "system": s.get("system", ""),
                   "user": p["text"],
                   "reasoning": reasoning, "content": content, "finish": fin, "usage": usage, "ctx_tokens": ctx,
                   "end": end, "temp": a.temp, "max_tokens": a.max_tokens, "time": time.strftime("%H:%M:%S")})
            with lock:
                stats["turns"] += 1
                stats["tok"] += int(usage.get("completion_tokens", 0))
                el = time.time() - stats["t0"]
                print("[%s t%d] %s ctx %d, +%s tok, reasoning %d ch, answer %d ch%s | total %d turns %d tok %.1f tok/s"
                      % (s["sid"], t, fin, ctx, usage.get("completion_tokens"), len(reasoning), len(content),
                         " END " + end if end else "", stats["turns"], stats["tok"], stats["tok"] / max(el, 1)),
                      flush=True)
            if end:
                return

    ex = ThreadPoolExecutor(a.workers)
    list(ex.map(one, todo))
    turns, closed = load_done(a.out)
    print("DONE: %d/%d sessions closed" % (len(closed & {s["sid"] for s in order}), len(order)), flush=True)


if __name__ == "__main__":
    main()
