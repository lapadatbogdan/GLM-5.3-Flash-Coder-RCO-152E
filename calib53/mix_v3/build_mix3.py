#!/usr/bin/env python3
"""Calibration pool v3 (calib53/mix_v3), built so the number of windows N can be chosen by measurement.

Same rendering and windows as every earlier set (build_calib.render/windows: chat template with clear_thinking=False,
1024-token windows from token 0, tail dropped, mask = model-written tokens incl. reasoning, >= 20% model tokens).

Sources: agent sessions calib53/traj_py, traj_js, mix_v3/traj_py (agentic); CommitPackFT edits calib53/traj_poly and
the generated chat sessions of mix_v1, mix_v2, mix_v3 (code / math / general).

Outputs (never overwritten):
  calib_heldout.pt  one fixed held-out set, task-disjoint from the pool: whole sessions (tasks) are held out.
                    Default 256 = agentic 100 (round-robin over repositories) + code 105 (5 per language x 21) +
                    math 26 + general 25. General is a MONITOR only (not in the pool, not in the weighted KL or
                    the convergence rule); the convergence analysis weights agentic/code/math 45/35/20.
  calib_search.pt   the search POOL (agentic / code / math, NO general) in a deterministic order such that EVERY
                    prefix of N windows keeps the 45/35/20 shares (largest-deficit scheduler) and is balanced inside each bucket (round-robin
                    over repositories / languages / sources, then over tasks, then a seeded window order).
                    N = 256, 512, 768, ... are prefixes; meta[i]["order"] == i. The pool stops at --max-pool or when
                    a bucket runs out (then later prefixes would break the shares).
  pool_report.json  counts per bucket / language and the largest N with exact shares.
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/work/glm53/calib53/mix_v1")
sys.path.insert(0, "/work/glm53/calib53")
import build_mix as v1  # noqa: E402  (generated_sessions)
from build_calib import render, windows  # noqa: E402

SHARES = {"agentic": 0.45, "code": 0.35, "math": 0.20}   # search pool: NO general (user decision 15 oct)
BUCKETS = ["agentic", "code", "math", "general"]          # general: held-out only, as a monitor
HELD = {"agentic": 100, "math": 26, "general": 25}   # code: --held-per-lang x number of languages
LANG = {"c++": "cpp", "js": "javascript"}


def agent_sessions(dirs):
    """(lang, task, repo, msgs) from agent trajectory dirs, filtered like build_calib (reasoning present)."""
    for tdir in dirs:
        for f in sorted(glob.glob(os.path.join(tdir, "*.json"))):
            d = json.load(open(f))
            msgs = d.get("messages") or []
            asst = [m for m in msgs if m["role"] == "assistant"]
            if not asst or not any((m.get("reasoning_content") or "").strip() for m in asst):
                continue
            yield d.get("lang") or tdir.rstrip("/").rsplit("_", 1)[-1], d.get("task", os.path.basename(f)), \
                d.get("repo", ""), msgs


def rr(groups, rnd):
    """Deterministic round-robin stream over {group: {task: [windows]}}: groups in seeded order, tasks in seeded
    order inside a group, windows in seeded order inside a task. Yields (group, task, window)."""
    order = sorted(groups)
    rnd.shuffle(order)
    tasks = {}
    for g in order:
        ts = sorted(groups[g])
        rnd.shuffle(ts)
        tasks[g] = [[t, list(groups[g][t])] for t in ts]
        for t in tasks[g]:
            rnd.shuffle(t[1])
    pos = Counter()
    while any(any(w for _, w in tasks[g]) for g in order):
        for g in order:
            live = [t for t in tasks[g] if t[1]]
            if not live:
                continue
            t = live[pos[g] % len(live)]
            pos[g] += 1
            yield g, t[0], t[1].pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib53", required=True)
    ap.add_argument("--gen", nargs="+", required=True, help="generated turn files or globs (mix_v1/v2/v3 shards)")
    ap.add_argument("--agent-dirs", nargs="+", required=True, help="agent trajectory dirs (traj_py traj_js mix_v3/traj_py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pool", type=int, default=1500)
    ap.add_argument("--held-per-lang", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run:
        for n in ("calib_search.pt", "calib_heldout.pt"):
            if os.path.exists(os.path.join(a.out, n)):
                raise SystemExit(f"{a.out}/{n} exists, refusing to overwrite")
    tok = AutoTokenizer.from_pretrained(a.model)
    # bucket -> group -> task -> windows ; group = repo (agentic), language (code), source (math / general)
    G = {b: defaultdict(lambda: defaultdict(list)) for b in BUCKETS}
    meta_of = {}
    bad = Counter()

    def add(b, group, task, msgs, meta):
        try:
            text, spans = render(tok, msgs)
        except Exception:
            bad[b] += 1
            return
        w = windows(tok, text, spans, a.seq)
        if w:
            G[b][group][task].extend(w)
            meta_of[task] = meta

    for lang, task, repo, msgs in agent_sessions(a.agent_dirs):
        add("agentic", repo or lang, ("agentic", task), msgs, {"bucket": "agentic", "lang": lang, "repo": repo,
                                                                "source": "agent"})
    for lang, task, repo, msgs in agent_sessions([os.path.join(a.calib53, "traj_poly")]):
        lang = LANG.get(lang, lang)
        add("code", lang, ("poly", task), msgs, {"bucket": "code", "lang": lang, "repo": repo, "source": "traj_poly"})
    files = sorted({f for g in a.gen for f in glob.glob(g)})
    for b, lang, sid, split, src, msgs in v1.generated_sessions(files):
        b = "code" if b.startswith("code") else b
        lang = LANG.get(lang, lang)
        group = lang if b == "code" else src
        add(b, group, ("gen", sid), msgs, {"bucket": b, "lang": lang, "repo": "", "source": src})
    rep = {"files": files, "render_failures": dict(bad), "windows": {}, "groups": {}}
    for b in BUCKETS:
        rep["windows"][b] = sum(len(w) for g in G[b].values() for w in g.values())
        rep["groups"][b] = {g: {"tasks": len(ts), "windows": sum(len(w) for w in ts.values())} for g, ts in sorted(G[b].items())}
    rnd = random.Random(a.seed)

    # ---- held-out: whole tasks, per group, deterministic
    held_quota = dict(HELD)
    held_quota["code"] = a.held_per_lang * len(G["code"])
    held, held_tasks = [], set()
    for b in BUCKETS:
        groups = sorted(G[b])
        if b == "code":  # held_per_lang windows from each language, whole tasks
            per = {g: a.held_per_lang for g in groups}
        else:            # spread over the groups (repositories / sources)
            per = {g: held_quota[b] // len(groups) + (1 if i < held_quota[b] % len(groups) else 0)
                   for i, g in enumerate(groups)}
        for g in groups:
            ts = sorted(G[b][g])
            rnd.shuffle(ts)
            # smallest set of whole tasks (in seeded order) covering per[g] windows; keep >= 1 task for the pool
            got, chosen = 0, []
            for t in ts[:-1]:
                if got >= per[g]:
                    break
                chosen.append(t)
                got += len(G[b][g][t])
            stream = rr({g: {t: G[b][g][t] for t in chosen}}, random.Random(f"{a.seed}-held-{b}-{g}"))
            sel = [(t, w) for _, t, w in stream][:per[g]]
            if len(sel) < per[g]:
                print(f"WARNING held-out {b}/{g}: {len(sel)} of {per[g]} windows")
            held += [(b, t, w) for t, w in sel]
            held_tasks |= set(chosen)
            for t in chosen:
                del G[b][g][t]
    # ---- pool in prefix-stable order
    streams = {b: rr(G[b], random.Random(f"{a.seed}-pool-{b}")) for b in SHARES}
    avail = {b: sum(len(w) for g in G[b].values() for w in g.values()) for b in SHARES}
    n_exact = min(int(avail[b] / SHARES[b]) for b in SHARES)
    n_pool = min(a.max_pool, n_exact)
    cnt, pool = Counter(), []
    for i in range(n_pool):  # largest deficit: the bucket furthest below its share of i+1 goes next
        b = max(SHARES, key=lambda k: (SHARES[k] * (i + 1) - cnt[k], SHARES[k]))
        g, t, w = next(streams[b])
        pool.append((b, t, w))
        cnt[b] += 1
    rep.update({"available_after_heldout": avail, "largest_exact_N": n_exact, "pool": n_pool,
                "pool_buckets": dict(cnt), "heldout": len(held),
                "heldout_buckets": dict(Counter(b for b, _, _ in held)),
                "heldout_code_langs": dict(Counter(meta_of[t]["lang"] for b, t, _ in held if b == "code")),
                "pool_code_langs_at": {}})
    for N in (256, 512, 768, 1024, 1280, 1500):
        if N <= n_pool:
            rep["pool_code_langs_at"][N] = dict(sorted(Counter(meta_of[t]["lang"] for b, t, _ in pool[:N]
                                                               if b == "code").items()))
    assert not ({t for _, t, _ in pool} & held_tasks), "pool and held-out share a task"
    print(json.dumps({k: rep[k] for k in ("windows", "available_after_heldout", "largest_exact_N", "pool",
                                          "pool_buckets", "heldout", "heldout_buckets", "heldout_code_langs")}))
    if a.dry_run:
        return
    os.makedirs(a.out, exist_ok=True)
    for name, sel in (("calib_search.pt", pool), ("calib_heldout.pt", held)):
        ids = torch.tensor([w[0] for _, _, w in sel], dtype=torch.long)
        mask = torch.tensor([w[1] for _, _, w in sel], dtype=torch.bool)
        meta = [dict(meta_of[t], task=t[1], order=i) for i, (b, t, w) in enumerate(sel)]
        torch.save({"ids": ids, "mask": mask, "meta": meta}, os.path.join(a.out, name))
    json.dump(rep, open(os.path.join(a.out, "pool_report.json"), "w"), indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
