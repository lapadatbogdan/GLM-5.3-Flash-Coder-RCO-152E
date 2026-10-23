#!/usr/bin/env python3
"""Mixed calibration windows (calib53/mix_v1): same rendering and windowing as calib53/build_calib.py (imported, not
copied: chat template with clear_thinking=False, SEQ-token windows cut from token 0, tail dropped, mask = tokens the
model wrote incl. reasoning, a window kept only if >= 20% of it is model-written).

Buckets and window counts (search / heldout, by window count):
    agentic  45%   traj_py + traj_js agent sessions (existing)                   230 / 29
    code     25%   traj_poly CommitPackFT edits (existing) 64 / 8  +  evol-codealpaca sessions (new) 64 / 8
    math     15%   GSM8K + MATH train sessions (new)                              77 / 10
    general  15%   Dolly + FineWeb-Edu sessions (new)                             77 / 9
Held-out windows come from tasks / sessions never used for the search (new sessions: split fixed at prompt time,
prompts disjoint; existing sessions: whole tasks held out, round-robin over languages, as build_calib.py).

    build_mix.py --model $P/full --calib53 $P/calib53 --gen $P/calib53/mix_v1/gen/turns.jsonl --out $P/calib53/mix_v1
writes calib_search.pt / calib_heldout.pt: {"ids": LongTensor[N, SEQ], "mask": BoolTensor[N, SEQ], "meta": [...]}
(meta adds "bucket" and "source" to build_calib's lang/task/repo). Never overwrites an existing output.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from build_calib import render, windows  # noqa: E402  (the exact functions behind calib53/calib_*.pt)

QUOTA = {  # bucket/source -> (search, heldout)
    "agentic": (230, 29),
    "code/poly": (64, 8),
    "code/evol": (64, 8),
    "math": (77, 10),
    "general": (77, 9),
}


def existing_sessions(calib53):
    """(bucket, lang, task, repo, msgs) from the agent and CommitPackFT trajectories, filtered like build_calib."""
    for tdir, bucket in (("traj_py", "agentic"), ("traj_js", "agentic"), ("traj_go", "agentic"),
                         ("traj_poly", "code/poly")):
        for f in sorted(glob.glob(os.path.join(calib53, tdir, "*.json"))):
            d = json.load(open(f))
            msgs = d.get("messages") or []
            asst = [m for m in msgs if m["role"] == "assistant"]
            if not asst or not any((m.get("reasoning_content") or "").strip() for m in asst):
                continue
            lang = d.get("lang") or tdir.rsplit("_", 1)[-1]
            yield bucket, lang, d.get("task", os.path.basename(f)), d.get("repo", ""), msgs


def generated_sessions(paths):
    """(bucket, lang, sid, split, source, msgs) from gen_mix.py's turn files (one or many shards; torn lines
    skipped; a (sid, turn) seen twice, e.g. after a re-shard, is taken once, first file wins)."""
    turns, seen = defaultdict(list), set()
    for path in paths:
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r["sid"], r["turn"]) in seen:
                continue
            seen.add((r["sid"], r["turn"]))
            turns[r["sid"]].append(r)
    for sid, rs in sorted(turns.items()):
        rs.sort(key=lambda r: r["turn"])
        rs = [r for i, r in enumerate(rs) if r["turn"] == i]  # contiguous turns only
        if not rs:
            continue
        msgs = [{"role": "system", "content": rs[0]["system"]}] if rs[0].get("system") else []
        for r in rs:
            msgs.append({"role": "user", "content": r["user"]})
            msgs.append({"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]})
        b = "code/evol" if rs[0]["bucket"] == "code" else rs[0]["bucket"]
        yield b, rs[0]["lang"], sid, rs[0]["split"], rs[0]["source"], msgs


def hold_out(per_task, n, rnd, group=lambda k: k[0], info=None, cap=None):
    """build_calib's rule: whole tasks, round-robin over languages (agentic: over repositories), until n windows."""
    keys = sorted(per_task)
    rnd.shuffle(keys)
    by_lang = defaultdict(list)
    for k in keys:
        by_lang[group(k)].append(k)
    langs, held, i = sorted(by_lang), set(), 0
    def usable():  # windows pick() can actually take from the held tasks under the per-repo cap
        if cap is None:
            return sum(len(per_task[k]) for k in held)
        per = Counter()
        for k in held:
            per[info[k]] += len(per_task[k])
        return sum(min(cap, v) for v in per.values())

    while usable() < n and any(by_lang.values()):
        l = langs[i % len(langs)]
        i += 1
        if by_lang[l]:
            held.add(by_lang[l].pop())
    return held


def pick(rows, n, info, cap, rnd):
    """build_calib's pick: round-robin over tasks, at most `cap` windows per repository."""
    bucket = defaultdict(list)
    for k, w in rows:
        bucket[k].append(w)
    order = sorted(bucket)
    rnd.shuffle(order)
    per_repo, out = Counter(), []
    while len(out) < n and any(bucket[k] for k in order if per_repo[info[k]] < cap):
        for k in order:
            if bucket[k] and len(out) < n and per_repo[info[k]] < cap:
                out.append((k, bucket[k].pop(rnd.randrange(len(bucket[k])))))
                per_repo[info[k]] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib53", required=True)
    ap.add_argument("--gen", required=True, nargs="+",
                    help="gen_mix.py turn files or globs, e.g. 'gen/turns_shard*.jsonl' (may be fake files for tests)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-repo-frac", type=float, default=0.15, help="of the TOTAL search/heldout size, per repo")
    ap.add_argument("--agentic-repo-frac", type=float, default=0.20,
                    help="agentic sessions come from only 4 repos (NodeBB 223 windows, qutebrowser, ansible, "
                         "openlibrary); at 0.15 (76/repo) 230 search windows are not reachable")
    ap.add_argument("--scale", type=float, default=1.0, help="tests: multiply every quota")
    ap.add_argument("--dry-run", action="store_true", help="count windows per bucket, write nothing")
    ap.add_argument("--allow-short", action="store_true", help="write even if a bucket misses its quota")
    a = ap.parse_args()
    quota = {k: (max(1, round(s * a.scale)), max(1, round(h * a.scale))) for k, (s, h) in QUOTA.items()}
    if not a.dry_run:
        for n in ("search", "heldout"):
            if os.path.exists(os.path.join(a.out, f"calib_{n}.pt")):
                raise SystemExit(f"{a.out}/calib_{n}.pt exists, refusing to overwrite")
        if os.path.realpath(a.out) == os.path.realpath(a.calib53):
            raise SystemExit("--out must not be calib53 itself")
    rnd = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)

    per_task = {b: defaultdict(list) for b in quota}   # bucket -> (lang, task) -> windows
    info, fixed_split = {}, {}
    bad = Counter()
    srcs = list(existing_sessions(a.calib53))
    gen_files = sorted({f for g in a.gen for f in glob.glob(g)})
    print(f"generated turn files: {gen_files}")
    if gen_files:
        srcs += [(b, l, sid, "", msgs, split, src) for b, l, sid, split, src, msgs in generated_sessions(gen_files)]
    for row in srcs:
        b, lang, task, repo, msgs = row[:5]
        if b not in quota:
            continue
        try:
            text, spans = render(tok, msgs)
        except Exception:
            bad[b] += 1
            continue
        w = windows(tok, text, spans, a.seq)
        if w:
            k = (lang, task)
            per_task[b][k].extend(w)
            info[k] = repo or task  # new sessions: one "repo" per session, so the repo cap never binds
            if len(row) > 5:
                fixed_split[k] = row[5]
    print(f"render failures per bucket: {dict(bad)}")
    short = []
    for b, pt in per_task.items():
        nw = sum(len(v) for v in pt.values())
        langs = Counter(k[0] for k, v in pt.items() for _ in v)
        print(f"{b:10s} sessions {len(pt):4d} windows {nw:5d} need {sum(quota[b])} ({quota[b][0]}+{quota[b][1]}) "
              f"langs {dict(langs)}")
    if a.dry_run:
        return

    res = {"search": ([], []), "heldout": ([], [])}
    n_tot = {"search": sum(q[0] for q in quota.values()), "heldout": sum(q[1] for q in quota.values())}
    for b, pt in per_task.items():
        ns, nh = quota[b]
        if any(k in fixed_split for k in pt):
            held = {k for k in pt if fixed_split.get(k) == "heldout"}
        else:
            if b == "agentic":
                held = hold_out(pt, nh, rnd, lambda k: info[k], info,
                                max(1, int(a.agentic_repo_frac * n_tot["heldout"])))
            else:
                held = hold_out(pt, nh, rnd)
        rows = {"heldout": [(k, w) for k in sorted(held) for w in pt[k]],
                "search": [(k, w) for k in sorted(set(pt) - held) for w in pt[k]]}
        for split, n in (("search", ns), ("heldout", nh)):
            frac = a.agentic_repo_frac if b == "agentic" else a.max_repo_frac
            cap = max(1, int(frac * n_tot[split]))
            sel = pick(rows[split], n, info, cap, rnd)
            if len(sel) < n:
                short.append(f"{b}/{split}: {len(sel)} of {n}")
            res[split][0].extend((b, k, w) for k, w in sel)
        assert not ({k for k, _ in rows["heldout"]} & {k for k, _ in rows["search"]})
    if short and not a.allow_short:
        raise SystemExit("quota not met (nothing written): " + "; ".join(short))
    os.makedirs(a.out, exist_ok=True)
    for split, (sel, _) in res.items():
        ids = torch.tensor([w[0] for _, _, w in sel], dtype=torch.long)
        mask = torch.tensor([w[1] for _, _, w in sel], dtype=torch.bool)
        meta = [{"bucket": b.split("/")[0], "source": b, "lang": k[0], "task": k[1], "repo": info[k]}
                for b, k, _ in sel]
        torch.save({"ids": ids, "mask": mask, "meta": meta}, os.path.join(a.out, f"calib_{split}.pt"))
        print(f"{split}: {len(sel)} windows; buckets {dict(Counter(m['source'] for m in meta))}; "
              f"langs {dict(Counter(m['lang'] for m in meta))}; tasks {len({(m['lang'], m['task']) for m in meta})}; "
              f"model-written tokens {mask.float().mean().item():.1%}")


if __name__ == "__main__":
    main()
