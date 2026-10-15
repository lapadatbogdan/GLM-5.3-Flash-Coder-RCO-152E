#!/usr/bin/env python3
"""RCO calibration set from GLM-5.3-Flash's own agent sessions (traj_py / traj_go / traj_js).

Each session is rendered with the model's chat template (clear_thinking=false keeps the reasoning of every
turn), tokenized, and cut into SEQ-token windows. The mask marks the tokens the model wrote itself (its
<think> reasoning and its answers); the prompt, tool outputs and template tags are context only.
Held-out windows come from sessions (tasks) never used for the search, so the check is not a leak.

    python build_calib.py --model .../full --out .../calib53 [--seq 1024 --n-search 512 --n-heldout 64]
writes calib_search.pt and calib_heldout.pt: {"ids": LongTensor[N, SEQ], "mask": BoolTensor[N, SEQ], "meta": [...]}
"""
import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer


def sessions(root):
    # agent sessions (traj_py / traj_go / traj_js) and the multi-language CommitPackFT edits (traj_poly)
    for tdir in sorted(glob.glob(os.path.join(root, "traj_*"))):
        if tdir.endswith(("_smoke",)) or not os.path.isdir(tdir):
            continue
        for f in sorted(glob.glob(os.path.join(tdir, "*.json"))):
            d = json.load(open(f))
            lang = d.get("lang") or tdir.rsplit("_", 1)[-1]
            msgs = d.get("messages") or []
            asst = [m for m in msgs if m["role"] == "assistant"]
            # a session without the model's reasoning is not what we calibrate on
            if not asst or not any((m.get("reasoning_content") or "").strip() for m in asst):
                continue
            yield lang, d.get("task", os.path.basename(f)), d.get("repo", ""), msgs


def render(tok, msgs):
    """Full text + the character spans of every assistant turn (from prefix renders)."""
    full = tok.apply_chat_template(msgs, tokenize=False, clear_thinking=False)
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        before = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True, clear_thinking=False)
        upto = tok.apply_chat_template(msgs[:i + 1], tokenize=False, clear_thinking=False)
        start = before.removesuffix("<think>")  # the generation prompt opens the think block
        if not full.startswith(upto) or not upto.startswith(start):
            raise ValueError("template is not prefix-stable")
        spans.append((len(start), len(upto)))
    return full, spans


def windows(tok, text, spans, seq):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    own = [any(a <= s < b for a, b in spans) for s, _ in offs]
    out = []
    for i in range(0, len(ids) - seq + 1, seq):
        m = own[i:i + seq]
        if sum(m) >= seq // 5:  # at least 20% of the window written by the model
            out.append((ids[i:i + seq], m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--n-search", type=int, default=512)
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-repo-frac", type=float, default=0.15, help="at most this share of windows from one repo")
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)

    per_task = defaultdict(list)  # (lang, task) -> windows
    info = {}
    bad = 0
    for lang, task, repo, msgs in sessions(args.out):
        try:
            text, spans = render(tok, msgs)
        except Exception as e:
            bad += 1
            continue
        w = windows(tok, text, spans, args.seq)
        if w:
            per_task[(lang, task)].extend(w)
            info[(lang, task)] = repo
    print(f"sessions with windows: {len(per_task)}, render failures: {bad}")
    print("windows per language:", dict(Counter(l for (l, _), w in per_task.items() for _ in w)))

    # held-out = whole tasks, spread over languages, until n_heldout windows
    keys = sorted(per_task)
    rnd.shuffle(keys)
    held, search = [], []
    held_tasks = set()
    by_lang = defaultdict(list)
    for k in keys:
        by_lang[k[0]].append(k)
    langs = sorted(by_lang)
    i = 0
    while sum(len(per_task[k]) for k in held_tasks) < args.n_heldout and any(by_lang.values()):
        l = langs[i % len(langs)]
        i += 1
        if by_lang[l]:
            held_tasks.add(by_lang[l].pop())
    for k in keys:
        (held if k in held_tasks else search).extend((k, w) for w in per_task[k])

    def pick(rows, n):
        # round-robin over tasks so no single long session dominates, and no repository above max_repo_frac
        bucket = defaultdict(list)
        for k, w in rows:
            bucket[k].append(w)
        order = sorted(bucket)
        rnd.shuffle(order)
        cap = max(1, int(args.max_repo_frac * n))
        per_repo = Counter()
        out = []
        while len(out) < n and any(bucket[k] for k in order if per_repo[info[k]] < cap):
            for k in order:
                if bucket[k] and len(out) < n and per_repo[info[k]] < cap:
                    out.append((k, bucket[k].pop(rnd.randrange(len(bucket[k])))))
                    per_repo[info[k]] += 1
        return out

    for name, rows, n in (("search", search, args.n_search), ("heldout", held, args.n_heldout)):
        sel = pick(rows, n)
        ids = torch.tensor([w[0] for _, w in sel], dtype=torch.long)
        mask = torch.tensor([w[1] for _, w in sel], dtype=torch.bool)
        meta = [{"lang": k[0], "task": k[1], "repo": info[k]} for k, _ in sel]
        torch.save({"ids": ids, "mask": mask, "meta": meta}, os.path.join(args.out, f"calib_{name}.pt"))
        print(f"{name}: {len(sel)} windows (asked {n}) from {len({m['task'] for m in meta})} tasks; "
              f"languages {dict(Counter(m['lang'] for m in meta))}; repos {dict(Counter(m['repo'] for m in meta))}; "
              f"model-written tokens {mask.float().mean().item():.1%}")


if __name__ == "__main__":
    main()
