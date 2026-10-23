#!/usr/bin/env python3
"""General-text calibration windows for the "generalist experts" test (calib53/general_v1), CPU, ~3 min.

Reuses what already exists: the general sessions the FULL model wrote for mix_v1 and mix_v3 (Dolly open/general QA
+ FineWeb-Edu explanations, ~290K model-written tokens, thinking on), rendered and windowed exactly like every
calibration set (build_calib.render/windows, 1024 tokens, mask = model-written tokens, >= 20%).
Excluded: every task that is in the mix_v3 held-out set (the 25 general monitor windows stay unseen), and any session
whose prompt shares an 8-word shingle with the IFEval items of stdbench/data/ifeval.jsonl.
Writes calib_search.pt (all general windows) and calib_heldout.pt (ONE mix_v3 held-out window: rco_prep_v2_mn.py
needs a held-out file; one window costs ~20 s). Refuses to overwrite.
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import torch
from transformers import AutoTokenizer

P = "/work/glm53"
sys.path.insert(0, P + "/calib53/mix_v1")
sys.path.insert(0, P + "/calib53")
import build_mix as v1  # noqa: E402
from build_calib import render, windows  # noqa: E402

OUT = P + "/calib53/general_v1"
for n in ("calib_search.pt", "calib_heldout.pt"):
    if os.path.exists(os.path.join(OUT, n)):
        raise SystemExit(f"{OUT}/{n} exists, refusing to overwrite")
H = torch.load(P + "/calib53/mix_v3/calib_heldout.pt", weights_only=False)
held_tasks = {m["task"] for m in H["meta"]}


def sh8(t):
    w = re.findall(r"[a-z0-9]+", t.lower())
    return {" ".join(w[i:i + 8]) for i in range(len(w) - 7)}


ife = set()
for l in open(P + "/stdbench/data/ifeval.jsonl"):
    ife |= sh8(json.loads(l)["prompt"])
tok = AutoTokenizer.from_pretrained(P + "/full")
files = sorted(glob.glob(P + "/calib53/mix_v1/gen/turns_shard*.jsonl") + glob.glob(P + "/calib53/mix_v3/gen/turns_shard*.jsonl"))
ids, masks, meta, drop = [], [], [], Counter()
for b, lang, sid, split, src, msgs in v1.generated_sessions(files):
    if b != "general":
        continue
    if sid in held_tasks:
        drop["mix_v3_heldout_task"] += 1
        continue
    if any(sh8(m["content"]) & ife for m in msgs if m["role"] == "user"):
        drop["ifeval_overlap"] += 1
        continue
    text, spans = render(tok, msgs)
    for w_ids, w_mask in windows(tok, text, spans, 1024):
        ids.append(w_ids)
        masks.append(w_mask)
        meta.append({"bucket": "general", "lang": "general", "source": src, "task": sid, "repo": ""})
os.makedirs(OUT, exist_ok=True)
mask_t = torch.tensor(masks, dtype=torch.bool)
torch.save({"ids": torch.tensor(ids, dtype=torch.long), "mask": mask_t, "meta": meta}, os.path.join(OUT, "calib_search.pt"))
torch.save({"ids": H["ids"][:1].clone(), "mask": H["mask"][:1].clone(), "meta": H["meta"][:1]},
           os.path.join(OUT, "calib_heldout.pt"))
print(f"general windows {len(ids)} from {len({m['task'] for m in meta})} sessions "
      f"({dict(Counter(m['source'] for m in meta))}); model-written tokens {int(mask_t.sum())} "
      f"({mask_t.float().mean():.1%}); dropped {dict(drop)}")
