#!/usr/bin/env python3
"""Calibration dir for the general-text heal continuation (CPU, seconds): N_MIX windows of mix_v3/N1270 (its prefix:
build_mix3 ordered the pool so every prefix keeps agentic/code/math 45/35/20 and balanced languages) + N_GEN general
windows of calib53/general_v1 (round-robin over its sessions; those sessions are task-disjoint from the held-out set
and share no 8-gram with IFEval), with the matching full-model top-20 references:
  OUT/calib_search.pt  N_MIX + N_GEN windows, meta["bucket"] = agentic/code/math/general (heal_v3 shuffles per epoch)
  OUT/calib_heldout.pt the same 256 mix_v3 held-out windows (so before/after numbers compare with every earlier heal)
  OUT/ref_top20.pt     {"search": refs in that order, "heldout": N1270's held-out refs}
   make_heal_mix.py --n-mix 672 --n-gen 96 --out calib53/heal_mixg_v1      (12.5% general, 96 steps of 8 windows)
Refuses to overwrite."""
import argparse
import os
from collections import defaultdict

import torch

P = "/work/glm53"
ap = argparse.ArgumentParser()
ap.add_argument("--n-mix", type=int, default=672)
ap.add_argument("--n-gen", type=int, default=96)
ap.add_argument("--mix-dir", default=P + "/calib53/mix_v3/N1270")
ap.add_argument("--gen-dir", default=P + "/calib53/general_v1")
ap.add_argument("--out", default=P + "/calib53/heal_mixg_v1")
a = ap.parse_args()
if os.path.exists(a.out):
    raise SystemExit(f"{a.out} exists, refusing to overwrite")
L = lambda p: torch.load(p, weights_only=False)
MS, MH, MR = L(a.mix_dir + "/calib_search.pt"), L(a.mix_dir + "/calib_heldout.pt"), L(a.mix_dir + "/ref_top20.pt")
GS, GR = L(a.gen_dir + "/calib_search.pt"), L(a.gen_dir + "/ref_top20.pt")
assert a.n_mix <= MS["ids"].size(0) and len(GR["search"]) == GS["ids"].size(0)
by = defaultdict(list)
for i, m in enumerate(GS["meta"]):
    by[m["task"]].append(i)
order, k = [], 0
tasks = sorted(by)
while len(order) < min(a.n_gen, GS["ids"].size(0)):           # round-robin over sessions: spread, deterministic
    for t in tasks:
        if k < len(by[t]) and len(order) < a.n_gen:
            order.append(by[t][k])
    k += 1
assert len(order) == a.n_gen, f"only {len(order)} general windows"
ids = torch.cat([MS["ids"][:a.n_mix], GS["ids"][order]])
mask = torch.cat([MS["mask"][:a.n_mix], GS["mask"][order]])
meta = MS["meta"][:a.n_mix] + [dict(GS["meta"][i], order=a.n_mix + j) for j, i in enumerate(order)]
refs = list(MR["search"][:a.n_mix]) + [GR["search"][i] for i in order]
os.makedirs(a.out)
torch.save({"ids": ids, "mask": mask, "meta": meta}, a.out + "/calib_search.pt")
torch.save(MH, a.out + "/calib_heldout.pt")
torch.save({"search": refs, "heldout": MR["heldout"]}, a.out + "/ref_top20.pt")
from collections import Counter
c = Counter(m["bucket"] for m in meta)
print(f"{a.out}: train {ids.size(0)} windows {dict(c)} (general {c['general'] / ids.size(0):.1%}, "
      f"from {len({GS['meta'][i]['task'] for i in order})} sessions); held-out {MH['ids'].size(0)}")
