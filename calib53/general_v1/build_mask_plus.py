#!/usr/bin/env python3
"""k144 REAP mask + n "generalist" experts per layer (CPU, seconds). Masks are PRUNE masks (True = pruned).
  gen_k{144+n}  : base mask with the n pruned experts of each layer that have the highest REAP sum ON GENERAL TEXT
                  (calib53/general_v1/reap_windows.pt, search windows) un-pruned
  mixreap_k152  : CONTROL, same size: base + the next 8 by the ORIGINAL mix_v3 REAP sum (stats.pt) - tells whether a
                  gain comes from general experts or just from 8 more experts
  base_k144     : the base mask itself (re-scored in the same job, for a same-run comparison)
Writes OUT/mask_*.pt + OUT/mask_list.json (for eval_masks_mn) + OUT/selection.json (chosen experts, overlap
gen vs control). Refuses to overwrite.
"""
import json
import os
import sys

import torch

P = "/work/glm53"
BASE = sys.argv[1] if len(sys.argv) > 1 else P + "/rco_runs/reap_mix3_N1270/mask_reap_sum_k144.pt"
MIXSTATS = P + "/rco_runs/reap_mix3_N1270/stats.pt"
GEN = os.environ.get("GEN", P + "/calib53/general_v1/reap_windows.pt")
OUT = sys.argv[2] if len(sys.argv) > 2 else P + "/rco_runs/general_plus"
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} exists, refusing to overwrite")
base = torch.load(BASE)
assert base.dtype == torch.bool and base.shape == (42, 288) and ((~base).sum(1) == 144).all()
g = torch.load(GEN)["search"].double().sum(0)[2]          # general reap_sum (L, E)
mix = torch.load(MIXSTATS)["reap"].double()


def add(score, n):
    m = base.clone()
    s = score.masked_fill(~base, float("-inf"))           # only currently pruned experts are candidates
    idx = s.topk(n, dim=1).indices
    m.scatter_(1, idx, False)
    return m, idx


os.makedirs(OUT)
masks, sel = {"base_k144": base}, {}
for n in (4, 8):
    m, idx = add(g, n)
    masks[f"gen_k{144 + n}"] = m
    sel[f"gen_k{144 + n}"] = idx.tolist()
m, idx = add(mix, 8)
masks["mixreap_k152"] = m
sel["mixreap_k152"] = idx.tolist()
ml = {}
for k, m in masks.items():
    p = os.path.join(OUT, f"mask_{k}.pt")
    torch.save(m, p)
    ml[k] = p
json.dump(ml, open(os.path.join(OUT, "mask_list.json"), "w"), indent=1)
gk, ck = ~masks["gen_k152"] & base, ~masks["mixreap_k152"] & base     # the 8 added per layer
same = (gk & ck).sum().item()
# how "general" are the added experts: their share of all general reap mass vs share in the mix
share = lambda m: ((g * m).sum() / g.sum()).item()
info = {"added_same_in_gen_and_control": same, "added_total": int(gk.sum()),
        "general_mass_share": {k: share(~m) for k, m in masks.items()},
        "selection": sel}
json.dump(info, open(os.path.join(OUT, "selection.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "selection"}, indent=1))
