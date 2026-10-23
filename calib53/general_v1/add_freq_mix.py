#!/usr/bin/env python3
"""Adds freq/score/freq_h/score_h to a make_heal_mix.py output (it copied only search/heldout refs), in the same
window order, so RCO (--init-from freq, final frequency-mask eval) can use it. Order recomputed exactly as
make_heal_mix.py (round-robin over general sessions) and checked against the saved ids. Rewrites ref_top20.pt
atomically (tmp + rename)."""
import os, sys
from collections import defaultdict
import torch
P = "/work/glm53"
out, n_mix, n_gen = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
L = lambda p: torch.load(p, weights_only=False)
MR, GS, GR = L(P + "/calib53/mix_v3/N1270/ref_top20.pt"), L(P + "/calib53/general_v1/calib_search.pt"), L(P + "/calib53/general_v1/ref_top20.pt")
by = defaultdict(list)
for i, m in enumerate(GS["meta"]):
    by[m["task"]].append(i)
order, k, tasks = [], 0, sorted(by)
while len(order) < n_gen:
    for t in tasks:
        if k < len(by[t]) and len(order) < n_gen:
            order.append(by[t][k])
    k += 1
S = L(out + "/calib_search.pt"); R = L(out + "/ref_top20.pt")
assert torch.equal(S["ids"][n_mix:], GS["ids"][order]), "order mismatch"
assert torch.equal(S["ids"][:n_mix], L(P + "/calib53/mix_v3/N1270/calib_search.pt")["ids"][:n_mix])
idx = torch.tensor(order)
for key in ("freq", "score"):
    R[key] = torch.cat([MR[key][:n_mix], GR[key][idx]])
R["freq_h"], R["score_h"] = MR["freq_h"], MR["score_h"]
assert R["freq"].shape[0] == len(R["search"]) == S["ids"].size(0)
torch.save(R, out + "/ref_top20.pt.tmp"); os.replace(out + "/ref_top20.pt.tmp", out + "/ref_top20.pt")
print("ok", {k: tuple(v.shape) for k, v in R.items() if hasattr(v, "shape")})
