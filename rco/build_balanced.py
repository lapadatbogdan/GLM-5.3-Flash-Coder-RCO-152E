#!/usr/bin/env python3
"""Domain/language-BALANCED expert masks for GLM-5.3 (CPU, seconds), from the saved per-window REAP sums.

Why: the REAP-sum k144 mask sums reap over all windows, so groups with many windows dominate (N1270: ~35% Python
agentic, each of the 21 code languages ~1.7%, general 0%): C++/Java/Rust/JS and generalist experts were cut.

Data (never the 256 held-out windows): calib53/mix_v3/reap_windows.pt "search" (the whole mix_v3 pool: agentic,
21 code languages, math; the held-out set is task-disjoint by construction) + calib53/general_v1/reap_windows.pt
"search" (general sessions, task-disjoint from the held-out monitor). [:, 2] = reap sum per window (g * ||f||).

Groups (finest level): agentic per repository, code per language (21), math per source (gsm8k, MATH), general per
source (dolly, fineweb_edu). Per group g: s_g[l, e] = mean reap per window, normalized per layer to sum 1
(= the group's REAP mass distribution over the experts of that layer). Domain weights (default) agentic 0.35,
code 0.35, math 0.15, general 0.15, each split EQUALLY over the domain's groups; combined score S = sum_g w_g s_g.
Masks (True = pruned, keep exactly K per layer):
  bal_k{K}     (a) top-K of S per layer
  balq_k{K}    (b) quota: every group keeps its own top-q experts per layer (union), the rest filled by S
Reports per mask: overlap with base_k144 (kept-set |A & B| / 144, mean over layers) and per-group coverage =
share of the group's normalized REAP mass kept (mean over layers), for base_k144, gen_k152 and every new mask.
Writes OUT/mask_*.pt, OUT/mask_list.json (eval_masks_mn), OUT/report.json. Refuses to overwrite.
"""
import argparse
import json
import os
from collections import defaultdict

import torch

P = "/work/glm53"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=P + "/rco_runs/balanced_v1")
ap.add_argument("--keeps", default="144,160")
ap.add_argument("--weights", default="agentic=0.35,code=0.35,math=0.15,general=0.15")
ap.add_argument("--quota", default="144=3,160=4", help="K=q: experts per layer guaranteed to every group")
a = ap.parse_args()
if os.path.exists(a.out):
    raise SystemExit(f"{a.out} exists, refusing to overwrite")
W = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
Q = {int(k): int(v) for k, v in (x.split("=") for x in a.quota.split(","))}
L = lambda p: torch.load(p, weights_only=False)

MS = L(P + "/calib53/mix_v3/calib_search.pt")["meta"]
MW = L(P + "/calib53/mix_v3/reap_windows.pt")["search"][:, 2].double()       # (n, 42, 288)
GS = L(P + "/calib53/general_v1/calib_search.pt")["meta"]
GW = L(P + "/calib53/general_v1/reap_windows.pt")["search"][:, 2].double()
assert MW.size(0) == len(MS) and GW.size(0) == len(GS)
H = L(P + "/calib53/mix_v3/calib_heldout.pt")["meta"]
held = {m["task"] for m in H}
assert not held & {m["task"] for m in MS} and not held & {m["task"] for m in GS}, "held-out task in the data"


def group(m):
    b = m["bucket"]
    if b == "agentic":
        return "agentic", "agentic/" + (m.get("repo") or m["lang"])
    if b == "code":
        return "code", "code/" + m["lang"]
    return b, f"{b}/{m['source']}"


idx = defaultdict(list)
dom = {}
for src, metas in (("mix", MS), ("gen", GS)):
    for i, m in enumerate(metas):
        d, g = group(m)
        idx[(src, g)].append(i)
        dom[g] = d
s = {}
for (src, g), ii in idx.items():
    X = (MW if src == "mix" else GW)[torch.tensor(ii)].mean(0)                 # mean reap per window (L, E)
    s[g] = X / X.sum(1, keepdim=True).clamp(min=1e-30)                           # per-layer mass distribution
groups = sorted(s)
n_dom = defaultdict(int)
for g in groups:
    n_dom[dom[g]] += 1
w = {g: W[dom[g]] / n_dom[dom[g]] for g in groups}
S = sum(w[g] * s[g] for g in groups)
Lr, E = S.shape


def keep_top(score, k):
    m = torch.ones(Lr, E, dtype=torch.bool)
    m.scatter_(1, score.topk(k, dim=1).indices, False)
    return m


def quota_mask(k, q):
    keep = torch.zeros(Lr, E, dtype=torch.bool)
    for g in groups:
        keep.scatter_(1, s[g].topk(q, dim=1).indices, True)
    nq = keep.sum(1)
    assert (nq <= k).all(), f"quota union {int(nq.max())} > {k}: lower q"
    rest = S.masked_fill(keep, float("-inf"))
    for l in range(Lr):
        keep[l, rest[l].topk(k - int(nq[l])).indices] = True
    return ~keep, nq


masks = {"base_k144": L(P + "/rco_runs/reap_mix3_N1270/mask_reap_sum_k144.pt")}
gk = P + "/rco_runs/general_plus/mask_gen_k152.pt"
if os.path.exists(gk):
    masks["gen_k152"] = L(gk)
info = {"weights": W, "group_weight": w, "windows": {g: len(idx[k]) for k in idx for g in [k[1]]}}
for k in [int(x) for x in a.keeps.split(",")]:
    masks[f"bal_k{k}"] = keep_top(S, k)
    masks[f"balq_k{k}"], nq = quota_mask(k, Q[k])
    info[f"balq_k{k}_quota_union"] = {"q": Q[k], "min": int(nq.min()), "max": int(nq.max()), "mean": nq.float().mean().item()}
base_keep = ~masks["base_k144"]
rep = {}
for name, m in masks.items():
    kp = ~m
    ov = ((kp & base_keep).sum(1).double() / 144).mean().item()
    cov = {g: (s[g] * kp).sum(1).mean().item() for g in groups}
    rep[name] = {"kept_per_layer": int(kp.sum(1)[0]), "overlap_with_base_k144": ov, "coverage": cov}
os.makedirs(a.out)
ml = {}
for name, m in masks.items():
    p = os.path.join(a.out, f"mask_{name}.pt")
    torch.save(m, p)
    ml[name] = p
json.dump(ml, open(os.path.join(a.out, "mask_list.json"), "w"), indent=1)
json.dump({"info": info, "report": rep}, open(os.path.join(a.out, "report.json"), "w"), indent=1)
show = ["agentic/" + r for r in ("NodeBB/NodeBB", "ansible/ansible", "qutebrowser/qutebrowser", "internetarchive/openlibrary")]
show = [g for g in show if g in s] + [f"code/{x}" for x in ("cpp", "java", "rust", "javascript", "go", "typescript", "c",
                                                             "csharp", "python", "kotlin")] + \
       ["math/gsm8k", "math/math", "general/dolly", "general/fineweb_edu"]
show = [g for g in show if g in s]
print(f"groups {len(groups)}: " + ", ".join(f"{g}({len(idx[('gen' if dom[g] == 'general' else 'mix', g)])})" for g in groups))
print(f"{'mask':12s} {'K':>4} {'ov144':>6} " + " ".join(f"{g.split('/')[-1][:7]:>7}" for g in show) + "   min-group")
for name, r in rep.items():
    c = r["coverage"]
    mg = min(c, key=c.get)
    print(f"{name:12s} {r['kept_per_layer']:>4} {r['overlap_with_base_k144']:>6.3f} " + " ".join(f"{c[g]:>7.3f}" for g in show)
          + f"   {mg} {c[mg]:.3f}")
print({k: v for k, v in info.items() if k.startswith("balq")})
