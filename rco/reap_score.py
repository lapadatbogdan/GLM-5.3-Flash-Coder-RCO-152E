#!/usr/bin/env python3
"""REAP-style expert scores for GLM-5.3-Flash on our own code sessions, then held-out KL of the masks they give.

For every sparse layer and expert, over the 512 search windows (full model, deployment routing):
    count[e]   tokens routed to e
    gsum[e]    sum of its routing weight g
    reap[e]    sum of g * ||f_e(x)||   (f_e = the expert's output before the weight)
Masks (keep K per layer):
    freq       most often chosen (the baseline that beat RCO: 0.94 at 144, 1.47 at 96)
    reap_mean  REAP (Lasby et al.): mean of g * ||f|| over the tokens routed to the expert
    reap_sum   total contribution: sum of g * ||f|| (frequency x strength)
Each mask is scored with the same held-out KL as the RCO runs (deployment routing, model-written tokens).
Deterministic: no sampling, one forward pass over the data. Stats are saved every 32 windows (resumable).

    reap_score.py --model .../full --calib-dir .../calib53 --out .../rco_runs/reap --keeps 144,96
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "src"))
sys.path.insert(0, str(_P(__file__).resolve().parent))

import argparse
import json
import logging
import os
import time

import torch
import torch.nn.functional as F

import glm5_next_rco as glm
from common import get_input_device
from rco_glm_prune import eval_mask

log = logging.getLogger("reap")


def install_stats(model, n_exp):
    """Wrap each sparse layer's FP8 experts so the forward also accumulates count / gsum / reap per expert."""
    blocks = glm.moe_blocks(model)
    st = {k: torch.zeros(len(blocks), n_exp, dtype=torch.float64) for k in ("count", "gsum", "reap")}
    for li, (_, blk) in enumerate(blocks):
        ex = blk.experts

        def fwd(hidden_states, top_k_index, top_k_weights, ex=ex, li=li):
            final = torch.zeros_like(hidden_states)
            mask = F.one_hot(top_k_index, num_classes=ex.num_experts).permute(2, 1, 0)
            hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
            for e in hit[:, 0].tolist():
                pos, tok = torch.where(mask[e])
                cur = glm._Fp8Linear.apply(hidden_states[tok], ex.gate_up[e], ex.gate_up_s[e], ex.block)
                g, u = cur.chunk(2, dim=-1)
                g = g.clamp(max=ex.swiglu_limit)
                u = u.clamp(min=-ex.swiglu_limit, max=ex.swiglu_limit)
                cur = glm._Fp8Linear.apply(F.silu(g) * u, ex.down[e], ex.down_s[e], ex.block)
                w = top_k_weights[tok, pos]
                st["count"][li, e] += tok.numel()
                st["gsum"][li, e] += w.float().sum().item()
                st["reap"][li, e] += (w.float() * cur.float().norm(dim=-1)).sum().item()
                final.index_add_(0, tok, (cur * w[:, None]).to(final.dtype))
            return final
        ex.forward = fwd
    return st


def keep_top(score, keep):
    """bool (L, E) prune mask, True = pruned: keep the `keep` highest scores per layer."""
    m = torch.ones_like(score, dtype=torch.bool)
    idx = score.topk(keep, dim=1).indices
    m.scatter_(1, idx, False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keeps", default="144,96")
    ap.add_argument("--masks", default="freq,reap_mean,reap_sum", help="which masks to score")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(a.out, "log.txt")), logging.StreamHandler()])
    S = torch.load(os.path.join(a.calib_dir, "calib_search.pt"))
    H = torch.load(os.path.join(a.calib_dir, "calib_heldout.pt"))
    if a.limit:
        S = {k: v[:a.limit] for k, v in S.items()}
        H = {k: v[:a.limit] for k, v in H.items()}
    t0 = time.time()
    model = glm.load_glm_fp8(a.model, n_layers=a.n_layers)
    n_exp = model.config.num_local_experts
    log.info(f"loaded in {time.time() - t0:.0f}s")

    stats_path = os.path.join(a.out, "stats.pt")
    st = install_stats(model, n_exp)
    done = 0
    if os.path.exists(stats_path):
        c = torch.load(stats_path)
        for k in st:
            st[k].copy_(c[k])
        done = c["done"]
        log.info(f"resumed at window {done}")
    dev = get_input_device(model)
    N = S["ids"].size(0)
    t1 = time.time()
    with torch.no_grad():
        for i in range(done, N):
            model(input_ids=S["ids"][i:i + 1].to(dev))
            if (i + 1) % 32 == 0 or i + 1 == N:
                tmp = stats_path + ".tmp"
                torch.save({**st, "done": i + 1}, tmp)
                os.replace(tmp, stats_path)
                log.info(f"stats {i + 1}/{N} windows, {time.time() - t1:.0f}s")
    # back to the stock experts forward for the KL evaluation (hard_prune_hooks only change the router)
    for _, blk in glm.moe_blocks(model):
        del blk.experts.forward

    cnt, gs, rp = st["count"], st["gsum"], st["reap"]
    scores = {"freq": cnt, "reap_mean": rp / cnt.clamp(min=1), "reap_sum": rp}
    scores = {n: scores[n] for n in ["freq"] + [m for m in a.masks.split(",") if m != "freq"]}  # freq: overlap reference
    ref_path = os.path.join(a.calib_dir, "ref_top20.pt" if not (a.limit or a.n_layers) else "ref_top20_test.pt")
    rh = torch.load(ref_path)["heldout"]
    res = {"windows": N, "keeps": {}}
    for kp in [int(x) for x in a.keeps.split(",")]:
        res["keeps"][kp] = {}
        masks = {n: keep_top(s, kp) for n, s in scores.items()}
        for n, m in masks.items():
            torch.save(m, os.path.join(a.out, f"mask_{n}_k{kp}.pt"))
            ov = ((~m) & (~masks["freq"])).sum().item() / (kp * m.size(0))
            kl = eval_mask(model, m, H["ids"], H["mask"], rh)
            res["keeps"][kp][n] = {"heldout_kl": kl, "overlap_with_freq": ov}
            log.info(f"keep {kp} {n:9s}: held-out KL {kl:.4f}  (same experts as freq: {ov:.0%})")
            json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    log.info("DONE")


if __name__ == "__main__":
    main()
