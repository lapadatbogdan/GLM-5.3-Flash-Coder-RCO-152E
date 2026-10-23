#!/usr/bin/env python3
"""The chosen calibration size N as a normal calibration dir (CPU, seconds), so the existing jobs run on it unchanged:
    CALIB/N{N}/calib_search.pt   first N windows of the prefix-ordered pool (build_mix3.py)
    CALIB/N{N}/calib_heldout.pt  the fixed held-out set (copy)
    CALIB/N{N}/ref_top20.pt      the pool reference sliced to those windows (same format as rco_glm_prune's)
    REAP_DIR/stats.pt + mask_{freq,reap_mean,reap_sum}_k{144,120,96}.pt   from the per-window REAP sums of the prefix
                                 (same masks as reap_score.py would give on CALIB/N{N})
    REAP_DIR/mask_list.json      for eval_masks_mn.py
Refuses to overwrite.   make_prefix.py --calib-dir $P/calib53/mix_v3 --n 768 --reap-dir $P/rco_runs/reap_mix3_N768
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "/work/glm53/rco")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--reap-dir", required=True)
    ap.add_argument("--keeps", default="144,120,96")
    a = ap.parse_args()
    from convergence import keep_top
    dst = os.path.join(a.calib_dir, f"N{a.n}")
    for p in (dst, a.reap_dir):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists, refusing to overwrite")
    S = torch.load(os.path.join(a.calib_dir, "calib_search.pt"), weights_only=False)
    H = torch.load(os.path.join(a.calib_dir, "calib_heldout.pt"), weights_only=False)
    R = torch.load(os.path.join(a.calib_dir, "ref_top20.pt"), weights_only=False)
    W = torch.load(os.path.join(a.calib_dir, "reap_windows.pt"), weights_only=False)
    n = a.n
    assert n <= S["ids"].size(0), f"pool has {S['ids'].size(0)} windows"
    assert [m["order"] for m in S["meta"][:n]] == list(range(n))
    os.makedirs(dst)
    torch.save({"ids": S["ids"][:n].clone(), "mask": S["mask"][:n].clone(), "meta": S["meta"][:n]},
               os.path.join(dst, "calib_search.pt"))
    torch.save(H, os.path.join(dst, "calib_heldout.pt"))
    torch.save({"search": R["search"][:n], "heldout": R["heldout"], "freq": R["freq"][:n].clone(),
                "score": R["score"][:n].clone(), "freq_h": R["freq_h"], "score_h": R["score_h"]},
               os.path.join(dst, "ref_top20.pt"))
    s = W["search"][:n].double().sum(0)
    st = {"count": s[0], "gsum": s[1], "reap": s[2], "done": n}
    os.makedirs(a.reap_dir)
    torch.save(st, os.path.join(a.reap_dir, "stats.pt"))
    scores = {"freq": st["count"], "reap_mean": st["reap"] / st["count"].clamp(min=1), "reap_sum": st["reap"]}
    ml = {}
    for kp in [int(x) for x in a.keeps.split(",")]:
        for name, sc in scores.items():
            p = os.path.join(a.reap_dir, f"mask_{name}_k{kp}.pt")
            torch.save(keep_top(sc, kp), p)
            ml[f"{name}_k{kp}"] = p
    json.dump(ml, open(os.path.join(a.reap_dir, "mask_list.json"), "w"), indent=1)
    print(f"{dst}: search {n}, held-out {H['ids'].size(0)}; {a.reap_dir}: {len(ml)} masks")


if __name__ == "__main__":
    main()
