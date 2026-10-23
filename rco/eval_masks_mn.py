#!/usr/bin/env python3
"""Per-window held-out KL of many prune masks, sharded over independent jobs (DSA fix, deployment routing via
hard_prune_hooks, the same kl_topk as rco_glm_prune.eval_mask). Job i of n takes held-out windows i, i+n, ...; for
every mask in --mask-list it records (KL, model-written tokens) per window, so the totals and any per-domain /
per-language breakdown can be recomputed exactly: eval_mask's value = sum(kl*w) / sum(w).
Incremental + resumable: OUT/kl_shard{i}of{n}.pt after every mask.

    eval_masks_mn.py --model $P/full --calib-dir $P/calib53/mix_v3 --mask-list masks.json --out DIR --shard 0 --nshards 8
mask-list: JSON {name: path_to_bool_mask (n_sparse, 288), ...}. RCO_CPU=1: model on CPU (test only).
"""
import argparse
import json
import logging
import os
import sys
import time

P = "/work/glm53"
for p in (P + "/heal", P + "/rco", P + "/rco/src"):
    sys.path.insert(0, p)
import torch

log = logging.getLogger("eval_masks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--mask-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.shard}of{a.nshards}"
    path = os.path.join(a.out, f"kl_shard{tag}.pt")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(a.out, f"log_shard{tag}.txt")), logging.StreamHandler()])
    masks = json.load(open(a.mask_list))
    st = torch.load(path, weights_only=False) if os.path.exists(path) else {"masks": {}, "shard": a.shard,
                                                                              "nshards": a.nshards}
    todo = [n for n in masks if n not in st["masks"]]
    if not todo:
        log.info("all masks done")
        return
    import dsa_fix
    dsa_fix.install_long_context_patches(int(os.environ.get("Q_CHUNK", "256")), int(os.environ.get("KDA_GROUP", "8")))
    import glm5_next_rco as glm
    import rco_glm_prune as R
    from common import get_input_device
    H = torch.load(os.path.join(a.calib_dir, "calib_heldout.pt"), weights_only=False)
    test = a.limit or a.n_layers
    rh = torch.load(os.path.join(a.calib_dir, "ref_top20_test.pt" if test else "ref_top20.pt"), weights_only=False)["heldout"]
    n = min(a.limit, H["ids"].size(0)) if a.limit else H["ids"].size(0)
    wins = list(range(a.shard, n, a.nshards))
    t0 = time.time()
    devices = [torch.device("cpu")] if os.environ.get("RCO_CPU") == "1" else None
    model = glm.load_glm_fp8(a.model, devices=devices, n_layers=a.n_layers)
    dev = get_input_device(model)
    log.info(f"loaded in {time.time() - t0:.0f}s; {len(todo)} masks x {len(wins)} windows")
    with torch.no_grad():
        for name in todo:
            m = torch.load(masks[name], weights_only=False)
            hooks = glm.hard_prune_hooks(model, m)
            rows = {}
            try:
                for i in wins:
                    kl = R.kl_topk(model(input_ids=H["ids"][i:i + 1].to(dev)).logits, rh[i][0], rh[i][1],
                                   H["mask"][i:i + 1])
                    rows[i] = (kl.item(), H["mask"][i, 1:].float().sum().item())
            finally:
                glm.remove_hard_prune_hooks(hooks)
            st["masks"][name] = rows
            torch.save(st, path + ".tmp")
            os.replace(path + ".tmp", path)
            tot = sum(k * w for k, w in rows.values()) / max(1, sum(w for _, w in rows.values()))
            log.info(f"{name}: {len(rows)} windows, shard KL {tot:.4f}, {time.time() - t0:.0f}s")
    log.info("DONE")


if __name__ == "__main__":
    main()
