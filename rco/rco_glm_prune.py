#!/usr/bin/env python3
"""RCO expert pruning for GLM-5.3-Flash on one 8-GPU node (FP8 experts), on the model's own agent sessions.

Two modes, sharing one reference:
    --mode prep    one forward pass over the search and held-out windows: the full model's top-k next-token
                   log-probs (the KL reference, ~100 MB instead of ~160 GB for the full vocabulary) and the
                   router statistics per window. Reports how much the two halves of the data agree on which
                   experts to keep (is the data enough?) and the frequency baseline's KL on the held-out windows.
    --mode search  loads that reference and runs the RCO search for --keep experts per layer; resumable
                   (run/ckpt.pt every --save-every steps). The final mask is scored on search AND held-out windows.

The KL is measured only on the tokens the model wrote itself (the calibration mask).
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "src"))

import argparse
import json
import logging
import os
import time

import torch
import torch.nn.functional as F

import glm5_next_rco as glm
from common import get_input_device
from search import prune

log = logging.getLogger("rco_glm")


# ----------------------------------------------------------------------------
# KL against a stored top-k reference
# ----------------------------------------------------------------------------


def kl_topk(logits, vals, idx, mask):
    """KL(ref || model) on the reference's top-k tokens (renormalized), mean over the masked positions.
    logits (1, T, V); vals/idx (1, T-1, k) for positions 0..T-2; mask (1, T) marks model-written tokens."""
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    dev = lp.device
    mp = lp.gather(-1, idx.to(dev).long())
    rp = vals.to(dev).float()
    rp = rp - rp.logsumexp(-1, keepdim=True)
    kl = (rp.exp() * (rp - mp)).sum(-1)
    m = mask[:, 1:].to(dev).float()
    return (kl * m).sum() / m.sum().clamp(min=1)


def _kl_loss(model, input_ids, ref, mask=None, topk=0):
    return kl_topk(model(input_ids=input_ids).logits, ref[0], ref[1], mask)


@torch.no_grad()
def eval_mask(model, prune_mask, ids, masks, ref):
    hooks = glm.hard_prune_hooks(model, prune_mask) if prune_mask is not None else []
    dev = get_input_device(model)
    tot, n = 0.0, 0.0
    try:
        for i in range(ids.size(0)):
            kl = kl_topk(model(input_ids=ids[i:i + 1].to(dev)).logits, ref[i][0], ref[i][1], masks[i:i + 1])
            w = masks[i, 1:].float().sum().item()
            tot += kl.item() * w
            n += w
    finally:
        glm.remove_hard_prune_hooks(hooks)
    return tot / max(n, 1)


# RCO's search calls these through its own module: route them to the top-k reference and GLM's routing
prune.compute_kl_loss = _kl_loss
REROUTE = [False]  # --reroute: the search routes like deployment (pruned experts replaced by the next best one)


def _install(model, n_layers, n_experts, shared):
    shared["reroute"] = REROUTE[0]
    return glm.install_wrappers(model, shared)


prune.install_wrappers = _install


@torch.no_grad()
def wrapper_kl(model, keep_mask, ids, masks, ref, reroute):
    """KL with the SEARCH's routing under a fixed 0/1 keep mask (n_sparse, n_exp): what the optimizer sees."""
    shared = {"ste_masks": keep_mask.float(), "reroute": reroute}
    ws = glm.install_wrappers(model, shared)
    dev = get_input_device(model)
    tot, n = 0.0, 0.0
    try:
        for i in range(ids.size(0)):
            kl = kl_topk(model(input_ids=ids[i:i + 1].to(dev)).logits, ref[i][0], ref[i][1], masks[i:i + 1])
            w = masks[i, 1:].float().sum().item()
            tot += kl.item() * w
            n += w
    finally:
        for w_ in ws:
            w_.restore()
    return tot / max(n, 1)


# ----------------------------------------------------------------------------
# Reference + router statistics
# ----------------------------------------------------------------------------


@torch.no_grad()
def reference(model, ids, k, n_exp, label):
    """Top-k reference per window, plus per-window router counts and gate sums (so halves can be compared)."""
    targets = glm.router_hook_targets(model)
    cur = {"freq": torch.zeros(len(targets), n_exp), "score": torch.zeros(len(targets), n_exp)}
    hooks = []
    for kk, g in targets:
        def h(mod, inp, out, kk=kk):
            logits, _, idx = out
            cur["freq"][kk] += torch.bincount(idx.flatten().cpu(), minlength=n_exp).float()
            cur["score"][kk] += logits.float().sigmoid().sum(0).cpu()
        hooks.append(g.register_forward_hook(h))
    dev = get_input_device(model)
    refs, freqs, scores = [], [], []
    t0 = time.time()
    try:
        for i in range(ids.size(0)):
            cur["freq"].zero_()
            cur["score"].zero_()
            lp = F.log_softmax(model(input_ids=ids[i:i + 1].to(dev)).logits[:, :-1].float(), dim=-1)
            v, ix = lp.topk(k, dim=-1)
            refs.append((v.half().cpu(), ix.int().cpu()))
            freqs.append(cur["freq"].clone())
            scores.append(cur["score"].clone())
            if i % 32 == 0:
                log.info(f"{label}: {i + 1}/{ids.size(0)} windows, {time.time() - t0:.0f}s")
    finally:
        for h in hooks:
            h.remove()
    return refs, torch.stack(freqs), torch.stack(scores)


def keep_overlap(fa, fb, keep):
    """Mean over layers of |top-keep(A) & top-keep(B)| / keep, and the worst layer."""
    a = fa.topk(keep, dim=1).indices
    b = fb.topk(keep, dim=1).indices
    ov = [len(set(a[l].tolist()) & set(b[l].tolist())) / keep for l in range(fa.size(0))]
    return sum(ov) / len(ov), min(ov)


def mem_report():
    return "GiB max/GPU " + " ".join(f"{i}:{torch.cuda.max_memory_allocated(i) / 2**30:.1f}"
                                      for i in range(torch.cuda.device_count()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["prep", "search", "check"], required=True,
                    help="check: the search's routing (old and --reroute) vs deployment on the frequency mask")
    ap.add_argument("--reroute", action="store_true",
                    help="search: pruned experts are replaced by the router's next choice, as in deployment")
    ap.add_argument("--init-from", choices=["router", "freq"], default="router",
                    help="search: alpha prior from router gate sums (old) or from routing frequency ranks")
    ap.add_argument("--calib-dir", required=True, help="calib_search.pt / calib_heldout.pt; the reference goes here")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=72, help="search: experts kept per sparse layer (of 288)")
    ap.add_argument("--nonuniform", action="store_true",
                    help="search: same total budget (keep x layers) but each layer gets its own count (global top-k)")
    ap.add_argument("--keeps", default="72,96,144", help="prep: keep counts for the overlap and baseline report")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-gumbel-samples", type=int, default=4)
    ap.add_argument("--router-spread", type=float, default=5.0)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--n-layers", type=int, default=None, help="test: only the first n decoder layers")
    ap.add_argument("--limit", type=int, default=0, help="test: only the first n windows of each set")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(args.out, "log.txt")), logging.StreamHandler()])
    torch.manual_seed(42)

    S = torch.load(os.path.join(args.calib_dir, "calib_search.pt"))
    H = torch.load(os.path.join(args.calib_dir, "calib_heldout.pt"))
    if args.limit:
        S = {k: v[:args.limit] for k, v in S.items()}
        H = {k: v[:args.limit] for k, v in H.items()}
    log.info(f"windows: search {tuple(S['ids'].shape)}, held-out {tuple(H['ids'].shape)}; "
             f"model-written {S['mask'].float().mean():.1%}")

    t0 = time.time()
    model = glm.load_glm_fp8(args.model, n_layers=args.n_layers)
    log.info(f"loaded in {time.time() - t0:.0f}s; {mem_report()}")
    n_sparse = len(glm.moe_blocks(model))
    n_exp = model.config.num_local_experts
    test = "_test" if (args.limit or args.n_layers) else ""
    ref_path = os.path.join(args.calib_dir, f"ref_top{args.topk}{test}.pt")
    keeps = [int(x) for x in args.keeps.split(",")]

    if args.mode == "prep":
        rs, fs, ss = reference(model, S["ids"], args.topk, n_exp, "search")
        rh, fh, sh = reference(model, H["ids"], args.topk, n_exp, "held-out")
        tmp = ref_path + ".tmp"
        torch.save({"search": rs, "heldout": rh, "freq": fs, "score": ss, "freq_h": fh, "score_h": sh}, tmp)
        os.replace(tmp, ref_path)
        log.info(f"reference saved to {ref_path}; {mem_report()}")
        n = fs.size(0) // 2
        rep = {"windows": [fs.size(0), fh.size(0)], "overlap": {}, "freq_heldout_kl": {}}
        for kp in keeps:
            mean_ov, min_ov = keep_overlap(fs[:n].sum(0), fs[n:].sum(0), kp)
            rep["overlap"][kp] = {"mean": mean_ov, "min_layer": min_ov}
            log.info(f"halves agree on the top-{kp} experts: mean {mean_ov:.1%}, worst layer {min_ov:.1%}")
        json.dump(rep, open(os.path.join(args.out, "prep_report.json"), "w"), indent=1)
        freq = fs.sum(0)
        for kp in keeps:
            fm = prune.frequency_uniform_mask(freq, n_sparse, n_exp, (n_exp - kp) * n_sparse)
            rep["freq_heldout_kl"][kp] = eval_mask(model, fm, H["ids"], H["mask"], rh)
            log.info(f"frequency baseline keep {kp}: held-out KL {rep['freq_heldout_kl'][kp]:.4f}")
            json.dump(rep, open(os.path.join(args.out, "prep_report.json"), "w"), indent=1)
        log.info("DONE")
        return

    R = torch.load(ref_path)
    rs, rh = R["search"], R["heldout"]
    REROUTE[0] = args.reroute

    if args.mode == "check":
        # Same fixed mask through the three routings. The search must optimize what we deploy: --reroute has to
        # match deployment (hard_prune_hooks) to numerical noise; the old routing showed how far off it was.
        n = min(args.limit or 8, H["ids"].size(0))
        hi, hm, hr = H["ids"][:n], H["mask"][:n], rh[:n]
        for kp in keeps:
            fm = prune.frequency_uniform_mask(R["freq"].sum(0), n_sparse, n_exp, (n_exp - kp) * n_sparse)
            dep = eval_mask(model, fm, hi, hm, hr)
            new = wrapper_kl(model, ~fm, hi, hm, hr, True)
            old = wrapper_kl(model, ~fm, hi, hm, hr, False)
            log.info(f"check keep {kp}, {n} held-out windows: deployment KL {dep:.4f} | search --reroute {new:.4f} "
                     f"| search old {old:.4f}")
            json.dump({"keep": kp, "windows": n, "deploy": dep, "reroute": new, "old": old},
                      open(os.path.join(args.out, f"check_k{kp}.json"), "w"), indent=1)
        log.info("DONE")
        return

    ref_masks = [S["mask"][i:i + 1] for i in range(S["ids"].size(0))]
    target = (n_exp - args.keep) * n_sparse

    if args.init_from == "freq":  # rank of how often each expert is chosen, 0..1 per layer (raw counts are skewed)
        prior = R["freq"].sum(0).argsort(dim=1).argsort(dim=1).float()
    else:
        prior = R["score"].sum(0)
    t1 = time.time()
    alpha, mask, hist = prune.optimize(
        model, S["ids"], rs, ref_masks, n_sparse, n_exp, target,
        n_steps=args.n_steps, lr=args.lr, n_gumbel_samples=args.n_gumbel_samples,
        batch_size=1, log_interval=1, kl_topk=args.topk,
        router_scores=prior, router_spread=args.router_spread, per_layer_budget=not args.nonuniform,
        ckpt_path=os.path.join(args.out, "ckpt.pt"), save_every=args.save_every)
    search_s = time.time() - t1
    torch.save(mask.cpu(), os.path.join(args.out, "mask.pt"))
    torch.save(alpha.detach().cpu(), os.path.join(args.out, "alpha.pt"))
    log.info(f"search done in {search_s:.0f}s; mask saved; scoring ...")

    fm = prune.frequency_uniform_mask(R["freq"].sum(0), n_sparse, n_exp, target)
    m64 = min(64, S["ids"].size(0))
    res = {"keep": args.keep, "n_sparse": n_sparse, "search_s": search_s, "steps": args.n_steps,
           "nonuniform": args.nonuniform, "kept_per_layer": (n_exp - mask.sum(1)).tolist()}
    res["rco_heldout_kl"] = eval_mask(model, mask, H["ids"], H["mask"], rh)
    res["freq_heldout_kl"] = eval_mask(model, fm, H["ids"], H["mask"], rh)
    res["rco_search_kl"] = eval_mask(model, mask, S["ids"][:m64], S["mask"][:m64], rs[:m64])
    res["freq_search_kl"] = eval_mask(model, fm, S["ids"][:m64], S["mask"][:m64], rs[:m64])
    res["history"], res["args"] = hist, vars(args)
    log.info(f"keep {args.keep}: held-out KL RCO {res['rco_heldout_kl']:.4f} vs frequency {res['freq_heldout_kl']:.4f}; "
             f"search-set KL RCO {res['rco_search_kl']:.4f} vs frequency {res['freq_search_kl']:.4f}; {mem_report()}")
    json.dump(res, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    log.info("DONE")


if __name__ == "__main__":
    main()
