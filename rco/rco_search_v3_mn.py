#!/usr/bin/env python3
"""v3 of rco_search_v2_mn.py (that file is left unchanged). New flags, all off by default, so the
defaults reproduce rco_search_v2_mn.py exactly (checked on CPU, bit-identical alpha):
  --batch-size B        B windows per Gumbel sample (random, from the calibration pool), loss = token-weighted KL over
                        the B windows (= kl_topk on a batch of B), computed by gradient accumulation one window at a
                        time, so memory stays that of batch 1 (42 GB max/GCD measured)
  --antithetic          upstream's antithetic pairs: each base noise g is also used as -g (n_gumbel_samples / 2 bases)
  --init-from-mask P    alpha starts at that prune mask (True = pruned, e.g. REAP sum k144): pruned experts get prune
  --init-scores S       logit +margin + spread * rank, kept ones -(margin + spread * rank), ranks inside each class from S
  --init-margin m       (stats.pt "reap" = reap_sum, or the router prior if absent), then the exact per-layer budget is
  --init-spread s       restored (retraction_per_layer). The deterministic mask of the initial alpha IS the given mask.
  --tau-min t           end of the exponential tau schedule 1.0 -> t (rco_glm_prune's default 0.05; paper 0.01)
  --mask-every k        rank 0 saves OUT/masks/step{k*i}.pt = {mask (True = pruned), alpha, step, tau, decided}
Work unit = (Gumbel sample, window); units go round-robin to the ranks; all ranks draw the same noise and windows, so
antithetic pairs and batches stay consistent whatever the split. Final: held-out KL per window for the RCO and the
frequency mask (rank 0 writes OUT/kl/kl_shard0of1.pt, the eval_masks_mn format) and per domain (convergence.py
weights, OUT/heldout_domains.json), plus the overlap of the final mask with the init mask.

Data-parallel RCO search over K nodes: rco_glm_prune.py --mode search with the DSA fix (as rco_search_v2.py),
rco_glm_prune.py and search/prune.py untouched (monkeypatched from here).

Why this split: one search step = n_gumbel_samples (default 4) forward+backward passes, each on ONE random window
(batch_size 1), so the work per step is the sample list. Every rank draws the SAME random numbers in the same order
(same seed, same calls: Gumbel noise on the control device, window index on the CPU generator), so every rank knows
the full list (noise_j, window_j); rank r computes the samples j with j % K == r. The gradients on alpha (N x 2,
42*288*2 floats = 97 KB) are summed with one all-reduce (gloo, CPU tensors) and rank 0's result is broadcast, so
alpha / Adam state are bit-identical on all ranks and equal to the single-node run up to the order of the float
sum of the K partial gradients. With K = n_gumbel_samples each rank does 1 pass per step (4 nodes: ~4x faster).

gloo over TCP (not a file barrier): the payload is tiny, gloo ships with torch, works on CPU tensors (no RCCL), and
the collectives are blocking with a timeout, so a dead rank aborts the job instead of hanging forever. Interface:
GLOO_SOCKET_IFNAME (default hsn0, the Slingshot NIC of our cluster). Launch: srun, one task per node (rco_search_v2_mn.sbatch).

Rank 0 writes log.txt, ckpt.pt (now also with the control device's CUDA RNG state, so a resumed run continues the
same noise sequence), mask.pt, alpha.pt, result.json in --out; ranks > 0 write only their own log in --out/rank{r}.
Resume: every rank loads rank 0's ckpt.pt. The final scoring (eval_mask) is also split by window across ranks.

Env: RANK/WORLD_SIZE from SLURM_PROCID/SLURM_NTASKS (or RANK/WORLD_SIZE), MASTER_ADDR, MASTER_PORT;
RCO_CPU=1 for the CPU test (model on CPU, control device CPU).
"""
import datetime
import logging
import math  # noqa: F401  (kept for parity with prune.optimize)
import os
import sys
import time

P = "/work/glm53"
for p in (P + "/heal", P + "/rco", P + "/rco/src"):
    sys.path.insert(0, p)

import torch
import torch.distributed as dist
import torch.nn.functional as F

RANK = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
WORLD = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
CPU = os.environ.get("RCO_CPU") == "1"
if WORLD > 1:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "hsn0")
    dist.init_process_group("gloo", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '29511')}",
                            rank=RANK, world_size=WORLD, timeout=datetime.timedelta(hours=3))

import dsa_fix  # noqa: E402
dsa_fix.install_long_context_patches(int(os.environ.get("Q_CHUNK", "256")), int(os.environ.get("KDA_GROUP", "8")))

import glm5_next_rco as glm  # noqa: E402
import rco_glm_prune as R  # noqa: E402  (its import installs the KL loss + GLM wrappers into search.prune)
from common import get_input_device  # noqa: E402
from search import prune  # noqa: E402

log = logging.getLogger("rco_mn")


def _sum(t):
    """All-reduce SUM of a CPU float64/float32 tensor, then rank 0's value everywhere (bit-identical on all ranks)."""
    if WORLD > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        dist.broadcast(t, src=0)
    return t


if CPU:
    _load = glm.load_glm_fp8
    glm.load_glm_fp8 = lambda path, devices=None, n_layers=None: _load(path, devices=[torch.device("cpu")],
                                                                       n_layers=n_layers)


@torch.no_grad()
def eval_mask_mn(model, prune_mask, ids, masks, ref):
    """rco_glm_prune.eval_mask with the windows split across ranks (i % WORLD == RANK)."""
    hooks = glm.hard_prune_hooks(model, prune_mask) if prune_mask is not None else []
    dev = get_input_device(model)
    acc = torch.zeros(2, dtype=torch.float64)
    rows = {}
    try:
        for i in range(RANK, ids.size(0), WORLD):
            kl = R.kl_topk(model(input_ids=ids[i:i + 1].to(dev)).logits, ref[i][0], ref[i][1], masks[i:i + 1])
            w = masks[i, 1:].float().sum().item()
            acc[0] += kl.item() * w
            acc[1] += w
            rows[i] = (kl.item(), w)
    finally:
        glm.remove_hard_prune_hooks(hooks)
    _sum(acc)
    if WORLD > 1:
        allr = [None] * WORLD
        dist.all_gather_object(allr, rows)
        rows = {k: v for r in allr for k, v in r.items()}
    EVAL_LOG.append((ids.size(0), rows))
    return (acc[0] / acc[1].clamp(min=1)).item()


CFG = {"batch_size": 1, "antithetic": False, "init_from_mask": None, "init_scores": None, "init_margin": 1.5,
       "init_spread": 1.0, "tau_min": None, "mask_every": 0}
EVAL_LOG = []  # per eval_mask call: (n windows, {window: (kl, tokens)}) gathered over ranks


def init_from_mask(alpha, prune_mask, scores, margin, spread, target_per_layer, n_layers, n_experts, costs):
    """alpha so that the deterministic assignment equals prune_mask; not saturated (|logit| in [margin, margin+spread])."""
    pm = prune_mask.bool().view(n_layers, n_experts)
    assert (pm.sum(1) == target_per_layer).all(), f"init mask must prune {target_per_layer} per layer"
    sc = scores.double().view(n_layers, n_experts)
    off = torch.zeros(n_layers, n_experts, dtype=torch.float64)
    for l in range(n_layers):
        for cls, sign in ((pm[l], 1.0), (~pm[l], -1.0)):
            idx = cls.nonzero().flatten()
            v = sc[l, idx]
            r = v.argsort().argsort().double() / max(1, idx.numel() - 1)    # 0 = lowest score .. 1 = highest
            # pruned: low score -> most prunable; kept: high score -> most kept
            off[l, idx] = sign * (margin + spread * ((1 - r) if sign > 0 else r))
    with torch.no_grad():
        alpha[:, 0] = 0.0
        alpha[:, 1] = off.view(-1).to(alpha.dtype).to(alpha.device)
    prune.retraction_per_layer(alpha, costs, target_per_layer, n_layers, n_experts)
    with torch.no_grad():
        pp = torch.softmax(alpha, dim=1)[:, 1].view(n_layers, n_experts)
    hard = prune.budget_assignment_per_layer(alpha.detach(), target_per_layer, n_layers, n_experts).view(n_layers, n_experts).bool()
    ov = ((~hard.cpu()) & (~pm)).sum(1).double() / (n_experts - target_per_layer)
    prune.logger.info(f"[MN] alpha init from mask: E[prune] per layer min {pp.sum(1).min():.2f} max {pp.sum(1).max():.2f} "
                      f"(target {target_per_layer}); prune prob kept-class max {pp[~pm.to(pp.device)].max():.3f}, "
                      f"pruned-class min {pp[pm.to(pp.device)].min():.3f}; deterministic mask vs init mask overlap "
                      f"mean {ov.mean():.4f} min {ov.min():.4f}")


def optimize_mn(model, cal_data, ref_log_probs, ref_masks,
                n_layers, n_experts, target_budget,
                n_steps=200, lr=0.1, tau_init=1.0, tau_min=0.05,
                n_gumbel_samples=4, batch_size=4,
                log_interval=5, kl_topk=0,
                entropy_reg=0.0, antithetic=False,
                router_scores=None, router_spread=3.0,
                per_layer_budget=False, ckpt_path=None, save_every=5):
    """search.prune.optimize, data-parallel over the Gumbel samples (see module docstring). Differences from the
    original are marked MN; everything else is the same code, calling the same search.prune helpers."""
    N = n_layers * n_experts
    ctrl = torch.device("cpu") if CPU else torch.device("cuda:0")  # MN: CPU test
    costs = torch.tensor([0.0, 1.0], device=ctrl)
    ckpt_path = os.environ.get("RCO_MN_CKPT", ckpt_path)  # MN: rank 0's checkpoint, read by every rank

    if per_layer_budget:
        assert target_budget % n_layers == 0
        target_per_layer = target_budget // n_layers
        assert 0 < target_per_layer < n_experts
        prune.logger.info(f"PER-LAYER mode: prune {target_per_layer}/{n_experts} per layer "
                          f"({n_layers} layers x {target_per_layer} = {target_budget} total)")

    alpha = torch.zeros(N, 2, dtype=torch.float32, device=ctrl, requires_grad=True)
    if router_scores is not None:
        prune.init_alpha_from_router_scores(alpha, router_scores, target_budget, n_layers, n_experts,
                                            spread=router_spread)
    else:
        prune.init_alpha(alpha, target_budget, N)
    batch_size = CFG["batch_size"]                    # v3: windows per Gumbel sample (rco_glm_prune passes 1)
    if CFG["tau_min"] is not None:                    # v3: paper's tau_min 0.01 (rco_glm_prune passes nothing: 0.05)
        tau_min = CFG["tau_min"]
    antithetic = CFG["antithetic"] or antithetic
    if CFG["init_from_mask"]:
        assert per_layer_budget, "--init-from-mask needs the per-layer budget"
        pm = torch.load(CFG["init_from_mask"], weights_only=False)
        sc = router_scores
        if CFG["init_scores"]:
            sc = torch.load(CFG["init_scores"], weights_only=False)["reap"]
        init_from_mask(alpha, pm, sc, CFG["init_margin"], CFG["init_spread"], target_per_layer, n_layers, n_experts,
                       costs)
    optimizer = torch.optim.Adam([alpha], lr=lr)
    model.requires_grad_(False)
    model.eval()
    shared_state = {"ste_masks": None}
    wrappers = prune.install_wrappers(model, n_layers, n_experts, shared_state)
    # same RNG consumption on every rank (it draws Gumbel noise and runs one pass on window 0)
    prune.gradient_sanity_check(model, alpha, shared_state, n_layers, n_experts, cal_data, ref_log_probs,
                                ref_masks, 1, target_budget, kl_topk=kl_topk)

    input_device = get_input_device(model)
    all_bi = list(range(0, cal_data.size(0), 1))      # one window per index (old code: batch_size 1)
    n_total_batches = len(all_bi)
    prune.logger.info(f"[MN rank {RANK}/{WORLD}] Gumbel-STE: {n_steps} steps, lr={lr}, tau {tau_init}->{tau_min}, "
                      f"{n_gumbel_samples} samples/step x {batch_size} windows, antithetic {antithetic}, "
                      f"split over {WORLD} ranks, {cal_data.size(0)} windows")

    history = []
    t0 = time.time()
    start = 0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=ctrl, weights_only=False)
        with torch.no_grad():
            alpha.copy_(ck["alpha"])
        optimizer.load_state_dict(ck["opt"])
        history = ck["history"]
        start = ck["step"] + 1
        torch.set_rng_state(ck["rng"])
        if "cuda_rng" in ck and ctrl.type == "cuda":  # MN: the Gumbel noise comes from this generator
            torch.cuda.set_rng_state(ck["cuda_rng"], ctrl)
        prune.logger.info(f"resumed from {ckpt_path} at step {start}")

    for step in range(start, n_steps):
        progress = step / max(n_steps - 1, 1)
        tau = max(tau_min, tau_init * (tau_min / tau_init) ** progress)
        optimizer.zero_grad()
        n_base = max(n_gumbel_samples // 2 if antithetic else n_gumbel_samples, 1)
        n_evals = n_base * 2 if antithetic else n_base
        scale = 1.0 / n_evals

        # MN: every rank draws the full sample list in the original order
        samples = []
        for _gs in range(n_base):
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(alpha).clamp(1e-20)) + 1e-20)
            noise_variants = [gumbel_noise] + ([-gumbel_noise] if antithetic else [])
            for gn_var in noise_variants:
                if batch_size == 1:   # exactly the old draw
                    bis = [all_bi[torch.randint(n_total_batches, (1,)).item()]]
                else:                 # v3: B random windows of the pool
                    bis = torch.randint(n_total_batches, (batch_size,)).tolist()
                samples.append((gn_var, bis))
        # v3: work units (sample, window), token weights so a sample's loss = kl_topk over its B windows
        units = []
        for j, (gn_var, bis) in enumerate(samples):
            wt = [ref_masks[b][:, 1:].float().sum().item() for b in bis]
            for b, w in zip(bis, wt):
                units.append((j, gn_var, b, (w / max(sum(wt), 1.0)) if batch_size > 1 else None))

        local_loss = 0.0
        for u, (j, gn_var, bi, wfrac) in enumerate(units):
            if u % WORLD != RANK:  # MN: not this rank's unit
                continue
            noisy = (alpha + gn_var) / tau
            soft_keep = F.softmax(noisy, dim=1)[:, 0]
            if per_layer_budget:
                hard_asgn = prune.budget_assignment_per_layer(noisy.detach(), target_per_layer, n_layers, n_experts)
            else:
                hard_asgn = prune.budget_assignment(noisy.detach(), target_budget)
            hard_keep = 1.0 - hard_asgn.float()
            ste_keep = hard_keep + (soft_keep - soft_keep.detach())
            shared_state["ste_masks"] = ste_keep.view(n_layers, n_experts)
            batch = cal_data[bi:bi + 1].to(input_device)
            ref_idx = min(bi, len(ref_log_probs) - 1)
            loss = prune.compute_kl_loss(model, batch, ref_log_probs[ref_idx], mask=ref_masks[ref_idx], topk=kl_topk)
            if wfrac is not None:
                loss = loss * wfrac
            (loss * scale).backward()
            local_loss += loss.item()

        # MN: sum the partial gradients and losses of all ranks
        g = alpha.grad.detach().cpu().clone() if alpha.grad is not None else torch.zeros(N, 2)
        packed = _sum(torch.cat([g.flatten().double(), torch.tensor([local_loss], dtype=torch.float64)]))
        with torch.no_grad():
            alpha.grad = packed[:-1].float().view(N, 2).to(ctrl)
        avg_loss = packed[-1].item() / n_evals

        if entropy_reg > 0:
            probs_for_ent = torch.softmax(alpha, dim=1)
            ent_loss = -(probs_for_ent * (probs_for_ent + 1e-10).log()).sum(1).mean()
            (entropy_reg * ent_loss).backward()

        if per_layer_budget:
            proj_coeff = prune.project_gradient_per_layer(alpha, costs, n_layers, n_experts)
        else:
            proj_coeff, _, _ = prune.project_gradient(alpha, costs)
        gn = alpha.grad.norm().item()
        optimizer.step()
        if per_layer_budget:
            bud = prune.retraction_per_layer(alpha, costs, target_per_layer, n_layers, n_experts) * n_layers
            prune.vector_transport_per_layer(optimizer, alpha, costs, n_layers, n_experts)
        else:
            bud = prune.retraction(alpha, costs, target_budget)
            prune.vector_transport(optimizer, alpha, costs)

        with torch.no_grad():
            probs = torch.softmax(alpha, dim=1)
            prune_probs = probs[:, 1]
            ent = -(probs * (probs + 1e-10).log()).sum(1).mean().item()
            decided = (probs.max(1).values > 0.9).sum().item()
            n_hard = (probs.max(1).values > 0.99).sum().item()
            per_layer = prune_probs.view(n_layers, n_experts).sum(1)
        history.append(dict(step=step, loss=avg_loss, budget=bud, tau=tau, grad_norm=gn, entropy=ent,
                            decided=decided, n_hard=n_hard, proj_coeff=proj_coeff))
        if step % log_interval == 0 or step == n_steps - 1:
            el = time.time() - t0
            eta = (el / (step + 1 - start)) * (n_steps - step - 1)
            em, es = divmod(int(eta), 60)
            pl = " ".join(f"{per_layer[l].item():.1f}" for l in range(n_layers))
            prune.logger.info(f"[{step:4d}] KL={avg_loss:.4f} bud={bud:.1f} tau={tau:.3f} ent={ent:.3f} "
                              f"dec={decided}/{N} hard={n_hard} grad={gn:.4f} proj={proj_coeff:.3f} "
                              f"E[r]=[{pl}]  ETA {em}m{es:02d}s")
        if RANK == 0 and ckpt_path and (step % save_every == 0 or step == n_steps - 1):  # MN: rank 0 only
            ck = {"alpha": alpha.detach(), "opt": optimizer.state_dict(), "history": history, "step": step,
                  "rng": torch.get_rng_state(), "world": WORLD}
            if ctrl.type == "cuda":
                ck["cuda_rng"] = torch.cuda.get_rng_state(ctrl)
            tmp = ckpt_path + ".tmp"
            torch.save(ck, tmp)
            os.replace(tmp, ckpt_path)
        if RANK == 0 and CFG["mask_every"] and ((step + 1) % CFG["mask_every"] == 0 or step == n_steps - 1):
            # v3: the deterministic hard mask (True = pruned, as mask.pt) + alpha, for held-out convergence afterwards
            md = os.path.join(os.path.dirname(ckpt_path), "masks")
            os.makedirs(md, exist_ok=True)
            if per_layer_budget:
                ha = prune.budget_assignment_per_layer(alpha.detach(), target_per_layer, n_layers, n_experts)
            else:
                ha = prune.budget_assignment(alpha.detach(), target_budget)
            torch.save({"mask": ha.view(n_layers, n_experts).bool().cpu(), "alpha": alpha.detach().cpu(), "step": step + 1,
                        "tau": tau, "decided": decided}, os.path.join(md, f"step{step + 1}.pt"))
        if WORLD > 1 and ckpt_path:
            dist.barrier()  # MN: nobody runs ahead of a checkpoint being written
        if os.environ.get("RCO_MN_STOP_AFTER") == str(step):  # test hook: simulate a kill right after this step
            prune.logger.info(f"[MN] test stop after step {step}")
            os._exit(0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prune.remove_wrappers(wrappers)
    if per_layer_budget:
        final_asgn = prune.budget_assignment_per_layer(alpha.detach(), target_per_layer, n_layers, n_experts)
    else:
        final_asgn = prune.budget_assignment(alpha.detach(), target_budget)
    final_mask = final_asgn.view(n_layers, n_experts).bool()
    # MN: proof that the ranks never diverged
    if WORLD > 1:
        a = alpha.detach().cpu().double()
        lo, hi = a.clone(), a.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        prune.logger.info(f"[MN] alpha identical on all {WORLD} ranks: {bool(torch.equal(lo, hi))}")
        assert torch.equal(lo, hi), "ranks diverged"
    return alpha, final_mask, history


def _pop(argv, flag, has_value=True, conv=str):
    if flag not in argv:
        return None
    i = argv.index(flag)
    if not has_value:
        del argv[i]
        return True
    v = conv(argv[i + 1])
    del argv[i:i + 2]
    return v


def final_reports(out0, calib_dir):
    """rank 0: per-window held-out KL of the RCO and the frequency mask (the first two eval_mask calls of
    rco_glm_prune.main are held-out RCO, held-out frequency) + per domain + overlap with the init mask."""
    sys.path.insert(0, P + "/rco")
    import convergence as C
    import json
    H = torch.load(os.path.join(calib_dir, "calib_heldout.pt"), weights_only=False)
    os.makedirs(os.path.join(out0, "kl"), exist_ok=True)
    masks = {}
    for name, (n, rows) in zip(("rco", "freq"), EVAL_LOG[:2]):
        assert n == len(H["meta"]), (n, len(H["meta"]))
        masks[name] = rows
    torch.save({"masks": masks, "shard": 0, "nshards": 1}, os.path.join(out0, "kl", "kl_shard0of1.pt"))
    res = C.kl_table(os.path.join(out0, "kl"), H) if "bucket" in H["meta"][0] else {}
    final = torch.load(os.path.join(out0, "mask.pt"))
    if CFG["init_from_mask"]:
        im = torch.load(CFG["init_from_mask"], weights_only=False)
        keep = (~im).sum(1)[0].item()
        ov = ((~final) & (~im)).sum(1).double() / keep
        res["overlap_with_init_mask"] = {"mean": ov.mean().item(), "min_layer": ov.min().item(),
                                         "experts_changed": int((final != im).sum().item()) // 2}
    json.dump(res, open(os.path.join(out0, "heldout_domains.json"), "w"), indent=1)
    for k, v in res.items():
        log.warning(f"[v3 final] {k}: {v}")


def main():
    argv = sys.argv[1:]
    bs = _pop(argv, "--batch-size", conv=int)
    CFG["batch_size"] = bs or 1
    CFG["antithetic"] = bool(_pop(argv, "--antithetic", has_value=False))
    CFG["init_from_mask"] = _pop(argv, "--init-from-mask")
    CFG["init_scores"] = _pop(argv, "--init-scores")
    m = _pop(argv, "--init-margin", conv=float)
    sp = _pop(argv, "--init-spread", conv=float)
    if m is not None:
        CFG["init_margin"] = m
    if sp is not None:
        CFG["init_spread"] = sp
    CFG["tau_min"] = _pop(argv, "--tau-min", conv=float)
    CFG["mask_every"] = _pop(argv, "--mask-every", conv=int) or 0
    if "--out" not in argv:
        raise SystemExit("--out required")
    i = argv.index("--out")
    out0 = argv[i + 1]
    os.environ.setdefault("RCO_MN_CKPT", os.path.join(out0, "ckpt.pt"))
    if RANK > 0:  # MN: own directory for the log and the (identical) final files
        argv[i + 1] = os.path.join(out0, f"rank{RANK}")
    if RANK == 0:
        os.makedirs(out0, exist_ok=True)
    prune.optimize = optimize_mn
    R.eval_mask = eval_mask_mn
    sys.argv = [P + "/rco/rco_glm_prune.py"] + argv
    log.info(f"rank {RANK}/{WORLD} out {argv[i + 1]}")
    R.main()
    if RANK == 0:
        final_reports(out0, argv[argv.index("--calib-dir") + 1])
    if WORLD > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
