#!/usr/bin/env python3
"""heal_v3.py: derived from heal_v2.py (left unchanged) for healing the mix_v3 masks (15 Oct). Changes vs heal_v2:
  A. mix_v3 data: --calib-dir with calib_search.pt / calib_heldout.pt (meta["bucket"] = agentic / code / math / general)
     and --ref with the full model's top-20 (corrected torch path);
  B. the held-out evaluation is the one of eval_masks_mn.py: the same kl_topk on every window, weight = model-written
     tokens (mask[1:]), aggregation per domain as in rco/convergence.kl_table, weighted KL 45/35/20 (agentic/code/math),
     general is a monitor only. Per-window rows are written to OUT/kl/kl_shard0of1.pt in the eval_masks_mn format,
     so `convergence.kl_table(OUT/kl, H)` reads them directly. delta_best selection and stopping use the WEIGHTED KL;
  C. --eval-subset N: periodic evaluations on N held-out windows stratified by domain (deterministic); at step 0 a
     window-by-window comparison with --compare-kl-dir (eval_masks_mn rows of the same mask on the full model with
     hard_prune_hooks): checks that the pruned checkpoint (router with the rows removed, as in the llama.cpp GGUF)
     gives the same KL as the mask scorer; --final-full-eval: delta_best evaluated on all windows;
  D. --gguf-experts: routed experts from any GGUF (gguf_experts_any: Q8_0/Q4_K/Q3_K/IQ2_S/IQ2_XXS) instead of FP8.

heal_v2.py: derived from heal.py (left unchanged) with three changes (14 Oct):
  1. the DSA fix + long-context memory savings from heal/dsa_fix.py, installed before loading (without it the DSA
     attention is wrong on ROCm: CE 3.52 instead of 1.07; see dsa_fix.py);
  2. --iq2-gguf GGUF: routed experts replaced with the IQ2_XXS bytes from the GGUF (heal/gguf_experts.py, frozen,
     bit-identical to llama.cpp), so healing sees exactly the 2-bit damage of the deployed file;
  3. defaults: v2 reference (calib53/v2/ref_top20.pt, made with dsa_fix), no router training,
     LoRA LR 2e-5, held-out KL at step 0 and every --eval-every steps,
     delta_best by held-out KL and automatic stop after --patience evaluations in a row without improvement.

Healing (knowledge distillation) of an expert-pruned GLM-5.3-Flash against the FULL model's stored top-20.

Train mode (default), one node, 8 GPUs (MI250X GCDs):
    heal.py --model PRUNED_DIR --calib-dir calib53 --out OUT [--lora-r 16 --lora-alpha 32] [--router] [--norms]
  Trainable: (a) LoRA on the attention projections + shared-expert MLPs (--lora-r 0 = off), (b) router weight of
  every sparse layer (--router, default on), (c) RMSNorm weights (--norms, default off). Routed experts stay frozen
  FP8 bytes. Loss = kl_topk (masked KL vs the full model's top-20) on the model-written tokens of the 512 search
  windows; held-out KL on the 64 held-out windows before training and every --eval-every optimizer steps.
  Resumable: OUT/ckpt.pt (trainables + optimizer + scheduler) every --save-every steps, atomic rename;
  OUT/delta.pt (trainables only, for --merge) and OUT/delta_best.pt (best held-out KL) alongside.

  Note on e_score_correction_bias: in Glm5NextTextTopkRouter it only shifts the top-k CHOICE, the routing weights
  come from the unbiased sigmoid scores, so it receives no gradient. It is therefore not trained here.

Merge mode (CPU, no model load):
    heal.py --merge --model PRUNED_DIR --delta OUT/delta.pt --merged-out DIR [--layers-below N] [--link-unchanged]
  Writes an HF checkpoint in the pruned model's format: every tensor not touched by healing is copied
  byte-identical (FP8 experts and their scale_inv included); LoRA targets become BF16 W + alpha/r * B @ A (their
  weight_scale_inv is dropped, so llama.cpp's convert_hf_to_gguf.py takes them as plain BF16, exactly like the
  checkpoint's other BF16 tensors); router / norm weights are written as F32 (small, keeps updates below BF16 ulp).
  --layers-below N keeps only decoder layers < N plus the non-layer tensors (a cheap test checkpoint).

Extra windows: --extra CALIB.pt:REF.pt adds windows whose top-20 reference the FULL model produced in a separate job
  (TODO for new data: build CALIB.pt {"ids","mask"} and run rco/rco_glm_prune.py --mode prep with --model full on it;
  the full and pruned models do not fit on one node together, so the reference is never computed in this process).
"""

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "rco" / "src", HERE.parent / "rco", HERE.parent / "prune_build"):
    sys.path.insert(0, str(p))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

import glm5_next_rco as glm
import dsa_fix
import gguf_experts as ge
import gguf_experts_any as gea

SH = {"agentic": 0.45, "code": 0.35, "math": 0.20}   # = rco/convergence.py; general = monitor
DOMS = ["agentic", "code", "math", "general"]


def kl_summary(rows, meta):
    """rco/convergence.kl_table on a single set of rows {i: (kl, w)}."""
    def agg(sel):
        ws = [rows[i] for i in sel if i in rows]
        return sum(k * w for k, w in ws) / max(1e-9, sum(w for _, w in ws)) if ws else None
    dom = {d: agg([i for i, m in enumerate(meta) if m["bucket"] == d]) for d in DOMS}
    present = [d for d in SH if dom[d] is not None]
    return {"total": agg([i for i, m in enumerate(meta) if m["bucket"] in SH]), "domain": dom,
            "weighted": sum(SH[d] * dom[d] for d in present) / sum(SH[d] for d in present) if present else None}


def stratified(meta, n):
    """n held-out windows, proportional per domain (at least 2 per domain), deterministic (every k-th)."""
    by = {}
    for i, m in enumerate(meta):
        by.setdefault(m["bucket"], []).append(i)
    out = []
    for d, idx in sorted(by.items()):
        k = max(2, round(n * len(idx) / len(meta)))
        step = max(1, len(idx) // k)
        out += idx[::step][:k]
    return sorted(out)


def fmt(sm):
    d = sm["domain"]
    return (f"weighted {sm['weighted']:.4f} total {sm['total']:.4f} | " +
            " ".join(f"{k} {d[k]:.4f}" if d[k] is not None else f"{k} -" for k in DOMS))

log = logging.getLogger("heal")
PFX = glm.PFX  # "model.language_model."

DEFAULT_LORA = (r"^layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)"
                r"|mlp\.shared_experts\.(gate_proj|up_proj|down_proj))$")


def kl_topk(logits, vals, idx, mask):
    """Same as rco_glm_prune.kl_topk (imported from there when possible, kept here as a fallback because importing
    rco_glm_prune pulls in the RCO search package and patches it)."""
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    dev = lp.device
    mp = lp.gather(-1, idx.to(dev).long())
    rp = vals.to(dev).float()
    rp = rp - rp.logsumexp(-1, keepdim=True)
    kl = (rp.exp() * (rp - mp)).sum(-1)
    m = mask[:, 1:].to(dev).float()
    return (kl * m).sum() / m.sum().clamp(min=1)


try:
    from rco_glm_prune import kl_topk  # noqa: F811  the canonical one
except Exception as e:  # pragma: no cover
    log.warning(f"using local kl_topk copy ({e})")


# ----------------------------------------------------------------------------
# Trainable parameters
# ----------------------------------------------------------------------------


def ckpt_name(path):
    """Module path inside Glm5NextTextModel ('layers.3.self_attn.q_a_proj') -> checkpoint weight name."""
    p = path.replace("self_attn.forget_gate.", "self_attn.")  # inverse of glm.RENAMES for the projections
    return PFX + p + ".weight"


class _Cast(nn.Module):
    """fp32 master weight, used in the module's own dtype (RMSNorm multiplies weight * h in that dtype)."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, w):
        return w.to(self.dtype)


def add_lora(lin, r, alpha):
    """Minimal LoRA on an existing nn.Linear: forward patched in place, so .weight and the module path stay."""
    dev = lin.weight.device
    A = nn.Parameter(torch.empty(r, lin.in_features, device=dev, dtype=torch.float32))
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    B = nn.Parameter(torch.zeros(lin.out_features, r, device=dev, dtype=torch.float32))
    lin.register_parameter("lora_A", A)
    lin.register_parameter("lora_B", B)
    lin.lora_scale = alpha / r

    def fwd(x, lin=lin):
        y = F.linear(x, lin.weight, lin.bias)
        d = F.linear(F.linear(x.float(), lin.lora_A), lin.lora_B) * lin.lora_scale
        return y + d.to(y.dtype)
    lin.forward = fwd


def setup_trainable(model, lora_r, lora_alpha, lora_targets, router, norms):
    """Returns {"lora": {ckpt: linear}, "router": {ckpt: gate}, "norm": {ckpt: module}} and param groups."""
    tm = model.model
    tm.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    T = {"lora": {}, "router": {}, "norm": {}}
    rx = re.compile(lora_targets)
    if lora_r > 0:
        for name, mod in tm.named_modules():
            if isinstance(mod, nn.Linear) and rx.search(name):
                add_lora(mod, lora_r, lora_alpha)
                T["lora"][ckpt_name(name)] = mod
    if router:
        for i, layer in enumerate(tm.layers):
            if hasattr(layer.mlp, "gate"):
                g = layer.mlp.gate
                g.weight = nn.Parameter(g.weight.detach().float(), requires_grad=True)  # router computes in fp32
                T["router"][ckpt_name(f"layers.{i}.mlp.gate")] = g
    if norms:
        for name, mod in tm.named_modules():
            if "RMSNorm" in type(mod).__name__ and isinstance(getattr(mod, "weight", None), nn.Parameter):
                dt = mod.weight.dtype
                mod.weight = nn.Parameter(mod.weight.detach().float(), requires_grad=True)
                parametrize.register_parametrization(mod, "weight", _Cast(dt), unsafe=True)
                T["norm"][ckpt_name(name)] = mod
    return T


def trainable_params(T):
    out = {"lora": [], "router": [], "norm": []}
    for n, m in T["lora"].items():
        out["lora"] += [m.lora_A, m.lora_B]
    for n, g in T["router"].items():
        out["router"].append(g.weight)
    for n, m in T["norm"].items():
        out["norm"].append(m.parametrizations.weight.original)
    return out


def trainable_state(T):
    s = {"lora": {}, "full": {}}
    for n, m in T["lora"].items():
        s["lora"][n] = {"A": m.lora_A.detach().cpu().clone(), "B": m.lora_B.detach().cpu().clone(),
                        "scale": m.lora_scale}
    for n, g in T["router"].items():
        s["full"][n] = g.weight.detach().cpu().clone()
    for n, m in T["norm"].items():
        s["full"][n] = m.parametrizations.weight.original.detach().cpu().clone()
    return s


@torch.no_grad()
def load_trainable_state(T, s):
    for n, m in T["lora"].items():
        m.lora_A.copy_(s["lora"][n]["A"])
        m.lora_B.copy_(s["lora"][n]["B"])
    for n, g in T["router"].items():
        g.weight.copy_(s["full"][n])
    for n, m in T["norm"].items():
        m.parametrizations.weight.original.copy_(s["full"][n])


def grad_report(model, T):
    """(trainables with zero/None grad, frozen tensors that got a grad or require one)."""
    want = {id(p) for ps in trainable_params(T).values() for p in ps}
    dead, leaks = [], []
    for n, p in model.named_parameters():
        if id(p) in want:
            if p.grad is None or p.grad.abs().sum().item() == 0:
                dead.append(n)
        elif p.requires_grad or p.grad is not None:
            leaks.append(n)
    return dead, leaks


def expert_digest(model):
    h = hashlib.sha256()
    for _, blk in glm.moe_blocks(model):
        for b in (blk.experts.gate_up, blk.experts.gate_up_s, blk.experts.down, blk.experts.down_s):
            h.update(b.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def load_windows(calib_path, refs, max_len=0, limit=0):
    """[(ids (T,), mask (T,), (vals (1,T-1,k), idx (1,T-1,k)))]"""
    C = torch.load(calib_path)
    n = C["ids"].size(0) if not limit else min(limit, C["ids"].size(0))
    assert len(refs) >= n, f"{calib_path}: {len(refs)} references for {n} windows"
    out = []
    for i in range(n):
        ids, m = C["ids"][i], C["mask"][i]
        v, ix = refs[i]
        assert v.shape[1] == ids.size(0) - 1, (v.shape, ids.shape)
        if max_len:
            ids, m, v, ix = ids[:max_len], m[:max_len], v[:, :max_len - 1], ix[:, :max_len - 1]
        out.append((ids, m, (v, ix)))
    return out


def mem_report():
    if not torch.cuda.is_available():
        return ""
    return "GiB max/GCD " + " ".join(f"{i}:{torch.cuda.max_memory_allocated(i) / 2**30:.1f}"
                                      for i in range(torch.cuda.device_count()))


def input_device(model):
    return model.model.embed_tokens.weight.device


@torch.no_grad()
def eval_kl(model, windows):
    dev = input_device(model)
    tot, n = 0.0, 0.0
    for ids, m, (v, ix) in windows:
        kl = kl_topk(model(input_ids=ids[None].to(dev)).logits, v, ix, m[None])
        w = m[1:].float().sum().item()
        tot += kl.item() * w
        n += w
    return tot / max(n, 1)


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------


class Trainer:
    def __init__(self, model, T, train, heldout, out, a):
        self.model, self.T, self.train, self.heldout, self.out, self.a = model, T, train, heldout, out, a
        P = trainable_params(T)
        groups = [g for g in (
            {"params": P["lora"], "lr": a.lr, "weight_decay": a.weight_decay, "name": "lora"},
            {"params": P["router"], "lr": a.router_lr, "weight_decay": 0.0, "name": "router"},
            {"params": P["norm"], "lr": a.norm_lr, "weight_decay": 0.0, "name": "norm"}) if g["params"]]
        assert groups, "nothing to train"
        self.params = [p for g in groups for p in g["params"]]
        self.opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        self.spe = max(len(train) // a.grad_accum, 1)  # optimizer steps per epoch
        self.total = a.max_steps or int(a.epochs * self.spe)
        warm = a.warmup_steps

        def lam(s):
            if s < warm:
                return (s + 1) / warm
            p = min((s - warm) / max(self.total - warm, 1), 1.0)
            return a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lam)
        self.step = 0
        self.hist = {"train": [], "heldout": []}
        self.best = float("inf")

    # one window per micro-step; the order is a function of the global micro index only, so resume is exact
    def window(self, g):
        n_micro = self.spe * self.a.grad_accum
        ep, pos = divmod(g, n_micro)
        gen = torch.Generator().manual_seed(self.a.seed + ep)
        return self.train[torch.randperm(len(self.train), generator=gen)[pos].item()]

    def ckpt_path(self):
        return os.path.join(self.out, "ckpt.pt")

    def _atomic(self, obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save(self):
        s = trainable_state(self.T)
        meta = {"step": self.step, "lora_r": self.a.lora_r, "lora_alpha": self.a.lora_alpha,
                "lora_targets": self.a.lora_targets, "model": os.path.abspath(self.a.model)}
        self._atomic({"trainable": s, "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                      "step": self.step, "hist": self.hist, "best": self.best, "meta": meta}, self.ckpt_path())
        self._atomic({**s, "meta": meta}, os.path.join(self.out, "delta.pt"))

    def load(self):
        c = torch.load(self.ckpt_path(), map_location="cpu", weights_only=False)
        load_trainable_state(self.T, c["trainable"])
        self.opt.load_state_dict(c["opt"])
        self.sched.load_state_dict(c["sched"])
        self.step, self.hist, self.best = c["step"], c["hist"], c["best"]
        self.bad_evals = self.hist.get("bad_evals", 0)
        log.info(f"resumed at optimizer step {self.step}")

    def eval_rows(self, idx):
        """{i: (kl, w)} as in eval_masks_mn, plus {i: (top-1 agreement, w)}: the fraction of model-written positions where
        the model's argmax = the full model's argmax (ix[..., 0], the top-20 is sorted in descending order)."""
        dev = input_device(self.model)
        rows, top1 = {}, {}
        with torch.no_grad():
            for i in idx:
                ids, m, (v, ix) = self.heldout[i]
                lg = self.model(input_ids=ids[None].to(dev)).logits
                kl = kl_topk(lg, v, ix, m[None])
                w = m[1:].float().sum().item()
                mm = m[1:].to(lg.device).bool()
                agree = ((lg[0, :-1].argmax(-1) == ix[0, :, 0].to(lg.device).long()) & mm).sum().item()
                rows[i] = (kl.item(), w)
                top1[i] = (agree / max(w, 1), w)
        self._top1 = top1
        return rows

    def save_rows(self, name, rows):
        os.makedirs(os.path.join(self.out, "kl"), exist_ok=True)
        p = os.path.join(self.out, "kl", "kl_shard0of1.pt")
        st = torch.load(p, weights_only=False) if os.path.exists(p) else {"masks": {}, "shard": 0, "nshards": 1}
        st["masks"][name] = rows
        st.setdefault("top1", {})[name] = dict(getattr(self, "_top1", {}))  # outside "masks": kl_table does not see it

        self._atomic(st, p)

    def evaluate(self, tag):
        t0 = time.time()
        full0 = self.step == 0 and self.a.full_before and len(self.ho_idx) < len(self.heldout)
        rows = self.eval_rows(list(range(len(self.heldout))) if full0 else self.ho_idx)
        self.save_rows(f"heal_step{self.step}" + ("_full" if full0 else ""), rows)
        if full0:
            fs = kl_summary(rows, self.meta)
            fs["top1"] = kl_summary(self._top1, self.meta)
            self.hist["full_before"] = fs
            log.info(f"[{tag}] FULL unhealed (all {len(rows)} held-out windows): KL {fmt(fs)} | top-1 {fmt(fs['top1'])}")
            rows = {i: rows[i] for i in self.ho_idx}
            self._top1 = {i: self._top1[i] for i in self.ho_idx}
        sm = kl_summary(rows, self.meta)
        sm["top1"] = kl_summary(self._top1, self.meta)
        kl = sm["weighted"]
        self.hist["heldout"].append({"step": self.step, "kl": kl, "summary": sm})
        base = self.hist["heldout"][0]["summary"]
        worse = [d for d in DOMS if sm["domain"][d] is not None and base["domain"][d] is not None
                 and sm["domain"][d] > base["domain"][d] * 1.005]
        log.info(f"[{tag}] held-out KL {fmt(sm)} | top-1 agreement {fmt(sm['top1'])} "
                 f"({len(rows)} windows, {time.time() - t0:.0f}s)"
                 + (f"; WORSE than step 0: {worse}" if worse and self.step > 0 else ""))
        if self.step == 0 and self.ref_rows:
            common = [i for i in rows if i in self.ref_rows]
            d = [abs(rows[i][0] - self.ref_rows[i][0]) for i in common]
            rs = kl_summary({i: self.ref_rows[i] for i in common}, self.meta)
            log.info(f"STEP0 CHECK vs eval_masks_mn ({self.a.compare_mask}) on the same {len(common)} windows: "
                     f"scorer {fmt(rs)}; per-window |dKL| mean {sum(d) / max(1, len(d)):.5f} max {max(d or [0]):.5f}")
        if kl < self.best - self.a.min_delta:
            self.bad_evals = 0
        else:
            self.bad_evals = getattr(self, "bad_evals", 0) + 1
            log.info(f"held-out weighted KL not improved ({self.bad_evals}/{self.a.patience})")
        self.hist["bad_evals"] = self.bad_evals
        if kl < self.best:
            self.best = kl
            self._atomic({**trainable_state(self.T), "meta": {"step": self.step, "heldout_kl_weighted": kl,
                                                               "summary": sm, "lora_r": self.a.lora_r,
                                                               "lora_alpha": self.a.lora_alpha}},
                         os.path.join(self.out, "delta_best.pt"))
        return kl

    def run(self, max_steps=None, deadline=None, on_first_grads=None):
        a, model = self.a, self.model
        dev = input_device(model)
        stop = min(max_steps or self.total, self.total)
        while self.step < stop:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
            t0 = time.time()
            tot_loss, ntok, nmask = 0.0, 0, 0
            for j in range(a.grad_accum):
                ids, m, (v, ix) = self.window(self.step * a.grad_accum + j)
                loss = kl_topk(model(input_ids=ids[None].to(dev)).logits, v, ix, m[None])
                (loss / a.grad_accum).backward()
                tot_loss += loss.item() / a.grad_accum
                ntok += ids.numel()
                nmask += int(m[1:].sum())
            if on_first_grads is not None:
                on_first_grads(self)
            gn = torch.nn.utils.clip_grad_norm_(self.params, a.clip) if a.clip > 0 else torch.tensor(0.0)
            self.opt.step()
            self.sched.step()
            self.opt.zero_grad(set_to_none=True)
            self.step += 1
            dt = time.time() - t0
            self.hist["train"].append({"step": self.step, "loss": tot_loss, "s": dt})
            lrs = " ".join(f"{g['name']}={g['lr']:.2e}" for g in self.opt.param_groups)
            log.info(f"step {self.step}/{self.total} loss {tot_loss:.5f} gnorm {float(gn):.3f} lr {lrs} "
                     f"{dt:.1f}s {ntok / dt:.1f} tok/s ({nmask / dt:.1f} masked tok/s) {mem_report()}")
            if a.eval_every and self.step % a.eval_every == 0 and self.step < self.total:
                self.evaluate(f"step {self.step}")
                if a.patience and self.bad_evals >= a.patience:
                    self.save()
                    log.info(f"early stop: held-out KL did not improve for {a.patience} evals (best {self.best:.5f})")
                    return True
            if self.step % a.save_every == 0 or self.step == stop:
                self.save()
            if deadline and time.time() > deadline:
                self.save()
                log.info("deadline reached, checkpoint saved; resubmit to continue")
                return False
        return self.step >= self.total


# ----------------------------------------------------------------------------
# Merge (file level, CPU)
# ----------------------------------------------------------------------------


def _tensor_bytes(t, dtype):
    if dtype == "BF16":
        return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    assert dtype == "F32"
    return t.float().contiguous().numpy().tobytes()


def merge(src, delta_path, dst, layers_below=None, link_unchanged=False):
    from safetensors import safe_open
    from build_pruned import out_header, read_header, sha256, CHUNK

    D = torch.load(delta_path, map_location="cpu", weights_only=False)
    cfg = json.load(open(os.path.join(src, "config.json")))
    qcfg = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config") or {}
    block = tuple(qcfg.get("weight_block_size", (128, 128)))
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    changed = {n: "BF16" for n in D["lora"]} | {n: "F32" for n in D["full"]}
    for n in changed:
        assert n in wm, f"delta tensor {n} not in {src}"
    drop = {n + "_scale_inv" for n in D["lora"] if n + "_scale_inv" in wm}
    lay_re = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.")

    def keep(n):
        if n in drop:
            return False
        if layers_below is None:
            return True
        m = lay_re.match(n)
        return m is None or int(m.group(1)) < layers_below
    assert not (link_unchanged and layers_below is not None), "--link-unchanged copies whole shards"

    os.makedirs(dst, exist_ok=True)
    manifest = {"src": os.path.abspath(src), "delta_sha256": sha256(delta_path), "layers_below": layers_below}
    mp = os.path.join(dst, "heal_manifest.json")
    if os.path.exists(mp) and json.load(open(mp)) != manifest:
        sys.exit(f"refusing to resume: {mp} differs from this run; use a fresh --merged-out")
    json.dump(manifest, open(mp, "w"), indent=1)

    new_wm, total = {}, 0
    for k, sh in enumerate(sorted(set(wm.values()))):
        sp = os.path.join(src, sh)
        hdr, base = read_header(sp)
        names = [n for n in hdr if n != "__metadata__" and keep(n)]
        if not names:
            continue
        touched = [n for n in names if n in changed] or [n for n in hdr if n in drop]
        dp = os.path.join(dst, sh)
        if link_unchanged and not touched:
            if not os.path.lexists(dp):
                os.symlink(os.path.abspath(sp), dp)
            for n in names:
                new_wm[n] = sh
            total += sum(hdr[n]["data_offsets"][1] - hdr[n]["data_offsets"][0] for n in names)
            continue
        ops, payload = [], {}
        with safe_open(sp, "pt") as f:
            for n in names:
                b0, b1 = hdr[n]["data_offsets"]
                if n in changed:
                    if n in D["lora"]:
                        L = D["lora"][n]
                        w = f.get_tensor(n)
                        if w.dtype == torch.float8_e4m3fn:
                            w = glm.dequant(w.view(torch.uint8), f.get_tensor(n + "_scale_inv"), block, torch.float32)
                        w = w.float() + L["scale"] * (L["B"].float() @ L["A"].float())
                    else:
                        w = D["full"][n]
                        assert list(w.shape) == hdr[n]["shape"], (n, w.shape, hdr[n]["shape"])
                    payload[n] = _tensor_bytes(w, changed[n])
                    ops.append((n, None, changed[n], hdr[n]["shape"], None, 0, len(payload[n]), "mem"))
                else:
                    ops.append((n, n, hdr[n]["dtype"], hdr[n]["shape"], base + b0, b1 - b0, b1 - b0, None))
        meta = hdr.get("__metadata__") or {"format": "pt"}
        raw, data_len = out_header(ops, meta)
        size = 8 + len(raw) + data_len
        for op in ops:
            new_wm[op[0]] = sh
        total += data_len
        if os.path.exists(dp) and not os.path.islink(dp) and os.path.getsize(dp) == size:
            print(f"[{k + 1}] {sh}: exists, skipped", flush=True)
            continue
        tmp = dp + ".tmp"
        with open(sp, "rb") as fi, open(tmp, "wb") as fo:
            fo.write(struct.pack("<Q", len(raw)))
            fo.write(raw)
            for dst_n, _, _, _, off, src_nb, _, kind in ops:
                if kind == "mem":
                    fo.write(payload[dst_n])
                    continue
                fi.seek(off)
                left = src_nb
                while left:
                    b = fi.read(min(CHUNK, left))
                    assert b, "short read"
                    fo.write(b)
                    left -= len(b)
            fo.flush()
            os.fsync(fo.fileno())
        assert os.path.getsize(tmp) == size
        os.replace(tmp, dp)
        print(f"[{k + 1}] {sh}: {len(ops)} tensors, {sum(1 for o in ops if o[7] == 'mem')} healed", flush=True)

    # verify the healed tensors read back as written
    for n, dt in changed.items():
        if n not in new_wm:
            continue
        with safe_open(os.path.join(dst, new_wm[n]), "pt") as f:
            t = f.get_tensor(n)
        assert t.dtype == {"BF16": torch.bfloat16, "F32": torch.float32}[dt], (n, t.dtype)
        assert n + "_scale_inv" not in new_wm

    # config: healed FP8 linears are BF16 now, so FP8-aware loaders (transformers, vLLM) must not expect a scale
    lin = sorted(n[:-len(".weight")].replace("language_model.", "") for n in D["lora"] if n + "_scale_inv" in wm)
    for d in (cfg, cfg.get("text_config", {})):
        q = d.get("quantization_config")
        if q is not None and "modules_to_not_convert" in q:
            q["modules_to_not_convert"] = sorted(set(q["modules_to_not_convert"]) | set(lin))
    cfg["heal"] = {"delta_sha256": manifest["delta_sha256"], "layers_below": layers_below,
                   "n_healed_tensors": len(changed)}
    for fn in sorted(os.listdir(src)):
        p = os.path.join(src, fn)
        if (os.path.isfile(p) and not fn.startswith(".") and not fn.endswith(".safetensors")
                and fn not in ("config.json", "model.safetensors.index.json")):
            shutil.copy2(p, os.path.join(dst, fn))
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    tmp = os.path.join(dst, "model.safetensors.index.json.tmp")
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_wm.items()))}, open(tmp, "w"), indent=2)
    os.replace(tmp, os.path.join(dst, "model.safetensors.index.json"))
    print(f"merge done: {len(new_wm)} tensors ({len(changed)} healed, {len(drop)} scale_inv dropped), "
          f"{total / 1e9:.2f} GB in {dst}", flush=True)
    return new_wm


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pruned FP8 HF checkpoint dir")
    ap.add_argument("--out", help="train: output dir (checkpoints, deltas, log)")
    ap.add_argument("--calib-dir", default="/work/glm53/calib53")
    ap.add_argument("--ref", default="/work/glm53/calib53/v2/ref_top20.pt",
                    help="top-20 reference (v2 = made with dsa_fix; the old ref_top20.pt has the ROCm SDPA bug)")
    ap.add_argument("--extra", action="append", default=[], help="CALIB.pt:REF.pt[:key] extra training windows")
    # trainables
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32)
    ap.add_argument("--lora-targets", default=DEFAULT_LORA, help="regex on module paths inside the text model")
    ap.add_argument("--router", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--norms", action=argparse.BooleanOptionalAction, default=False)
    # optimization
    ap.add_argument("--lr", type=float, default=2e-5, help="LoRA lr")
    ap.add_argument("--router-lr", type=float, default=1e-5)
    ap.add_argument("--norm-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="LoRA only; router/norms always 0")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps; 0 = from --epochs")
    ap.add_argument("--warmup-steps", type=int, default=4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=4, help="optimizer steps; 0 = only before/after")
    ap.add_argument("--patience", type=int, default=3, help="stop after this many evals without improvement (0 = off)")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="improvement smaller than this counts as none")
    ap.add_argument("--eval-limit", type=int, default=0, help="only the first n held-out windows (0 = all 64)")
    ap.add_argument("--gguf-experts", default=None, help="routed experts from this GGUF (any type in gguf_experts_any)")
    ap.add_argument("--eval-subset", type=int, default=0, help="periodic evals on n stratified held-out windows (0 = all)")
    ap.add_argument("--final-full-eval", action=argparse.BooleanOptionalAction, default=True,
                    help="at the end: delta_best on ALL held-out windows")
    ap.add_argument("--full-before", action=argparse.BooleanOptionalAction, default=True,
                    help="step-0 eval on ALL held-out windows (unhealed KL + top-1 on the full set)")
    ap.add_argument("--compare-kl-dir", default=None, help="eval_masks_mn kl dir with the same mask's per-window rows")
    ap.add_argument("--compare-mask", default=None, help="mask name inside --compare-kl-dir")
    ap.add_argument("--iq2-gguf", default=None, help="replace the routed experts with this GGUF's IQ2_XXS bytes")
    ap.add_argument("--q-chunk", type=int, default=256, help="dsa_fix: DSA queries per chunk")
    ap.add_argument("--kda-group", type=int, default=8, help="dsa_fix: KDA blocks per intra-chunk group")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--deadline-min", type=float, default=0, help="save and exit after this many minutes (0 = off)")
    # test knobs
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=0, help="truncate windows to this many tokens")
    ap.add_argument("--limit", type=int, default=0, help="only the first n windows of each set")
    # merge
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--delta", help="merge: delta.pt or delta_best.pt")
    ap.add_argument("--merged-out")
    ap.add_argument("--layers-below", type=int, default=None)
    ap.add_argument("--link-unchanged", action="store_true", help="merge: symlink shards with no healed tensor")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.merge:
        merge(a.model, a.delta, a.merged_out, a.layers_below, a.link_unchanged)
        return
    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(a.out, "log.txt")), logging.StreamHandler()])
    torch.manual_seed(a.seed)
    t_start = time.time()
    R = torch.load(a.ref or os.path.join(a.calib_dir, "ref_top20.pt"))
    train = load_windows(os.path.join(a.calib_dir, "calib_search.pt"), R["search"], a.max_len, a.limit)
    heldout = load_windows(os.path.join(a.calib_dir, "calib_heldout.pt"), R["heldout"], a.max_len, a.limit)
    if a.eval_limit:
        heldout = heldout[:a.eval_limit]
    meta = torch.load(os.path.join(a.calib_dir, "calib_heldout.pt"), weights_only=False)["meta"][:len(heldout)]
    ho_idx = stratified(meta, a.eval_subset) if a.eval_subset else list(range(len(heldout)))
    ref_rows = {}
    if a.compare_kl_dir and a.compare_mask:
        for f in sorted(os.listdir(a.compare_kl_dir)):
            if f.startswith("kl_shard") and f.endswith(".pt"):
                ref_rows.update(torch.load(os.path.join(a.compare_kl_dir, f), weights_only=False)["masks"].get(a.compare_mask, {}))
    log.info(f"held-out eval windows {len(ho_idx)} of {len(heldout)} "
             f"({ {d: sum(1 for i in ho_idx if meta[i]['bucket'] == d) for d in DOMS} }); "
             f"scorer rows for the step-0 check: {len(ref_rows)}")
    for spec in a.extra:
        c, r, *key = spec.split(":")
        train += load_windows(c, torch.load(r)[key[0] if key else "search"], a.max_len)
    del R
    log.info(f"windows: train {len(train)}, held-out {len(heldout)}; "
             f"model-written tokens train {sum(int(w[1][1:].sum()) for w in train)}")

    dsa_fix.install_long_context_patches(a.q_chunk, a.kda_group)
    log.info(f"dsa_fix installed (q-chunk {a.q_chunk}, kda-group {a.kda_group}); ref {a.ref}")
    devices = [torch.device("cpu")] if a.cpu else None
    model = glm.load_glm_fp8(a.model, devices=devices, n_layers=a.n_layers)
    if a.gguf_experts:
        eb = gea.ExpertBytesAny(a.gguf_experts)
        done_l = gea.install_any_experts(model, eb)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info(f"GGUF experts {dict(eb.types())} from {a.gguf_experts} in {len(done_l)} sparse layers; {mem_report()}")
    if a.iq2_gguf:
        done_l = ge.install_iq2_experts(model, ge.ExpertBytes(a.iq2_gguf))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info(f"IQ2_XXS experts from {a.iq2_gguf} installed in {len(done_l)} sparse layers; {mem_report()}")
    n_exp = [blk.gate.weight.shape[0] for _, blk in glm.moe_blocks(model)]
    log.info(f"loaded in {time.time() - t_start:.0f}s; routed experts per sparse layer {sorted(set(n_exp))}; {mem_report()}")
    T = setup_trainable(model, a.lora_r, a.lora_alpha, a.lora_targets, a.router, a.norms)
    P = trainable_params(T)
    log.info("trainable: " + ", ".join(f"{k} {len(T[k])} tensors / {sum(p.numel() for p in P[k]) / 1e6:.2f}M params"
                                       for k in T))
    tr = Trainer(model, T, train, heldout, a.out, a)
    tr.bad_evals = 0
    tr.meta, tr.ho_idx, tr.ref_rows = meta, ho_idx, ref_rows
    if os.path.exists(tr.ckpt_path()):
        tr.load()
    if not tr.hist["heldout"]:
        tr.evaluate("before")
        tr.save()
    deadline = t_start + 60 * a.deadline_min if a.deadline_min else None
    done = tr.run(deadline=deadline)
    if done:
        if not tr.hist["heldout"] or tr.hist["heldout"][-1]["step"] != tr.step:
            tr.evaluate("after")
            tr.save()
        h = tr.hist["heldout"]
        log.info(f"held-out weighted KL before {h[0]['kl']:.5f} -> after {h[-1]['kl']:.5f} (best {tr.best:.5f}); "
                 f"{mem_report()}")
        full_p = os.path.join(a.out, "final_full_eval.json")
        if a.final_full_eval and len(ho_idx) < len(heldout) and not os.path.exists(full_p):
            best = torch.load(os.path.join(a.out, "delta_best.pt"), map_location="cpu", weights_only=False)
            load_trainable_state(T, best)
            t0 = time.time()
            rows = tr.eval_rows(list(range(len(heldout))))
            sm = kl_summary(rows, meta)
            sm["top1"] = kl_summary(tr._top1, meta)
            tr.save_rows(f"heal_best_step{best['meta']['step']}_full", rows)
            res = {"best_step": best["meta"]["step"], "healed_full": sm}
            if ref_rows and len(ref_rows) == len(heldout):
                res["scorer_unhealed_full"] = kl_summary(ref_rows, meta)
                log.info(f"FULL unhealed (scorer {a.compare_mask}): {fmt(res['scorer_unhealed_full'])}")
            log.info(f"FULL healed (delta_best, step {best['meta']['step']}): {fmt(sm)} | top-1 {fmt(sm['top1'])} ({len(rows)} windows, "
                     f"{time.time() - t0:.0f}s)")
            json.dump(res, open(full_p + ".tmp", "w"), indent=1)
            os.replace(full_p + ".tmp", full_p)
        open(os.path.join(a.out, "DONE"), "w").write(json.dumps(h) + "\n")
        log.info("DONE")


if __name__ == "__main__":
    main()
