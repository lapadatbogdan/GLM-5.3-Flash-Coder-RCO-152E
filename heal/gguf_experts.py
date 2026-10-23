#!/usr/bin/env python3
"""Experts of a pruned GLM-5.3-Flash exactly as they run on the box: IQ2_XXS bytes read from the GGUF, dequantized
on the fly in torch (forward and backward), frozen. Used to train LoRA on the non-expert parts against the real
2-bit quantization error (RLVR / RFT for the box variant).

    from gguf_experts import ExpertBytes, install_iq2_experts
    eb = ExpertBytes(".../out/GLM-5.3-Flash-Coder-r144-IQ2XXS.gguf")
    install_iq2_experts(model, eb)   # model from glm5_next_rco.load_glm_fp8(pruned/r144): FP8 experts -> IQ2 bytes

Layout (gguf-py reader, numpy shape = reversed ggml ne):
    blk.L.ffn_gate_exps / ffn_up_exps: (E, inter, hidden) -> Fp8Experts.gate_up (E, 2*inter, hidden), gate rows first
    blk.L.ffn_down_exps:              (E, hidden, inter)  -> Fp8Experts.down    (E, hidden, inter)
IQ2_XXS block = 256 weights in 66 bytes: fp16 d, then 8 groups of (uint32 grid bytes, uint32 signs|scale).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = "/work/glm53"
sys.path.insert(0, os.path.join(P, "llama.cpp", "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402
from gguf.quants import IQ2_XXS  # noqa: E402

QK = 256
BLOCK_BYTES = 66
_CONST = {}


def _consts(device):
    """IQ2_XXS lookup tables (from gguf-py, which mirrors ggml-common.h) on `device`."""
    if device not in _CONST:
        if IQ2_XXS.grid is None:
            IQ2_XXS.init_grid()
        grid = torch.from_numpy(np.asarray(IQ2_XXS.grid, dtype=np.float32).reshape(256, 8)).to(device)
        ksigns = torch.from_numpy(np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).astype(np.int64)).to(device)
        _CONST[device] = (grid, ksigns,
                          torch.tensor([0, 7, 14, 21], device=device, dtype=torch.int64),
                          torch.arange(8, device=device, dtype=torch.int64),
                          torch.tensor([0, 8, 16, 24], device=device, dtype=torch.int64))
    return _CONST[device]


def dequant_iq2xxs(raw, rows, cols, dtype=torch.bfloat16):
    """raw: uint8 tensor holding rows*cols/256 IQ2_XXS blocks (any shape, contiguous) -> (rows, cols) dtype."""
    dev = raw.device
    grid, ksigns, sshift, bit8, bshift = _consts(dev)
    b = raw.reshape(-1, BLOCK_BYTES)
    n = b.shape[0]
    d = b[:, :2].contiguous().view(torch.float16).float()                     # (n, 1)
    q = b[:, 2:].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF  # (n, 16)
    q = q.view(n, 8, 2)
    q0, q1 = q[..., 0], q[..., 1]
    db = d * (0.5 + (q1 >> 28).float()) * 0.25                               # (n, 8)
    sidx = (q1.unsqueeze(-1) >> sshift) & 0x7F                                 # (n, 8, 4)
    sbyte = ksigns[sidx]                                                      # (n, 8, 4)
    sign = 1.0 - 2.0 * ((sbyte.unsqueeze(-1) >> bit8) & 1).float()           # (n, 8, 4, 8)
    gidx = (q0.unsqueeze(-1) >> bshift) & 0xFF                                 # (n, 8, 4)
    g = grid[gidx]                                                            # (n, 8, 4, 8)
    w = db[..., None, None] * g * sign
    return w.reshape(rows, cols).to(dtype)


class ExpertBytes:
    """Raw IQ2_XXS bytes of every sparse layer's experts, straight from the GGUF (memory-mapped)."""

    def __init__(self, path):
        self.path = path
        self.r = GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors if "_exps." in t.name}
        for name, t in self.t.items():
            if t.tensor_type != GGMLQuantizationType.IQ2_XXS:
                raise ValueError(f"{name} is {t.tensor_type.name}, expected IQ2_XXS")

    def layer(self, L):
        """{'gate','up','down'}: (uint8 array (E, rows, row_bytes), rows, cols)."""
        out = {}
        for k in ("gate", "up", "down"):
            t = self.t[f"blk.{L}.ffn_{k}_exps.weight"]
            ne = [int(x) for x in t.shape]            # ggml order: cols, rows, E
            cols, rows, E = ne
            arr = np.asarray(t.data).reshape(E, rows, cols // QK * BLOCK_BYTES)
            out[k] = (arr, rows, cols)
        return out


class _QLinear(torch.autograd.Function):
    """y = x @ W^T with W dequantized from IQ2_XXS bytes; backward dequantizes again, no weight gradient."""

    @staticmethod
    def forward(ctx, x, raw, rows, cols):
        ctx.save_for_backward(raw)
        ctx.rc = (rows, cols)
        return F.linear(x, dequant_iq2xxs(raw, rows, cols, x.dtype))

    @staticmethod
    def backward(ctx, gy):
        (raw,) = ctx.saved_tensors
        rows, cols = ctx.rc
        return gy @ dequant_iq2xxs(raw, rows, cols, gy.dtype), None, None, None


class IQ2Experts(nn.Module):
    """Drop-in for glm5_next_rco.Fp8Experts: same forward(hidden, top_k_index, top_k_weights), IQ2_XXS bytes inside."""

    def __init__(self, layer_bytes, device, swiglu_limit):
        super().__init__()
        g, u, dn = layer_bytes["gate"], layer_bytes["up"], layer_bytes["down"]
        self.num_experts = g[0].shape[0]
        self.inter, self.hidden = g[1], g[2]
        assert u[1:] == g[1:] and dn[1] == self.hidden and dn[2] == self.inter, (g[1:], u[1:], dn[1:])
        self.swiglu_limit = swiglu_limit
        self.register_buffer("gate", torch.from_numpy(np.ascontiguousarray(g[0])).to(device))
        self.register_buffer("up", torch.from_numpy(np.ascontiguousarray(u[0])).to(device))
        self.register_buffer("down", torch.from_numpy(np.ascontiguousarray(dn[0])).to(device))

    def weights(self, e, dtype=torch.float32):
        """Dequantized (gate_up (2*inter, hidden), down (hidden, inter)) of expert e, for checks."""
        gu = torch.cat([dequant_iq2xxs(self.gate[e], self.inter, self.hidden, dtype),
                        dequant_iq2xxs(self.up[e], self.inter, self.hidden, dtype)], 0)
        return gu, dequant_iq2xxs(self.down[e], self.hidden, self.inter, dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        for e in hit[:, 0].tolist():
            pos, tok = torch.where(mask[e])
            x = hidden_states[tok]
            g = _QLinear.apply(x, self.gate[e], self.inter, self.hidden)
            u = _QLinear.apply(x, self.up[e], self.inter, self.hidden)
            g = g.clamp(max=self.swiglu_limit)
            u = u.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            cur = _QLinear.apply(F.silu(g) * u, self.down[e], self.hidden, self.inter)
            cur = cur * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final


def install_iq2_experts(model, eb):
    """Replace every sparse layer's Fp8Experts with IQ2Experts (on the same device); returns the layer indices."""
    import glm5_next_rco as glm
    done = []
    for L, blk in glm.moe_blocks(model):
        old = blk.experts
        dev = old.gate_up.device
        new = IQ2Experts(eb.layer(L), dev, old.swiglu_limit)
        assert new.num_experts == old.num_experts, (L, new.num_experts, old.num_experts)
        assert (2 * new.inter, new.hidden) == tuple(old.gate_up.shape[1:]), (L, old.gate_up.shape)
        blk.experts = new
        del old
        done.append(L)
    return done


def expert_digest(model):
    """Checksum of the raw IQ2 bytes, to prove training never touched them."""
    import glm5_next_rco as glm
    h = 0
    for _, blk in glm.moe_blocks(model):
        for buf in (blk.experts.gate, blk.experts.up, blk.experts.down):
            h = (h * 1000003 + int(buf.to(torch.int64).sum().item())) % (1 << 61)
    return h
