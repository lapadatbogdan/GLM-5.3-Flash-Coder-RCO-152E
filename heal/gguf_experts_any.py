#!/usr/bin/env python3
"""Frozen GGUF experts of any published r144 quant (Q8_0, Q4_K_M, Q3_K, IQ2_S, IQ2_XXS), dequantized on the fly in
torch, bit-identical to llama.cpp's dequantize_row_* (fp32). Generalizes gguf_experts.py (IQ2_XXS only).

    from gguf_experts_any import ExpertBytesAny, install_any_experts
    eb = ExpertBytesAny(".../out/GLM-5.3-Flash-Coder-r144-Q4KM.gguf")
    install_any_experts(model, eb)   # model from glm5_next_rco.load_glm_fp8(pruned/r144): FP8 experts -> GGUF bytes

The GGUF is never memory-mapped: the header is parsed with buffered reads and each tensor's bytes are read with
seek/read (the login node has a ~64 GB virtual memory limit; the files are 45-97 GB).

Expert ggml types found in the r144 GGUFs (gate/up/down):
    pruned-r144-q8_0.gguf   Q8_0 / Q8_0 / Q8_0
    r144-Q4KM.gguf          Q4_K / Q4_K / Q4_K or Q6_K (down: 23 layers Q4_K, 19 layers Q6_K)
    r144-Q3K.gguf           Q3_K / Q3_K / Q3_K
    r144-IQ2S.gguf          IQ2_S / IQ2_S / IQ2_S
    r144-IQ2XXS.gguf        IQ2_XXS (reuses gguf_experts.dequant_iq2xxs)
Also implemented (same verification path, not used by the files above): Q5_K, Q2_K.

Layout (ggml ne = cols, rows, E):  gate/up (E, inter, hidden), down (E, hidden, inter); rows are contiguous blocks.
"""
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = "/work/glm53"
sys.path.insert(0, os.path.join(P, "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGMLQuantizationType as Q  # noqa: E402
from gguf.quants import IQ2_S  # noqa: E402

import gguf_experts as _ge  # noqa: E402  (IQ2_XXS dequant, unchanged)

# ----------------------------------------------------------------------------
# GGUF header (no mmap)
# ----------------------------------------------------------------------------

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _Stream:
    def __init__(self, f):
        self.f, self.buf, self.pos, self.base = f, b"", 0, 0

    def take(self, n):
        while len(self.buf) - self.pos < n:
            chunk = self.f.read(max(n, 16 << 20))
            if not chunk:
                raise EOFError("truncated GGUF header")
            self.base += self.pos
            self.buf, self.pos = self.buf[self.pos:] + chunk, 0
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def tell(self):
        return self.base + self.pos


def read_gguf_header(path):
    """-> (kv dict (arrays summarized), [(name, ne list, ggml type int, abs byte offset)], data_start)."""
    with open(path, "rb") as f:
        s = _Stream(f)
        if s.take(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF")
        s.u("<I")
        nt, nkv = s.u("<Q"), s.u("<Q")

        def val(t):
            if t == 8:
                return s.take(s.u("<Q")).decode("utf-8", "replace")
            if t == 9:
                it, n = s.u("<I"), s.u("<Q")
                if it in _SCALAR:
                    s.take(struct.calcsize(_SCALAR[it]) * n)
                else:
                    for _ in range(n):
                        val(it)
                return f"<array {n}>"
            return s.u(_SCALAR[t])

        kv = {}
        for _ in range(nkv):
            k = s.take(s.u("<Q")).decode()
            kv[k] = val(s.u("<I"))
        infos = []
        for _ in range(nt):
            name = s.take(s.u("<Q")).decode()
            ne = [s.u("<Q") for _ in range(s.u("<I"))]
            infos.append((name, ne, s.u("<I"), s.u("<Q")))
        al = kv.get("general.alignment", 32)
        start = (s.tell() + al - 1) // al * al
    return kv, [(n, ne, t, start + o) for n, ne, t, o in infos], start


# ----------------------------------------------------------------------------
# Dequantization (torch, vectorized, fp32 math in the same order as ggml-quants.c)
# ----------------------------------------------------------------------------

def _f16(b, i):
    """fp16 field at byte offset i of each block row -> (n, 1) fp32 (exact)."""
    return b[:, i:i + 2].contiguous().view(torch.float16).float()


def _i32(x):
    return x.to(torch.int32)


def dq_q8_0(b):                       # 34 bytes / 32: fp16 d, int8 qs[32]
    d = _f16(b, 0)
    q = b[:, 2:34].contiguous().view(torch.int8).float()
    return q * d


def _scale_min_k4(s):
    """get_scale_min_k4 for j = 0..7 on (n, 12) uint8 -> sc (n, 8), m (n, 8) int32."""
    s = _i32(s)
    sc = torch.cat([s[:, 0:4] & 63, (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)], 1)
    m = torch.cat([s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)], 1)
    return sc, m


def dq_q4_k(b):                       # 144 bytes: d, dmin, scales[12], qs[128]
    n = b.shape[0]
    d, dmin = _f16(b, 0), _f16(b, 2)
    sc, m = _scale_min_k4(b[:, 4:16])
    d1 = d * sc.float()                                       # (n, 8)  d * sc
    m1 = dmin * m.float()                                     # (n, 8)  min * m
    q = _i32(b[:, 16:144]).view(n, 4, 1, 32)
    q = torch.cat([q & 0xF, q >> 4], 2).view(n, 8, 32).float()  # sub-block 2j: low nibbles, 2j+1: high
    return d1[..., None] * q - m1[..., None]


def dq_q5_k(b):                       # 176 bytes: d, dmin, scales[12], qh[32], qs[128]
    n = b.shape[0]
    d, dmin = _f16(b, 0), _f16(b, 2)
    sc, m = _scale_min_k4(b[:, 4:16])
    d1, m1 = d * sc.float(), dmin * m.float()
    qh = _i32(b[:, 16:48]).view(n, 1, 32)
    ql = _i32(b[:, 48:176]).view(n, 4, 1, 32)
    lo = torch.cat([ql & 0xF, ql >> 4], 2).view(n, 8, 32)
    hb = (qh >> torch.arange(8, device=b.device, dtype=torch.int32).view(1, 8, 1)) & 1   # sub-block k uses bit k
    q = (lo + (hb << 4)).float()
    return d1[..., None] * q - m1[..., None]


def dq_q6_k(b):                       # 210 bytes: ql[128], qh[64], scales[16] int8, d
    n = b.shape[0]
    dev = b.device
    ql = _i32(b[:, 0:128]).view(n, 2, 1, 2, 32)               # [half n][.][ql half (l / l+32)][l]
    qh = _i32(b[:, 128:192]).view(n, 2, 1, 32)
    sc = b[:, 192:208].contiguous().view(torch.int8).float().view(n, 2, 4, 2)  # [half][k][l//16]
    d = _f16(b, 208)
    k = torch.arange(4, device=dev, dtype=torch.int32)
    lo = torch.cat([ql & 0xF, ql >> 4], 2).view(n, 2, 4, 32)   # k = 0: ql[l]&F, 1: ql[l+32]&F, 2: ql[l]>>4, 3: ql[l+32]>>4
    hi = (qh >> (2 * k).view(1, 1, 4, 1)) & 3                  # (n, 2, 4, 32)
    q = ((lo | (hi << 4)) - 32).float().view(n, 2, 4, 2, 16)
    return ((d.view(n, 1, 1, 1, 1) * sc[..., None]) * q).view(n, 256)


def dq_q3_k(b):                       # 110 bytes: hmask[32], qs[64], scales[12], d
    n = b.shape[0]
    dev = b.device
    hm = _i32(b[:, 0:32]).view(n, 1, 1, 2, 16)
    qs = _i32(b[:, 32:96]).view(n, 2, 1, 2, 16)               # [n128][.][h][l]
    s = _i32(b[:, 96:108])
    i = torch.arange(16, device=dev)
    low = (s[:, i % 8] >> (4 * (i // 8)).to(torch.int32)) & 0xF
    high = (s[:, 8 + i % 4] >> (2 * (i // 4)).to(torch.int32)) & 3
    scl = ((low | (high << 4)) - 32).float()                  # (n, 16) index n128*8 + j*2 + h
    d = _f16(b, 108)
    dl = (d * scl).view(n, 2, 4, 2, 1)
    j = torch.arange(4, device=dev, dtype=torch.int32).view(1, 1, 4, 1, 1)
    q = (qs >> (2 * j)) & 3                                   # (n, 2, 4, 2, 16)
    bit = (torch.arange(2, device=dev, dtype=torch.int32).view(1, 2, 1, 1, 1) * 4 + j)
    h = (hm >> bit) & 1
    q = (q - 4 * (1 - h)).float()
    return (dl * q).view(n, 256)


def dq_q2_k(b):                       # 84 bytes: scales[16], qs[64], d, dmin
    n = b.shape[0]
    dev = b.device
    s = _i32(b[:, 0:16]).view(n, 2, 4, 2)                     # [n128][shift j][l//16 half]
    qs = _i32(b[:, 16:80]).view(n, 2, 1, 2, 16)
    d, dmin = _f16(b, 80), _f16(b, 82)
    dl = (d.view(n, 1, 1, 1) * (s & 0xF).float())[..., None]
    ml = (dmin.view(n, 1, 1, 1) * (s >> 4).float())[..., None]
    j = torch.arange(4, device=dev, dtype=torch.int32).view(1, 1, 4, 1, 1)
    q = ((qs >> (2 * j)) & 3).float()
    return (dl * q - ml).view(n, 256)


_IQ2S = {}


def _iq2s_consts(dev):
    if dev not in _IQ2S:
        if IQ2_S.grid is None:
            IQ2_S.init_grid()
        grid = torch.from_numpy(np.asarray(IQ2_S.grid, dtype=np.float32).reshape(1024, 8)).to(dev)
        _IQ2S[dev] = (grid, torch.arange(8, device=dev, dtype=torch.int32),
                      torch.tensor([8, 6, 4, 2], device=dev, dtype=torch.int32))
    return _IQ2S[dev]


def dq_iq2_s(b):                      # 82 bytes: d, qs[32] grid lo, signs[32], qh[8], scales[8]
    n = b.shape[0]
    grid, bit8, qhs = _iq2s_consts(b.device)
    d = _f16(b, 0)
    qs = _i32(b[:, 2:34]).view(n, 8, 4)
    sg = _i32(b[:, 34:66]).view(n, 8, 4)
    qh = _i32(b[:, 66:74]).view(n, 8, 1)
    sc = _i32(b[:, 74:82])
    db = torch.stack([d * (0.5 + (sc & 0xF).float()) * 0.25,
                      d * (0.5 + (sc >> 4).float()) * 0.25], -1)                  # (n, 8, 2)
    dl = db.repeat_interleave(2, -1)                                               # (n, 8, 4): l//2
    gi = qs | ((qh << qhs) & 0x300)
    g = grid[gi.long()]                                                            # (n, 8, 4, 8)
    sign = 1.0 - 2.0 * ((sg[..., None] >> bit8) & 1).float()
    return (dl[..., None] * g * sign).view(n, 256)


def _dq_iq2_xxs(b):
    return _ge.dequant_iq2xxs(b, b.shape[0], 256, torch.float32)


# type -> (block elements, block bytes, fn(uint8 (n, block_bytes)) -> fp32 (n, block_elems))
QTYPES = {
    Q.Q8_0: (32, 34, dq_q8_0),
    Q.Q2_K: (256, 84, dq_q2_k),
    Q.Q3_K: (256, 110, dq_q3_k),
    Q.Q4_K: (256, 144, dq_q4_k),
    Q.Q5_K: (256, 176, dq_q5_k),
    Q.Q6_K: (256, 210, dq_q6_k),
    Q.IQ2_S: (256, 82, dq_iq2_s),
    Q.IQ2_XXS: (256, 66, _dq_iq2_xxs),
}


def row_bytes(qtype, cols):
    be, bb, _ = QTYPES[Q(qtype)]
    assert cols % be == 0, (qtype, cols)
    return cols // be * bb


def dequant(raw, qtype, rows, cols, dtype=torch.bfloat16):
    """raw: uint8 tensor (any shape, contiguous) holding rows x cols of `qtype` -> (rows, cols) `dtype`."""
    be, bb, fn = QTYPES[Q(qtype)]
    w = fn(raw.reshape(-1, bb))
    return w.reshape(rows, cols).to(dtype)


# ----------------------------------------------------------------------------
# Expert bytes
# ----------------------------------------------------------------------------

class ExpertBytesAny:
    """Index of every blk.L.ffn_{gate,up,down}_exps.weight in a GGUF (offset, type, shape); bytes read on demand."""

    KINDS = ("gate", "up", "down")

    def __init__(self, path):
        self.path = path
        self.kv, infos, self.data_start = read_gguf_header(path)
        self.t = {}
        for name, ne, ty, off in infos:
            if "_exps." not in name:
                continue
            if Q(ty) not in QTYPES:
                raise ValueError(f"{name}: unsupported type {Q(ty).name}")
            cols, rows, E = ne
            self.t[name] = dict(off=off, qtype=Q(ty), rows=rows, cols=cols, E=E, rb=row_bytes(ty, cols))
        self.layers = sorted({int(n.split(".")[1]) for n in self.t})

    def info(self, L, k):
        return self.t[f"blk.{L}.ffn_{k}_exps.weight"]

    def types(self):
        from collections import Counter
        return Counter((n.split(".")[2], v["qtype"].name) for n, v in self.t.items())

    def nbytes(self):
        return sum(v["E"] * v["rows"] * v["rb"] for v in self.t.values())

    def _read(self, off, n):
        out = np.empty(n, dtype=np.uint8)
        with open(self.path, "rb") as f:
            f.seek(off)
            mv, got = memoryview(out), 0
            while got < n:
                r = f.readinto(mv[got:])
                if not r:
                    raise EOFError(f"{self.path}: short read at {off + got}")
                got += r
        return out

    def expert(self, L, k, e):
        """Raw bytes of one expert: (uint8 (rows, row_bytes), qtype, rows, cols)."""
        i = self.info(L, k)
        per = i["rows"] * i["rb"]
        return self._read(i["off"] + e * per, per).reshape(i["rows"], i["rb"]), i["qtype"], i["rows"], i["cols"]

    def layer(self, L):
        """{'gate','up','down'}: (uint8 array (E, rows, row_bytes), rows, cols, qtype)."""
        out = {}
        for k in self.KINDS:
            i = self.info(L, k)
            arr = self._read(i["off"], i["E"] * i["rows"] * i["rb"]).reshape(i["E"], i["rows"], i["rb"])
            out[k] = (arr, i["rows"], i["cols"], i["qtype"])
        return out


# ----------------------------------------------------------------------------
# Experts module
# ----------------------------------------------------------------------------

class _QLinear(torch.autograd.Function):
    """y = x @ W^T, W dequantized from GGUF bytes; backward dequantizes again, no weight gradient."""

    @staticmethod
    def forward(ctx, x, raw, qtype, rows, cols):
        ctx.save_for_backward(raw)
        ctx.meta = (qtype, rows, cols)
        return F.linear(x, dequant(raw, qtype, rows, cols, x.dtype))

    @staticmethod
    def backward(ctx, gy):
        (raw,) = ctx.saved_tensors
        qtype, rows, cols = ctx.meta
        return gy @ dequant(raw, qtype, rows, cols, gy.dtype), None, None, None, None


class GGUFExperts(nn.Module):
    """Drop-in for glm5_next_rco.Fp8Experts: forward(hidden, top_k_index, top_k_weights), GGUF bytes inside."""

    def __init__(self, layer_bytes, device, swiglu_limit):
        super().__init__()
        g, u, dn = layer_bytes["gate"], layer_bytes["up"], layer_bytes["down"]
        self.num_experts = g[0].shape[0]
        self.inter, self.hidden = g[1], g[2]
        assert u[1:3] == g[1:3] and dn[1] == self.hidden and dn[2] == self.inter, (g[1:3], u[1:3], dn[1:3])
        self.qt = {"gate": g[3], "up": u[3], "down": dn[3]}
        self.swiglu_limit = swiglu_limit
        self.register_buffer("gate", torch.from_numpy(np.ascontiguousarray(g[0])).to(device))
        self.register_buffer("up", torch.from_numpy(np.ascontiguousarray(u[0])).to(device))
        self.register_buffer("down", torch.from_numpy(np.ascontiguousarray(dn[0])).to(device))

    def weights(self, e, dtype=torch.float32):
        """Dequantized (gate_up (2*inter, hidden), down (hidden, inter)) of expert e, for checks."""
        gu = torch.cat([dequant(self.gate[e], self.qt["gate"], self.inter, self.hidden, dtype),
                        dequant(self.up[e], self.qt["up"], self.inter, self.hidden, dtype)], 0)
        return gu, dequant(self.down[e], self.qt["down"], self.hidden, self.inter, dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        qg, qu, qd = self.qt["gate"], self.qt["up"], self.qt["down"]
        for e in hit[:, 0].tolist():
            pos, tok = torch.where(mask[e])
            x = hidden_states[tok]
            g = _QLinear.apply(x, self.gate[e], qg, self.inter, self.hidden)
            u = _QLinear.apply(x, self.up[e], qu, self.inter, self.hidden)
            g = g.clamp(max=self.swiglu_limit)
            u = u.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            cur = _QLinear.apply(F.silu(g) * u, self.down[e], qd, self.hidden, self.inter)
            cur = cur * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final


def install_any_experts(model, eb):
    """Replace every sparse layer's Fp8Experts with GGUFExperts (on the same device); returns the layer indices."""
    import glm5_next_rco as glm
    done = []
    for L, blk in glm.moe_blocks(model):
        old = blk.experts
        dev = old.gate_up.device
        new = GGUFExperts(eb.layer(L), dev, old.swiglu_limit)
        assert new.num_experts == old.num_experts, (L, new.num_experts, old.num_experts)
        assert (2 * new.inter, new.hidden) == tuple(old.gate_up.shape[1:]), (L, old.gate_up.shape)
        blk.experts = new
        del old
        done.append(L)
    return done


expert_digest = _ge.expert_digest   # works unchanged: buffers are named gate/up/down
