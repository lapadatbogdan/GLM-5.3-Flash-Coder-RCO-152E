"""GLM-5.3-Flash (glm5_next) for the RCO expert-pruning search, on AMD ROCm GPUs without FP8 math.

The checkpoint is FP8 e4m3 with 128x128 block scales (weight_scale_inv). The routed experts (~305 GB) stay
FP8 on the GPUs as raw bytes and are dequantized one expert at a time when a token routes to it; everything
else (attention, dense layers, shared experts, router, embeddings) is dequantized to BF16 at load (~32 GB).
That puts the whole 321B model on one node (8 GPUs x 64 GB) with no CPU offload.

Pieces the RCO search needs:
    load_glm_fp8(path, devices, n_layers=None) -> TextLM   model(input_ids=...).logits, as RCO expects
    moe_blocks(model) -> list of (layer_idx, Glm5NextTextMoE) for the sparse layers only (dense 0-2 skipped)
    GlmPruneWrapper   STE survival mask on the routing weights (GLM: sigmoid router, shared expert ungated)
    hard_prune_hooks  deployment routing: pruned experts out of the top-k choice, kept weights renormalized
"""

import json
import logging
import os
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PFX = "model.language_model."

# checkpoint name -> module name (transformers' conversion_mapping.py, "glm5_next"; the experts and the q/k/v
# conv1d concatenation are handled in the loader)
RENAMES = [
    (r"self_attn\.f_a_proj\.", "self_attn.forget_gate.f_a_proj."),
    (r"self_attn\.f_b_proj\.", "self_attn.forget_gate.f_b_proj."),
    (r"self_attn\.dt_bias", "self_attn.forget_gate.dt_bias"),
    (r"self_attn\.A_log", "self_attn.forget_gate.A_log"),
    (r"hc_attn_fn", "attn_hc.fn"), (r"hc_attn_base", "attn_hc.base"), (r"hc_attn_scale", "attn_hc.scale"),
    (r"hc_ffn_fn", "ffn_hc.fn"), (r"hc_ffn_base", "ffn_hc.base"), (r"hc_ffn_scale", "ffn_hc.scale"),
]
CONV_RE = re.compile(r"^(layers\.\d+\.self_attn\.)([qkv])_conv1d\.weight$")


def rename(p):
    for a, b in RENAMES:
        p = re.sub(a, b, p)
    return p


# ----------------------------------------------------------------------------
# FP8 storage
# ----------------------------------------------------------------------------


def _fp8_lut(device):
    # all 256 e4m3fn byte values as float32; indexing avoids needing FP8 kernels on gfx90a
    b = torch.arange(256, dtype=torch.uint8)
    return b.view(torch.float8_e4m3fn).float().to(device)


_LUTS = {}


def dequant(w_u8, scale_inv, block, dtype=torch.bfloat16):
    """(rows, cols) uint8 e4m3 bytes + (ceil(rows/b0), ceil(cols/b1)) scales -> dtype."""
    dev = w_u8.device
    if dev not in _LUTS:
        _LUTS[dev] = _fp8_lut(dev)
    w = _LUTS[dev][w_u8.to(torch.int32)]
    s = scale_inv.float().to(dev)
    s = s.repeat_interleave(block[0], 0)[: w.shape[0]].repeat_interleave(block[1], 1)[:, : w.shape[1]]
    return (w * s).to(dtype)


class _Fp8Linear(torch.autograd.Function):
    """y = x @ W^T with W dequantized on the fly; the backward dequantizes W again instead of keeping the BF16
    copy (one layer's experts would otherwise hold ~14 GB of weights for the backward). No weight gradient."""

    @staticmethod
    def forward(ctx, x, w_u8, s, block):
        ctx.save_for_backward(w_u8, s)
        ctx.block = block
        return F.linear(x, dequant(w_u8, s, block, x.dtype))

    @staticmethod
    def backward(ctx, gy):
        w_u8, s = ctx.saved_tensors
        return gy @ dequant(w_u8, s, ctx.block, gy.dtype), None, None, None


class Fp8Experts(nn.Module):
    """Drop-in for Glm5NextTextExperts: same forward(hidden, top_k_index, top_k_weights), FP8 bytes inside."""

    def __init__(self, n_experts, hidden, inter, swiglu_limit, block, device):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim = n_experts, hidden, inter
        self.swiglu_limit, self.block = swiglu_limit, block
        b0, b1 = block
        e = n_experts
        self.register_buffer("gate_up", torch.empty(e, 2 * inter, hidden, dtype=torch.uint8, device=device))
        self.register_buffer("gate_up_s", torch.empty(e, 2 * (-(-inter // b0)), -(-hidden // b1), device=device))
        self.register_buffer("down", torch.empty(e, hidden, inter, dtype=torch.uint8, device=device))
        self.register_buffer("down_s", torch.empty(e, -(-hidden // b0), -(-inter // b1), device=device))

    def weights(self, e, dtype):
        return (dequant(self.gate_up[e], self.gate_up_s[e], self.block, dtype),
                dequant(self.down[e], self.down_s[e], self.block, dtype))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        for e in hit[:, 0].tolist():
            top_k_pos, token_idx = torch.where(mask[e])
            cur = _Fp8Linear.apply(hidden_states[token_idx], self.gate_up[e], self.gate_up_s[e], self.block)
            g, u = cur.chunk(2, dim=-1)
            g = g.clamp(max=self.swiglu_limit)
            u = u.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            cur = _Fp8Linear.apply(F.silu(g) * u, self.down[e], self.down_s[e], self.block)
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
        return final


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


class _Out:
    def __init__(self, logits):
        self.logits = logits


class TextLM(nn.Module):
    """Glm5NextTextModel + lm_head; the layers sit on several GPUs and the hidden state follows them."""

    def __init__(self, text_model, lm_head):
        super().__init__()
        self.model, self.lm_head = text_model, lm_head

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids=None, **kw):
        h = self.model(input_ids=input_ids, use_cache=False).last_hidden_state
        h = h.to(self.lm_head.weight.device)
        return _Out(self.lm_head(h))


def _move(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, (tuple, list)):
        return type(x)(_move(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: _move(v, dev) for k, v in x.items()}
    return x


def _layer_device_hook(dev):
    def pre(module, args, kwargs):
        return _move(args, dev), _move(kwargs, dev)
    return pre


def _shards_for(index, wanted):
    by = defaultdict(list)
    for name, f in index["weight_map"].items():
        if wanted(name):
            by[f].append(name)
    return by


def load_glm_fp8(path, devices=None, n_layers=None):
    """Build the text model from the FP8 checkpoint at `path`.

    n_layers: keep only the first n decoder layers (a cheap memory/time test); None = all.
    """
    from safetensors import safe_open
    from transformers.models.glm5_next import modeling_glm5_next as M
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    cfg_all = json.load(open(os.path.join(path, "config.json")))
    tcfg = dict(cfg_all.get("text_config", cfg_all))
    qcfg = cfg_all.get("quantization_config") or tcfg.get("quantization_config") or {}
    block = tuple(qcfg.get("weight_block_size", (128, 128)))
    tcfg.pop("quantization_config", None)
    L = tcfg["num_hidden_layers"] if n_layers is None else n_layers
    if n_layers is not None:
        tcfg["num_hidden_layers"] = L
        for k in ("mlp_layer_types", "layer_types"):
            if tcfg.get(k):
                tcfg[k] = tcfg[k][:L]
    cfg = Glm5NextTextConfig(**tcfg)
    cfg._attn_implementation = "sdpa"
    devices = devices or [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    # layer i -> device: experts dominate, so split the sparse layers evenly
    sparse = [i for i in range(L) if cfg.mlp_layer_types[i] == "sparse"]
    dev_of = {}
    for i in range(L):
        k = sparse.index(i) if i in sparse else 0
        # the first device also holds the embeddings and the dense layers, the last one the lm_head and the
        # logits: they count as OFF0 / OFF1 extra sparse layers
        OFF0, OFF1 = 2, 2
        dev_of[i] = (devices[(k + OFF0) * len(devices) // (max(len(sparse), 1) + OFF0 + OFF1)]
                     if i in sparse else devices[0])

    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        tm = M.Glm5NextTextModel(cfg)
        head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    torch.set_default_dtype(torch.float32)
    keep_fp32 = tuple(M.Glm5NextPreTrainedModel._keep_in_fp32_modules_strict)
    # build_pruned.py --per-layer: one routed expert count per decoder layer (0 on dense layers)
    per_layer = tcfg.get("n_routed_experts_per_layer")
    for i, layer in enumerate(tm.layers):
        if cfg.mlp_layer_types[i] == "sparse":
            n_e = cfg.num_local_experts
            if per_layer is not None:
                n_e = per_layer[i]
                assert cfg.num_experts_per_tok <= n_e <= cfg.num_local_experts, (i, n_e)
                g = layer.mlp.gate  # router sized to this layer, still on meta until loaded
                g.num_experts = n_e
                g.weight = nn.Parameter(torch.empty(n_e, cfg.hidden_size, device="meta"), requires_grad=False)
                g.e_score_correction_bias = torch.empty(n_e, device="meta", dtype=g.e_score_correction_bias.dtype)
            layer.mlp.experts = Fp8Experts(n_e, cfg.hidden_size, cfg.moe_intermediate_size,
                                           cfg.swiglu_limit, block, dev_of[i])
        layer.register_forward_pre_hook(_layer_device_hook(dev_of[i]), with_kwargs=True)
    tm.norm.register_forward_pre_hook(_layer_device_hook(devices[-1]), with_kwargs=True)
    tm.hc_head.register_forward_pre_hook(_layer_device_hook(devices[-1]), with_kwargs=True)

    def dev_for(pname):
        m = re.match(r"layers\.(\d+)\.", pname)
        if m:
            return dev_of[int(m.group(1))]
        if pname.startswith(("norm", "hc_head")):
            return devices[-1]
        return devices[0]

    index = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    keep = lambda n: (n.startswith(PFX) and (not re.match(PFX + r"layers\.(\d+)\.", n)
                                            or int(re.match(PFX + r"layers\.(\d+)\.", n).group(1)) < L)) \
        or n == "lm_head.weight"
    own = dict(tm.named_parameters()) | dict(tm.named_buffers())
    loaded = set()
    exp_re = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight(_scale_inv)?$")
    inter = cfg.moe_intermediate_size
    nb_i = -(-inter // block[0])

    convs = defaultdict(dict)
    for shard, names in sorted(_shards_for(index, keep).items()):
        with safe_open(os.path.join(path, shard), "pt") as f:
            for n in names:
                if n.endswith("weight_scale_inv") and not exp_re.search(n):
                    continue  # read together with its weight
                t = f.get_tensor(n)
                if n == "lm_head.weight":
                    head.to_empty(device=devices[-1])
                    head.weight.data.copy_(t.to(torch.bfloat16))
                    loaded.add(n)
                    continue
                p = n[len(PFX):]
                m = exp_re.search(p)
                if m:
                    li, e, kind, is_s = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
                    ex = tm.layers[li].mlp.experts
                    if is_s:
                        if kind == "gate_proj":
                            ex.gate_up_s[e, :nb_i].copy_(t)
                        elif kind == "up_proj":
                            ex.gate_up_s[e, nb_i:].copy_(t)
                        else:
                            ex.down_s[e].copy_(t)
                    else:
                        t = t.view(torch.uint8)
                        if kind == "gate_proj":
                            ex.gate_up[e, :inter].copy_(t)
                        elif kind == "up_proj":
                            ex.gate_up[e, inter:].copy_(t)
                        else:
                            ex.down[e].copy_(t)
                    loaded.add(n)
                    continue
                if t.dtype == torch.float8_e4m3fn:
                    s = f.get_tensor(n + "_scale_inv")
                    t = dequant(t.view(torch.uint8), s, block, torch.float32)
                    loaded.add(n + "_scale_inv")
                cm = CONV_RE.match(p)
                if cm:
                    convs[cm.group(1)][cm.group(2)] = t
                    loaded.add(n)
                    continue
                p = rename(p)
                if p not in own:
                    logger.warning(f"unused checkpoint tensor {n}")
                    continue
                tgt = own[p]
                d = dev_for(p)
                new = t.to(device=d, dtype=torch.float32 if any(k in p for k in keep_fp32) else torch.bfloat16)
                _set(tm, p, new, isinstance(tgt, nn.Parameter))
                loaded.add(n)

    for pre, qkv in convs.items():
        p = pre + "conv1d.weight"
        t = torch.cat([qkv["q"], qkv["k"], qkv["v"]], dim=0)
        _set(tm, p, t.to(device=dev_for(p), dtype=torch.float32 if any(k in p for k in keep_fp32)
                          else torch.bfloat16), isinstance(own[p], nn.Parameter))
    missing = [k for k, v in list(tm.named_parameters()) + list(tm.named_buffers()) if v.device.type == "meta"]
    if missing:
        raise RuntimeError(f"{len(missing)} tensors not loaded, e.g. {missing[:8]}")
    tm.requires_grad_(False)
    head.requires_grad_(False)
    # one layer's expert graph is ~14 GB (every expert hit in a batch keeps its dequantized weights for the
    # backward); recomputing each layer in the backward keeps only one alive at a time
    for layer in tm.layers:
        _checkpoint_when_grad(layer)
    model = TextLM(tm, head).eval()
    model.config = cfg
    logger.info(f"GLM loaded: {L} layers, {len(sparse)} sparse, devices {[str(d) for d in devices]}")
    return model


def _checkpoint_when_grad(layer):
    """Recompute the layer in the backward whenever gradients are on, whatever train()/eval() says (RCO's
    optimize() switches the model to eval, which turns transformers' own checkpointing off)."""
    from torch.utils.checkpoint import checkpoint
    orig = layer.forward

    def fwd(*a, **k):
        if torch.is_grad_enabled():
            return checkpoint(orig, *a, use_reentrant=False, **k)
        return orig(*a, **k)
    layer.forward = fwd


def _set(root, dotted, tensor, as_param):
    *parents, leaf = dotted.split(".")
    m = root
    for p in parents:
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    if as_param:
        setattr(m, leaf, nn.Parameter(tensor, requires_grad=False))
    else:
        m._buffers[leaf] = tensor


# ----------------------------------------------------------------------------
# RCO hooks
# ----------------------------------------------------------------------------


def moe_blocks(model):
    return [(i, l.mlp) for i, l in enumerate(model.model.layers) if hasattr(l.mlp, "gate")]


class GlmPruneWrapper:
    """STE survival mask on the routing weights of one GLM MoE block (RCO's MoEPruneWrapper, GLM layout).

    The routing is GLM's own (sigmoid + correction bias, top-8, normalized, x2.5); a pruned expert's
    contribution is multiplied by its 0/1 mask in the forward and by the soft keep-probability in the backward.
    The shared expert has no gate in GLM and passes through unchanged.
    """

    def __init__(self, block, idx, shared):
        self.block, self.idx, self.shared = block, idx, shared
        self._orig = block.forward
        block.forward = self._forward

    def _forward(self, hidden_states):
        b = self.block
        shape = hidden_states.shape
        if self.shared.get("reroute"):
            return self._forward_reroute(hidden_states)
        _, w, sel = b.gate(hidden_states)
        lm = self.shared["ste_masks"][self.idx].to(w.device)
        w = (w.float() * lm[sel].float()).to(hidden_states.dtype)
        h = hidden_states.reshape(-1, shape[-1])
        out = b.experts(h, sel, w).view(*shape)
        return out + b.shared_experts(hidden_states)

    def _forward_reroute(self, hidden_states):
        """Deployment routing during the search: an expert whose sampled mask is 0 can't be chosen, the router
        takes the next best one and the top-k weights are renormalized (exactly hard_prune_hooks in the forward).
        The mask multiplies the chosen experts' scores before the normalization, so its forward value (1) changes
        nothing and its STE backward tells how much more or less weight each kept expert should get.
        The default path (mask applied after the top-k, no replacement) optimized a different model than the one
        we deploy: 11 Oct, all 6 searches lost to the frequency baseline (KL 1.52 vs 0.94 at 144 kept)."""
        b, g = self.block, self.block.gate
        shape = hidden_states.shape
        h = hidden_states.reshape(-1, shape[-1])
        lm = self.shared["ste_masks"][self.idx].to(h.device)
        scores = F.linear(h.float(), g.weight.float()).sigmoid()
        choice = (scores + g.e_score_correction_bias).masked_fill(lm.detach() < 0.5, float("-inf"))
        sel = torch.topk(choice, k=g.top_k, dim=-1, sorted=False)[1]
        w = scores.gather(1, sel) * lm[sel].float()
        if g.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        w = (w * g.routed_scaling_factor).to(hidden_states.dtype)
        out = b.experts(h, sel, w).view(*shape)
        return out + b.shared_experts(hidden_states)

    def restore(self):
        self.block.forward = self._orig


def install_wrappers(model, shared):
    return [GlmPruneWrapper(blk, k, shared) for k, (_, blk) in enumerate(moe_blocks(model))]


def hard_prune_hooks(model, prune_mask):
    """Deployment routing: pruned experts can't be chosen (their score is -inf before the top-k), as when they
    are physically removed from the GGUF. prune_mask: (n_sparse_layers, n_experts) bool, True = pruned."""
    hooks = []
    for k, (_, blk) in enumerate(moe_blocks(model)):
        g = blk.gate
        m = prune_mask[k].to(g.weight.device)

        def fwd(hidden_states, g=g, m=m):
            h = hidden_states.view(-1, g.hidden_dim)
            logits = F.linear(h.float(), g.weight.float())
            scores = logits.sigmoid()
            choice = (scores + g.e_score_correction_bias).masked_fill(m, float("-inf"))
            idx = torch.topk(choice, k=g.top_k, dim=-1, sorted=False)[1]
            w = scores.gather(1, idx)
            if g.norm_topk_prob:
                w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
            return logits, w * g.routed_scaling_factor, idx

        orig = g.forward
        g.forward = fwd
        hooks.append((g, orig))
    return hooks


def remove_hard_prune_hooks(hooks):
    for g, orig in hooks:
        g.forward = orig


def router_hook_targets(model):
    """(sparse_index, gate) pairs, for the frequency / router-score baselines."""
    return [(k, blk.gate) for k, (_, blk) in enumerate(moe_blocks(model))]
