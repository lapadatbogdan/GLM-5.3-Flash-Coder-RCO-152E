#!/usr/bin/env python3
"""Fixes for the torch path of GLM-5.3-Flash (transformers glm5_next), installed at load time (site-packages untouched).
Used by the RCO scripts (rco/), the heal scripts (heal/) and the reference builders (rco/rco_prep_v2_mn.py).

    import dsa_fix; dsa_fix.install_long_context_patches(qchunk=256, kda_group=8)   # before the forward pass

1. CORRECTNESS (bug found 14 Oct): the DSA attention in transformers calls F.scaled_dot_product_attention with a boolean
   (L x L) mask and head_dim 256; on ROCm/MI250X the result is wrong (layer 3: rel 0.99 / cos 0.32 vs llama.cpp;
   the same code on CPU: rel 0.066). CE on calib_heldout 0-15: 3.52 -> 1.066 with the fix (llama.cpp 1.078).
   Replacement: explicit attention (q k^T, mask, softmax fp32, @ v), no SDPA.
2. MEMORY: the reference KDA materializes decay_mask (B,H,nC,C,C,K) fp32 = 2 MiB/token; the DSA indexer and mask are
   L x L. The replacements work on groups of blocks / chunks of queries, under checkpoint when gradients are needed.
Numerical equivalence: checked with a CPU test (tiny model, CPU, forward and backward).
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ----------------------------------------------------------------------------
# Long context: replacements without tensors proportional to L x (something large), same math
# ----------------------------------------------------------------------------
# 14 Oct: OOM "Tried to allocate 32 GiB" at 16K and 64 GiB at 32K (and 32 GiB at 16 x 1024 in the probe),
# so linear in tokens: 2 MiB/token = H(64) x C(64) x K(128) x 4 B = decay_mask (B,H,nC,C,C,K) fp32 from
# chunk_kimi_delta_attention, the reference variant in transformers (flash-linear-attention is missing). The DSA
# layers (indexer + masked dense attention) also have L x L tensors: indexer scores (B,L,32,L/4) fp32 = 32 GiB at 32K,
# int32 mask (B,L,L) = 4 GiB, and SDPA with an arbitrary mask and head_dim 256 falls back to the math path: (B,64,L,L).
# The replacements below: KDA with the intra-block part computed on groups of --kda-group blocks, under checkpoint
# (backward recomputes the group, does not store the decay); DSA on chunks of --q-chunk queries (indexer, mask and SDPA
# per chunk; attention under checkpoint). The result is identical (CPU test).
LC = {"qchunk": 256, "kda_group": 8}


def _kda_intra(q, k, kb, vb, g):
    """Intra-block part of chunk_kimi_delta_attention, for a group of blocks (B,H,n,C,*), as in the reference."""
    C = q.shape[-2]
    mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=0)
    strict = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=1)
    decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(strict[..., None], float("-inf")).exp().float()
    attn = -(kb.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(dim=-1).masked_fill(mask, 0)
    for i in range(1, C):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(C, dtype=attn.dtype, device=attn.device)
    v = attn @ vb
    kcum = attn @ (kb * g.exp())
    aqk = (q.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(dim=-1).masked_fill(strict, 0)
    return v, kcum, aqk


def kda_chunked(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, **kwargs):
    """Replacement for modeling_glm5_next.chunk_kimi_delta_attention (same signature and math)."""
    from transformers.models.glm5_next.modeling_glm5_next import l2norm
    initial_dtype = query.dtype
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                                  for x in (query, key, value, beta, g)]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    B, H, L, K = key.shape
    V = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    pad = (chunk_size - L % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad)) * scale
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    g = F.pad(g, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, g, k_beta, v_beta = [x.reshape(B, H, -1, chunk_size, x.shape[-1])
                                     for x in (query, key, g, k_beta, v_beta)]
    g = g.cumsum(dim=-2)
    n = query.shape[2]
    G = max(1, LC["kda_group"] // max(B, 1))  # the group is ~1 GiB fp32 at B=1; with a larger batch, smaller groups
    vs, kcs, aqs = [], [], []
    for s in range(0, n, G):
        args = (query[:, :, s:s + G], key[:, :, s:s + G], k_beta[:, :, s:s + G], v_beta[:, :, s:s + G], g[:, :, s:s + G])
        if torch.is_grad_enabled():
            v_, kc_, aq_ = checkpoint(_kda_intra, *args, use_reentrant=False)
        else:
            v_, kc_, aq_ = _kda_intra(*args)
        vs.append(v_)
        kcs.append(kc_)
        aqs.append(aq_)
    value, k_cumdecay, aqk = torch.cat(vs, 2), torch.cat(kcs, 2), torch.cat(aqs, 2)
    state = (torch.zeros(B, H, K, V, dtype=value.dtype, device=value.device) if initial_state is None
             else initial_state.to(value))
    outs = []
    for i in range(n):
        q_i, k_i, v_i, g_i = query[:, :, i], key[:, :, i], value[:, :, i], g[:, :, i]
        attn_inter = (q_i * g_i.exp()) @ state
        v_new = v_i - k_cumdecay[:, :, i] @ state
        outs.append(attn_inter + aqk[:, :, i] @ v_new)
        state = state * g_i[:, :, -1].exp().unsqueeze(-1) + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
    core = torch.stack(outs, 2).reshape(B, H, -1, V)[:, :, :L]
    return core.transpose(1, 2).contiguous().to(initial_dtype), (state if output_final_state else None)


@torch.no_grad()
def _indexer_chunked(ix, hidden_states, q_resid, attention_mask, qc):
    """Glm5NextTextIndexer.forward without cache, on chunks of qc queries (the same row-wise operations)."""
    B, S = hidden_states.shape[:2]
    q = ix.wq_b(q_resid).view(B, S, -1, ix.head_dim)
    k = ix.k_norm(ix.wk(hidden_states)).view(B, S, -1, ix.head_dim).squeeze(2)
    gate_scores = F.linear(hidden_states, ix.index_kpool_compress_gate)
    packed = torch.cat([k, gate_scores, attention_mask.to(k.dtype)[..., None]], dim=-1)
    valid_keys = packed[..., -1].bool()
    pool_keys, pool_indices, pool_valid = ix.get_pooled_states(packed_states=packed)
    pkT = pool_keys.transpose(-1, -2).float().unsqueeze(1)
    weights = ix.weights_proj(hidden_states.to(ix.weights_proj.weight.dtype)).float() * (ix.n_heads ** -0.5)
    pool_end = pool_indices[..., -1].clamp(0, S - 1)
    kv_pos = torch.arange(S, device=hidden_states.device)
    batch_idx = torch.arange(B, device=hidden_states.device)[:, None, None]
    width = ix.index_topk + (ix.index_kpool - 1 if ix.index_kpool_always_select_tail else 0)
    outs = []
    for s in range(0, S, qc):
        e = min(S, s + qc)
        c = e - s
        visible = (kv_pos[None, None, :] <= torch.arange(s, e, device=kv_pos.device)[None, :, None]) & valid_keys[:, None, :]
        scores = F.relu(torch.matmul(q[:, s:e].float(), pkT) * ix.softmax_scale)
        index_scores = torch.matmul(weights[:, s:e].unsqueeze(-2), scores).squeeze(-2)
        pool_visible = visible.gather(dim=-1, index=pool_end[:, None, :].expand(B, c, -1))
        valid_c = pool_visible & pool_valid[:, None]
        index_scores = index_scores.masked_fill(~valid_c, torch.finfo(index_scores.dtype).min)
        select_k = min(ix.index_topk // ix.index_kpool, index_scores.shape[-1])
        selected = index_scores.topk(select_k, dim=-1).indices
        selected_valid = valid_c.gather(-1, selected)
        selected_indices = pool_indices[batch_idx, selected]
        topk = selected_indices.flatten(-2).masked_fill(
            ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1)
        if ix.index_kpool_always_select_tail:
            topk = ix.append_visible_tail(topk, visible, valid_keys)
        topk = F.pad(topk, (0, width - topk.shape[-1]), value=-1)[..., :width]
        outs.append(topk.masked_fill(~attention_mask[:, s:e, None], -1).to(torch.int32))
    return torch.cat(outs, dim=1)


def _dsa_chunk(attn, q, k, v, topk):
    """Explicit attention (q k^T, mask, softmax fp32, @ v), WITHOUT F.scaled_dot_product_attention.
    Bisection of 14 Oct (vs llama.cpp, window 0 of calib_heldout, 1024 tokens): on GPU (ROCm,
    MI250X) the DSA attention output of layer 3 had rel 0.99 / cos 0.32 vs llama.cpp, the same code on CPU gives
    rel 0.066; all 11 DSA layers have cos 0.06-0.32 on GPU, and the CE rises from 1.2 (llama.cpp) to 3.5.
    SDPA on ROCm with an arbitrary boolean mask and head_dim 256 gives a wrong result; the explicit path is correct on
    every device."""
    mask = attn.build_attention_mask_from_topk(topk_indices=topk, query_states=q, kv_length=k.shape[2])
    s = torch.matmul(q, k.transpose(-1, -2)).float() * attn.scaling
    s = s.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else s + mask.float()
    return torch.matmul(torch.softmax(s, dim=-1).to(v.dtype), v)


def _dsa_forward(self, hidden_states, attention_mask, prev_topk_indices):
    B, S = hidden_states.shape[:-1]
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    query_states = self.q_b_proj(q_resid).view(B, S, -1, self.qk_head_dim).transpose(1, 2)
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_a_layernorm(kv_pass).view(B, 1, S, self.kv_lora_rank)
    k_rot = k_rot.view(B, 1, S, self.qk_rope_head_dim)
    key_states, value_states = self.expand_kv(k_pass, k_rot)
    if self.indexer is not None:
        topk = _indexer_chunked(self.indexer, hidden_states, q_resid, attention_mask, LC["qchunk"] if LC["qchunk"] > 0 else S)
    else:
        assert prev_topk_indices is not None, "shared DSA layer without top-k from the previous layer"
        topk = prev_topk_indices
    outs = []
    qc = LC["qchunk"] if LC["qchunk"] > 0 else S
    for s in range(0, S, qc):
        args = (query_states[:, :, s:s + qc], key_states, value_states, topk[:, s:s + qc])
        if torch.is_grad_enabled():
            outs.append(checkpoint(_dsa_chunk, self, *args, use_reentrant=False))
        else:
            outs.append(_dsa_chunk(self, *args))
    attn_output = torch.cat(outs, dim=2).transpose(1, 2).reshape(B, S, -1).contiguous()
    return self.o_proj(attn_output), None, topk if self.next_skip_topk else None


def install_long_context_patches(qchunk, kda_group):
    """Replaces at module level (site-packages untouched) the reference KDA and the cache-less DSA attention."""

    from transformers.models.glm5_next import modeling_glm5_next as M
    LC["qchunk"], LC["kda_group"] = qchunk, kda_group
    if not getattr(M, "_rft_patched", False):
        M._rft_orig_kda = M.chunk_kimi_delta_attention
        M._rft_orig_attn_forward = M.Glm5NextTextAttention.forward
        M.chunk_kimi_delta_attention = kda_chunked

        def forward(self, hidden_states, attention_mask, past_key_values=None, prev_topk_indices=None, **kwargs):
            if past_key_values is not None:
                return M._rft_orig_attn_forward(self, hidden_states, attention_mask, past_key_values=past_key_values,
                                                prev_topk_indices=prev_topk_indices, **kwargs)
            return _dsa_forward(self, hidden_states, attention_mask, prev_topk_indices)
        M.Glm5NextTextAttention.forward = forward
        M._rft_patched = True
    return M


def uninstall_long_context_patches():
    from transformers.models.glm5_next import modeling_glm5_next as M
    if getattr(M, "_rft_patched", False):
        M.chunk_kimi_delta_attention = M._rft_orig_kda
        M.Glm5NextTextAttention.forward = M._rft_orig_attn_forward
        M._rft_patched = False
