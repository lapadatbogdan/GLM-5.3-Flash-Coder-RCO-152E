# llama.cpp patch: `llama-imatrix` on GLM-5.3 (glm5_next)

**Problem.** Unpatched `llama-imatrix` segfaults on GLM-5.3-Flash. The graph callback reads the weight name
(`src0->name`) before checking the node type, and glm5_next graphs contain nodes without `src[0]` (linear attention,
indexer, hyper-connections). Found with gdb: the crash is in `filter_tensor_name(src0->name)` with `src0 == nullptr`.

**Fix.** Check that the node is a matrix multiplication (`MUL_MAT` / `MUL_MAT_ID`) with both operands before reading
the name. Only matrix multiplications were ever collected, so other models are unaffected.

**Verified.**
- GLM-5.3-Flash: imatrix runs to completion (it crashed before), used for all our quants.
- No change for other models: on Qwen2.5-0.5B-Instruct Q8_0 (llama.cpp `23b0202`, 20 chunks of 512 tokens, CPU),
  the imatrix from the patched and the unpatched build is identical on all 672 tensors (bit for bit, NaN treated as
  equal; the 35 NaN values are present in the unpatched output too), final PPL 11.4068 for both.

**Apply.**
```
cd llama.cpp && git apply /path/to/llama.cpp-imatrix-glm5_next.patch
```
Our build script applies it automatically.

# llama.cpp patch: per-layer routed expert count for glm5_next

`llama.cpp-glm5_next-per-layer-experts.patch` (on top of `79e2e74`, independent of the imatrix patch) lets a pruned
GLM-5.3-Flash keep a different number of routed experts in each MoE layer, without padding.

- GGUF: new key `glm5-next.expert_count_per_layer` (array, one entry per block incl. MTP, 0 on dense layers);
  `expert_count` stays the maximum, so scalar readers keep working. Written by the converter when the HF config has
  `n_routed_experts_per_layer` (from `build_pruned.py --per-layer`); absent key = uniform, GGUF unchanged.
- Loader: `hparams.n_expert_l(il)`, tensors sized per layer, `build_moe_ffn` gets the layer count and
  top-k = min(n_expert_used, n_expert_l) (warmup asks for k = n_expert and would assert on a smaller layer).
- imatrix and quantize need no change (sized from each tensor's ne[2]).

**Verified** (CPU, 6-layer real-weights cut of GLM-5.3-Flash, 24 experts, mask 8/20/13 + MTP 20): per-layer GGUF on the
patched build vs the padded GGUF on the unpatched build gives bit-identical logits with `GGML_LLAMAFILE=OFF`
(max |d| 4e-4 with llamafile sgemm, whose tiling depends on the router row count); uniform models bit-identical
patched vs unpatched; imatrix of the real experts identical to the padded one; Q4_K_M with imatrix runs.
Negative control: without the min() clamp, warmup aborts at `ggml.c:5455 GGML_ASSERT(a->ne[0] >= k)` on the 8-expert layer.
Test scripts: `../../perlayer/` (make_mini_src.py, dump_logits.cpp, step1-3 scripts, test_loader_perlayer.py).
