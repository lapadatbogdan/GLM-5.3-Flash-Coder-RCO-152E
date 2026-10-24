# GLM-5.3-Flash-Coder-RCO-152E

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (321B MoE) with **152 of 288 experts per layer**,
built for coding and agentic use on a home PC. **321B-A18B becomes ~170B-A17B.**

Weights (GGUF): [lapadatbogdan/GLM-5.3-Flash-Coder-RCO-152E-GGUF](https://huggingface.co/lapadatbogdan/GLM-5.3-Flash-Coder-RCO-152E-GGUF)

This repository is the recipe: calibration mix, expert selection with RCO, the final mask, healing, quantization,
the llama.cpp patches we needed, and the evaluation harness. The scripts are published as we ran them, with our
paths; expect to adapt them (see [Paths](#paths)).

## What changed from REAP144

| | REAP144 (old) | RCO-152E (this release) |
|---|---|---|
| Loop test: problems in 30 hard instruction-following runs | 22 | **4** (0 repetition loops) |
| 17 x 23, 1234 x 5678, Fibonacci, small executed code task, tool call | arithmetic errors | **pass** at every quant (Q3_K missed Fibonacci once in one run) |
| Experts per layer | 144 | 152 (incl. 8 generalists) |
| Calibration | Python agentic sessions | agentic sessions + code in 21 languages + math + 15% general text |

## Files

| Quant | Size | Fits (GPU VRAM + RAM) | KLD to Q8 |
|---|---|---|---|
| Q8_0 | 180.4 GB (4 parts) | 24 + 192, multi-GPU | 0 |
| Q4_K_M | 102.6 GB (3 parts) | 24 + 96, 32 + 96, 128 GB unified | 0.024 |
| Q3_K | 75.1 GB (2 parts) | 24 + 64, 32 + 64 | 0.045 |
| IQ2_XXS (GSQ) | ~47 GB | 24 + 32 | in progress |

KLD: llama.cpp `llama-perplexity --kl-divergence` against the Q8_0 of this model on held-out code and agentic text.
Same top token as Q8: Q4_K_M 94.5%, Q3_K 92.5%.

A 2-bit version for 24 GB VRAM + 32 GB RAM is in progress (GSQ vector quantization of the experts + quantization-aware
healing) and will be added when it passes the same tests.

## Results (Q8_0, no special settings)

Against the full GLM-5.3-Flash (Q8), same prompts, default sampling:

| Benchmark | RCO-152E | Full model |
|---|---|---|
| AIME 2025 (20) | 19 | 20 |
| MATH-500 subset (6) | 6 | 6 |
| LiveCodeBench subset (6) | 6 | 6 |
| IFEval (100) | 71 | 83 |

Agentic results (51 DeepSWE tasks, same harness as the full model): in progress, will be added.

Held-out KL divergence to the full model (top-20 logits, 256 windows): agentic 0.209, code 0.193, math 0.089,
general text 0.594.

## Recommended settings

- **Q8_0, code and agentic use:** works without special settings.
- **Q4_K_M and Q3_K, code:** ban the "hmm" tokens (see below). Without it, these files can loop on harder exercises.
- **Long prose (essays, poems) and non-English prompts:** the model can fall into repetition loops; this is where
  pruning hurts most. A presence penalty of 1.0 reduces it. This model is not meant for creative writing or other
  languages.

### The "hmm" loop at Q4_K_M and Q3_K

On harder coding exercises the quantized files can get stuck in their reasoning: the model hesitates ("hmm"),
repeats it ("hmm, hmm, hmm, ...") until the token limit and never writes the code. Once one attempt has looped, the
next attempts with that reasoning in the history loop again within a few thousand characters.

Aider Polyglot, up to 10 attempts per exercise, exercises on which a file looped (runs in progress):

| File | Exercises with a loop |
|---|---|
| Q8_0 | 0 of 19 |
| Q4_K_M | 2 of 15 |
| Q3_K | 4 of 14 |

Banning the three "hmm" tokens of this tokenizer removes it. With the ban, Q3_K solved all 3 exercises it had looped
on (2 of them on the first attempt). Q4_K_M loops the same way and should get the same setting.

- **llama.cpp** (applies to every request): add
  `--logit-bias 86776-100 --logit-bias 79380-100 --logit-bias 87459-100` to `llama-server`.
- **Per request** (OpenAI-compatible API, also Maya): `"logit_bias": {"86776": -100, "79380": -100, "87459": -100}`.

Token ids: 86776 = " hmm", 79380 = "Hmm", 87459 = " Hmm". What did not fix this loop in our tests: a presence penalty
(the reasoning no longer repeats but does not end either) and a reasoning budget of 16K tokens (helped on 1 of 3
exercises).

## Run

Split files load directly; point the runtime at the first part.

```bash
pip install -U huggingface_hub
hf download lapadatbogdan/GLM-5.3-Flash-Coder-RCO-152E-GGUF --include "*Q3_K*" --local-dir glm53-rco-q3k
```

- **llama.cpp** (tested at `79e2e74`): `llama-server -m glm53-rco-q3k/GLM-5.3-Flash-Coder-RCO-152E-Q3_K-00001-of-00002.gguf --jinja -c 131072 --logit-bias 86776-100 --logit-bias 79380-100 --logit-bias 87459-100`
  with expert tensors offloaded to CPU as needed (`--n-cpu-moe`).
- **Maya** (made for GLM-5.3-Flash, hottest experts on the GPU, the rest in RAM):
  `git clone https://github.com/mw00/project-maya && cd project-maya && ./maya.sh --gguf-dir ../glm53-rco-q3k --no-vision`
  (add `--backend hip` on AMD).

Text only: no vision projector (mmproj) is included.

## How it was made

1. **Calibration mix:** agentic coding sessions written by the full model, code in 21 languages, math, and 15% general
   text, all outside the evaluation sets (the held-out windows are task-disjoint from the search windows).
2. **Expert selection with RCO** (optimization of the expert mask against the full model's outputs), 152 experts per
   layer, started from a REAP-score mask with 8 generalist experts per layer added. Against a REAP-style mask of the
   same size: same code and agentic KL, 22% lower KL on general text, and far fewer loops. The released mask is the
   step-400 checkpoint of the search.
3. **Healing:** LoRA on attention and the shared expert, distilled from the full model's top-20 logits (96 steps),
   continued from the LoRA of an earlier heal; the router and the routed experts are not trained.
4. **Build:** prune to an FP8 checkpoint, merge the LoRA, convert to Q8_0, imatrix on the same mix, then Q4_K_M and
   Q3_K.

## Repository layout

Directory names match our working tree (`/work/glm53/<dir>`), so the paths inside the scripts point at the same
places as in this repository.

| Step | Scripts |
|---|---|
| Window format | `calib53/build_calib.py` (chat rendering with reasoning kept, 1024-token windows, loss mask = model-written tokens) |
| Agentic sessions | `calib53/agent_py.py`, `agent_js.py` (minimal bash-tool agents), `run_py.sh`, `run_js.sh`, `gen_py.sbatch`, `gen_js.sbatch`, `task_prep/pro_gen_prep*.sh` (clean-room task preparation); new sessions for this release: `calib53/mix_v3/agentic_tasks.txt`, `gen_agentic_v3.sbatch`, `run_agentic_v3.sh` |
| Multi-language edits | `calib53/gen_poly.py`, `gen_poly.sbatch` (CommitPackFT files edited by the full model) |
| Code / math / general chat sessions | `calib53/mix_v1/collect_prompts.py`, `mix_v2/collect_prompts_v2.py`, `mix_v3/collect_prompts_v3.py` (prompt pools, decontaminated), `mix_v1/gen_mix.py` + `gen_mix.sbatch`, `mix_v2/gen_mix2.sbatch`, `mix_v3/gen_mix3.sbatch` (the full model answers) |
| Calibration mix (`mix_v3`) | `calib53/mix_v3/build_mix3.py`, `build_mix3.sbatch` (prefix-ordered pool + fixed held-out set; imports `mix_v1/build_mix.py`), `pool_report.json`, `prompts/sources.json`; `rco/make_prefix.py` (the chosen pool size as `mix_v3/N1270`) |
| General text (`general_v1`) | `calib53/general_v1/build_general.py` |
| Mixed RCO and heal sets | `calib53/general_v1/make_heal_mix.py` + `add_freq_mix.py` (mix_v3 windows + general windows: `calib53/rco_mixg_v1` for RCO, `calib53/heal_mixg_v1` for healing) |
| Full-model reference | `rco/rco_prep_v2_mn.py`, `rco_prep_v2_mn.sbatch`, `rco_prep_merge.sbatch` (top-20 logits, expert frequency and REAP statistics, sharded) |
| Initial mask | `calib53/general_v1/build_mask_plus.py` (REAP-sum mask + 8 generalist experts per layer: `gen_k152`); `rco/build_balanced.py` writes the same `gen_k152` mask next to balanced variants; `rco/reap_score.py`, `reap_score_v2.py`, `reap_score_v2.sbatch` (standalone REAP scorer) |
| RCO search | `rco/rco_search_v3g_mn.sbatch` (the run used) and `rco_search_v3_mn.sbatch`, `rco/rco_search_v3_mn.py` (multi-node RCO), `rco/rco_glm_prune.py`, `rco/convergence.py`, `rco/src/` (RCO + GLM-5.3 model code), `heal/dsa_fix.py` (DSA attention fix for the torch path on ROCm) |
| Mask selection | `rco/rcog_k152_select.sh`, `rco/eval_masks_mn.py`, `eval_masks_mn.sbatch` (held-out KL of candidate masks) |
| Final mask | `masks/mask_rco_k152.pt` |
| Healing | `heal/heal_v3_rco.sbatch` (first heal, 144-expert RCO mask), `heal/heal_v3_cont.sbatch` (`VARIANT=d`: this release), `heal/heal_v3.py`, `heal/heal_v3_cont.py`, `heal/gguf_experts.py`, `heal/gguf_experts_any.py` |
| Build | `rebuild_v2/common.sh`, `a1_prune_conv.sbatch` (mask -> pruned FP8 checkpoint -> Q8_0), `h1_merge_conv.sbatch` (LoRA merge -> Q8_0), `a2_imatrix_mix3.sbatch`, `quant.sbatch`; `calib53/rco_mixg_v1/build_imatrix_text.py` (imatrix text); `prune_build/build_pruned.py`; `release/meta.sbatch` (metadata + split for upload) |
| Evaluation | `smoke/`, `stdbench/`, `swepro/` (see below) |
| llama.cpp patches | `patches/` |

`masks/mask_rco_k152.pt` is a bool tensor `[42, 288]` (42 MoE layers x 288 routed experts), `True` = pruned,
152 kept in every layer:

```python
import torch
m = torch.load("masks/mask_rco_k152.pt", weights_only=True)
kept = (~m[layer]).nonzero().flatten()   # routed expert ids kept in MoE layer `layer`
```

### The commands of this release

Reconstructed from the job logs; `P=/work/glm53`.

```bash
# RCO search, 152 kept per layer, mixed calibration (general text included), init from gen_k152
cd $P/rco && CALIB=calib53/rco_mixg_v1 KEEP=152 INIT=reap INITMASK=$P/rco_runs/general_plus/mask_gen_k152.pt \
  INITSCORES= OUT=$P/rco_runs/rcog_k152_v3 sbatch --nodes=2 rco_search_v3g_mn.sbatch
bash rcog_k152_select.sh                          # step 400 / 500 / 600 masks -> held-out KL; step 400 chosen
# prune, heal, build
cd $P/rebuild_v2
sbatch --export=ALL,TAG=r152v5,MASK=$P/masks/mask_rco_k152.pt a1_prune_conv.sbatch
sbatch --job-name=healg_d --export=ALL,VARIANT=d $P/heal/heal_v3_cont.sbatch
sbatch --export=ALL,TAG=r152v5g,SRC=$P/pruned/r152v5,DELTA=$P/heal_runs/r152v5_g1/delta.pt h1_merge_conv.sbatch
sbatch --export=ALL,TAG=r152v5g,TXT=$P/calib53/rco_mixg_v1/calib_text_search.txt,CHUNKS=1494 a2_imatrix_mix3.sbatch
sbatch --export=ALL,TAG=r152v5g quant.sbatch Q4KM; sbatch --export=ALL,TAG=r152v5g quant.sbatch Q3K
sbatch $P/release/meta.sbatch
```

### Paths

- `/work/glm53`: working tree (this repository), plus `full/` (the base model, HF FP8), `gguf/`, `pruned/`,
  `out/`, `rco_runs/`, `heal_runs/`, `llama.cpp/` (built at `79e2e74` with the patches below), `vllm-rocm.sif`
  (a ROCm PyTorch container) and `pylib/` (a transformers build with `glm5_next` support).
- `/work/tasks`: SWE task images and mined task instances used for the agentic calibration sessions (not included).
- `/work/swe_scratch`: scratch space for the DeepSWE harness overlays.
- The `.sbatch` files are Slurm jobs for nodes with 8 GPUs (MI250X GCDs, 64 GB each); `--account` and the
  partitions are placeholders.

## Evaluation harness

- **Basic regression** (`smoke/smoke.py`, `smoke.sbatch`): on every quant, 17 x 23, 1234 x 5678, the 20th Fibonacci
  number, one knowledge question, a small Python function that is executed against asserts, and one tool call.
- **Loop gate** (`stdbench/sbatch/std_dry2.sbatch` -> `run_dry2.sh`): hard IFEval prompts, 3 runs each
  (`data/drytest_ids.txt`), plus small LiveCodeBench and MATH-500 sets, once with default sampling and once with
  presence penalty 1.0 (`DRY_CFGFILE=data/dry_cfgs_basepres.txt`). This is the loop test in the table above.
- **Benchmarks** (`stdbench/sbatch/std_gate.sbatch` -> `run_gate.sh`, `run.py`, `benches/`, `report.py`): IFEval,
  AIME 2025, MATH-500 and LiveCodeBench subsets (`data/verif_ids_a.txt`, `verif_ids_b.txt`) against the full model with
  the same prompts; code is graded in a sandbox without network. `stdbench/prep/` downloads and prepares the data.
- **DeepSWE agentic harness** (`swepro/`): `agent_v2.sbatch deepswe <run> <gguf>` -> `run_bench_v2.sh` starts
  llama-server and the workers, `swe_attempt.py` runs one attempt with
  [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) in two containers: the agent works in a clean room
  (fresh copy of the repository, git history replaced by one commit, no network, checked on every attempt), then the
  task's official verifier runs in a new container on the collected patch. `deepswe51_ids.txt` lists the 51 tasks,
  `report.py` summarizes a run. `mini.yaml` is mini-swe-agent's default config (unchanged). `mini_verify.yaml`
  (made by `mkverify.py` from `mini.yaml`) is an optional step that asks the agent to check each requirement and run
  the full test suite and type checker before submitting. Task manifests and container images are not included.

**Security notes.** The harnesses execute commands written by the model. Run them only inside disposable, isolated
containers on a machine with nothing you care about: the model's commands see the container's environment (so do not
put API keys there; with a local llama.cpp server none is needed). In `swepro/` each attempt gets a fresh overlay that
is deleted afterwards and the agent container has no network; the calibration agents in `calib53/` reuse a writable
overlay between tasks of the same worker. The pass/fail check runs the task's own tests in a separate container, but a
determined model could still change test configuration it does not detect; review successful patches if you build on
these numbers.

## What did not work

So you do not have to try it. Rows marked REAP144 were measured on the earlier Python-only setup (30 SWE tasks,
2 attempts; see the `reap144` tag); * = not all attempts run.

| Tried | Result | Why it failed |
|---|---|---|
| REAP-score mask, 152 experts | same code and agentic KL as RCO, 22% higher general-text KL, more loops | A per-expert score ranks experts one by one; RCO scores the whole mask against the full model's outputs |
| Keep 120 experts (REAP144 setup, REAP-sum score) | 35/38 *, x2.32 turns; held-out KL 0.88 | The model starts "thinking" for tens of thousands of tokens without answering, and loops on commands |
| Keep 96 experts (REAP144 setup) | held-out KL 1.07; ~90 turns per conversation, stopped | Too slow to be useful even when it solves the task |
| Frequency mask, 144 experts (REAP144 setup) | 39/43 *, x2.21 turns; held-out KL 0.94 | Keeps the most used experts, not the ones that matter for code |
| Public REAP 50% (multi-domain), 144 experts (REAP144 setup) | 44/46 *, x2.29 turns | Generic calibration; at 2 bit (IQ2_XXS) it answered 4/4 short questions and solved 0/4 tasks |
| 120 experts at 2-2.5 bit (IQ2_S, mixed quant) | 9/12 on a screen; the mixed quant looped in its reasoning | Pruning and low bits compound |
| GSQ instead of imatrix at ~3 bit | KLD 0.0711 vs 0.0702 (8 of 42 layers) | No gain at 3 bit. GSQ is a scalar method: the lowest llama.cpp scalar format (Q2_K, ~2.6 bits) is above the 2.06-bit size of IQ2_XXS, so it cannot target the 24 + 32 GB variant |
| Tensor parallel in llama.cpp | not available for this architecture | Use layer split (`-sm layer`); with several concurrent conversations the decode micro-batch patch keeps the GPUs busy |
| Presence penalty for code | not needed | It helps only on long prose and non-English prompts (see Recommended settings) |

## llama.cpp patches

`patches/` (llama.cpp `79e2e74`, details in `patches/README.md`):
- `llama.cpp-imatrix-glm5_next.patch`: `llama-imatrix` segfaults on GLM-5.3-Flash (graph nodes without `src[0]`);
  fixed by checking the node type first. Needed to build the importance matrix, not to run the model.
- `llama.cpp-decode-microbatch.patch`: `LLAMA_DECODE_MICROBATCH=k` splits a multi-sequence decode batch into k
  micro-batches so a layer-split model keeps all GPUs busy. With 8 concurrent conversations on 8 MI250X GCDs, k=4 gave
  54.9 tokens/s per node vs 25.6 unpatched (aggregate). Used for the agentic evaluation, not needed to run the model.
- `llama.cpp-glm5_next-per-layer-experts.patch`: lets a pruned model keep a different number of experts per layer.
  Not needed for this release (152 in every layer).

## Limits

Built and tested for code and agentic use. Pruning removes knowledge outside that domain: creative writing and
non-English prompts are the weak spots.

## Previous release

REAP144 (144 experts, calibrated on Python agentic sessions only) is replaced by this release: it looped on general
questions and on other programming languages and could get simple arithmetic wrong. Its recipe, masks and harness
remain available in the git history under the tag `reap144`.

## Acknowledgements

- RCO and GSQ: IST Austria (ISTA-DASLab).
- REAP (Lasby et al., 2025) for the expert-saliency score used for the initial mask.
- This project was built with the help of Claude (Anthropic) as a coding and research assistant.
  The decisions are the author's.

## Credits and licenses

- Base model: GLM-5.3-Flash by Z.ai, MIT.
- RCO and GSQ: IST Austria (ISTA-DASLab). `rco/`: based on the RCO reference implementation by Helcig and Alistarh
  (Apache-2.0, see `rco/LICENSE`, `rco/NOTICE` and `rco/CITATION.cff`), extended with the GLM-5.3-Flash model code and
  the REAP scorer.
- REAP expert scoring: Lasby et al., 2025. Public REAP 50% comparison: `patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2`.
- mini-swe-agent (MIT): `swepro/mini.yaml` is its default config.
- Runtime: llama.cpp (MIT), Maya (MIT).
- Everything else in this repository: MIT.

Contact: lapadat.bogdanv@gmail.com or [LinkedIn](https://www.linkedin.com/in/bogdan-lapadat-57baa08/)
