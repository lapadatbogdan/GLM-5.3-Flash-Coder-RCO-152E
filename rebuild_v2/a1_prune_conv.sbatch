#!/bin/bash
#SBATCH --job-name=v2A1
#SBATCH --account=your_project
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/work/glm53/rebuild_v2/logs/a1_%x_%j.log
# Step A1 (CPU only): mask -> pruned FP8 HF checkpoint (build_pruned.py, as stageC step 1) -> Q8_0 GGUF
# (convert_hf_to_gguf --outtype q8_0 --fp8-as-q8, as stageC step 2) -> hardlink as the published Q8_0 name.
# Old timings on a GPU node: prune 14-17 min, convert 24-46 min; conv full model MaxRSS 29 GB (so small is enough).
# Resumable: build_pruned skips finished shards, each step has a marker. Usage:
#   sbatch --export=ALL,TAG=r96v2,MASK=/work/glm53/.../mask_reap_sum_k96.pt a1_prune_conv.sbatch
set -euo pipefail
. /work/glm53/rebuild_v2/common.sh
MASK=${MASK:?MASK missing}

stage "A1.0 mask check"
singularity exec -B /work $P/vllm-rocm.sif python3 -c "
import torch, sys
m = torch.load('$MASK', weights_only=True); k = (~m).sum(1).unique().tolist()
print('mask', tuple(m.shape), m.dtype, 'kept per layer', k)
assert m.dtype == torch.bool and m.shape == (42, 288) and len(k) == 1, 'expected uniform bool (42, 288) mask'
digits = ''.join(c for c in '$TAG'.split('v')[0] if c.isdigit())
assert int(digits) == k[0], f'TAG $TAG does not match kept count {k[0]}'
"

stage "A1.1 prune -> $PR"
if [ ! -f $M.build ]; then
  if [ -f $PR/model.safetensors.index.json ] && [ ! -f $M.build.started ]; then
    echo "$PR exists but was not started by this pipeline, refusing"; exit 3
  fi
  touch $M.build.started
  $P/venv/bin/python $P/prune_build/build_pruned.py --src $P/full --mask $MASK --out $PR --mtp drop 2>&1 | tail -5
  [ -s $PR/model.safetensors.index.json ] && touch $M.build
fi
[ -f $M.build ] || { echo "prune failed"; exit 1; }

stage "A1.2 convert -> $Q8"
if [ ! -f $M.conv ]; then
  is_gguf $Q8 && { echo "$Q8 exists without marker, refusing to overwrite"; exit 3; }
  . $P/venv/bin/activate
  python $P/llama.cpp/convert_hf_to_gguf.py $PR --outtype q8_0 --fp8-as-q8 --outfile $Q8.part 2>&1 \
    | grep -v -E "^INFO:hf-to-gguf:(blk|output|token)" | tail -8
  deactivate
  is_gguf $Q8.part && mv $Q8.part $Q8 && touch $M.conv
fi
[ -f $M.conv ] || { echo "convert failed"; exit 1; }

stage "A1.3 Q8_0 deliverable (hardlink, no copy)"
[ -e $OUTP-Q8_0.gguf ] || ln $Q8 $OUTP-Q8_0.gguf
ls -la $PR | head -3; ls -la $Q8 $OUTP-Q8_0.gguf
stage "A1 DONE"
