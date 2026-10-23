#!/bin/bash
#SBATCH --job-name=v2H1
#SBATCH --account=your_project
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/work/glm53/rebuild_v2/logs/h1_%x_%j.log
# Step H1 (CPU only): LoRA of a heal run merged into the pruned FP8 checkpoint (heal_v3.py --merge = heal.py's merge:
# LoRA targets -> BF16 W + alpha/r B A, their scale_inv dropped, every other tensor byte-identical) -> Q8_0 GGUF
# (convert_hf_to_gguf --outtype q8_0 --fp8-as-q8, as A1.2) -> hardlink as the published Q8_0 name. Then A2 (imatrix,
# use a2_imatrix_mix3.sbatch) and quant.sbatch run unchanged with the same TAG. Resumable via markers.
#   sbatch --export=ALL,TAG=r144v3h,SRC=$P/pruned/r144v3,DELTA=$P/heal_runs/r144v3_full/delta_best.pt h1_merge_conv.sbatch
set -euo pipefail
. /work/glm53/rebuild_v2/common.sh
SRC=${SRC:?SRC missing}; DELTA=${DELTA:?DELTA missing}
[ -s $SRC/model.safetensors.index.json ] && [ -s $DELTA ] || { echo "missing $SRC or $DELTA"; exit 1; }

stage "H1.1 merge $DELTA into $SRC -> $PR"
if [ ! -f $M.build ]; then
  if [ -f $PR/model.safetensors.index.json ] && [ ! -f $M.build.started ]; then
    echo "$PR exists but was not started by this pipeline, refusing"; exit 3
  fi
  touch $M.build.started
  singularity exec -B /work --env PYTHONPATH=$P/pylib $P/vllm-rocm.sif \
    python3 $P/heal/heal_v3.py --merge --model $SRC --delta $DELTA --merged-out $PR 2>&1 | tail -5
  [ -s $PR/model.safetensors.index.json ] && touch $M.build
fi
[ -f $M.build ] || { echo "merge failed"; exit 1; }

stage "H1.2 convert -> $Q8"
if [ ! -f $M.conv ]; then
  is_gguf $Q8 && { echo "$Q8 exists without marker, refusing to overwrite"; exit 3; }
  . $P/venv/bin/activate
  python $P/llama.cpp/convert_hf_to_gguf.py $PR --outtype q8_0 --fp8-as-q8 --outfile $Q8.part 2>&1 \
    | grep -v -E "^INFO:hf-to-gguf:(blk|output|token)" | tail -8
  deactivate
  is_gguf $Q8.part && mv $Q8.part $Q8 && touch $M.conv
fi
[ -f $M.conv ] || { echo "convert failed"; exit 1; }

stage "H1.3 Q8_0 deliverable (hardlink, no copy)"
[ -e $OUTP-Q8_0.gguf ] || ln $Q8 $OUTP-Q8_0.gguf
ls -la $PR | head -3; ls -la $Q8 $OUTP-Q8_0.gguf
stage "H1 DONE"
