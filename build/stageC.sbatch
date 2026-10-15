#!/bin/bash
#SBATCH --job-name=stageC
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=/work/glm53/logs/stageC_%j.log
# Stage C for an RCO mask: physical pruning -> Q8_0 -> imatrix on our calibration -> Q3_K for the box -> evaluation on
# the same 30 Python tasks as the full model (calib53/eval/eval_tasks_py.txt). Resumable: each step leaves a marker file.
#   TAG=k96 MASK=/work/.../rco_runs/k96/mask.pt sbatch stageC.sbatch
set -euo pipefail
P=/work/glm53; B=$P/llama.cpp/build/bin
TAG=${TAG:?}; MASK=${MASK:?}
PR=$P/pruned/$TAG; Q8=$P/gguf/pruned-$TAG-q8_0.gguf; IM=$P/gguf/pruned-$TAG.imatrix
Q3=$P/out/GLM-5.3-Flash-Coder-$TAG-Q3K.gguf; M=$P/pruned/.done_$TAG
mkdir -p $P/pruned
run() { singularity exec -B /work $P/vllm-rocm.sif bash -c "export LD_LIBRARY_PATH=/opt/rocm/lib:$B:\$LD_LIBRARY_PATH; $1"; }
stage() { echo "=== $(date +%T) $1"; }

stage "1 prune $TAG ($MASK)"
if [ ! -f $M.build ]; then
  $P/venv/bin/python $P/prune_build/build_pruned.py --src $P/full --mask $MASK --out $PR --mtp drop 2>&1 | tail -5
  [ -s $PR/model.safetensors.index.json ] && touch $M.build
fi
stage "2 convert to Q8_0"
if [ ! -f $M.conv ]; then
  . $P/venv/bin/activate
  python $P/llama.cpp/convert_hf_to_gguf.py $PR --outtype q8_0 --fp8-as-q8 --outfile $Q8 2>&1 | grep -v -E "^INFO:hf-to-gguf:(blk|output|token)" | tail -8
  deactivate
  [ "$(head -c 4 $Q8)" = "GGUF" ] && touch $M.conv
fi
stage "3 imatrix on the GLM-5.3 sessions"
if [ ! -f $M.imat ]; then
  run "$B/llama-imatrix -m $Q8 -f $P/calib53/calib_text.txt -o $IM -c 1024 --chunks 400 -ngl 99 -sm layer --load-mode none" 2>&1 | tail -4
  [ -s $IM ] && touch $M.imat
fi
stage "4 Q3_K experts (rest Q4_K_M) for the box"
if [ ! -f $M.q3 ]; then
  run "$B/llama-quantize --allow-requantize --imatrix $IM --tensor-type ffn_gate_exps=q3_k --tensor-type ffn_up_exps=q3_k --tensor-type ffn_down_exps=q3_k $Q8 $Q3 Q4_K_M 56" 2>&1 | tail -3
  [ "$(head -c 4 $Q3)" = "GGUF" ] && touch $M.q3
fi
ls -la $Q8 $Q3
stage "5 evaluate the pruned Q8 on the 30 tasks (2 attempts)"
# HELPER=1: a second node takes the second half of the tasks on the same model (eval_helper.sbatch), once
if [ "${HELPER:-0}" = 1 ] && [ ! -f $M.helper ]; then
  sbatch --job-name=hlp$TAG $P/calib53/eval_helper.sbatch pruned-$TAG-q8_0.gguf $TAG && touch $M.helper
fi
K=2 TEMP=1.0 EVAL_GGUF=$Q8 GEN_OUT=$P/calib53/eval/$TAG OVLPFX=c53e$TAG bash $P/calib53/run_eval_py.sh
stage "DONE"
