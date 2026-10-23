#!/bin/bash
#SBATCH --job-name=healv3
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/work/glm53/logs/healv3_%x_%j.log
# LoRA healing of a pruned checkpoint (default r144v4 = RCO k144 mask on mix_v3/N1270) against the full model's
# top-20 (corrected torch path). LoRA on attention + the shared expert, no router, routed experts FP8 from the pruned
# checkpoint. Evaluation = the eval_masks_mn convention (KL weighted 45/35/20, general = monitor) + top-1 agreement
# with the full model.
# MODE=pilot (default): 12 steps x 8 windows, eval at 0/4/8/12 on 64 stratified held-out windows, ~2.1 h.
# MODE=full: 1 epoch (159 steps), eval every 16 steps on 64 windows, all 256 before and at the end; resumable,
#   resubmit the same command until OUT/DONE appears (walltime 12 h, save at 11 h 40).

#   sbatch --job-name=pilot /work/glm53/heal/heal_v3.sbatch
#   sbatch --job-name=full --time=12:00:00 --export=ALL,MODE=full /work/glm53/heal/heal_v3.sbatch
set -uo pipefail
P=/work/glm53
MODE=${MODE:-pilot}
PRUNED=${PRUNED:-$P/pruned/r144v4}
CAL=$P/calib53/mix_v3/N1270
OUT=${OUT:-$P/heal_runs/r144v4_$MODE}
[ -s $PRUNED/model.safetensors.index.json ] || { echo "missing $PRUNED (rebuild_v2 A1 not done)"; exit 1; }
case $MODE in
  pilot) ARGS="--max-steps 12 --eval-every 4 --eval-subset 64 --patience 2 --no-full-before --no-final-full-eval --deadline-min 160";;
  full)  ARGS="--epochs 1 --eval-every 16 --eval-subset 64 --patience 3 --full-before --final-full-eval --deadline-min 700";;
  *) echo "MODE pilot|full"; exit 1;;
esac
mkdir -p $OUT
cd $P/heal
singularity exec -B /work --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True $P/vllm-rocm.sif \
  python3 heal_v3.py --model $PRUNED --out $OUT --calib-dir $CAL --ref $CAL/ref_top20.pt \
  --compare-kl-dir $P/rco_runs/rcov3_ckpt_eval600/kl --compare-mask rco_step600 \
  --lr ${LR:-2e-5} --warmup-steps 2 --grad-accum 8 $ARGS ${HEAL_ARGS:-}
echo "EXIT $?"
grep -E "STEP0 CHECK|held-out KL|FULL|not improved|early stop|DONE" $OUT/log.txt
