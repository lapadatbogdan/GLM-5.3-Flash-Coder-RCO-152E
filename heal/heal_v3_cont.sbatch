#!/bin/bash
#SBATCH --job-name=healg
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/work/glm53/logs/healg_%x_%j.log
# Heal continuation with general text (heal_v3_cont.py = heal_v3.py + --init-delta): starts from an existing heal's
# delta_best (step 0 = the healed model), trains on calib53/heal_mixg_v1 (mix_v3 N1270 prefix + ~12.5% general_v1
# windows), evaluates on the same 256 mix_v3 held-out windows (general = 25 monitor windows) as every heal so far.
# heal_v3 picks delta_best on the weighted KL (agentic/code/math), which does NOT see general: --patience 0 so the
# run always completes, and OUT/delta.pt (last step) is kept next to delta_best; the periodic logs show general.
#   VARIANT=a (REAP r144v3, from r144v3_full) | b (RCO r144v4, from r144v4_full) | c (k152 gen mask r152v3g, from r144v3_full) | d (RCO k152 general calib step400 r152v5, from r144v4_full)
#   sbatch --job-name=healg_a --export=ALL,VARIANT=a heal_v3_cont.sbatch
set -uo pipefail
P=/work/glm53
case ${VARIANT:?} in
  a) PRUNED=$P/pruned/r144v3;   INIT=$P/heal_runs/r144v3_full/delta_best.pt; TAGO=r144v3_g1;;
  b) PRUNED=$P/pruned/r144v4;   INIT=$P/heal_runs/r144v4_full/delta_best.pt; TAGO=r144v4_g1;;
  c) PRUNED=$P/pruned/r152v3g;  INIT=$P/heal_runs/r144v3_full/delta_best.pt; TAGO=r152v3g_g1;;
  d) PRUNED=$P/pruned/r152v5;   INIT=$P/heal_runs/r144v4_full/delta_best.pt; TAGO=r152v5_g1;;
  *) echo "VARIANT a|b|c|d"; exit 1;;
esac
CAL=${CAL:-$P/calib53/heal_mixg_v1}; OUT=${OUT:-$P/heal_runs/$TAGO}; STEPS=${STEPS:-96}
[ -s $PRUNED/model.safetensors.index.json ] || { echo "missing $PRUNED"; exit 1; }
[ -s $INIT ] && [ -s $CAL/ref_top20.pt ] || { echo "missing $INIT or $CAL/ref_top20.pt (make_heal_mix.py)"; exit 1; }
[ -f $OUT/DONE ] && { echo "$OUT already DONE"; exit 0; }
mkdir -p $OUT
cd $P/heal
singularity exec -B /work --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True $P/vllm-rocm.sif \
  python3 heal_v3_cont.py --init-delta $INIT --model $PRUNED --out $OUT --calib-dir $CAL --ref $CAL/ref_top20.pt \
  --lr ${LR:-2e-5} --warmup-steps 2 --grad-accum 8 --max-steps $STEPS --eval-every 16 --eval-subset 64 --patience 0 \
  --no-full-before --final-full-eval --deadline-min 700 ${HEAL_ARGS:-}
echo "EXIT $?"
grep -E "\[cont\]|held-out KL|FULL|DONE" $OUT/log.txt | tail -12
