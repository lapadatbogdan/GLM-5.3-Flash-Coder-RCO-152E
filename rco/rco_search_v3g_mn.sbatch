#!/bin/bash
#SBATCH --job-name=rcov3g
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=16:00:00
#SBATCH --output=/work/glm53/rco_runs/rcov3_%x_%j.log
# v3g (19 oct) = copy of rco_search_v3_mn.sbatch + INITMASK/INITSCORES overrides for INIT=reap (RCO k152 from gen_k152, calib with general text).
# RCO v3 (rco_search_v3_mn.py), defaults = the run decided 15 oct (paper-faithful, arXiv 2605.00649):
#   STE mask AFTER routing (no --reroute: the diagnosis showed reroute gives pruned experts no gradient), init from
#   expert activation FREQUENCY, 1300 steps, 4 Gumbel samples (1 per node), batch 1, no antithetic, exponential
#   tau 1.0 -> 0.01, per-layer budget, hard mask + alpha saved every 100 steps to OUT/masks/step{N}.pt.
#   CALIB=calib53/mix_v3 WINDOWS=1270 KEEP=144 sbatch rco_search_v3_mn.sbatch   -> rco_runs/mix3_N1270_k144_v3
# Env: STEPS (1300), SAMPLES (4), BATCH (1), ANTI (0), REROUTE (0), INIT (freq | router | reap), TAU_MIN (0.01),
#      MASK_EVERY (100); INIT=reap uses rco_runs/reap_mix3_N$WINDOWS/mask_reap_sum_k$KEEP.pt (+ stats.pt ordering),
#      MARGIN (1.5), SPREAD (1.0). Resumable from OUT/ckpt.pt.
. /work/glm53/rco/calib_names.sh
KEEP=${KEEP:-144}
# WINDOWS=N: the first N windows of a prefix-ordered pool (calib53/mix_v3), i.e. $CAL/N$N made by make_prefix.py
if [ -n "${WINDOWS:-}" ]; then
  CAL=$CAL/N$WINDOWS; PFX=${PFX}_N$WINDOWS
  [ -f $CAL/calib_search.pt ] || { echo "missing $CAL: run make_prefix.py --n $WINDOWS first"; exit 1; }
fi
OUT=${OUT:-$P/rco_runs/${PFX}_k${KEEP}_v3}
STEPS=${STEPS:-1300}; BATCH=${BATCH:-1}; SAMPLES=${SAMPLES:-4}; ANTI=${ANTI:-0}; REROUTE=${REROUTE:-0}; INIT=${INIT:-freq}
TAU_MIN=${TAU_MIN:-0.01}; MASK_EVERY=${MASK_EVERY:-100}; MARGIN=${MARGIN:-1.5}; SPREAD=${SPREAD:-1.0}
EXTRA="--n-steps $STEPS --n-gumbel-samples $SAMPLES --batch-size $BATCH --tau-min $TAU_MIN --mask-every $MASK_EVERY"
[ "$ANTI" = 1 ] && EXTRA="$EXTRA --antithetic"
[ "$REROUTE" = 1 ] && EXTRA="$EXTRA --reroute"
case $INIT in
  freq|router) EXTRA="$EXTRA --init-from $INIT";;
  reap) M=${INITMASK:-$P/rco_runs/reap_mix3_N${WINDOWS}/mask_reap_sum_k${KEEP}.pt}; SC=${INITSCORES-$P/rco_runs/reap_mix3_N${WINDOWS}/stats.pt}
        [ -f $M ] || { echo "missing $M"; exit 1; }
        EXTRA="$EXTRA --init-from freq --init-from-mask $M --init-margin $MARGIN --init-spread $SPREAD"
        [ -n "$SC" ] && [ -f "$SC" ] && EXTRA="$EXTRA --init-scores $SC";;
  *) echo "INIT must be freq, router or reap"; exit 1;;
esac
[ -f $CAL/ref_top20.pt ] || { echo "missing $CAL/ref_top20.pt (run the mix_v1 reference first)"; exit 1; }
[ -f $OUT/result.json ] && { echo "$OUT/result.json exists, refusing to overwrite"; exit 1; }
mkdir -p $OUT
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
echo "EXTRA $EXTRA"
echo "nodes $SLURM_JOB_NODELIST master $MASTER_ADDR:$MASTER_PORT keep $KEEP calib $CAL out $OUT"
cd $P/rco
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --gpus-per-node=8 --cpus-per-task=56 \
  singularity exec -B /work \
    --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    $P/vllm-rocm.sif python3 rco_search_v3_mn.py --mode search --keep $KEEP $EXTRA \
      --model $P/full --calib-dir $CAL --out $OUT
echo "EXIT $?"
grep -E "identical|held-out KL|alpha init from mask|v3 final" $OUT/log.txt | tail -8
