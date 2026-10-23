#!/bin/bash
#SBATCH --job-name=prepmerge
#SBATCH --account=your_project
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/glm53/rco_runs/prepmerge_%j.log
# Merge the reference shards into $CALIB/ref_top20.pt (refuses to overwrite) + prep_report.json (halves overlap).
# With REAP=1 (default) also sums the shards' REAP stats into rco_runs/reap_$PFX (stats.pt + mask_{freq,reap_mean,
# reap_sum}_k{144,120,96}.pt, as reap_score.py); refuses to overwrite.
#   CALIB=calib53/mix_v2 NSHARDS=4 sbatch --dependency=afterok:<shard jobs> rco_prep_merge.sbatch
. /work/glm53/rco/calib_names.sh
OUT=${OUT:-$P/rco_runs/prep_$NAME}; NSHARDS=${NSHARDS:?}; REAP_OUT=${REAP_OUT:-$P/rco_runs/reap_$PFX}
RO=""; [ "${REAP:-1}" = 1 ] && RO="--reap-out $REAP_OUT"   # REAP=0 with per-window shards (reap_windows.pt is automatic)
cd $P/rco
singularity exec -B /work --env PYTHONPATH=$P/pylib $P/vllm-rocm.sif \
  python3 rco_prep_v2_mn.py --merge --calib-dir $CAL --out $OUT --nshards $NSHARDS $RO
echo "EXIT $?"
