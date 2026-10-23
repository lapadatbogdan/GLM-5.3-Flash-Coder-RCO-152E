#!/bin/bash
#SBATCH --job-name=prepmn
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/work/glm53/rco_runs/prepmn_%x_%j.log
# One shard of the full model's top-20 reference on $CALIB (default calib53/mix_v1) (rco_prep_v2_mn.py: DSA fix, same forward as
# rco_prep_v2.py, windows SHARD, SHARD+NSHARDS, ...). Incremental + resumable (every 8 windows). The merge job
# (rco_prep_merge.sbatch) then writes calib53/mix_v1/ref_top20.pt. 576 windows x 18.5 s / 4 + ~18 min load = ~1 h.
# REAP=1 (default): the same forward also sums the REAP stats on the search windows (no extra pass; the experts
# compute exactly as before, checked bit-identical on CPU).
#   for i in 0 1 2 3; do CALIB=calib53/mix_v2 SHARD=$i NSHARDS=4 sbatch --job-name=prepmn$i rco_prep_v2_mn.sbatch; done
. /work/glm53/rco/calib_names.sh
OUT=${OUT:-$P/rco_runs/prep_$NAME}; RS=""; [ "${REAP:-1}" = 1 ] && RS="--reap-stats"
# REAPW=1: REAP sums PER WINDOW (search and held-out) instead of one total; the merge writes $CALIB/reap_windows.pt
[ "${REAPW:-0}" = 1 ] && RS="--reap-per-window"
SHARD=${SHARD:?}; NSHARDS=${NSHARDS:?}
[ -f $CAL/calib_search.pt ] && [ -f $CAL/calib_heldout.pt ] || { echo "missing calibration windows in $CAL"; exit 1; }
[ -e $CAL/ref_top20.pt ] && { echo "$CAL/ref_top20.pt exists, refusing to recompute"; exit 1; }
cd $P/rco
singularity exec -B /work --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  $P/vllm-rocm.sif python3 rco_prep_v2_mn.py --model $P/full --calib-dir $CAL --out $OUT --shard $SHARD --nshards $NSHARDS $RS
echo "EXIT $?"
