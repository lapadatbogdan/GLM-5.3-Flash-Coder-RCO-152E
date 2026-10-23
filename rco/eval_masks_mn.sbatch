#!/bin/bash
#SBATCH --job-name=evalmasks
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/work/glm53/rco_runs/evalmasks_%x_%j.log
# One shard of the per-window held-out KL of a list of masks (eval_masks_mn.py): held-out windows SHARD, SHARD+NSHARDS..
# of $CALIB, against $CALIB/ref_top20.pt. Resumable per mask. ~18 min load + masks x windows/NSHARDS x ~9.5 s.
#   CALIB=calib53/mix_v3 MASKLIST=<json> OUT=<dir> SHARD=i NSHARDS=8 sbatch eval_masks_mn.sbatch
. /work/glm53/rco/calib_names.sh
MASKLIST=${MASKLIST:?}; OUT=${OUT:?}; SHARD=${SHARD:?}; NSHARDS=${NSHARDS:?}
[ -f $CAL/ref_top20.pt ] || { echo "missing $CAL/ref_top20.pt"; exit 1; }
cd $P/rco
singularity exec -B /work --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  $P/vllm-rocm.sif python3 eval_masks_mn.py --model $P/full --calib-dir $CAL --mask-list $MASKLIST --out $OUT \
    --shard $SHARD --nshards $NSHARDS
echo "EXIT $?"
