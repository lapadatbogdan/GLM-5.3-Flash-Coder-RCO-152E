#!/bin/bash
#SBATCH --job-name=rcoprep
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=/work/glm53/logs/rcoprep_%j.log
# Stage B1: top-20 reference + per-window router stats, agreement of the two halves (>90% = enough data), held-out baseline.
P=/work/glm53
cd $P/rco
singularity exec -B /work --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True $P/vllm-rocm.sif \
  python3 rco_glm_prune.py --mode prep --model $P/full --calib-dir $P/calib53 --out $P/rco_runs/prep
echo "EXIT $?"
