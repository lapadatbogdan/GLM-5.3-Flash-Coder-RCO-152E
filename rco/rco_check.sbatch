#!/bin/bash
#SBATCH --job-name=rcocheck
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/work/glm53/logs/rcocheck_%j.log
# Question: does the fixed RCO search routing (--reroute) give the same KL as the real deployment on the same mask?
# Criterion: |reroute - deployment| < 1% on 8 held-out windows, at 144 and 96 (frequency mask).
# The old routing (mask after top-k, no replacement) is measured alongside: shows how far off it was.
P=/work/glm53
cd $P/rco
singularity exec -B /work --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True $P/vllm-rocm.sif \
  python3 rco_glm_prune.py --mode check --keeps 144,96 --model $P/full --calib-dir $P/calib53 --out $P/rco_runs/check
echo "EXIT $?"
