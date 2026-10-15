#!/bin/bash
#SBATCH --job-name=reapscore
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=/work/glm53/logs/reapscore_%j.log
# Question: does a REAP score (router weight x expert output norm) on our own code sessions give a better mask
# than frequency? Criterion: held-out KL below frequency (0.94 at 144, 1.47 at 96) by >= 5%.
# Deterministic, one pass over the 512 windows; stats saved every 32 windows (resumable).
P=/work/glm53
cd $P/rco
singularity exec -B /work --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True $P/vllm-rocm.sif \
  python3 reap_score.py --model $P/full --calib-dir $P/calib53 --out $P/rco_runs/reap --keeps 144,96
echo "EXIT $?"
