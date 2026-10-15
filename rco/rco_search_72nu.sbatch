#!/bin/bash
#SBATCH --job-name=rco72nu
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/work/glm53/logs/rco72nu_%j.log
# Stage B2: RCO search, 72 experts kept per layer on average (non-uniform); resumable from rco_runs/k72nu/ckpt.pt.
P=/work/glm53
cd $P/rco
singularity exec -B /work --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True $P/vllm-rocm.sif \
  python3 rco_glm_prune.py --mode search --keep 72 --nonuniform --model $P/full --calib-dir $P/calib53 --out $P/rco_runs/k72nu
echo "EXIT $?"
