#!/bin/bash
#SBATCH --job-name=reapmap
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/work/glm53/logs/reapmap_%j.log
# REAP map: held-out KL at 192 and 120 experts (frequency vs total REAP), stats reused from rco_runs/reap.
P=/work/glm53
cd $P/rco
singularity exec -B /work --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True $P/vllm-rocm.sif \
  python3 reap_score.py --model $P/full --calib-dir $P/calib53 --out $P/rco_runs/reap2 --keeps 192,120 --masks reap_sum
echo "EXIT $?"
