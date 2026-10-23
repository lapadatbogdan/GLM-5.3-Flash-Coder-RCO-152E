#!/bin/bash
#SBATCH --job-name=reapscore2
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=/work/glm53/rco_runs/reapscore2_%j.log
# REAP scores redone on the corrected torch path (SDPA bug on ROCm in the DSA layers). Masks 144/120/96 + held-out KL
# against the v2 reference (calib53/v2/ref_top20.pt). Runs after rco_prep_v2 (afterok). Writes to rco_runs/reap_v2.
P=/work/glm53
cd $P/rco
singularity exec -B /work --env PYTHONPATH=$P/pylib,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True $P/vllm-rocm.sif   python3 reap_score_v2.py --model $P/full --calib-dir $P/calib53/v2 --out $P/rco_runs/reap_v2 --keeps 144,120,96
echo "EXIT $?"
