#!/bin/bash
#SBATCH --job-name=c53py
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/work/glm53/calib53/logs/gen_py_%j.log
# RCO calibration: SWE agent sessions (py) written by GLM-5.3-Flash itself (Q8_0, thinking on).
# Resumable: one trajectory = one JSON file; a restart skips finished instances.
export K=1 TEMP=1.0
bash /work/glm53/calib53/run_py.sh
