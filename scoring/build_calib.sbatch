#!/bin/bash
#SBATCH --job-name=c53build
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/work/glm53/calib53/logs/build_%j.log
# Calibration set from the generated GLM-5.3 sessions (whatever the 3 jobs produced, even if one crashed).
P=/work/glm53
cd $P/calib53
for L in py go js; do echo "$L: $(ls traj_$L 2>/dev/null | wc -l) sessions"; done
singularity exec -B /work --env PYTHONPATH=$P/pylib $P/vllm-rocm.sif \
  python3 build_calib.py --model $P/full --out $P/calib53 --seq 1024 --n-search 512 --n-heldout 64
[ -s calib_search.pt ] && [ -s calib_heldout.pt ] || { echo "SET MISSING"; exit 1; }
echo DONE
