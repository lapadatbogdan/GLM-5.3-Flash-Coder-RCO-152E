#!/bin/bash
#SBATCH --job-name=mix3build
#SBATCH --account=your_project
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/glm53/calib53/mix_v3/logs/build_%j.log
# calib53/mix_v3/calib_search.pt (prefix-ordered pool, up to 1500) + calib_heldout.pt (fixed, ~256) + pool_report.json
# from every agent session and every generated shard (mix_v1, v2, v3). CPU, ~5 min. Refuses to overwrite.
P=/work/glm53; C=$P/calib53; D=$C/mix_v3
singularity exec -B /work --env PYTHONPATH=$P/pylib $P/vllm-rocm.sif \
  python3 $D/build_mix3.py --model $P/full --calib53 $C \
    --gen "$C/mix_v1/gen/turns_shard*.jsonl" "$C/mix_v2/gen/turns_shard*.jsonl" "$D/gen/turns_shard*.jsonl" \
    --agent-dirs $C/traj_py $C/traj_js $D/traj_py --out $D
