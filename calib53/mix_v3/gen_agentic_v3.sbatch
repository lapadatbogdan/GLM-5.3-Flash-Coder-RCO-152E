#!/bin/bash
#SBATCH --job-name=mix3agent
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/work/glm53/calib53/mix_v3/logs/agent_%j.log
# mix_v3 agentic sessions, made exactly like calib53/traj_py (gen_py.sbatch: run_py.sh, full Q8 llama-server, 8 workers,
# clean room, agent_py.py, K=1, TEMP=1.0) on 80 NEW Python instances (mix_v3/agentic_tasks.txt: not in calib53/eval,
# not in traj_py/traj_js). MAXTURN=30 = the cap the original traj_py sessions ran with (agent_py.py default is now 200).
# Output calib53/mix_v3/traj_py/<instance>_att0.json; resumable (done instances are skipped).
#   for i in 0 1 2; do SHARD=$i NSHARDS=3 sbatch --job-name=mix3agent$i gen_agentic_v3.sbatch; done
export K=1 TEMP=1.0 MAXTURN=30 GEN_OUT=/work/glm53/calib53/mix_v3/traj_py
export SHARD=${SHARD:?} NSHARDS=${NSHARDS:?} OVLPFX=v3a${SHARD:?}
bash /work/glm53/calib53/mix_v3/run_agentic_v3.sh
