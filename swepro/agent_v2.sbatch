#!/bin/bash
#SBATCH --job-name=dsw_v2
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=/work/glm53/swepro/logs/agent_%j.log
# Agentic run (mini-swe-agent, no turn cap, WALL=12 h per attempt (default), T=0.95 top_p=1.0, 400K context per slot).
#   sbatch agent.sbatch <bench> <run> <gguf> [ntask]
# Several jobs with the same <run> share the list (atomic claim); resubmitting resumes where it stopped.
# Full model: 1 server x 8 GCDs (NSERV=1); pruned Q8: NSERV=2; 2 bit: NSERV=8 (export before sbatch).
export BENCH=$1 RUN=$2 GGUF=$3 NTASK=${4:-0}
export NSERV=${NSERV:-1} SLOTS=${SLOTS:-8} K=${K:-1}
bash /work/glm53/swepro/run_bench_v2.sh
