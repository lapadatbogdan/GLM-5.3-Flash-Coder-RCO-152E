#!/bin/bash
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/work/glm53/calib53/logs/evalfill_%x_%j.log
# Reruns only the attempts moved out of a GEN_OUT (skips what exists).  sbatch eval_fill.sbatch GGUF_PATH NAME LIST NSERV SLOTS TAG
P=/work/glm53
export CTXSLOT=${7:-131072} NSERV=$4 SLOTS=$5 K=2 TEMP=1.0 EVAL_GGUF=$1 GEN_OUT=$P/calib53/eval/$2 OVLPFX=c53u$2$6 TASKLIST=$3
bash $P/calib53/run_eval_py.sh
