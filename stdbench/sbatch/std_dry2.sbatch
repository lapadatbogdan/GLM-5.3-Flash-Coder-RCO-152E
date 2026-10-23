#!/bin/bash
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=/work/glm53/stdbench/logs/std_%x_%j.log
# v3 (16 oct): full model, per-slot KV (run_dry2.sh), NP slots x CTXSLOT, one bench group per node.
# Needs TAG BENCHES; optional SHARD i/n. max_tokens floor 131072 for reasoning benches (was 32K/65K: 8 truncated runs).
export GGUF=${GGUF:-/work/glm53/gguf/full-q8_0.gguf} NSERV=${NSERV:-1} NP=${NP:-16} CTXSLOT=${CTXSLOT:-163840}
export SERVER_EXTRA="-ub 128 -fa on -cram 98304 ${XTRA:-}" SWEEP="" CONC=${CONC:-$((NP*NSERV))}
export MINTOK=${MINTOK:-lcb=131072,aime25=131072,math500=131072}
bash /work/glm53/stdbench/run_dry2.sh
