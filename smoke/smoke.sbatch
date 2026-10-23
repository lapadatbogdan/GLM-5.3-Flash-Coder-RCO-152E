#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=02:30:00
#SBATCH --output=/work/glm53/smoke/smoke_%j.log
# Basic regression on each published quant: arithmetic, fib, knowledge, a small code task executed, a tool call.
P=/work/glm53; B=$P/llama.cpp/build/bin
for v in Q8_0 Q4KM Q3K IQ2XXS; do
  G=$P/out/GLM-5.3-Flash-Coder-r152v5g-$v.gguf
  singularity exec -B /work $P/vllm-rocm.sif bash -c "export LD_LIBRARY_PATH=/opt/rocm/lib:$B:\$LD_LIBRARY_PATH; $B/llama-server -m $G -ngl 99 -sm layer --load-mode none --host 127.0.0.1 --port 8000 -c 65536 --parallel 1 --jinja" > $P/smoke/serve_${v}_$SLURM_JOB_ID.log 2>&1 &
  for i in $(seq 1 90); do curl -s -m 5 127.0.0.1:8000/health | grep -q ok && break; sleep 20; done
  python3 $P/smoke/smoke.py http://127.0.0.1:8000/v1/chat/completions $v
  pkill -f llama-server; sleep 20
done
