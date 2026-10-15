#!/bin/bash
#SBATCH --job-name=q4
#SBATCH --account=your_project
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work/glm53/logs/q4_%j.log
# Q4_K_M everywhere (with imatrix) from the pruned Q8: variant for 96-128 GB (Mac, Strix Halo, RTX 6000 Pro). sbatch q4.sbatch r144
set -euo pipefail
P=/work/glm53; B=$P/llama.cpp/build/bin; TAG=${1:-${TAG:?}}
OUT=$P/out/GLM-5.3-Flash-Coder-$TAG-Q4KM.gguf
if [ -s $OUT ] && [ "$(head -c 4 $OUT)" = GGUF ]; then echo exists; exit 0; fi
singularity exec -B /work $P/vllm-rocm.sif bash -c "export LD_LIBRARY_PATH=/opt/rocm/lib:$B:\$LD_LIBRARY_PATH; $B/llama-quantize --allow-requantize --imatrix $P/gguf/pruned-$TAG.imatrix $P/gguf/pruned-$TAG-q8_0.gguf $OUT.part Q4_K_M 32" 2>&1 | tail -4
[ "$(head -c 4 $OUT.part)" = GGUF ] && mv $OUT.part $OUT
ls -la $OUT
