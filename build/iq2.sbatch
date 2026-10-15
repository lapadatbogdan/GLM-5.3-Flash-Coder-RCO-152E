#!/bin/bash
#SBATCH --job-name=iq2
#SBATCH --account=your_project
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work/glm53/logs/iq2_%j.log
# Box variant like D (experts IQ2_XXS, rest Q4_K_M) from the pruned Q8 + its imatrix: sbatch iq2.sbatch f96
set -euo pipefail
P=/work/glm53; B=$P/llama.cpp/build/bin; TAG=${1:-${TAG:?}}
OUT=$P/out/GLM-5.3-Flash-Coder-$TAG-IQ2XXS.gguf
if [ -s $OUT ] && [ "$(head -c 4 $OUT)" = GGUF ]; then echo exists; exit 0; fi
singularity exec -B /work $P/vllm-rocm.sif bash -c "export LD_LIBRARY_PATH=/opt/rocm/lib:$B:\$LD_LIBRARY_PATH; $B/llama-quantize --allow-requantize --imatrix $P/gguf/pruned-$TAG.imatrix --tensor-type ffn_gate_exps=iq2_xxs --tensor-type ffn_up_exps=iq2_xxs --tensor-type ffn_down_exps=iq2_xxs $P/gguf/pruned-$TAG-q8_0.gguf $OUT.part Q4_K_M 32" 2>&1 | tail -4
[ "$(head -c 4 $OUT.part)" = GGUF ] && mv $OUT.part $OUT
ls -la $OUT
