#!/bin/bash
#SBATCH --job-name=qvar
#SBATCH --account=your_project
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work/glm53/logs/qvar_%j.log
# Quantization variants for the box (~46 GB) from the pruned Q8 + imatrix: sbatch quant_var.sbatch r120 MIX
#   MIX   = gate/up IQ2_XXS, down Q3_K (the sensitive part), rest Q4_K_M
#   IQ2S  = all experts IQ2_S (~2.5 bits), rest Q4_K_M
set -euo pipefail
P=/work/glm53; B=$P/llama.cpp/build/bin; TAG=$1; V=$2
case $V in
  MIX)  TT="--tensor-type ffn_gate_exps=iq2_xxs --tensor-type ffn_up_exps=iq2_xxs --tensor-type ffn_down_exps=q3_k";;
  IQ2S) TT="--tensor-type ffn_gate_exps=iq2_s --tensor-type ffn_up_exps=iq2_s --tensor-type ffn_down_exps=iq2_s";;
  *) echo "unknown variant $V"; exit 1;;
esac
OUT=$P/out/GLM-5.3-Flash-Coder-$TAG-$V.gguf
if [ -s $OUT ] && [ "$(head -c 4 $OUT)" = GGUF ]; then echo exists; exit 0; fi
singularity exec -B /work $P/vllm-rocm.sif bash -c "export LD_LIBRARY_PATH=/opt/rocm/lib:$B:\$LD_LIBRARY_PATH; $B/llama-quantize --allow-requantize --imatrix $P/gguf/pruned-$TAG.imatrix $TT $P/gguf/pruned-$TAG-q8_0.gguf $OUT.part Q4_K_M 32" 2>&1 | tail -4
[ "$(head -c 4 $OUT.part)" = GGUF ] && mv $OUT.part $OUT
ls -la $OUT
