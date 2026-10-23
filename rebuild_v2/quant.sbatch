#!/bin/bash
#SBATCH --job-name=v2Q
#SBATCH --account=your_project
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/glm53/rebuild_v2/logs/q_%x_%j.log
# Step Q (CPU): one quant of the pruned Q8_0 with its imatrix, same recipes as the published r144:
#   IQ2XXS  experts iq2_xxs, rest Q4_K_M   (prune_build/iq2.sbatch;        r144: 40 min)
#   IQ2S    experts iq2_s,   rest Q4_K_M   (prune_build/quant_var.sbatch;  r144: 26 min)
#   Q3K     experts q3_k,    rest Q4_K_M   (stageC step 4;                 r144: 7 min on 56 threads)
#   Q4KM    plain Q4_K_M (down_exps get Q6_K in some layers by llama.cpp's own rule) (q4.sbatch; r144: 19 min)
# Q8_0 is the A1 base itself (hardlinked there). Usage: sbatch --export=ALL,TAG=r96v2 quant.sbatch IQ2S
set -euo pipefail
. /work/glm53/rebuild_v2/common.sh
V=${1:?variant}
e() { echo "--tensor-type ffn_gate_exps=$1 --tensor-type ffn_up_exps=$1 --tensor-type ffn_down_exps=$1"; }
case $V in
  IQ2XXS) TT=$(e iq2_xxs);;
  IQ2S)   TT=$(e iq2_s);;
  Q3K)    TT=$(e q3_k);;
  Q4KM)   TT="";;
  *) echo "unknown variant $V"; exit 1;;
esac
OUT=$OUTP-$V.gguf
[ -f $M.imat ] && [ -s $IM ] || { echo "missing imatrix $IM (run A2)"; exit 1; }
if is_gguf $OUT; then echo "exists: $OUT"; ls -la $OUT; exit 0; fi
stage "quant $V -> $OUT"
run "$B/llama-quantize --allow-requantize --imatrix $IM $TT $Q8 $OUT.part Q4_K_M 32" 2>&1 | tail -4
is_gguf $OUT.part && mv $OUT.part $OUT
is_gguf $OUT || { echo "quant $V failed"; exit 1; }
ls -la $OUT
stage "quant $V DONE"
