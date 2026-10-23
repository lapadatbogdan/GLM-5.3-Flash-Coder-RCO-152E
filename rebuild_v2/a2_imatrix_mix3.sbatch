#!/bin/bash
#SBATCH --job-name=v2A2m3
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/work/glm53/rebuild_v2/logs/a2m3_%x_%j.log
# Step A2 with the mix_v3 calibration (replaces a2_imatrix.sbatch, which uses calib53/calib_text.txt = old code-only
# sessions, without --parse-special). Text: calib53/mix_v3/N1270/calib_text_search.txt = the 1270 search windows
# (agentic 572 / code 444 / math 254, held-out excluded), decoded with special tokens, shuffled (seed 0) so any
# --chunks prefix keeps the domain mix; --parse-special turns <|user|>, <think> ... back into special tokens.
# CHUNKS=1270 (all, ~2.4 h by the old 42-47 min / 400 chunks) or e.g. 640 (~1.2 h).
#   sbatch --export=ALL,TAG=r144v3h,CHUNKS=1270 a2_imatrix_mix3.sbatch
set -euo pipefail
. /work/glm53/rebuild_v2/common.sh
TXT=${TXT:-$P/calib53/mix_v3/N1270/calib_text_search.txt}  # TXT override (19 oct): calib53/rco_mixg_v1 = mix_v3 + 15% general
CH=${CHUNKS:-1270}
is_gguf $Q8 || { echo "missing $Q8 (run A1 / H1)"; exit 1; }
[ -s $TXT ] || { echo "missing $TXT (calib53/mix_v3/N1270/build_imatrix_text.py)"; exit 1; }
stage "A2 imatrix (mix_v3, $CH chunks, parse-special) -> $IM"
if [ ! -f $M.imat ]; then
  run "$B/llama-imatrix -m $Q8 -f $TXT --parse-special -o $IM -c 1024 --chunks $CH -ngl 99 -sm layer --load-mode none" 2>&1 | tail -6
  is_gguf $IM && touch $M.imat
fi
[ -f $M.imat ] || { echo "imatrix failed"; exit 1; }
ls -la $IM
stage "A2 DONE"
