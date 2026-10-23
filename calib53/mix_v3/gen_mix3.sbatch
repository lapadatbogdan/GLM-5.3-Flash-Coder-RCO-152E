#!/bin/bash
#SBATCH --job-name=mix3gen
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/work/glm53/calib53/mix_v3/logs/gen_%j.log
# mix_v3 chat sessions (collect_prompts_v3.py: +36 math, +18 general, +30 code sessions for the 13 thinnest languages),
# same server and generator as mix_v1/v2 (calib53/mix_v1/gen_mix.py, temperature 1.0, thinking on, max_tokens 8192).
# Writes calib53/mix_v3/gen/turns_shard{i}.jsonl, resumable. 84 sessions over 3 nodes x 16 slots: ~1.3 h.
#   for i in 0 1 2; do SHARD=$i NSHARDS=3 SLOTS=16 sbatch --job-name=mix3gen$i gen_mix3.sbatch; done
set -uo pipefail
P=/work/glm53; D=$P/calib53/mix_v3; G=$P/calib53/mix_v1/gen_mix.py; B=$P/llama.cpp/build/bin
SIF=/work/tasks/vllm-rocm.sif
SLOTS=${SLOTS:-8}; CTX=$((SLOTS * 32768))   # 32k per slot: a session ends near 5.2k + one answer <= 8k
SHARD=${SHARD:-0}; NSHARDS=${NSHARDS:-1}
if [ "$NSHARDS" = 1 ]; then OUT=$D/gen/turns.jsonl; else OUT=$D/gen/turns_shard${SHARD}.jsonl; fi
LIM=0
if [ -n "${SMOKE:-}" ]; then OUT=$D/gen/smoke_turns.jsonl; LIM=$SMOKE; fi
mkdir -p $D/gen $D/logs
for f in $D/prompts/sessions_pool.jsonl; do [ -s $f ] || { echo "missing $f"; exit 1; }; done

singularity exec -B /work -B /work $SIF bash -c "
  export LD_LIBRARY_PATH=/opt/rocm/lib:$B:\$LD_LIBRARY_PATH
  $B/llama-server -m $P/gguf/full-q8_0.gguf -ngl 99 -sm layer --load-mode none --host 127.0.0.1 --port 8000 \
    -c $CTX --parallel $SLOTS --jinja" > $D/logs/serve-$SLURM_JOB_ID.log 2>&1 &
ok=no
for t in $(seq 1 90); do
  curl -s --max-time 120 http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"q","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' 2>/dev/null | grep -q '"choices"' && { ok=yes; break; }
  sleep 20
done
[ $ok = yes ] || { echo "SERVER DEAD - ABORT"; pkill -f llama-server; exit 1; }
echo "server up after $((t*20))s, slots $SLOTS, ctx/slot 32768"
/usr/bin/python3 $G --sessions $D/prompts/sessions_pool.jsonl \
  --out $OUT --workers $SLOTS --max-tokens ${MAXTOK:-8192} --temp 1.0 --limit-sessions $LIM \
  --shard $SHARD --nshards $NSHARDS
pkill -f llama-server
grep -E "eval time" $D/logs/serve-$SLURM_JOB_ID.log | grep -v prompt | tail -3
echo DONE
