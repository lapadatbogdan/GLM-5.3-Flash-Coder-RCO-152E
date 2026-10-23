#!/bin/bash
# Agentic run of SWE-bench Pro (V2 / v1) or DeepSWE on one 8-GPU node.
# Resumable: every attempt is a file OUT/<id>/att<k>.json; existing ones are skipped.
# Several jobs can work on the SAME list at once: each task is claimed atomically (mkdir), and
# claims of jobs that are no longer running are released automatically. Resubmitting the same sbatch continues where it stopped.
#
# Variables (defaults in parentheses):
#   BENCH (sbp2) sbp2|sbp1|deepswe     MODE (agent) agent|gold|nop     RUN (run name, output directory)
#   GGUF  the model                    NSERV (1) servers/node          SLOTS (8) conversations/server
#   CTXSLOT (409600) max context per conversation (400K as at Z.ai)    CTXTOT (1638400) total KV per server (unified)
#   K (1) attempts/task                NTASK (0=all) first N of the manifest (stratified sample)
#   IDS file with ids (optional)       TEMP (0.95) TOPP (1.0) WALL (43200 s = 12 h; Z.ai: 6 h at API speed, us ~4 tok/s/slot) AGENT_CFG (mini.yaml)
#   NW (NSERV*SLOTS) workers; in MODE=gold|nop no server starts and NW defaults to 16
set -u
R=/work/glm53/swepro
P=/work/glm53
BENCH=${BENCH:-sbp2}; MODE=${MODE:-agent}; RUN=${RUN:?set RUN}; K=${K:-1}; NTASK=${NTASK:-0}
OUT=$R/results/$BENCH/$RUN; mkdir -p $OUT $R/logs
export SWEPRO_SCRATCH=/work/swe_scratch
export TEMP=${TEMP:-0.95} TOPP=${TOPP:-1.0} WALL=${WALL:-43200} AGENT_CFG=${AGENT_CFG:-mini.yaml} MODEL_TAG=${MODEL_TAG:-$(basename ${GGUF:-none})}
JOB=${SLURM_JOB_ID:-local$$}
source $R/venv/bin/activate
export LITELLM_LOCAL_MODEL_COST_MAP=True MSWEA_COST_TRACKING=ignore_errors MSWEA_SILENT_STARTUP=1
echo "=== $(date) job=$JOB bench=$BENCH mode=$MODE run=$RUN model=$MODEL_TAG K=$K T=$TEMP top_p=$TOPP wall=$WALL cfg=$AGENT_CFG ==="

# ---- task list (only tasks with an image) ----
LIST=$OUT/tasks_$JOB.tsv
if [ -n "${IDS:-}" ]; then grep -F -f "$IDS" $R/manifests/$BENCH.tsv > $LIST; else cp $R/manifests/$BENCH.tsv $LIST; fi
# only tasks with an OPENED image ready (the image pull leaves <sif>.open); column 7 of the manifest is informative only
while IFS=$'\t' read -r id repo lang cal tdir sif have; do [ -f "$sif.open" ] && printf '%s\n' "$id	$repo	$lang	$cal	$tdir	$sif	1"; done < $LIST > $LIST.ok; mv $LIST.ok $LIST
[ "$NTASK" -gt 0 ] && { head -$NTASK $LIST > $LIST.n; mv $LIST.n $LIST; }
NT=$(wc -l < $LIST); echo "=== $NT tasks with an image ==="; cut -f2 $LIST | sort | uniq -c
[ "$NT" -eq 0 ] && exit 1

# ---- LLM servers ----
if [ "$MODE" = agent ] && [ -n "${LLM_URL:-}" ]; then
  # external endpoint (test with a fake LLM or another server already running): no llama-server
  NSERV=1; NW=${NW:-2}
elif [ "$MODE" = agent ]; then
  NSERV=${NSERV:-1}; SLOTS=${SLOTS:-8}; G=$((8 / NSERV)); CTXSLOT=${CTXSLOT:-409600}; CTXTOT=${CTXTOT:-1638400}
  start_servers() {
    for i in $(seq 0 $((NSERV-1))); do
      devs=$(seq -s, $((i*G)) $((i*G+G-1)))
      singularity exec -B /work --env ROCR_VISIBLE_DEVICES=$devs --env LLAMA_DECODE_MICROBATCH=${LLAMA_DECODE_MICROBATCH:-0} $P/vllm-rocm.sif bash -c "
        export LD_LIBRARY_PATH=/opt/rocm/lib:${LLAMA_BIN:-$P/llama.cpp/build/bin}:\$LD_LIBRARY_PATH
        ${LLAMA_BIN:-$P/llama.cpp/build/bin}/llama-server -m ${GGUF:?} -ngl 99 -sm layer --load-mode none --host 127.0.0.1 --port $((8000+i)) \
          -c $1 --parallel $SLOTS $2 -cram ${CRAM:-98304} --jinja -lv 3 ${SERVER_EXTRA:-}" \
        > $R/logs/serve-$JOB-$i.log 2>&1 &
    done
  }
  wait_servers() {
    for i in $(seq 0 $((NSERV-1))); do
      ok=no
      for t in $(seq 1 120); do
        r=$(curl -s --max-time 120 http://127.0.0.1:$((8000+i))/v1/chat/completions -H "Content-Type: application/json" \
            -d '{"model":"q","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' 2>/dev/null)
        echo "$r" | grep -q '"choices"' && { ok=yes; break; }
        pgrep -f "port $((8000+i))" >/dev/null || break
        sleep 20
      done
      [ "$ok" = yes ] || return 1
    done
  }
  # v2 (16 Oct): NO unified KV. With -kvu the DSA indexer scans the whole shared buffer
  # (8 conversations x context), ~1.9x slower. Each slot has its own KV of CTXSLOT. KV f16 (same numbers as the old runs).
  # Try 8 slots, then 6; a config that falls back to "retrying without pipeline parallelism" is rejected (prefill ~4x slower).
  STARTED=no
  for sl in ${SLOTS_TRY:-8 6}; do
    SLOTS=$sl; c=$((CTXSLOT * SLOTS)); kvf="-no-kvu -ub ${UB:-128} --no-context-shift"
    echo "=== starting $NSERV server(s) x $G GCDs, $SLOTS slots, KV $c ($kvf) ==="
    start_servers $c "$kvf"
    if wait_servers; then
      if grep -q "retrying without pipeline parallelism" $R/logs/serve-$JOB-*.log; then
        echo "!!! $SLOTS slots: pipeline parallelism dropped - rejecting"
      else STARTED=yes; CTXTOT=$c; break; fi
    else echo "!!! does not start with $SLOTS slots:"; tail -3 $R/logs/serve-$JOB-0.log; fi
    pkill -f llama-server; sleep 30
  done
  [ $STARTED = yes ] || { echo "SERVER DEAD - ABORT"; exit 1; }
  grep -iE "kv.*(MiB|GiB)|n_ctx|buffer size" $R/logs/serve-$JOB-0.log | head -8
  # Tool-call probe: if llama.cpp does not extract tool_calls from the GLM format, mini.yaml cannot work.
  probe=$(curl -s --max-time 1800 http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model":"q","temperature":0.95,"top_p":1.0,
    "messages":[{"role":"user","content":"Use the bash tool to run: ls /tmp"}],
    "tools":[{"type":"function","function":{"name":"bash","description":"Execute a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]}')
  echo "$probe" | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('TOOLPROBE tool_calls=%s content=%r' % (json.dumps(m.get('tool_calls'))[:200], (m.get('content') or '')[:120]))" || echo "TOOLPROBE parse fail: ${probe:0:300}"
  if [ "$AGENT_CFG" = mini.yaml ] && ! echo "$probe" | grep -q '"tool_calls"'; then
    echo "!!! the server returns no tool_calls -> switching to mini_textbased.yaml"; export AGENT_CFG=mini_textbased.yaml
  fi
  NW=${NW:-$((NSERV*SLOTS))}
else
  NSERV=1; NW=${NW:-16}
fi
echo "=== workers: $NW ==="

# ---- memory guard: a runaway process (seen 12 Oct: 365 GB) kills the whole node; stop only that process ----
( while true; do
    ps -eo pid=,rss=,comm= --sort=-rss | awk '$2 > 41943040 && $3 !~ /llama-server/ {print $1}' | while read p; do
      echo "[guard] kill $p ($(ps -o rss=,args= -p $p | cut -c1-150))"; kill -9 $p; done
    sleep 10; done ) >> $OUT/guard_$JOB.log 2>&1 &
GUARD=$!

claim() {  # atomic claim; claims of jobs that no longer exist are released
  local d=$1
  if mkdir "$d" 2>/dev/null; then echo $JOB > "$d/job"; return 0; fi
  local j=$(cat "$d/job" 2>/dev/null)
  if [ -n "$j" ] && [ "$j" != "$JOB" ] && ! squeue -h -j "$j" -o %i 2>/dev/null | grep -q "$j"; then
    rm -rf "$d"; mkdir "$d" 2>/dev/null && { echo $JOB > "$d/job"; return 0; }
  fi
  return 1
}

worker() {
  local wid=$1 port=$((8000 + $1 % NSERV))
  for k in $(seq 0 $((K-1))); do
    while IFS=$'\t' read -r id repo lang cal tdir sif have; do
      [ -f "$OUT/$id/att$k.json" ] && continue
      # do not start an attempt that no longer has time to finish in the current job (the agent phase would be lost)
      if [ -n "${SLURM_JOB_ID:-}" ] && [ "$MODE" = agent ]; then
        left=$(squeue -h -j $SLURM_JOB_ID -o %L | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>=3) s+=3600*$(n-2); if(n==4) s+=86400*$1; print s}')
        [ -n "$left" ] && [ "$left" -lt $((WALL + 3600)) ] && { echo "[w$wid] $left s left < WALL+1h - not claiming more" >> $OUT/worker_${JOB}_$wid.log; return; }
      fi
      mkdir -p "$OUT/$id"
      claim "$OUT/$id/att$k.claim" || continue
      for try in 1 2 3; do
        python3 $R/swe_attempt.py --bench $BENCH --id "$id" --task-dir "$tdir" --sif "$sif" --out $OUT --attempt $k \
          --mode $MODE --url ${LLM_URL:-http://127.0.0.1:$port/v1} --wid $JOB-$wid \
          --meta "{\"repo\":\"$repo\",\"lang\":\"$lang\",\"calibrated\":$cal}" >> $OUT/worker_${JOB}_$wid.log 2>&1
        [ -f "$OUT/$id/att$k.json" ] && break
        echo "[w$wid] $id att$k try $try without result (infra) - retrying" >> $OUT/worker_${JOB}_$wid.log
        # dead server? then there is no point burning through the list
        if [ "$MODE" = agent ] && [ -z "${LLM_URL:-}" ] && ! curl -s --max-time 30 http://127.0.0.1:$port/health | grep -q ok; then
          echo "[w$wid] server $port not responding - stopping the worker" >> $OUT/worker_${JOB}_$wid.log; rm -rf "$OUT/$id/att$k.claim"; return; fi
        sleep 30
      done
      rm -rf "$OUT/$id/att$k.claim"
    done < $LIST
  done
}
WP=()
for w in $(seq 0 $((NW-1))); do worker $w & WP+=($!); sleep 5; done
for p in "${WP[@]}"; do wait $p; done
kill $GUARD 2>/dev/null
echo "=== $(date) DONE job=$JOB ==="

python3 $R/report.py $OUT
pkill -f llama-server 2>/dev/null; true
