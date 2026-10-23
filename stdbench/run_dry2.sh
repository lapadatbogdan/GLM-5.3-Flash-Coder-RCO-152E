#!/bin/bash
# v2 (16 oct): per-slot KV (no -kvu; unified KV made the DSA indexer scan all slots, ~2x slower), CTXSLOT per conversation,
# SHARD i/n, MINTOK per-bench max_tokens floor. Exit 3 = pipeline parallelism dropped.
# stdbench job body (called from sbatch/std_<tag>.sbatch). Starts NSERV llama-servers, each on 8/NSERV
# GCDs (64 GB each) with NP slots and one unified KV pool of CTX tokens, optionally runs the throughput
# sweep on server 0, then runs run.py on the host python3 (3.6). One JSON line per (item, sample) is
# appended to results/$TAG.jsonl as soon as it is scored; resubmitting resumes where it stopped.
# Needs: TAG GGUF NSERV NP CTX. Optional: CONC (total concurrent conversations, or "auto" = sweep pick
# x NSERV), FAKE (right|wrong|mixed: CPU self-test with fake_server.py), SWEEP (levels, e.g. 4,8,16,32,48,64; empty = no sweep), BENCHES, SAMPLES, LIMIT.
S=/work/glm53/stdbench
P=/work/glm53
SIF_LLM=/work/tasks/vllm-rocm.sif
: ${TAG:?} ${GGUF:?} ${NSERV:?} ${NP:?} ${CTXSLOT:?}
CTX=$((CTXSLOT*NP))
CONC=${CONC:-auto}
BASEPORT=${BASEPORT:-8000}
BENCHES=${BENCHES:-ifeval,math500,aime25,lcb}
G=$((8 / NSERV))
[ -f "$GGUF" ] || { echo "MISSING GGUF $GGUF"; exit 1; }
for b in ${BENCHES//,/ }; do [ -f $S/benches/$b.py ] || { echo "MISSING bench $b"; exit 1; }; done
mkdir -p $S/logs $S/results
echo "=== $TAG: $GGUF  NSERV=$NSERV x $G GCD, NP=$NP slots/server, KV pool=$CTX tokens/server, CONC=$CONC, benches=$BENCHES samples=${SAMPLES:-default} limit=${LIMIT:-none} ==="
date
PORTS=""
for i in $(seq 0 $((NSERV-1))); do
  devs=$(seq -s, $((i*G)) $((i*G+G-1)))
  if [ -n "$FAKE" ]; then  # CPU self-test of this script: fake server in mode $FAKE instead of llama-server
    /usr/bin/python3 $S/fake_server.py $((BASEPORT+i)) $FAKE $BENCHES > $S/logs/serve-$TAG-$SLURM_JOB_ID-$i.log 2>&1 &
    FPIDS="$FPIDS $!"; PORTS="$PORTS${PORTS:+,}$((BASEPORT+i))"; continue
  fi
  singularity exec -B /work -B /work --env ROCR_VISIBLE_DEVICES=$devs $SIF_LLM bash -c "
    export LD_LIBRARY_PATH=${XLIB:+$XLIB:}/opt/rocm/lib:$P/llama.cpp/build/bin:\$LD_LIBRARY_PATH
    $P/llama.cpp/build/bin/llama-server -m $GGUF -ngl 99 -sm layer --load-mode none --host 127.0.0.1 --port $((BASEPORT+i)) \
      -c $((CTXSLOT*NP)) --parallel $NP -no-kvu --no-context-shift --jinja --metrics ${SERVER_EXTRA:-}" > $S/logs/serve-$TAG-$SLURM_JOB_ID-$i.log 2>&1 &
  PORTS="$PORTS${PORTS:+,}$((BASEPORT+i))"
done
for i in $(seq 0 $((NSERV-1))); do
  ok=no
  for t in $(seq 1 120); do
    r=$(curl -s --max-time 300 http://127.0.0.1:$((BASEPORT+i))/v1/chat/completions -H "Content-Type: application/json" \
        -d '{"model":"q","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":2048}' 2>/dev/null)
    echo "$r" | grep -q '"choices"' && { ok=yes; break; }
    grep -q "exiting due to model loading error" $S/logs/serve-$TAG-$SLURM_JOB_ID-$i.log 2>/dev/null && break
    sleep 20
  done
  if [ "$ok" = yes ] && grep -q "retrying without pipeline parallelism" $S/logs/serve-$TAG-$SLURM_JOB_ID-$i.log; then echo "!!! pipeline parallelism dropped (prefill ~4x slower) - ABORT, lower NP"; pkill -f llama-server; exit 3; fi
  [ "$ok" = yes ] || { echo "SERVER $i DEAD - ABORT"; tail -5 $S/logs/serve-$TAG-$SLURM_JOB_ID-$i.log; pkill -f llama-server; exit 1; }
  echo "server $i OK (port $((BASEPORT+i))): $(echo "$r" | /usr/bin/python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(repr((m.get("content") or "")[-80:]), "reasoning_chars=%d" % len(m.get("reasoning_content") or ""))' 2>&1)"
done
date
if [ -n "$SWEEP" ]; then
  /usr/bin/python3 $S/sweep.py --tag $TAG --port $BASEPORT --levels $SWEEP --tokens ${SWEEP_TOKENS:-512} | tee $S/logs/sweep-$TAG-$SLURM_JOB_ID.txt
  date
fi
if [ "$CONC" = auto ]; then
  if [ -f $S/results/sweep_$TAG.json ]; then
    per=$(/usr/bin/python3 -c "import json; print(json.load(open('$S/results/sweep_$TAG.json'))['chosen_conc_per_server'])")
  else
    per=$NP
  fi
  CONC=$((per * NSERV))
fi
[ $((CONC)) -gt $((NP * NSERV)) ] && CONC=$((NP * NSERV))
echo "driver concurrency: $CONC"
# first log line of the driver shows item counts per bench (check it right after launch)
export STDBENCH_LOOPSTOP=1
# configs: file with lines "name|sampling-json" (DRY_CFGFILE), default = base/dry/rep
CFGF=${DRY_CFGFILE:-$S/data/dry_cfgs_default.txt}
while IFS='|' read -r name js; do
  [ -z "$name" ] && continue
  STDBENCH_SAMPLING="$js" /usr/bin/python3 $S/run.py --tag ${TAG}_$name --ports $PORTS --conc $CONC --benches $BENCHES \
    --ids ${IDS:-$S/data/drytest_ids.txt} --samples ${SAMPLESARG:-ifeval=3} ${MINTOK:+--min-max-tokens $MINTOK} --http-timeout 172800 < /dev/null
done < $CFGF
grep -h -E "slot print_timing.*eval time" $S/logs/serve-$TAG-$SLURM_JOB_ID-*.log | tail -3
if [ -n "$FAKE" ]; then kill $FPIDS 2>/dev/null; else pkill -f llama-server 2>/dev/null; fi
date
echo "=== $TAG DONE ==="
