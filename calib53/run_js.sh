#!/bin/bash
# Agent session generation pass@K on MINED JS instances (NodeBB/mocha).
# Clean room: shared file system not mounted, /app masked, git history deleted, node_modules copied into the overlay.
# Tasks are sorted BY image, so node_modules is copied only once per image per worker.
W=/work/tasks; SIF_LLM=$W/vllm-rocm.sif; GGUF=$W/Qwen3.8-27B-Q8_0.gguf
module load cray-python/3.11.7 2>/dev/null
K=${K:-4}; TEMP=${TEMP:-0.8}; GEN_OUT=${GEN_OUT:-/work/glm53/calib53/traj_js}
# Optional LoRA adapter. EMPTY = exactly the run that produced the baseline; set = the same
# run with the adapter loaded. The only difference allowed between the two measurements.
LORA=${LORA:-}
LORA_ARG=""
[ -n "$LORA" ] && { [ -f "$LORA" ] || { echo "ADAPTER MISSING: $LORA"; exit 1; }; LORA_ARG="--lora $LORA"; echo "=== adapter: $LORA ==="; }
mkdir -p $GEN_OUT $W/out $W/emptydir

python3 - "$W" > $W/out/gen_js_tasks_${OVLPFX:-c53js}.txt <<'PY'
import json, glob, sys, hashlib, os
W = sys.argv[1]
rows = []
for d in sorted(glob.glob(W + "/mined_js/*/instance_*")):
    try: m = json.load(open(d + "/meta.json"))
    except Exception: continue
    img = m.get("img", "")
    if not img or not os.path.exists(img): continue
    key = hashlib.md5(os.path.basename(os.path.dirname(img)).encode()).hexdigest()[:10]
    rows.append((img, key, d))
rows.sort()                      # grouped by image -> node_modules copied once
for img, key, d in rows: print(img, key, d)
PY
shuf -n 28 --random-source=<(yes) $W/out/gen_js_tasks_${OVLPFX:-c53js}.txt > $W/out/gen_js_tasks_${OVLPFX:-c53js}.txt.s && mv $W/out/gen_js_tasks_${OVLPFX:-c53js}.txt.s $W/out/gen_js_tasks_${OVLPFX:-c53js}.txt
NT=$(wc -l < $W/out/gen_js_tasks_${OVLPFX:-c53js}.txt)
echo "=== $NT JS instances, K=$K, TEMP=$TEMP ==="
[ "$NT" -eq 0 ] && { echo "nothing to run"; exit 1; }

echo "=== starting 1 llama-server GLM-5.3-Flash Q8_0 on 8 GCDs ==="
singularity exec -B /work
 $SIF_LLM bash -c "
  export LD_LIBRARY_PATH=/opt/rocm/lib:/work/glm53/llama.cpp/build/bin:\$LD_LIBRARY_PATH
  /work/glm53/llama.cpp/build/bin/llama-server -m /work/glm53/gguf/full-q8_0.gguf -ngl 99 -sm layer --load-mode none --host 127.0.0.1 --port 8000 -c 524288 --parallel 8 --jinja" > /work/glm53/calib53/logs/serve-$SLURM_JOB_ID.log 2>&1 &
ok=no
for t in $(seq 1 90); do
  r=$(curl -s --max-time 120 http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
      -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" 2>/dev/null)
  echo "$r" | grep -q '"choices"' && { ok=yes; break; }
  sleep 20
done
[ "$ok" = yes ] || { echo "SERVER DEAD - ABORT"; pkill -f llama-server; exit 1; }
echo "server OK after $((t*20))s"
GOOD="0 1 2 3 4 5 6 7"
GOODARR=($GOOD); NW=${#GOODARR[@]}
mapfile -t TASKS < $W/out/gen_js_tasks_${OVLPFX:-c53js}.txt

worker() {
  local wid=$1 slot=$2 port=8000 inst=genjs_$1 prep=prepjs_$1 ovl=$W/work_js$1.img
  [ -f "$ovl" ] || singularity overlay create --size 40960 "$ovl" >/dev/null 2>&1
  local idx=$slot
  while [ $idx -lt ${#TASKS[@]} ]; do
    local img=$(echo "${TASKS[$idx]}" | awk '{print $1}')
    local key=$(echo "${TASKS[$idx]}" | awk '{print $2}')
    local d=$(echo "${TASKS[$idx]}" | awk '{print $3}')
    local iid=$(basename "${d%/}")
    local todo=0
    for k in $(seq 0 $((K-1))); do [ -f "$GEN_OUT/${iid}_att${k}.json" ] || todo=1; done
    if [ $todo -eq 1 ]; then
      singularity instance start --overlay "$ovl" -B /work "$img" $prep >/dev/null 2>&1
      pr=$(singularity exec instance://$prep bash $W/pro_gen_prep_js.sh "${d%/}" "$key" 2>&1 | tail -1)
      singularity instance stop $prep >/dev/null 2>&1
      echo "[w$wid] prep $iid: $pr"
      case "$pr" in PREP_OK*) ;; *) echo "[w$wid] SKIPPING $iid ($pr)"; idx=$((idx+NW)); continue;; esac
      for k in $(seq 0 $((K-1))); do
        [ -f "$GEN_OUT/${iid}_att${k}.json" ] && continue
        singularity instance start --overlay "$ovl" -B $W/emptydir:/app "$img" $inst >/dev/null 2>&1
        ATTEMPT=$k TASK=$iid TASK_DIR="${d%/}" INST=$inst IMGKEY=$key TEMP=$TEMP GEN_OUT=$GEN_OUT \
          VLLM_URL=http://127.0.0.1:$port/v1/chat/completions \
          timeout 7200 python3 /work/glm53/calib53/agent_js.py >> $W/out/genjs-w$wid.log 2>&1
        singularity instance stop $inst >/dev/null 2>&1
      done
    fi
    echo "[w$wid] done $iid ($((idx+1))/${#TASKS[@]})"
    idx=$((idx+NW))
  done
}
# Do NOT use a bare `wait`: it would also wait for the llama-servers started with &, which never exit
# (the job stays stuck for hours after the workers finish). Wait EXPLICITLY only for the worker PIDs.
slot=0; WPIDS=()
for wid in "${GOODARR[@]}"; do worker $wid $slot & WPIDS+=($!); slot=$((slot+1)); done
for p in "${WPIDS[@]}"; do wait "$p"; done
echo "=== JS GENERATION DONE ==="

pkill -f llama-server 2>/dev/null
