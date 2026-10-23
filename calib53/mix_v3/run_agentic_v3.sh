#!/bin/bash
# Agent session generation pass@K on MINED Python instances (ansible/openlibrary/qutebrowser).
# Same architecture as for Go: work_w<id>.img (writable overlay, 0 inodes on the shared file system)
# + CLEAN ROOM in the solve phase (shared file system not mounted, /app masked, git history deleted).
W=/work/tasks; SIF_LLM=$W/vllm-rocm.sif; GGUF=$W/Qwen3.8-27B-Q8_0.gguf
module load cray-python/3.11.7 2>/dev/null
K=${K:-4}; TEMP=${TEMP:-0.8}; GEN_OUT=${GEN_OUT:-/work/glm53/calib53/traj_py}
# Optional LoRA adapter. EMPTY = exactly the run that produced the baseline; set = the same
# run with the adapter loaded. The only difference allowed between the two measurements.
AGENT=${AGENT:-pro_gen_py.py}
# The overlay prefix MUST differ between concurrent jobs. Two runs with the same
# prefix overwrite each other and instances fail with PREP_COPY_FAIL (lost ~3 hours, 9 Sep).
OVLPFX=${OVLPFX:-c53py}
NTASK=${NTASK:-0}   # 0 = all
LORA=${LORA:-}
LORA_ARG=""
[ -n "$LORA" ] && { [ -f "$LORA" ] || { echo "ADAPTER MISSING: $LORA"; exit 1; }; LORA_ARG="--lora $LORA"; echo "=== adapter: $LORA ==="; }
mkdir -p $GEN_OUT $W/out $W/emptydir

# --- task list: instance + the image of its repository ---
# mix_v3: the task list is fixed in calib53/mix_v3/agentic_tasks.txt (80 Python instances, none in calib53/eval/*.txt,
# none already in traj_py/traj_js); this job takes the lines with (line number % NSHARDS == SHARD).
awk -v s=${SHARD:-0} -v n=${NSHARDS:-1} "(NR-1)%n==s" /work/glm53/calib53/mix_v3/agentic_tasks.txt > $W/out/gen_py_tasks_${OVLPFX:-c53py}.txt
NT=$(wc -l < $W/out/gen_py_tasks_${OVLPFX:-c53py}.txt)
echo "=== $NT Python instances, K=$K, TEMP=$TEMP ==="
awk '{print $1}' $W/out/gen_py_tasks_${OVLPFX:-c53py}.txt | sort | uniq -c
[ "$NT" -eq 0 ] && { echo "nothing to run"; exit 1; }

if [ "${GOLD_ONLY:-0}" = "1" ]; then
  # Gold patch validation calls no model: skip the servers and run on CPU.
  echo "=== GOLD_ONLY MODE: no servers, only apply gold.patch and run the tests ==="
  # Workers are launched from the list of servers that answered. Without servers the list is empty
  # and nobody starts - so fill it by hand with all 8.
  GOOD="0 1 2 3 4 5 6 7"
else
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
fi
# The worker list must be built OUTSIDE the branch: in GOLD_ONLY mode the server branch
# does not run, so GOODARR stayed empty and no worker started.
GOODARR=($GOOD); NW=${#GOODARR[@]}
echo "=== workers: $NW ==="

mapfile -t TASKS < $W/out/gen_py_tasks_${OVLPFX:-c53py}.txt

worker() {
  local wid=$1 slot=$2 port=8000 inst=genpy_$1 prep=prepy_$1 ovl=$W/work_${OVLPFX:-c53py}$1.img
  [ -f "$ovl" ] || singularity overlay create --size 20480 "$ovl" >/dev/null 2>&1
  local idx=$slot
  while [ $idx -lt ${#TASKS[@]} ]; do
    local d=$(echo "${TASKS[$idx]}" | awk '{print $2}')
    local sif=$(echo "${TASKS[$idx]}" | awk '{print $3}')
    local iid=$(basename "${d%/}")
    local todo=0
    for k in $(seq 0 $((K-1))); do [ -f "$GEN_OUT/${iid}_att${k}.json" ] || todo=1; done
    if [ $todo -eq 1 ]; then
      # PHASE 1: preparation (has /app + the shared file system), deletes the git history
      singularity instance start --overlay "$ovl" -B /work "$sif" $prep >/dev/null 2>&1
      pr=$(SINGULARITYENV_GOLD_ONLY=${GOLD_ONLY:-0} singularity exec instance://$prep bash $W/pro_gen_prep.sh "${d%/}" 2>&1 | tail -1)
      singularity instance stop $prep >/dev/null 2>&1
      echo "[w$wid] prep $iid: $pr"
      case "$pr" in PREP_OK*) ;; *) echo "[w$wid] SKIPPING $iid ($pr)"; idx=$((idx+NW)); continue;; esac
      # PHASE 2: clean room (no shared file system, /app masked)
      for k in $(seq 0 $((K-1))); do
        [ -f "$GEN_OUT/${iid}_att${k}.json" ] && continue
        singularity instance start --overlay "$ovl" -B $W/emptydir:/app "$sif" $inst >/dev/null 2>&1
        ATTEMPT=$k TASK=$iid TASK_DIR="${d%/}" INST=$inst TEMP=$TEMP GEN_OUT=$GEN_OUT \
          VLLM_URL=http://127.0.0.1:$port/v1/chat/completions \
          timeout 7200 python3 /work/glm53/calib53/agent_py.py >> $W/out/genpy-w$wid.log 2>&1
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
echo "=== PYTHON GENERATION DONE ==="
python3 - "$GEN_OUT" <<'PY'
import json, glob, sys
from collections import Counter, defaultdict
r=[json.load(open(f)) for f in glob.glob(sys.argv[1]+"/*_att*.json")]
ok=[x for x in r if x.get("resolved")]
per=defaultdict(list)
for x in r: per[x["task"]].append(bool(x.get("resolved")))
h=Counter(sum(v) for v in per.values())
print("trajectories=%d resolved=%d (%.1f%%) instances=%d"%(len(r),len(ok),100*len(ok)/max(1,len(r)),len(per)))
print("pass@K histogram:", dict(sorted(h.items())))
PY
pkill -f llama-server 2>/dev/null
