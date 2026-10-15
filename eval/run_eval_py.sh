#!/bin/bash
# RFT generation pass@K on the MINED Python instances (ansible/openlibrary/qutebrowser).
# Same architecture as for Go: work_w<id>.img (writable overlay, 0 scratch inodes)
# + CLEAN ROOM in the solve phase (no /work, /app masked, git history deleted).
W=/work/seclab; SIF_LLM=$W/vllm-rocm.sif; GGUF=$W/Qwen3.8-27B-Q8_0.gguf
true  # needs python3 >= 3.10
# AGENT=agent_py.py (default, all reported results) or agent_py_v2.py (anti-loop guard).
AGENT=${AGENT:-agent_py.py}
K=${K:-4}; TEMP=${TEMP:-0.8}; GEN_OUT=${GEN_OUT:-/work/glm53/calib53/traj_py}
# Optional LoRA adapter. EMPTY = exactly the run that produced the baseline; set = the same
# run with the adapter loaded. The only difference allowed between the two measurements.
# The overlay prefix MUST differ between concurrent jobs. Two runs with the same
# prefix clobber each other and instances fail with PREP_COPY_FAIL. 
OVLPFX=${OVLPFX:-c53ev}
NTASK=${NTASK:-0}   # 0 = all
LORA=${LORA:-}
LORA_ARG=""
[ -n "$LORA" ] && { [ -f "$LORA" ] || { echo "ADAPTER NOT FOUND: $LORA"; exit 1; }; LORA_ARG="--lora $LORA"; echo "=== adapter: $LORA ==="; }
mkdir -p $GEN_OUT $W/out $W/emptydir

# --- task list: instance + the image of its repo ---
# TASKLIST=file: a helper on another node takes a given list (the second half, easy ones first) and writes to the same
# GEN_OUT; it skips attempts already saved, so the two nodes meet in the middle.
if [ -n "${TASKLIST:-}" ]; then cp "$TASKLIST" $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt; else
: > $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt
for repo in ansible openlibrary qutebrowser; do
  sif=$(ls -d $W/pro_tasks/*${repo}*/img.sif 2>/dev/null | head -1)
  [ -z "$sif" ] && continue
  for d in $W/mined_py/$repo/instance_*/; do
    [ -f "${d}f2p.txt" ] && echo "$repo $d $sif" >> $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt
  done
done
if [ "$NTASK" -gt 0 ]; then head -$NTASK $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt > $W/out/gen_py_tasks_sub.txt; mv $W/out/gen_py_tasks_sub.txt $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt; fi
# evaluation: only instances NOT used for calibration, 30 fixed (the same list for the full and the pruned model)
grep -v -F -f /work/glm53/calib53/eval/calib_tasks_py.txt $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt > $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt.f && mv $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt.f $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt
python3 - $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt <<"PYPICK"
import random, sys, collections
rows=[l for l in open(sys.argv[1]) if l.strip()]
by=collections.defaultdict(list)
[by[r.split()[0]].append(r) for r in rows]
rnd=random.Random(7)
out=[]
for repo in sorted(by):
    rnd.shuffle(by[repo]); out+=by[repo][:10]
open(sys.argv[1],"w").writelines(out)
PYPICK
cp $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt /work/glm53/calib53/eval/eval_tasks_py.txt
fi
NT=$(wc -l < $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt)
echo "=== $NT Python instances, K=$K, TEMP=$TEMP ==="
awk '{print $1}' $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt | sort | uniq -c
[ "$NT" -eq 0 ] && { echo "nothing to run"; exit 1; }

if [ "${GOLD_ONLY:-0}" = "1" ]; then
  # Gold patch validation calls no model: skip the servers and run on CPU.
  echo "=== GOLD_ONLY MODE: no servers, only applying gold.patch and running the tests ==="
  # Workers are launched from the list of servers that answered. Without servers the list is empty
  # and nobody starts - so fill it in manually with all 8.
  GOOD="0 1 2 3 4 5 6 7"
else
# NSERV servers per node, each on 8/NSERV GPUs (64 GB/GPU), SLOTS conversations per server.
# Full model (341 GB): NSERV=1 SLOTS=8. Pruned Q8 (~150-175 GB): NSERV=2 SLOTS=4. 2 bits (~45 GB): NSERV=8 SLOTS=2.
# (12 Oct: a single server on 8 GPUs for models that fit on 1-4 GPUs = ~4 tok/s and GPUs at 0%.)
NSERV=${NSERV:-1}; SLOTS=${SLOTS:-$((8 / NSERV))}; G=$((8 / NSERV)); CTXSLOT=${CTXSLOT:-131072}
echo "=== starting $NSERV llama-server x $G GPU, $SLOTS conversations each ==="
for i in $(seq 0 $((NSERV-1))); do
  devs=$(seq -s, $((i*G)) $((i*G+G-1)))
  singularity exec -B /work -B /work --env ROCR_VISIBLE_DEVICES=$devs $SIF_LLM bash -c "
    export LD_LIBRARY_PATH=/opt/rocm/lib:/work/glm53/llama.cpp/build/bin:\$LD_LIBRARY_PATH
    /work/glm53/llama.cpp/build/bin/llama-server -m ${EVAL_GGUF:?} -ngl 99 -sm layer --load-mode none --host 127.0.0.1 --port $((8000+i)) -c $((CTXSLOT*SLOTS)) --parallel $SLOTS --jinja" > /work/glm53/calib53/logs/serve-$SLURM_JOB_ID-$i.log 2>&1 &
done
for i in $(seq 0 $((NSERV-1))); do
  ok=no
  for t in $(seq 1 90); do
    r=$(curl -s --max-time 120 http://127.0.0.1:$((8000+i))/v1/chat/completions -H "Content-Type: application/json" \
        -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" 2>/dev/null)
    echo "$r" | grep -q '"choices"' && { ok=yes; break; }
    sleep 20
  done
  [ "$ok" = yes ] || { echo "SERVER $i DEAD - ABORT"; tail -5 /work/glm53/calib53/logs/serve-$SLURM_JOB_ID-$i.log; pkill -f llama-server; exit 1; }
  echo "server $i OK (port $((8000+i)))"
done
GOOD=$(seq -s " " 0 $((NSERV*SLOTS-1)))
GOODARR=($GOOD); NW=${#GOODARR[@]}
fi
# Building the worker list must be OUTSIDE the branch: in GOLD_ONLY mode the server
# branch does not run, so GOODARR stayed empty and no worker started.
GOODARR=($GOOD); NW=${#GOODARR[@]}
echo "=== workers: $NW ==="

mapfile -t TASKS < $W/out/gen_py_tasks_${OVLPFX:-c53ev}.txt

worker() {
  local wid=$1 slot=$2 port=$((8000 + $1 % ${NSERV:-1})) inst=genpy_$1 prep=prepy_$1 ovl=$W/work_${OVLPFX:-c53ev}$1.img
  [ -f "$ovl" ] || singularity overlay create --size 20480 "$ovl" >/dev/null 2>&1
  local idx=$slot
  while [ $idx -lt ${#TASKS[@]} ]; do
    local d=$(echo "${TASKS[$idx]}" | awk '{print $2}')
    local sif=$(echo "${TASKS[$idx]}" | awk '{print $3}')
    local iid=$(basename "${d%/}")
    local todo=0
    for k in $(seq 0 $((K-1))); do [ -f "$GEN_OUT/${iid}_att${k}.json" ] || todo=1; done
    if [ $todo -eq 1 ]; then
      # PHASE 1: prep (has /app + /work), deletes the git history
      # prep sometimes fails at random (PREP_REVERT_FAIL on valid tasks, 12 Oct: ansible-004c skipped at r144):
      # up to 3 tries, only then skip the task
      for ptry in 1 2 3; do
        singularity instance start --overlay "$ovl" -B /work "$sif" $prep >/dev/null 2>&1
        pr=$(SINGULARITYENV_GOLD_ONLY=${GOLD_ONLY:-0} singularity exec instance://$prep bash $W/pro_gen_prep.sh "${d%/}" 2>&1 | tail -1)
        singularity instance stop $prep >/dev/null 2>&1
        case "$pr" in PREP_OK*) break;; esac
        echo "[w$wid] prep $iid try $ptry: $pr - retrying"; sleep 10
      done
      echo "[w$wid] prep $iid: $pr"
      case "$pr" in PREP_OK*) ;; *) echo "[w$wid] SKIPPING $iid ($pr)"; idx=$((idx+NW)); continue;; esac
      # PHASE 2: clean room (no /work, /app masked)
      for k in $(seq 0 $((K-1))); do
        [ -f "$GEN_OUT/${iid}_att${k}.json" ] && continue
        singularity instance start --overlay "$ovl" -B $W/emptydir:/app "$sif" $inst >/dev/null 2>&1
        ATTEMPT=$k TASK=$iid TASK_DIR="${d%/}" INST=$inst TEMP=$TEMP GEN_OUT=$GEN_OUT \
          VLLM_URL=http://127.0.0.1:$port/v1/chat/completions \
          timeout ${ATT_TIMEOUT:-43200} python3 /work/glm53/calib53/$AGENT >> $W/out/genpy-w$wid.log 2>&1
        singularity instance stop $inst >/dev/null 2>&1
      done
    fi
    echo "[w$wid] done $iid ($((idx+1))/${#TASKS[@]})"
    idx=$((idx+NW))
  done
}
# Do NOT use a bare `wait`: it would also wait for the 8 llama-servers started with &, which never exit
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
