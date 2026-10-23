#!/bin/bash
#SBATCH --job-name=c53poly
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=/work/glm53/calib53/logs/gen_poly_%j.log
# Multi-language calibration: GLM-5.3-Flash (Q8_0, thinking on) edits real files from CommitPackFT
# (go/js/ts/rust/java/c++/sql/shell). Resumable: one JSON file per task.
P=/work/glm53
singularity exec -B /work $P/vllm-rocm.sif bash -c "
  export LD_LIBRARY_PATH=/opt/rocm/lib:$P/llama.cpp/build/bin:\$LD_LIBRARY_PATH
  $P/llama.cpp/build/bin/llama-server -m $P/gguf/full-q8_0.gguf -ngl 99 -sm layer --load-mode none --host 127.0.0.1 --port 8000 -c 524288 --parallel 8 --jinja" > $P/calib53/logs/serve-$SLURM_JOB_ID.log 2>&1 &
for t in $(seq 1 90); do
  curl -s --max-time 120 http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"q","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' | grep -q '"choices"' && break
  sleep 20
done
echo "server up after $((t*20))s"
module load cray-python/3.11.7 2>/dev/null
python3 $P/calib53/gen_poly.py --data $P/calib53/commitpackft/data --out $P/calib53/traj_poly --shard ${SHARD:-0/1}
pkill -f llama-server
