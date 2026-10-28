#!/bin/bash
#SBATCH --job-name=relmeta
#SBATCH --account=your_project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=110G
#SBATCH --time=05:00:00
#SBATCH --output=/work/glm53/release/meta_%j.log
# Release files: clean metadata (general.name set, cluster paths quantize.imatrix.file/.dataset removed), then split
# into <=48 GB parts for Hugging Face (50 GB per-file limit). Writes release/<name>-0000k-of-0000n.gguf + <name>.DONE.
P=/work/glm53; B=$P/llama.cpp/build/bin
run() { singularity exec -B /work --env PYTHONPATH=$P/llama.cpp/gguf-py $P/vllm-rocm.sif bash -c "export LD_LIBRARY_PATH=/opt/rocm/lib:$B:\$LD_LIBRARY_PATH; $1"; }
for v in Q8_0:Q8_0 Q4KM:Q4_K_M Q3K:Q3_K_M; do
  src=$P/out/GLM-5.3-Flash-Coder-r152v5g-${v%%:*}.gguf; name=GLM-5.3-Flash-Coder-RCO-152E-${v##*:}
  [ -f /work/glm53/release/$name.DONE ] && { echo "done $name"; continue; }
  run "python3 $P/llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.py $src /work/glm53/release/tmp_$name.gguf --general-name GLM-5.3-Flash-Coder-RCO-152E --remove-metadata quantize.imatrix.file --remove-metadata quantize.imatrix.dataset --force" 2>&1 | tail -2
  run "$B/llama-gguf-split --split --split-max-size 48G /work/glm53/release/tmp_$name.gguf /work/glm53/release/$name" 2>&1 | tail -3
  if ls /work/glm53/release/$name-0*.gguf; then rm -f "/work/glm53/release/tmp_$name.gguf"; touch /work/glm53/release/$name.DONE; fi
done
