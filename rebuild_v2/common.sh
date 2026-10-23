# Shared paths and guards for the v2 rebuild (sourced by every step). TAG = new variant name, e.g. r96v2.
P=/work/glm53
B=$P/llama.cpp/build/bin            # same build as r144/r120 (commit 79e2e74 + imatrix.cpp fix, built 10 oct)
R=$P/rebuild_v2
TAG=${TAG:?TAG missing}
PR=$P/pruned/$TAG                   # pruned FP8 HF checkpoint
Q8=$P/gguf/pruned-$TAG-q8_0.gguf    # Q8_0 base (FP8 -> Q8_0 directly, --fp8-as-q8)
IM=$P/gguf/pruned-$TAG.imatrix
OUTP=$P/out/GLM-5.3-Flash-Coder-$TAG
M=$R/state/$TAG                     # step markers (resume)
mkdir -p $R/state $R/logs

# never touch the published / old builds
case $TAG in r144|r120|r96|f144|f96|hr144b|reap50|full) echo "TAG $TAG is an existing build, refusing"; exit 2;; esac
[[ $TAG =~ ^[rf][0-9]+v[0-9]+[a-z]*$ ]] || { echo "TAG $TAG: expected like r96v2"; exit 2; }

run() { singularity exec -B /work $P/vllm-rocm.sif bash -c "export LD_LIBRARY_PATH=/opt/rocm/lib:$B:\$LD_LIBRARY_PATH; $1"; }
stage() { echo "=== $(date '+%F %T') $TAG $1"; }
is_gguf() { [ -s "$1" ] && [ "$(head -c 4 "$1")" = GGUF ]; }
