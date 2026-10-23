#!/bin/bash
# RCO k152 (general calib) mask selection: masks step400/500/600 of rco_runs/rcog_k152_v3 -> bool
# mask files + mask_list.json, then eval_masks_mn on the 256 mix_v3 held-out windows (same as gen_k152 rows in
# rco_runs/general_plus/kl), 4 shards. Refuses if a mask is missing or the list exists.
set -euo pipefail
P=/work/glm53; D=$P/rco_runs/rcog_k152_eval
[ -f $D/mask_list.json ] && { echo "mask_list exists"; exit 1; }
singularity exec -B /work $P/vllm-rocm.sif python3 - <<PY
import torch, json
D="$D"; src="$P/rco_runs/rcog_k152_v3/masks"
L={}
for s in (400,500,600):
    d=torch.load(f"{src}/step{s}.pt",weights_only=False); m=d["mask"]
    assert m.dtype==torch.bool and m.shape==(42,288) and ((~m).sum(1)==152).all()
    torch.save(m.clone(), f"{D}/rcog_step{s}.pt"); L[f"rcog152_step{s}"]=f"{D}/rcog_step{s}.pt"
json.dump(L, open(f"{D}/mask_list.json","w")); print(L)
PY
cd $P/rco
for i in 0 1 2 3; do CALIB=calib53/mix_v3 MASKLIST=$D/mask_list.json OUT=$D/kl SHARD=$i NSHARDS=4 sbatch --parsable --job-name=rcogev$i eval_masks_mn.sbatch; done
