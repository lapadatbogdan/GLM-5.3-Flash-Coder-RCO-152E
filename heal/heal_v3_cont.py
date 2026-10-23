#!/usr/bin/env python3
"""heal_v3.py (untouched) continued from an existing heal: --init-delta DELTA loads that run's LoRA (delta_best.pt)
into the freshly built trainables before training, so step 0 IS the healed model (same function as the merged
checkpoint), then heal_v3 trains on as usual (new optimizer + LR schedule, same LoRA shape). Everything else, flags
included, is heal_v3.py's.   heal_v3_cont.py --init-delta heal_runs/r144v3_full/delta_best.pt <heal_v3 args>
On resume (OUT/ckpt.pt exists) heal_v3 loads its own checkpoint after this, so the init does not override progress."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import heal_v3 as H

argv = sys.argv[1:]
INIT = None
if "--init-delta" in argv:   # without it this is exactly heal_v3.py
    i = argv.index("--init-delta")
    INIT = argv[i + 1]
    del argv[i:i + 2]
_setup = H.setup_trainable


def setup_and_init(*a, **k):
    T = _setup(*a, **k)
    s = torch.load(INIT, map_location="cpu", weights_only=False)
    with torch.no_grad():
        H.load_trainable_state(T, s)
    n = len(T["lora"])
    missing = [x for x in T["lora"] if x not in s["lora"]]
    assert not missing, f"{len(missing)} LoRA targets not in {INIT}: {missing[:3]}"
    H.log.info(f"[cont] LoRA of {n} targets initialized from {INIT} (step {s.get('meta', {}).get('step')})")
    return T


if INIT:
    H.setup_trainable = setup_and_init
if __name__ == "__main__":
    H.main(argv)
