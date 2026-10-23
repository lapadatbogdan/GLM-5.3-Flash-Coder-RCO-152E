#!/usr/bin/env python3
"""Calibration text for llama-imatrix from the calib_search.pt windows of this directory (NOT from the old calib_text.txt,
which had only code sessions). Each 1024-token window is decoded with its special tokens as text (<|user|>, <think> ...);
llama-imatrix must be run with --parse-special to turn them back into special tokens, as at inference.
The held-out windows (calib_heldout.pt) are NOT included. Order: deterministic shuffle (seed 0), so any --chunks prefix
keeps the domain shares.
    build_imatrix_text.py   -> calib_text_search.txt + calib_text_search.json (statistics, round-trip check)"""
import json, os, random, collections
import torch
from transformers import AutoTokenizer
D = os.path.dirname(os.path.abspath(__file__))
S = torch.load(os.path.join(D, "calib_search.pt"), weights_only=False)
tok = AutoTokenizer.from_pretrained("/work/glm53/pruned/r144")
order = list(range(S["ids"].size(0)))
random.Random(0).shuffle(order)
texts, ok, n_tok = [], 0, 0
for k, i in enumerate(order):
    ids = S["ids"][i].tolist()
    t = tok.decode(ids, skip_special_tokens=False)
    texts.append(t)
    if k < 64:  # round trip on a sample (tokenization with special tokens)

        ok += int(tok(t, add_special_tokens=False)["input_ids"] == ids)
    n_tok += len(ids)
out = os.path.join(D, "calib_text_search.txt")
open(out + ".tmp", "w").write("".join(texts))
os.replace(out + ".tmp", out)
st = {"windows": len(order), "tokens": n_tok, "roundtrip_exact_of_64": ok,
      "buckets": collections.Counter(S["meta"][i]["bucket"] for i in order),
      "bytes": os.path.getsize(out)}
json.dump(st, open(os.path.join(D, "calib_text_search.json"), "w"), indent=1)
print(st)
