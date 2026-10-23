"""Physically remove pruned MoE experts from the GLM-5.3-Flash FP8 HF checkpoint.

Raw safetensors I/O: every kept tensor is copied as a byte range (FP8 stays FP8 with its
scale_inv, bit-exact), router weight / e_score_correction_bias are sliced row-wise, kept
experts are renumbered 0..K-1 in original order. Streams one shard at a time, writes each
output shard to *.tmp then renames; finished shards are skipped on rerun.

  build_pruned.py --src DIR --mask mask.pt --out DIR [--mtp drop|usage --mtp-scores f.pt] [--pad | --per-layer [--mtp-k K]] [--dry-run]

mask.pt: bool (n_sparse_layers, n_experts), True = pruned, row k = k-th sparse decoder layer.

--pad (non-uniform masks): GGUF needs one expert count, so every layer gets K = max kept count.
Layer L keeps its k_L real experts as 0..k_L-1, then K - k_L dead experts: zero FP8 weights,
weight_scale_inv 1.0, router row 0 and e_score_correction_bias -1e4. Selection is top-k over
sigmoid(gate @ h) + bias, so a dead slot scores ~ -1e4 + 0.5 and never beats a real expert while
k_L >= num_experts_per_tok; routing weights use the unbiased scores of the chosen experts only.

--per-layer (non-uniform masks, no padding): layer L keeps exactly its k_L real experts as 0..k_L-1,
router rows / bias sliced to k_L. Needs a per-layer aware runtime (llama.cpp with the
glm5_next per-layer expert patch, glm5_next_rco.load_glm_fp8). config text_config gets
n_routed_experts = max k_L and n_routed_experts_per_layer: one int per decoder layer
(num_hidden_layers, then the kept MTP layers), 0 on dense layers. The MTP layer keeps --mtp-k
experts by score (default max k_L). pruned_experts.json repeats the list.
"""
import argparse, hashlib, json, os, re, shutil, struct, sys

LAYER_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")
ROUTER_SUFFIXES = (".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")
EXPERT_COUNT_KEYS = ("n_routed_experts", "num_local_experts", "num_experts")
CHUNK = 64 << 20
DEAD_BIAS = -1e4
SCALAR_FMT = {"F64": "<d", "F32": "<f", "F16": "<e"}


def scalar_bytes(dtype, v):
    """One scalar in a safetensors dtype (BF16 = high half of the F32 bits)."""
    if dtype == "BF16":
        return struct.pack("<f", v)[2:]
    assert dtype in SCALAR_FMT, f"no scalar encoding for {dtype}"
    return struct.pack(SCALAR_FMT[dtype], v)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def load_tensor_file(path):
    import torch
    t = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(t, dict):  # accept {"mask": t} / {"scores": t}
        assert len(t) == 1, f"{path}: expected one tensor, got keys {list(t)}"
        t = next(iter(t.values()))
    return t


def text_cfg(cfg):
    return cfg.get("text_config", cfg)


def layer_plan(cfg, mask_path, mtp, scores_path, pad=False, per_layer=False, mtp_k=None):
    """Return (n_experts, K, {decoder_layer: kept original ids}, set of dropped layers).

    K is the output expert count per layer; with pad it is the max kept count and shorter
    layers are filled up to K with dead experts by plan_shards. With per_layer K is the max
    kept count, each layer keeps len(kept[L]) experts and nothing is padded."""
    tc = text_cfg(cfg)
    n_exp = tc["n_routed_experts"]
    nhl = tc["num_hidden_layers"]
    types = tc.get("mlp_layer_types")
    if types is not None:
        sparse = [i for i, t in enumerate(types) if t == "sparse"]
    else:
        sparse = list(range(tc.get("first_k_dense_replace", 0), nhl))
    n_mtp = tc.get("num_nextn_predict_layers", 0)
    mtp_layers = list(range(nhl, nhl + n_mtp))

    mask = load_tensor_file(mask_path).bool()
    assert tuple(mask.shape) == (len(sparse), n_exp), f"mask shape {tuple(mask.shape)} != ({len(sparse)}, {n_exp})"
    n_kept = (~mask).sum(1).tolist()
    assert not (pad and per_layer), "--pad and --per-layer are exclusive"
    if per_layer:
        K = max(n_kept)
        low = {sparse[k]: n for k, n in enumerate(n_kept) if n < tc["num_experts_per_tok"]}
        assert not low, f"layers keep fewer than num_experts_per_tok: {low}"
        ng = tc.get("n_group", 1)
        assert len(set(n_kept)) == 1 or ng == 1, "per-layer counts need n_group == 1"
        assert all(n % ng == 0 for n in n_kept), f"kept counts not divisible by n_group={ng}"
    elif pad:
        K = max(n_kept)
        low = {sparse[k]: n for k, n in enumerate(n_kept) if n < tc["num_experts_per_tok"]}
        assert not low, f"layers keep fewer than num_experts_per_tok, a dead slot could be routed: {low}"
        # group-limited routing would rank groups that hold dead slots; only plain top-k is safe
        assert len(set(n_kept)) == 1 or tc.get("n_group", 1) == 1, "padding needs n_group == 1"
    else:
        assert len(set(n_kept)) == 1, f"non-uniform kept count per layer: {sorted(set(n_kept))} (see --pad)"
        K = n_kept[0]
    assert K >= tc["num_experts_per_tok"], f"K={K} < num_experts_per_tok"
    assert K % tc.get("n_group", 1) == 0, f"K={K} not divisible by n_group"

    kept = {L: [e for e in range(n_exp) if not mask[k, e]] for k, L in enumerate(sparse)}
    dropped = set()
    if mtp == "drop":
        dropped = set(mtp_layers)
    else:
        s = load_tensor_file(scores_path).float().flatten()
        assert s.numel() == n_exp, f"scores numel {s.numel()} != {n_exp}"
        # stable top-K (ties -> lower id), then original order
        order = sorted(range(n_exp), key=lambda e: (-float(s[e]), e))
        k_mtp = mtp_k if (per_layer and mtp_k) else K
        assert tc["num_experts_per_tok"] <= k_mtp <= n_exp, f"MTP keeps {k_mtp} experts"
        for L in mtp_layers:
            kept[L] = sorted(order[:k_mtp])
    return n_exp, K, kept, dropped


def plan_shards(src, n_exp, kept, dropped, K=None, per_layer=False):
    """Header-only plan. Returns {shard: [op]}, op = (dst, src, dtype, shape, abs_off, src_nbytes, nbytes, rows).

    rows: None = byte-range copy; (row ids, bytes per row, pad bytes) = router row slice plus
    dead rows; ("fill", unit) = synthetic dead expert tensor, unit repeated to nbytes (src None)."""
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    remap = {L: {e: i for i, e in enumerate(ids)} for L, ids in kept.items()}
    K = K or max(len(ids) for ids in kept.values())
    n_dead = {L: 0 if per_layer else K - len(ids) for L, ids in kept.items()}
    tmpl, router_shard = {}, {}  # padded layer -> {(prefix, suffix): (dtype, shape, nbytes)}, gate shard
    plan, seen_exp = {}, {}
    for sh in shards:
        hdr, base = read_header(os.path.join(src, sh))
        ops = []
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            assert idx["weight_map"].get(name) == sh, f"{name} in {sh} but index says {idx['weight_map'].get(name)}"
            b0, b1 = info["data_offsets"]
            dtype, shape = info["dtype"], list(info["shape"])
            dst, rows = name, None
            m = LAYER_RE.match(name)
            L = int(m.group(1)) if m else None
            if L in dropped:
                continue
            if L in remap:
                me = EXPERT_RE.search(name)
                if me:
                    e = int(me.group(1))
                    seen_exp.setdefault(L, set()).add(e)
                    if n_dead[L]:
                        tmpl.setdefault(L, {})[(name[:me.start()], name[me.end():])] = (dtype, shape, b1 - b0)
                    if e not in remap[L]:
                        continue
                    dst = name[:me.start()] + f".mlp.experts.{remap[L][e]}." + name[me.end():]
                elif name.endswith(ROUTER_SUFFIXES):
                    assert shape[0] == n_exp, f"{name}: dim0 {shape[0]} != {n_exp}"
                    assert (b1 - b0) % n_exp == 0
                    rb = (b1 - b0) // n_exp
                    if name.endswith(ROUTER_SUFFIXES[0]):
                        pad = b"\0" * (rb * n_dead[L])  # dead gate rows: logit 0
                        router_shard[L] = sh
                    elif n_dead[L]:
                        unit = scalar_bytes(dtype, DEAD_BIAS)
                        assert len(unit) == rb, f"{name}: bias row is {rb} bytes, expected one {dtype}"
                        pad = unit * n_dead[L]
                    else:
                        pad = b""
                    rows = (kept[L], rb, pad)  # (row ids, bytes per row, dead rows)
                    shape = [len(kept[L]) + n_dead[L]] + shape[1:]  # = K unless per_layer
            nbytes = (b1 - b0) if rows is None else rows[1] * len(rows[0]) + len(rows[2])
            ops.append((dst, name, dtype, shape, base + b0, b1 - b0, nbytes, rows))
        plan[sh] = ops
    # every sparse/kept layer must have had all n_exp experts in the source
    for L in kept:
        assert seen_exp.get(L) == set(range(n_exp)), f"layer {L}: source experts incomplete"
    # dead experts K-n_dead..K-1, same shapes/dtypes as the real ones, next to the layer's router
    for L in sorted(n_dead):
        if not n_dead[L]:
            continue
        assert L in router_shard, f"layer {L}: no router gate to pad"
        for e in range(len(kept[L]), K):
            for (pre, suf), (dtype, shape, nb) in sorted(tmpl[L].items()):
                unit = scalar_bytes(dtype, 1.0) if suf.endswith("scale_inv") else b"\0"
                assert nb % len(unit) == 0
                plan[router_shard[L]].append((f"{pre}.mlp.experts.{e}.{suf}", None, dtype, shape, None, 0, nb, ("fill", unit)))
    return idx, plan


def out_header(ops, metadata):
    hdr, off = {}, 0
    if metadata is not None:
        hdr["__metadata__"] = metadata
    for dst, _, dtype, shape, _, _, nbytes, _ in ops:
        hdr[dst] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nbytes]}
        off += nbytes
    raw = json.dumps(hdr, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # 8-byte alignment of the data section
    return raw, off


def write_shard(src_path, dst_path, ops, metadata):
    raw, data_len = out_header(ops, metadata)
    tmp = dst_path + ".tmp"
    with open(src_path, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(raw)))
        fo.write(raw)
        for _, _, _, _, off, src_nb, nbytes, rows in ops:
            if rows is not None and rows[0] == "fill":
                unit = rows[1]
                block = unit * max(1, CHUNK // len(unit))
                left = nbytes
                while left:
                    n = min(len(block), left)
                    fo.write(block[:n])
                    left -= n
            elif rows is None:
                fi.seek(off)
                left = src_nb
                while left:
                    b = fi.read(min(CHUNK, left))
                    assert b, "short read"
                    fo.write(b)
                    left -= len(b)
            else:
                ids, rb, pad = rows
                fi.seek(off)
                buf = fi.read(src_nb)  # router tensors are small (288 x 4096 bf16)
                for r in ids:
                    fo.write(buf[r * rb:(r + 1) * rb])
                fo.write(pad)
        fo.flush()
        os.fsync(fo.fileno())
    assert os.path.getsize(tmp) == 8 + len(raw) + data_len, f"{tmp}: size mismatch"
    os.replace(tmp, dst_path)
    return 8 + len(raw) + data_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mtp", choices=["drop", "usage"], default="drop")
    ap.add_argument("--mtp-scores")
    ap.add_argument("--pad", action="store_true", help="allow non-uniform masks: pad layers to max kept with dead experts")
    ap.add_argument("--per-layer", action="store_true",
                    help="allow non-uniform masks: each layer keeps its own expert count, no padding")
    ap.add_argument("--mtp-k", type=int, help="--per-layer --mtp usage: experts kept in the MTP layer (default max)")
    ap.add_argument("--dry-run", action="store_true", help="plan from headers only, write nothing")
    a = ap.parse_args()
    if a.mtp == "usage" and not a.mtp_scores:
        ap.error("--mtp usage needs --mtp-scores")
    if a.pad and a.per_layer:
        ap.error("--pad and --per-layer are exclusive")
    if a.mtp_k is not None and not (a.per_layer and a.mtp == "usage"):
        ap.error("--mtp-k needs --per-layer --mtp usage")

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    n_exp, K, kept, dropped = layer_plan(cfg, a.mask, a.mtp, a.mtp_scores, a.pad, a.per_layer, a.mtp_k)
    idx, plan = plan_shards(a.src, n_exp, kept, dropped, K, a.per_layer)
    n_dead = {L: 0 if a.per_layer else K - len(ids) for L, ids in kept.items()}

    n_src = len(idx["weight_map"])
    n_out = sum(len(ops) for ops in plan.values())
    empty = [sh for sh, ops in plan.items() if not ops]
    total = sum(op[6] for ops in plan.values() for op in ops)
    print(f"experts {n_exp} -> {K}, pruned layers {len(kept)}, "
          f"mtp {a.mtp} (dropped layers {sorted(dropped)}), tensors {n_src} -> {n_out}, "
          f"bytes {idx['metadata']['total_size']} -> {total}, empty shards {empty}", flush=True)
    if a.pad:
        real = [len(ids) for ids in kept.values()]
        dead_b = sum(op[6] for ops in plan.values() for op in ops if op[1] is None)
        dead_b += sum(len(op[7][2]) for ops in plan.values() for op in ops if op[7] and op[7][0] != "fill")
        print(f"pad: K {K}, real experts {sum(real)} (per layer min {min(real)} max {max(real)}), "
              f"dead slots {sum(n_dead.values())} in {sum(1 for n in n_dead.values() if n)} layers, "
              f"padding overhead {dead_b / 1e9:.2f} GB", flush=True)
    if a.per_layer:
        real = [len(ids) for ids in kept.values()]
        print(f"per-layer: real experts {sum(real)} (per layer min {min(real)} max {max(real)}), "
              f"counts {[len(kept[L]) for L in sorted(kept)]}", flush=True)
    if a.dry_run:
        return

    os.makedirs(a.out, exist_ok=True)
    manifest = {"src": os.path.abspath(a.src), "mask_sha256": sha256(a.mask), "mtp": a.mtp,
                "mtp_scores_sha256": sha256(a.mtp_scores) if a.mtp_scores else None, "K": K}
    if a.pad:  # only when set, so uniform manifests stay as before
        manifest["pad"] = True
    if a.per_layer:
        manifest["per_layer"] = True
        if a.mtp_k is not None:
            manifest["mtp_k"] = a.mtp_k
    mpath = os.path.join(a.out, "prune_manifest.json")
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        if old != manifest:
            sys.exit(f"refusing to resume: {mpath} differs from this run ({old} vs {manifest}); use a fresh --out")
    else:
        json.dump(manifest, open(mpath, "w"), indent=1)

    weight_map = {}
    for i, (sh, ops) in enumerate(plan.items()):
        if not ops:
            print(f"[{i + 1}/{len(plan)}] {sh}: empty after pruning, skipped", flush=True)
            continue
        for op in ops:
            weight_map[op[0]] = sh
        dst = os.path.join(a.out, sh)
        hdr, _ = read_header(os.path.join(a.src, sh))
        meta = hdr.get("__metadata__") or {"format": "pt"}
        raw, data_len = out_header(ops, meta)
        if os.path.exists(dst) and os.path.getsize(dst) == 8 + len(raw) + data_len:
            print(f"[{i + 1}/{len(plan)}] {sh}: exists, skipped", flush=True)
            continue
        sz = write_shard(os.path.join(a.src, sh), dst, ops, meta)
        print(f"[{i + 1}/{len(plan)}] {sh}: {len(ops)} tensors, {sz / 1e9:.2f} GB", flush=True)

    # config: expert count everywhere it appears, MTP fields
    nhl = text_cfg(cfg)["num_hidden_layers"]
    for d in (cfg, cfg.get("text_config", {})):
        for k in EXPERT_COUNT_KEYS:
            if d.get(k) == n_exp:
                d[k] = K
    per_list = None
    if a.per_layer:  # one count per decoder layer, then per kept MTP layer; 0 = dense
        n_mtp = 0 if dropped else text_cfg(cfg).get("num_nextn_predict_layers", 0)
        per_list = [len(kept[L]) if L in kept else 0 for L in range(nhl + n_mtp)]
        text_cfg(cfg)["n_routed_experts_per_layer"] = per_list
    if dropped:
        text_cfg(cfg)["num_nextn_predict_layers"] = 0
        qc = cfg.get("quantization_config", {})
        if "modules_to_not_convert" in qc:
            qc["modules_to_not_convert"] = [m for m in qc["modules_to_not_convert"]
                                            if not (LAYER_RE.match(m) and int(LAYER_RE.match(m).group(1)) >= nhl)]

    # small files (tokenizer, chat template, generation_config, ...)
    for f in sorted(os.listdir(a.src)):
        p = os.path.join(a.src, f)
        if (os.path.isfile(p) and not f.startswith(".") and not f.endswith(".safetensors")
                and f not in ("config.json", "model.safetensors.index.json")):
            shutil.copy2(p, os.path.join(a.out, f))

    pj = {"n_experts_orig": n_exp, "n_experts_kept": K, "mtp": a.mtp,
          "kept_original_ids": {str(L): ids for L, ids in sorted(kept.items())}}
    if a.pad:  # slot ids >= len(kept_original_ids[L]) are dead
        pj["n_real_experts"] = sum(len(ids) for ids in kept.values())
        pj["n_dead_slots"] = {str(L): n for L, n in sorted(n_dead.items())}
    if a.per_layer:  # n_experts_kept is the max; the real count of layer L is the list entry
        pj["per_layer"] = True
        pj["n_real_experts"] = sum(len(ids) for ids in kept.values())
        pj["n_routed_experts_per_layer"] = per_list
    json.dump(pj, open(os.path.join(a.out, "pruned_experts.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    # index last: its presence marks a complete build
    tmp = os.path.join(a.out, "model.safetensors.index.json.tmp")
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, open(tmp, "w"), indent=2)
    os.replace(tmp, os.path.join(a.out, "model.safetensors.index.json"))
    print(f"done: {len(weight_map)} tensors, {total / 1e9:.2f} GB in {a.out}", flush=True)


if __name__ == "__main__":
    main()
