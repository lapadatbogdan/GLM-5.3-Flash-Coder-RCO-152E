#!/usr/bin/env python3
"""Sharded RCO reference (the --mode prep forward pass of rco_glm_prune.py, with the DSA fix like rco_prep_v2.py):
N independent jobs, job i computes the full model's top-k reference and router statistics for the windows
i, i+N, i+2N, ... of calib_search.pt and calib_heldout.pt (batch 1, each window independent, so the result is
identical to the single-node run). Incremental and resumable: progress saved every --chunk windows.
Then `--merge` (CPU, no model) assembles CALIB_DIR/ref_top{k}.pt exactly in rco_glm_prune's format and the
halves-overlap part of prep_report.json; it refuses to overwrite an existing reference.

    shard:  rco_prep_v2_mn.py --model $P/full --calib-dir $P/calib53/mix_v1 --out $P/rco_runs/prep_mix --shard 0 --nshards 4
    merge:  rco_prep_v2_mn.py --merge --calib-dir $P/calib53/mix_v1 --out $P/rco_runs/prep_mix --nshards 4
RCO_CPU=1: model on CPU (test only). --n-layers / --limit as in rco_glm_prune (test reference name *_test.pt).
"""
import argparse
import json
import logging
import os
import sys
import time

P = "/work/glm53"
for p in (P + "/heal", P + "/rco", P + "/rco/src"):
    sys.path.insert(0, p)
import torch
import torch.nn.functional as F

log = logging.getLogger("prep_mn")


def load_sets(calib_dir, limit):
    S = torch.load(os.path.join(calib_dir, "calib_search.pt"), weights_only=False)
    H = torch.load(os.path.join(calib_dir, "calib_heldout.pt"), weights_only=False)
    if limit:
        S = {k: v[:limit] for k, v in S.items()}
        H = {k: v[:limit] for k, v in H.items()}
    return S, H


def ref_name(a):
    return f"ref_top{a.topk}{'_test' if (a.limit or a.n_layers) else ''}.pt"


def merge(a):
    import rco_glm_prune as R
    dst = os.path.join(a.calib_dir, ref_name(a))
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists, refusing to overwrite")
    S, H = load_sets(a.calib_dir, a.limit)
    parts = []
    for i in range(a.nshards):
        f = os.path.join(a.out, f"shard{i}of{a.nshards}.pt")
        d = torch.load(f, weights_only=False)
        if not d.get("done"):
            raise SystemExit(f"{f} is not finished")
        parts.append(d)
    out = {}
    for key, n in (("search", S["ids"].size(0)), ("heldout", H["ids"].size(0))):
        rows = {}
        for d in parts:
            for w, r, fq, sc in zip(d[key]["idx"], d[key]["ref"], d[key]["freq"], d[key]["score"]):
                assert w not in rows, (key, w)
                rows[w] = (r, fq, sc)
        assert sorted(rows) == list(range(n)), f"{key}: windows missing {sorted(set(range(n)) - set(rows))[:10]}"
        out[key] = [rows[w][0] for w in range(n)]
        out["freq" if key == "search" else "freq_h"] = torch.stack([rows[w][1] for w in range(n)])
        out["score" if key == "search" else "score_h"] = torch.stack([rows[w][2] for w in range(n)])
    tmp = dst + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, dst)
    fs = out["freq"]
    n = fs.size(0) // 2
    rep = {"windows": [fs.size(0), out["freq_h"].size(0)], "overlap": {}, "shards": a.nshards}
    for kp in [int(x) for x in a.keeps.split(",")]:
        mean_ov, min_ov = R.keep_overlap(fs[:n].sum(0), fs[n:].sum(0), kp)
        rep["overlap"][kp] = {"mean": mean_ov, "min_layer": min_ov}
        print(f"halves agree on the top-{kp} experts: mean {mean_ov:.1%}, worst layer {min_ov:.1%}")
    json.dump(rep, open(os.path.join(a.out, "prep_report.json"), "w"), indent=1)
    if a.reap_out:
        merge_reap(a, parts, S["ids"].size(0))
    if "reap_w" in parts[0]["search"]:
        merge_reap_windows(a, parts, S["ids"].size(0), H["ids"].size(0))
    print(f"reference saved to {dst}: search {len(out['search'])}, held-out {len(out['heldout'])} windows")


def merge_reap(a, parts, n_search):
    """Sum the shards' REAP stats (float64) and write stats.pt + the keep-K masks exactly like reap_score.py."""
    import reap_score
    if os.path.exists(os.path.join(a.reap_out, "stats.pt")):
        raise SystemExit(f"{a.reap_out}/stats.pt exists, refusing to overwrite")
    wins = sorted(w for d in parts for w in d["reap"]["windows"])
    assert wins == list(range(n_search)), "REAP stats do not cover every search window once"
    st = {k: sum(d["reap"][k] for d in parts) for k in ("count", "gsum", "reap")}
    os.makedirs(a.reap_out, exist_ok=True)
    torch.save({**st, "done": n_search, "shards": a.nshards}, os.path.join(a.reap_out, "stats.pt"))
    cnt, rp = st["count"], st["reap"]
    scores = {"freq": cnt, "reap_mean": rp / cnt.clamp(min=1), "reap_sum": rp}
    for kp in [int(x) for x in a.reap_keeps.split(",")]:
        masks = {n: reap_score.keep_top(sc, kp) for n, sc in scores.items()}
        for n, m in masks.items():
            torch.save(m, os.path.join(a.reap_out, f"mask_{n}_k{kp}.pt"))
            ov = ((~m) & (~masks["freq"])).sum().item() / (kp * m.size(0))
            print(f"REAP keep {kp} {n:9s}: same experts as freq {ov:.0%}")
    print(f"REAP stats + masks saved to {a.reap_out}")


def merge_reap_windows(a, parts, n_s, n_h):
    """CALIB_DIR/reap_windows.pt: {"search"|"heldout": float64 (n, 3, L, E) in window order}, the 3 = count, gsum,
    reap (same sums as reap_score.install_stats, but per window)."""
    dst = os.path.join(a.calib_dir, "reap_windows.pt")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists, refusing to overwrite")
    out = {}
    for key, n in (("search", n_s), ("heldout", n_h)):
        rows = {}
        for d in parts:
            for w, t in zip(d[key]["idx"], d[key]["reap_w"]):
                rows[w] = t
        assert sorted(rows) == list(range(n)), key
        out[key] = torch.stack([rows[w] for w in range(n)])
    out["fields"] = ["count", "gsum", "reap"]
    torch.save(out, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    print(f"per-window REAP stats saved to {dst}: search {tuple(out['search'].shape)}, held-out {tuple(out['heldout'].shape)}")


def shard(a):
    import dsa_fix
    dsa_fix.install_long_context_patches(int(os.environ.get("Q_CHUNK", "256")), int(os.environ.get("KDA_GROUP", "8")))
    import glm5_next_rco as glm
    import rco_glm_prune as R
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"shard{a.shard}of{a.nshards}.pt")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(a.out, f"log_shard{a.shard}of{a.nshards}.txt")),
                                  logging.StreamHandler()])
    st = torch.load(path, weights_only=False) if os.path.exists(path) else None
    if st and st.get("done"):
        log.info(f"{path} already done")
        return
    S, H = load_sets(a.calib_dir, a.limit)
    todo = {"search": list(range(a.shard, S["ids"].size(0), a.nshards)),
            "heldout": list(range(a.shard, H["ids"].size(0), a.nshards))}
    st = st or {k: {"idx": [], "ref": [], "freq": [], "score": []} for k in todo}
    log.info(f"shard {a.shard}/{a.nshards}: search {len(todo['search'])}, held-out {len(todo['heldout'])} windows; "
             f"already {len(st['search']['idx'])} + {len(st['heldout']['idx'])}")
    t0 = time.time()
    devices = [torch.device("cpu")] if os.environ.get("RCO_CPU") == "1" else None
    model = glm.load_glm_fp8(a.model, devices=devices, n_layers=a.n_layers)
    log.info(f"loaded in {time.time() - t0:.0f}s")
    n_exp = model.config.num_local_experts
    dev = get_dev(model)
    if a.shard == 0 and not st["search"]["idx"]:  # same SANITY line as rco_prep_v2.py, once
        with torch.no_grad():
            for w in range(min(2, H["ids"].size(0))):
                ids = H["ids"][w:w + 1].long()
                m = H["mask"][w].clone().bool()
                m[0] = False
                lp = F.log_softmax(model(input_ids=ids.to(dev)).logits[0, :-1].float(), -1).cpu()
                pos = m[1:].nonzero().flatten()
                tgt = ids[0, 1:][pos]
                ce = -lp[pos].gather(-1, tgt[:, None]).squeeze(-1)
                log.info(f"SANITY full model (dsa_fix) calib_heldout window {w}: CE {ce.mean():.4f} on {pos.numel()} "
                         f"tok, top-1 {(lp[pos].argmax(-1) == tgt).float().mean().item():.3f}")
    reap = None
    if a.reap_per_window:
        a.reap_stats = True
    if a.reap_stats:  # REAP count/gsum/reap sums over the SEARCH windows, collected in the same forward pass
        import reap_score
        reap = reap_score.install_stats(model, n_exp)  # same expert math as Fp8Experts.forward, plus the sums
        if "reap" in st and not a.reap_per_window:
            for k in reap:
                reap[k].copy_(st["reap"][k])
    sets = {"search": S, "heldout": H}
    for key in ("search", "heldout"):
        rest = [w for w in todo[key] if w not in set(st[key]["idx"])]
        for c in range(0, len(rest), a.chunk):
            ws = rest[c:c + a.chunk]
            if a.reap_per_window:  # one window at a time: zero the sums, forward, keep this window's sums
                refs, fq, sc = [], [], []
                for w in ws:
                    for v in reap.values():
                        v.zero_()
                    r1, f1, s1 = R.reference(model, sets[key]["ids"][[w]], a.topk, n_exp, f"{key} shard {a.shard}")
                    refs += r1
                    fq.append(f1[0])
                    sc.append(s1[0])
                    st[key].setdefault("reap_w", []).append(torch.stack([reap[k].clone() for k in ("count", "gsum", "reap")]))
                fq, sc = torch.stack(fq), torch.stack(sc)
            else:
                refs, fq, sc = R.reference(model, sets[key]["ids"][ws], a.topk, n_exp, f"{key} shard {a.shard}")
            st[key]["idx"] += ws
            st[key]["ref"] += refs
            st[key]["freq"] += list(fq)
            st[key]["score"] += list(sc)
            if reap is not None and key == "search" and not a.reap_per_window:
                st["reap"] = {k: v.clone() for k, v in reap.items()}
                st["reap"]["windows"] = list(st["search"]["idx"])
            tmp = path + ".tmp"
            torch.save(st, tmp)
            os.replace(tmp, path)
            log.info(f"{key}: {len(st[key]['idx'])}/{len(todo[key])} windows, {time.time() - t0:.0f}s")
    st["done"] = True
    torch.save(st, path + ".tmp")
    os.replace(path + ".tmp", path)
    log.info("DONE shard")


def get_dev(model):
    from common import get_input_device
    return get_input_device(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--keeps", default="96,120,144")
    ap.add_argument("--reap-stats", action="store_true", help="shard: also REAP stats on the search windows")
    ap.add_argument("--reap-per-window", action="store_true",
                    help="shard: REAP sums per window (search AND held-out), count/gsum/reap x layer x expert, float64; "
                         "merge then writes CALIB_DIR/reap_windows.pt")
    ap.add_argument("--reap-out", help="merge: write summed REAP stats + masks (freq/reap_mean/reap_sum) here")
    ap.add_argument("--reap-keeps", default="144,120,96")
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    merge(a) if a.merge else shard(a)


if __name__ == "__main__":
    main()
