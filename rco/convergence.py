#!/usr/bin/env python3
"""How many calibration windows are enough? Measured on the mix_v3 pool (CPU only; the KL comes from eval_masks_mn).

Inputs: CALIB/calib_search.pt (prefix-ordered pool, build_mix3.py), CALIB/calib_heldout.pt and CALIB/reap_windows.pt
(per-window REAP sums from rco_prep_v2_mn.py --reap-per-window; [:, 0] count, [:, 1] gsum, [:, 2] reap).

  --stage masks   for every N: the reap_sum mask (keep K per layer) of the N-window prefix, plus B bootstrap draws
                  (N windows resampled with replacement from the prefix). Prints, per N: mean / worst-layer overlap
                  between draws and against the N=max mask, how many experts flip between draws and which
                  domains / languages drive the flips. Writes OUT/masks/*.pt and OUT/mask_list.json for
                  eval_masks_mn.py (prefix masks only: one held-out KL per N).
  --stage report  reads eval_masks_mn's per-window KL (OUT/kl/*.pt): KL per N in total (token-weighted, agentic/code/
                  math), per domain and per code language, and the domain-weighted KL (45/35/20). General is in the
                  held-out set only as a monitor: printed in its own column, never in the weighted KL or the rule. Rule: the smallest N whose mean
                  between-draw overlap is >= 98% AND whose weighted held-out KL differs by < 1% from the next N.
                  Prints the table and the chosen N; writes OUT/convergence.json.
overlap(A, B) for one layer = |kept(A) & kept(B)| / K.
"""
import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

import torch

SH = {"agentic": 0.45, "code": 0.35, "math": 0.20}   # weighted KL + rule; general = held-out monitor only
DOMS = ["agentic", "code", "math", "general"]


def keep_top(score, keep):
    m = torch.ones_like(score, dtype=torch.bool)
    m.scatter_(1, score.topk(keep, dim=1).indices, False)
    return m  # True = pruned (as reap_score.keep_top)


def score_of(rw, idx, kind):
    s = rw[idx].sum(0)  # (3, L, E)
    if kind == "reap_sum":
        return s[2]
    if kind == "freq":
        return s[0]
    return s[2] / s[0].clamp(min=1)  # reap_mean


def overlap(ma, mb, keep):
    """per layer |kept & kept| / keep -> (mean, min)."""
    o = ((~ma) & (~mb)).sum(1).double() / keep
    return o.mean().item(), o.min().item()


def stage_masks(a, S, rw):
    n_pool = rw.size(0)
    Ns = [n for n in a.ns if n <= n_pool] + ([n_pool] if n_pool not in a.ns else [])
    Ns = sorted(set(Ns))
    groups = [m["bucket"] if m["bucket"] != "code" else "code/" + m["lang"] for m in S["meta"]]
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    ref = {k: keep_top(score_of(rw, torch.arange(n_pool), "reap_sum"), k) for k in a.keeps}
    mask_list, table = {}, []
    g = torch.Generator().manual_seed(a.seed)
    for N in Ns:
        idx = torch.arange(N)
        row = {"N": N}
        for k in a.keeps:
            mN = keep_top(score_of(rw, idx, "reap_sum"), k)
            p = os.path.join(a.out, "masks", f"reap_sum_k{k}_N{N}.pt")
            torch.save(mN, p)
            if k == a.kl_keep:
                mask_list[f"reap_sum_k{k}_N{N}"] = p
            draws = [keep_top(score_of(rw, torch.randint(0, N, (N,), generator=g), "reap_sum"), k) for _ in range(a.draws)]
            pair = [overlap(x, y, k) for x, y in itertools.combinations(draws, 2)]
            row[f"k{k}_between_mean"] = sum(p_[0] for p_ in pair) / len(pair)
            row[f"k{k}_between_minlayer"] = min(p_[1] for p_ in pair)
            row[f"k{k}_vs_max"] = overlap(mN, ref[k], k)[0]
            if k == a.kl_keep:
                kept = torch.stack([~d for d in draws]).float().mean(0)          # (L, E) fraction of draws keeping
                flip = (kept > 0) & (kept < 1)
                row["flipping_experts"] = int(flip.sum())
                # who drives a flipping expert: the group whose share of that expert's reap mass in the prefix most
                # exceeds the group's share of the prefix windows
                share_w = Counter(groups[:N])
                tot = rw[:N, 2].sum(0).clamp(min=1e-12)
                gsum = {gn: rw[torch.tensor([i for i in range(N) if groups[i] == gn]), 2].sum(0) for gn in share_w}
                over = torch.stack([gsum[gn] / tot / (share_w[gn] / N) for gn in share_w])  # (G, L, E)
                names_ = list(share_w)
                drivers = Counter(names_[j] for j in over[:, flip].argmax(0).tolist())
                row["flip_drivers"] = dict(drivers.most_common(8))
        table.append(row)
        print(json.dumps(row), flush=True)
    for k in a.keeps:  # the frequency baseline at N = max, for reference in the KL table
        p = os.path.join(a.out, "masks", f"freq_k{k}_N{n_pool}.pt")
        torch.save(keep_top(score_of(rw, torch.arange(n_pool), "freq"), k), p)
        if k == a.kl_keep:
            mask_list[f"freq_k{k}_N{n_pool}"] = p
    json.dump(mask_list, open(os.path.join(a.out, "mask_list.json"), "w"), indent=1)
    json.dump(table, open(os.path.join(a.out, "overlap_table.json"), "w"), indent=1)
    print(f"{len(mask_list)} masks for eval_masks_mn.py -> {a.out}/mask_list.json")


def kl_table(kl_dir, H):
    """{mask: {total, weighted (45/25/15/15 over the domains present), domain, code_lang}} from eval_masks_mn shards."""
    rows = defaultdict(dict)
    for f in sorted(os.listdir(kl_dir)):
        if f.startswith("kl_shard") and f.endswith(".pt"):
            d = torch.load(os.path.join(kl_dir, f), weights_only=False)
            for name, r in d["masks"].items():
                rows[name].update(r)
    n_h = len(H["meta"])
    res = {}
    for name, r in sorted(rows.items()):
        if len(r) != n_h:
            print(f"WARNING {name}: {len(r)} of {n_h} held-out windows evaluated")

        def agg(sel):
            ws = [r[i] for i in sel if i in r]
            return sum(k * w for k, w in ws) / max(1e-9, sum(w for _, w in ws)) if ws else None
        per_dom = {d: agg([i for i, m in enumerate(H["meta"]) if m["bucket"] == d]) for d in DOMS}
        per_lang = {l: agg([i for i, m in enumerate(H["meta"]) if m["bucket"] == "code" and m["lang"] == l])
                    for l in sorted({m["lang"] for m in H["meta"] if m["bucket"] == "code"})}
        present = [d for d in SH if per_dom[d] is not None]
        res[name] = {"total": agg([i for i, m in enumerate(H["meta"]) if m["bucket"] in SH]),
                     "general_monitor": per_dom.get("general"), "domain": per_dom, "code_lang": per_lang,
                     "weighted": sum(SH[d] * per_dom[d] for d in present) / sum(SH[d] for d in present)}
    return res


def stage_summary(a, H):
    res = kl_table(a.kl_dir, H)
    print(f"{'mask':28s} {'KL_w':>8} {'total':>8} " + " ".join(f"{d[:7]:>8}" for d in DOMS) + "  (general = monitor, not weighted)")
    for n, r in res.items():
        print(f"{n:28s} {r['weighted']:>8.4f} {r['total']:>8.4f} "
              + " ".join(f"{(r['domain'][d] if r['domain'][d] is not None else float('nan')):>8.4f}" for d in DOMS))
    json.dump(res, open(os.path.join(a.kl_dir, "summary.json"), "w"), indent=1)


def stage_report(a, H):
    res = kl_table(os.path.join(a.out, "kl"), H)
    table = json.load(open(os.path.join(a.out, "overlap_table.json")))
    k = a.kl_keep
    Ns = [t["N"] for t in table]
    print(f"\n{'N':>6} {'between':>8} {'worstL':>7} {'vs_max':>7} {'flips':>6} {'KL_w':>8} {'dKL_next':>9} "
          + " ".join(f"{d[:7]:>8}" for d in DOMS) + "  (general = monitor, not in KL_w / rule)")
    chosen = None
    for i, t in enumerate(table):
        r = res.get(f"reap_sum_k{k}_N{t['N']}")
        nxt = res.get(f"reap_sum_k{k}_N{Ns[i + 1]}") if i + 1 < len(Ns) else None
        d = abs(r["weighted"] - nxt["weighted"]) / nxt["weighted"] if (r and nxt) else None
        t["kl_weighted"] = r["weighted"] if r else None
        t["dkl_next"] = d
        print(f"{t['N']:>6} {t[f'k{k}_between_mean']:>8.4f} {t[f'k{k}_between_minlayer']:>7.3f} {t[f'k{k}_vs_max']:>7.4f} "
              f"{t['flipping_experts']:>6} {(r['weighted'] if r else float('nan')):>8.4f} "
              f"{(d if d is not None else float('nan')):>9.4f} "
              + " ".join(f"{(r['domain'][x] if r and r['domain'][x] is not None else float('nan')):>8.4f}" for x in DOMS))
        if chosen is None and d is not None and t[f"k{k}_between_mean"] >= a.min_overlap and d < a.max_dkl:
            chosen = t["N"]
    fr = [n for n in res if n.startswith("freq_")]
    for n in fr:
        print(f"reference {n}: weighted KL {res[n]['weighted']:.4f}, total {res[n]['total']:.4f}")
    worst = sorted(res.get(f"reap_sum_k{k}_N{chosen or Ns[-1]}", {}).get("code_lang", {}).items(), key=lambda x: -x[1])[:6]
    print("highest held-out KL per code language at the chosen N:", [(l, round(v, 4)) for l, v in worst])
    if chosen is None:
        print(f"\nNO N meets the rule (overlap >= {a.min_overlap:.0%} and dKL < {a.max_dkl:.0%}); the pool is not enough,"
              f" or the rule is too strict: look at the table before choosing")
    else:
        print(f"\nCHOSEN N = {chosen}  (smallest N with between-draw overlap >= {a.min_overlap:.0%} and |dKL| < "
              f"{a.max_dkl:.0%} to the next N)")
    json.dump({"chosen_N": chosen, "table": table, "kl": res, "rule": {"min_overlap": a.min_overlap,
               "max_dkl": a.max_dkl, "keep": k}}, open(os.path.join(a.out, "convergence.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["masks", "report", "summary"], required=True)
    ap.add_argument("--kl-dir", help="summary: an eval_masks_mn output dir (e.g. the REAP eval of the chosen N)")
    ap.add_argument("--ns", default="256,512,768,1024,1280,1500")
    ap.add_argument("--keeps", default="144,120,96")
    ap.add_argument("--kl-keep", type=int, default=144)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-overlap", type=float, default=0.98)
    ap.add_argument("--max-dkl", type=float, default=0.01)
    a = ap.parse_args()
    a.ns = [int(x) for x in a.ns.split(",")]
    a.keeps = [int(x) for x in a.keeps.split(",")]
    S = torch.load(os.path.join(a.calib_dir, "calib_search.pt"), weights_only=False)
    H = torch.load(os.path.join(a.calib_dir, "calib_heldout.pt"), weights_only=False)
    if a.stage == "masks":
        rw = torch.load(os.path.join(a.calib_dir, "reap_windows.pt"), weights_only=False)["search"].double()
        assert rw.size(0) == len(S["meta"]), (rw.shape, len(S["meta"]))
        stage_masks(a, S, rw)
    elif a.stage == "summary":
        stage_summary(a, H)
    else:
        stage_report(a, H)


if __name__ == "__main__":
    main()
