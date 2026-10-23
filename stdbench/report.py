# stdbench report: per-benchmark scores for each model and model-vs-reference deltas on common runs.
# python3 report.py --ref full --variants r144q8,r144v2q8 [--groups] [--json out.json]
# - Only runs with status ok count. A run = (item, sample); AIME is avg@k over its samples.
# - "common" = run ids answered (status ok) by BOTH the reference and the variant; deltas use only those.
# - trunc% = runs where any turn ended with finish_reason=length (reported, never hidden or retried).
# - metrics (per-run floats) are averaged; metrics_w ([num, den]) are summed then divided (IFEval
#   instruction-level). The headline per bench is the mean of "pass" (IFEval: prompt-level strict;
#   MMLU-Pro: None counted wrong, see acc_official_expected).
# Python 3.6.
import argparse, json, math, os, sys
from collections import defaultdict, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["ifeval", "math500", "aime25", "gpqa", "mmlupro", "lcb"]


def load(tag):
    p = os.path.join(HERE, "results", "%s.jsonl" % tag)
    if not os.path.exists(p):
        sys.exit("missing results file: " + p)
    res, errs = {}, defaultdict(int)
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok":
                res[r["id"]] = r  # last write wins
            else:
                errs[r.get("bench")] += 1
    return res, errs


def summarize(rs):
    n = len(rs)
    if not n:
        return None
    p = sum(1.0 for r in rs if r.get("pass")) / n
    out = OrderedDict()
    out["n"] = n
    out["score"] = 100.0 * p
    out["se"] = 100.0 * math.sqrt(p * (1 - p) / n)
    out["trunc"] = 100.0 * sum(1 for r in rs if r.get("truncated")) / n
    out["avg_tok"] = sum(r.get("completion_tokens", 0) for r in rs) / float(n)
    out["avg_s"] = sum(r.get("seconds", 0) for r in rs) / float(n)
    mets = defaultdict(list)
    mw = defaultdict(lambda: [0.0, 0.0])
    for r in rs:
        for k, v in (r.get("metrics") or {}).items():
            mets[k].append(v)
        for k, v in (r.get("metrics_w") or {}).items():
            mw[k][0] += v[0]
            mw[k][1] += v[1]
    out["metrics"] = OrderedDict((k, 100.0 * sum(v) / len(v)) for k, v in sorted(mets.items()))
    for k, v in sorted(mw.items()):
        out["metrics"][k] = 100.0 * v[0] / max(v[1], 1)
    return out


def fmt(s):
    if not s:
        return "%-44s" % "-"
    m = " ".join("%s=%.1f" % (k, v) for k, v in s["metrics"].items())
    return "n=%4d %5.1f%% (se %.1f) trunc=%4.1f%% tok=%6.0f %s" % (s["n"], s["score"], s["se"], s["trunc"], s["avg_tok"], m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="full")
    ap.add_argument("--variants", default="")
    ap.add_argument("--groups", action="store_true", help="also break down by item group (subject, difficulty, language)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    tags = [a.ref] + [v for v in a.variants.split(",") if v]
    data = {t: load(t) for t in tags}
    benches = [b for b in ORDER if any(r["bench"] == b for t in tags for r in data[t][0].values())]
    benches += sorted(set(r["bench"] for t in tags for r in data[t][0].values()) - set(benches))
    report = OrderedDict()
    print("== per-benchmark scores (all ok runs of each model) ==")
    for b in benches:
        print("[%s]" % b)
        for t in tags:
            rs = [r for r in data[t][0].values() if r["bench"] == b]
            s = summarize(rs)
            report.setdefault(b, OrderedDict())[t] = s
            print("  %-12s %s  errors=%d" % (t, fmt(s), data[t][1].get(b, 0)))
            if a.groups and rs:
                g = defaultdict(list)
                for r in rs:
                    g[r.get("group") or "-"].append(r)
                for k in sorted(g):
                    gs = summarize(g[k])
                    print("      %-28s n=%4d %5.1f%% trunc=%4.1f%%" % (k[:28], gs["n"], gs["score"], gs["trunc"]))
    if len(tags) > 1:
        print("\n== deltas vs %s on common runs (both models status ok) ==" % a.ref)
        ref = data[a.ref][0]
        for v in tags[1:]:
            var = data[v][0]
            for b in benches:
                ids = [i for i, r in ref.items() if r["bench"] == b and i in var]
                if not ids:
                    print("  %-10s %-12s no common runs" % (b, v))
                    continue
                sr = summarize([ref[i] for i in ids])
                sv = summarize([var[i] for i in ids])
                lost = sum(1 for i in ids if ref[i].get("pass") and not var[i].get("pass"))
                gained = sum(1 for i in ids if var[i].get("pass") and not ref[i].get("pass"))
                # paired standard error of the difference
                d = [float(bool(var[i].get("pass"))) - float(bool(ref[i].get("pass"))) for i in ids]
                md = sum(d) / len(d)
                se = math.sqrt(sum((x - md) ** 2 for x in d) / max(len(d) - 1, 1) / len(d)) * 100
                extra = " ".join("%s %+.1f" % (k, sv["metrics"][k] - sr["metrics"][k]) for k in sv["metrics"] if k in sr["metrics"])
                print("  %-10s %-12s n=%4d  ref %5.1f%%  var %5.1f%%  delta %+5.1f (paired se %.1f)  lost=%d gained=%d  "
                      "trunc ref %.1f%% var %.1f%%  %s" % (b, v, len(ids), sr["score"], sv["score"], sv["score"] - sr["score"],
                                                            se, lost, gained, sr["trunc"], sv["trunc"], extra))
                report[b]["delta_%s" % v] = {"n": len(ids), "ref": sr["score"], "var": sv["score"],
                                             "delta": sv["score"] - sr["score"], "paired_se": se,
                                             "lost": lost, "gained": gained}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
