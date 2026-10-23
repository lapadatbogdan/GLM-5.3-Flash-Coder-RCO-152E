#!/usr/bin/env python3
"""Report for one run: python3 report.py results/<bench>/<run> [--gold results/<bench>/<gold_run>] [--nop results/<bench>/<nop_run>]

Denominator: with --gold (and optionally --nop), only tasks where the reference patch passes in OUR harness
(and the empty patch fails) are counted. Otherwise the denominator = all graded tasks.
Reports: resolved in total, per language, per repo, calibration repos (ansible/openlibrary/qutebrowser) separately
from the rest, the distribution of turns and time, exit reasons, mean pass@1 over K attempts.
"""
import json, glob, sys, os, collections, statistics, argparse

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--gold"); ap.add_argument("--nop")
ap.add_argument("--json", help="also write the summary as json")
a = ap.parse_args()


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "*", "att*.json")):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    return rows


rows = load(a.run)
denom = None
gold_info = ""
if a.gold:
    g = {r["id"]: r["resolved"] for r in load(a.gold)}
    n = {r["id"]: r["resolved"] for r in load(a.nop)} if a.nop else {}
    denom = {i for i, ok in g.items() if ok and not n.get(i, False)}
    spurious = [i for i, ok in n.items() if ok]
    # the denominator list, on disk, so it can also be used as IDS= for runs
    open(os.path.join(os.path.dirname(a.gold.rstrip("/")), "denominator.txt"), "w").write("".join(i + "\n" for i in sorted(denom)))
    gold_info = "gold graded %d, passes %d (%.1f%%); nop graded %d, false passes %d -> denominator %d" % (
        len(g), sum(g.values()), 100 * sum(g.values()) / max(1, len(g)), len(n), len(spurious), len(denom))

if denom is not None:
    rows = [r for r in rows if r["id"] in denom]

per_task = collections.defaultdict(list)
for r in rows:
    per_task[r["id"]].append(r)


def rate(rs):
    # mean pass@1: the mean over tasks of the fraction of resolved attempts
    t = collections.defaultdict(list)
    for r in rs:
        t[r["id"]].append(bool(r.get("resolved")))
    if not t:
        return 0, 0, 0.0
    p = [sum(v) / len(v) for v in t.values()]
    return sum(p), len(t), 100 * sum(p) / len(t)


def line(name, rs, total=None):
    s, n, pct = rate(rs)
    extra = ""
    if total is not None and n < total:
        extra = "  (graded %d of %d in the denominator)" % (n, total)
    return "  %-34s %6.1f / %-4d = %5.1f%%%s" % (name, s, n, pct, extra)


print("=" * 78)
print("RUN:", a.run)
if gold_info:
    print("DENOMINATOR:", gold_info)
ms = collections.Counter(r.get("model", "") for r in rows)
print("model:", dict(ms), "| attempts:", len(rows), "| tasks:", len(per_task))
print(line("TOTAL", rows, len(denom) if denom is not None else None))
print("-- calibration vs the rest (the calibration sessions came from ansible/openlibrary/qutebrowser)")
print(line("calibration repos", [r for r in rows if r.get("calibrated") == 1]))
print(line("NON-calibration repos", [r for r in rows if r.get("calibrated") != 1]))
print("-- per language")
for lang in sorted({r.get("lang", "?") for r in rows}):
    print(line(lang, [r for r in rows if r.get("lang") == lang]))
print("-- per repo")
for repo in sorted({r.get("repo", "?") for r in rows}):
    print(line(repo, [r for r in rows if r.get("repo") == repo]))

ag = [r for r in rows if r.get("mode") == "agent"]
if ag:
    calls = sorted(r.get("n_calls", 0) for r in ag)
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
    print("-- turns (model calls) per attempt: min %d p25 %d median %d p75 %d p90 %d max %d" % (
        calls[0], q(calls, .25), q(calls, .5), q(calls, .75), q(calls, .9), calls[-1]))
    bins = [0, 25, 50, 100, 150, 200, 300, 500, 10 ** 9]
    h = collections.Counter()
    hr = collections.Counter()
    for r in ag:
        c = r.get("n_calls", 0)
        b = next(i for i in range(len(bins) - 1) if bins[i] <= c < bins[i + 1])
        h[b] += 1; hr[b] += bool(r.get("resolved"))
    for i in sorted(h):
        print("     %4d-%-6s %4d attempts, resolved %d" % (bins[i], str(bins[i + 1] - 1) if bins[i + 1] < 10 ** 9 else "+", h[i], hr[i]))
    w = sorted(r.get("wall_agent_s", 0) for r in ag)
    sl = sum(r.get("wall_agent_s", 0) + r.get("wall_grade_s", 0) for r in ag) / 3600 / len(ag)
    print("-- agent time (min): median %.0f p90 %.0f max %.0f | slot-hours/attempt (agent+grading): %.2f" % (
        q(w, .5) / 60, q(w, .9) / 60, w[-1] / 60, sl))
    slots = int(os.environ.get("SLOTS", "8"))
    for name, n in (("Pro V2 (642)", 642), ("Pro v1 (731)", 731), ("DeepSWE (113)", 113)):
        nh = sl * n / slots / 0.8   # 0.8 = utilization (server start ~5 min, tail at the end, walltime guard)
        print("   estimate %s at %d slots/node: %.0f node-hours = %.0f GPU hours (4/node)" % (name, slots, nh, 4 * nh))
    print("-- exit reason:", dict(collections.Counter(r.get("exit_status") for r in ag)))
    hit_wall = sum(1 for r in ag if r.get("exit_status") == "TimeExceeded")
    print("   hit the time cap: %d (if large, the cap hides points)" % hit_wall)
    tok = [r.get("usage", {}).get("completion_tokens", 0) for r in ag]
    lp = sorted(r.get("last_prompt_tokens", 0) for r in ag)
    print("-- generated tokens/attempt: median %d | max context reached: median %d p90 %d max %d" % (
        statistics.median(tok) if tok else 0, q(lp, .5), q(lp, .9), lp[-1]))
    print("-- empty patch: %d | patch does not apply: %d" % (sum(1 for r in ag if not r.get("patch_bytes")),
                                                        sum(1 for r in ag if r.get("apply_failed"))))
infra = glob.glob(os.path.join(a.run, "*", "infra_att*.log"))
done_ids = {(r["id"], r["attempt"]) for r in load(a.run)}
pend = [f for f in infra if (os.path.basename(os.path.dirname(f)), int(f.split("infra_att")[1].split(".")[0])) not in done_ids]
print("-- unresolved infrastructure errors (retried on resubmit): %d" % len(pend))

if a.json:
    s, n, pct = rate(rows)
    json.dump({"run": a.run, "resolved": s, "n": n, "pct": pct, "gold": gold_info}, open(a.json, "w"))
