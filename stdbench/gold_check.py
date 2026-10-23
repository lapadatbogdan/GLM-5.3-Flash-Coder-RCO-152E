# Gold check: every item's known-correct answer (bench.gold_reply) must pass the bench's grader and,
# where defined, a known-wrong answer (bench.wrong_reply) must fail. CPU only, login node is fine for
# small benches; code benches start the no-network sandbox. Python 3.6.
# python3 gold_check.py BENCH [--threads 16] [--limit N]
# Writes data/goldcheck_<bench>.json (counts + ids that misbehave).
import argparse, json, os, sys, threading, queue, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sandboxes", type=int, default=0, help="override SANDBOX n")
    a = ap.parse_args()
    m = core.bench_module(a.bench)
    items = core.load_items(a.bench)
    if a.limit:
        items = items[:a.limit]
    spec = getattr(m, "SANDBOX", None)
    box = None
    if spec:
        box = core.Sandbox(sif=spec.get("sif"), n=a.sandboxes or spec.get("n", 8), binds=spec.get("binds"),
                           env=spec.get("env"), tag="gc").start()
    res = {"gold_pass": [], "gold_fail": [], "wrong_pass": [], "wrong_fail": [], "no_gold": [], "errors": []}
    detail = {}
    lock = threading.Lock()
    q = queue.Queue()
    for it in items:
        q.put(it)

    def one(it):
        g = m.gold_reply(it, 0)
        if g is None:
            with lock:
                res["no_gold"].append(it["id"])
            return
        r = m.score(it, {"content": g, "reasoning_content": "", "finish_reason": "stop"}, box)
        with lock:
            (res["gold_pass"] if r.get("pass") else res["gold_fail"]).append(it["id"])
            if not r.get("pass"):
                detail[it["id"]] = {k: (str(v)[-800:]) for k, v in r.items()}
        if hasattr(m, "wrong_reply"):
            w = m.wrong_reply(it, 0)
            if w is not None:
                r = m.score(it, {"content": w, "reasoning_content": "", "finish_reason": "stop"}, box)
                with lock:
                    (res["wrong_pass"] if r.get("pass") else res["wrong_fail"]).append(it["id"])

    def worker():
        while True:
            try:
                it = q.get_nowait()
            except queue.Empty:
                return
            try:
                one(it)
            except Exception as e:
                with lock:
                    res["errors"].append("%s: %s: %s" % (it["id"], type(e).__name__, str(e)[:300]))

    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(a.threads)]
    try:
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    finally:
        if box:
            box.stop()
    summary = {k: len(v) for k, v in res.items()}
    summary["items"] = len(items)
    summary["seconds"] = round(time.time() - t0, 1)
    print(a.bench, json.dumps(summary))
    for k in ("gold_fail", "wrong_pass", "errors", "no_gold"):
        if res[k]:
            print("  %s (%d): %s" % (k, len(res[k]), res[k][:15]))
    if not a.limit:
        with open(os.path.join(core.DATA, "goldcheck_%s.json" % a.bench), "w") as f:
            json.dump({"summary": summary, "lists": {k: v for k, v in res.items() if k not in ("gold_pass", "wrong_fail")},
                       "gold_fail_detail": detail}, f, indent=1)


if __name__ == "__main__":
    main()
