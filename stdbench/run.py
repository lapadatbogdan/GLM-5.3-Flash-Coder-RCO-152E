# stdbench driver: dispatches (item, sample) runs to llama-servers, scores each final answer with the
# bench's official-style grader (code runs in the no-network sandbox), appends ONE JSON line per run.
# Resumable: runs already present with status=ok in the output file are skipped. Python 3.6.
#
# python3 run.py --tag full --ports 8000 --conc 32 [--benches ifeval,math500,...] [--samples aime25=8]
#                [--limit N  (first N items per bench, by stable hash; smoke runs)]
import argparse, hashlib, json, os, queue, sys, threading, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

HERE = core.HERE


def parse_samples(s):
    out = {}
    for kv in (s or "").split(","):
        if kv.strip():
            k, v = kv.split("=")
            out[k.strip()] = int(v)
    return out


def _start_pools(sbx, pend, mods):
    """One sandbox pool per distinct SANDBOX spec, only for benches with pending runs."""
    for b in sorted(set(x[0] for x in pend)):
        spec = getattr(mods[b], "SANDBOX", None)
        if spec:
            key = json.dumps(spec, sort_keys=True)
            if key not in sbx:
                n_sbx = int(os.environ.get("STDBENCH_SBX_N", 0)) or spec.get("n", 8)  # login-node tests: small
                sbx[key] = core.Sandbox(sif=spec.get("sif"), n=n_sbx, binds=spec.get("binds"),
                                        env=spec.get("env"), tag="sb").start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ports", required=True, help="comma list of llama-server ports")
    ap.add_argument("--conc", type=int, required=True, help="concurrent conversations in total (spread over ports)")
    ap.add_argument("--benches", default=",".join(core.ALL_BENCHES))
    ap.add_argument("--samples", default="", help="per-bench sample count overrides, e.g. aime25=8")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items per bench (stable hash order)")
    ap.add_argument("--data", default=core.DATA)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", default="", help="i/n: keep only pending runs with md5(run id) %% n == i")
    ap.add_argument("--min-max-tokens", default="", help="per-bench floor for MAX_TOKENS, e.g. lcb=131072,aime25=131072")
    ap.add_argument("--ids", default="", help="file with item ids to keep (one per line)")
    ap.add_argument("--http-timeout", type=int, default=6 * 3600)
    a = ap.parse_args()
    ports = [int(p) for p in a.ports.split(",")]
    out = a.out or os.path.join(HERE, "results", "%s.jsonl" % a.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    samples = parse_samples(a.samples)
    benches = [b for b in a.benches.split(",") if b]

    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # torn last line from a killed job
                if r.get("status") == "ok":
                    done.add(r["id"])

    mods = {}
    pend = []
    counts = []
    for b in benches:
        m = core.bench_module(b)
        mods[b] = m
        mm = parse_samples(a.min_max_tokens)
        if b in mm and mm[b] > m.MAX_TOKENS:
            m.MAX_TOKENS = mm[b]
        items = core.load_items(b, a.data)
        items.sort(key=lambda it: hashlib.md5(it["id"].encode()).hexdigest())
        if a.limit:
            items = items[:a.limit]
        if a.ids:
            keep = set(x.strip() for x in open(a.ids) if x.strip())
            items = [it for it in items if it["id"] in keep]
        k = samples.get(b, getattr(m, "SAMPLES", 1))
        n_b = 0
        for it in items:
            for s in range(k):
                rid = core.run_id(it, s)
                if rid not in done:
                    pend.append((b, it, s, rid))
                    n_b += 1
        counts.append("%s=%dx%d(pending %d)" % (b, len(items), k, n_b))
    if a.shard:
        si, sn = [int(x) for x in a.shard.split("/")]
        pend = [x for x in pend if int(hashlib.md5(x[3].encode()).hexdigest(), 16) % sn == si]
        print("[%s] shard %d/%d: %d pending" % (a.tag, si, sn, len(pend)), flush=True)
    # interleave benches by a stable hash so a partial run covers every bench evenly
    pend.sort(key=lambda x: hashlib.md5(x[3].encode()).hexdigest())
    print("[%s] %s | done=%d pending=%d servers=%d conc=%d" %
          (a.tag, " ".join(counts), len(done), len(pend), len(ports), a.conc), flush=True)
    if not pend:
        return

    # sandboxes, one pool per distinct spec, started only if a pending bench needs it
    sbx = {}
    try:
        _start_pools(sbx, pend, mods)
    except Exception:
        for v in sbx.values():
            v.stop()
        raise
    lock = threading.Lock()
    fout = open(out, "a")
    stats = {"n": 0, "pass": 0, "trunc": 0, "err": 0, "tok": 0}
    t_start = time.time()

    def work(port, b, it, s, rid):
        m = mods[b]
        it = dict(it, _sample=s)  # benches with per-sample variants (GPQA permutations) read it
        spec = getattr(m, "SANDBOX", None)
        box = sbx[json.dumps(spec, sort_keys=True)] if spec else None
        rec = {"id": rid, "item": it["id"], "bench": b, "sample": s, "tag": a.tag,
               "group": it.get("group"), "max_tokens": m.MAX_TOKENS}
        try:
            msgs = m.messages(it)
            turns = []
            res = None
            while True:
                r = None
                for attempt in range(3):
                    try:
                        r = core.chat(port, msgs, m.MAX_TOKENS, core.seed_for(rid) + len(turns), a.http_timeout)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(30)
                res = m.score(it, r, box)
                turns.append({"content": r["content"], "reasoning_chars": len(r["reasoning_content"]),
                              "reasoning_tail": r["reasoning_content"][-1500:],
                              "finish_reason": r["finish_reason"], "usage": r["usage"], "seconds": r["seconds"],
                              "pass": bool(res.get("pass"))})
                nxt = m.next_messages(it, msgs, r, res) if hasattr(m, "next_messages") else None
                if not nxt:
                    break
                msgs = nxt
            rec.update(res)
            rec["pass"] = bool(res.get("pass"))
            rec["turns"] = turns
            rec["truncated"] = any(t["finish_reason"] == "length" for t in turns)
            rec["loop"] = any(t["finish_reason"] == "loop" for t in turns)
            rec["truncated"] = rec["truncated"] or rec["loop"]
            rec["completion_tokens"] = sum((t["usage"] or {}).get("completion_tokens", 0) for t in turns)
            rec["seconds"] = round(sum(t["seconds"] for t in turns), 1)
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:500])
            traceback.print_exc()
        with lock:
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            stats["n"] += 1
            stats["pass"] += int(bool(rec.get("pass")))
            stats["trunc"] += int(bool(rec.get("truncated")))
            stats["err"] += int(rec["status"] != "ok")
            stats["tok"] += rec.get("completion_tokens", 0)
            if stats["n"] % 10 == 0 or stats["n"] == len(pend):
                el = time.time() - t_start
                print("[%s] %d/%d pass=%d trunc=%d err=%d  %.0fs elapsed, %.2f runs/min, %.1f gen tok/s (finished runs)" %
                      (a.tag, stats["n"], len(pend), stats["pass"], stats["trunc"], stats["err"], el,
                       60.0 * stats["n"] / max(el, 1), stats["tok"] / max(el, 1)), flush=True)

    q = queue.Queue()
    for x in pend:
        q.put(x)

    def worker(port):
        while True:
            try:
                x = q.get_nowait()
            except queue.Empty:
                return
            work(port, *x)

    threads = [threading.Thread(target=worker, args=(ports[k % len(ports)],)) for k in range(a.conc)]
    try:
        for t in threads:
            t.daemon = True
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(60)
    finally:
        fout.close()
        for v in sbx.values():
            try:
                v.stop()
            except Exception:
                traceback.print_exc()
    print("[%s] DONE pass=%d/%d trunc=%d err=%d" % (a.tag, stats["pass"], stats["n"], stats["trunc"], stats["err"]),
          flush=True)


if __name__ == "__main__":
    main()
