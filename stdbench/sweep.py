# Throughput sweep against one llama-server: for each concurrency level C, fire C identical-length
# generations at once (ignore_eos, fixed max_tokens) and measure aggregate and per-slot decode speed.
# Writes results/sweep_<tag>.json and prints the chosen concurrency: the smallest C whose aggregate
# tok/s is within 5% of the best (fewer slots = shorter tail latency for long reasoning items).
# python3 sweep.py --tag full --port 8000 --levels 4,8,16,32,48,64 --tokens 512
import argparse, json, os, sys, threading, time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = ("Write a very long, detailed essay about the history of mathematics, from Babylonian tablets to "
          "modern proof assistants. Do not stop early.")


def one(port, tokens, out, i):
    body = {"model": "m", "messages": [{"role": "user", "content": PROMPT + " (variant %d)" % i}],
            "max_tokens": tokens, "ignore_eos": True, "temperature": 1.0, "top_p": 0.95, "top_k": 0,
            "min_p": 0.0, "seed": i}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=7200).read().decode())
        out[i] = {"tok": d["usage"]["completion_tokens"], "s": time.time() - t0,
                  "timings": d.get("timings", {})}
    except Exception as e:
        out[i] = {"err": str(e)[:200], "s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--levels", default="4,8,16,32,48,64")
    ap.add_argument("--tokens", type=int, default=512)
    a = ap.parse_args()
    res = []
    for c in [int(x) for x in a.levels.split(",")]:
        out = {}
        ths = [threading.Thread(target=one, args=(a.port, a.tokens, out, i)) for i in range(c)]
        t0 = time.time()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        ok = [v for v in out.values() if "tok" in v]
        tok = sum(v["tok"] for v in ok)
        pred = [v["timings"].get("predicted_per_second") for v in ok if v["timings"].get("predicted_per_second")]
        r = {"conc": c, "ok": len(ok), "errors": c - len(ok), "wall_s": round(wall, 1), "agg_tok_s": round(tok / wall, 2),
             "per_slot_tok_s": round(sum(pred) / len(pred), 2) if pred else None}
        res.append(r)
        print("[sweep %s] %s" % (a.tag, json.dumps(r)), flush=True)
        if c - len(ok):
            print("[sweep] errors at conc=%d, stopping: %s" % (c, [v.get("err") for v in out.values() if "err" in v][:3]), flush=True)
            break
    good = [r for r in res if not r["errors"]]
    best = max(r["agg_tok_s"] for r in good) if good else 0
    pick = min([r["conc"] for r in good if r["agg_tok_s"] >= 0.95 * best] or [4])
    with open(os.path.join(HERE, "results", "sweep_%s.json" % a.tag), "w") as f:
        json.dump({"levels": res, "chosen_conc_per_server": pick, "tokens": a.tokens}, f, indent=1)
    print("[sweep %s] chosen concurrency per server: %d (best aggregate %.1f tok/s)" % (a.tag, pick, best), flush=True)
    print("CHOSEN=%d" % pick)


if __name__ == "__main__":
    main()
