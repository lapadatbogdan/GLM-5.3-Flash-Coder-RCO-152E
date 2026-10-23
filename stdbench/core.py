# Standard benchmark suite (stdbench) core: bench registry, sandbox, chat calls.
# Must stay Python 3.6 compatible (the host /usr/bin/python3 of our cluster is 3.6.15): no f-strings with
# backslashes, no walrus, no capture_output/text= in subprocess, no dataclasses.
#
# Bench module interface (benches/<name>.py, imported on the host):
#   NAME, MAX_TOKENS, SAMPLES (default samples per item), SANDBOX (None or dict(sif=..., binds=[...], n=...))
#   load(data_dir) -> list of items (dict with unique "id")              [default: data/<NAME>.jsonl]
#   messages(item) -> chat messages for turn 1
#   score(item, reply, sbx) -> dict with "pass" (bool) and optionally "score" (float in [0,1]),
#        "metrics" {name: float} (averaged per item), "metrics_w" {name: [num, den]} (summed),
#        "extracted", "exec"...; reply = {"content", "reasoning_content", "finish_reason"}
#   next_messages(item, msgs, reply, result) -> messages for the next turn or None   [optional]
#   gold_reply(item, turn) -> a correct answer text (fake server, gold checks)
#   wrong_reply(item, turn) -> an incorrect answer text                            [optional]
import hashlib, importlib, json, os, random, shutil, subprocess, sys, threading, time, uuid
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TOOLS = os.path.join(HERE, "tools")
SBX_ROOT = os.environ.get("STDBENCH_SBX_ROOT", os.path.join(HERE, "sbx"))
DEFAULT_SIF = os.environ.get("STDBENCH_SIF", os.path.join(HERE, "..", "regsuite", "sif", "multiple-eval.sif"))
ALL_BENCHES = ["ifeval", "math500", "aime25", "lcb"]  # gpqa, mmlupro built but dropped from the suite (scope: code + math)

# Official recommended sampling for GLM-5.3-Flash (generation_config.json + docs.z.ai):
# temperature 1.0, top_p 0.95, no top_k, no min_p. llama-server defaults are top_k=40, min_p=0.05,
# so they are switched off explicitly per request.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0}
if os.environ.get("STDBENCH_SAMPLING"):  # test overrides, e.g. DRY / repeat penalty
    SAMPLING = dict(SAMPLING, **json.loads(os.environ["STDBENCH_SAMPLING"]))

sys.path.insert(0, HERE)


def bench_module(name):
    return importlib.import_module("benches." + name)


def load_items(name, data_dir=DATA):
    m = bench_module(name)
    if hasattr(m, "load"):
        return m.load(data_dir)
    items = []
    with open(os.path.join(data_dir, "%s.jsonl" % name)) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def run_id(item, k):
    return "%s#s%d" % (item["id"], k)


def seed_for(rid):
    return int(hashlib.md5(rid.encode()).hexdigest()[:8], 16)


LOOPSTOP = os.environ.get("STDBENCH_LOOPSTOP", "") == "1"
LOOP_WIN = 2000   # chars of tail checked
LOOP_PMAX = 64    # max period (chars) of the repeated unit


def is_loop(text):
    """Degenerate repetition: the last LOOP_WIN chars are exactly periodic with a period <= LOOP_PMAX
    (e.g. 'California California ...', 'amamam'). Such a run never recovers; it would only hit max_tokens."""
    if len(text) < LOOP_WIN:
        return False
    t = text[-LOOP_WIN:]
    for p in range(1, LOOP_PMAX + 1):
        if t[p:] == t[:-p]:
            return True
    return False


def _chat_stream(port, body, timeout):
    """Streaming chat that stops a degenerate loop early (client disconnect cancels the slot).
    Returns the same dict as chat(); finish_reason 'loop' when stopped."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % body.pop("_port"),
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    content, reasoning, finish, usage, n = [], [], None, {}, 0
    tail = ""
    resp = urllib.request.urlopen(req, timeout=timeout)
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                c, r = dl.get("content") or "", dl.get("reasoning_content") or ""
                if c or r:
                    n += 1
                    content.append(c)
                    reasoning.append(r)
                    tail = (tail + r + c)[-(2 * LOOP_WIN):]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if n and n % 64 == 0 and is_loop(tail):
                finish = "loop"
                usage = dict(usage or {}, completion_tokens=n)
                break
    finally:
        resp.close()
    return {"content": "".join(content), "reasoning_content": "".join(reasoning),
            "finish_reason": finish, "usage": usage or {"completion_tokens": n},
            "seconds": round(time.time() - t0, 1)}


def chat(port, messages, max_tokens, seed, timeout, sampling=None):
    body = {"model": "m", "messages": messages, "max_tokens": max_tokens, "seed": seed}
    body.update(sampling or SAMPLING)
    if LOOPSTOP:
        return _chat_stream(port, dict(body, _port=port), timeout)
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))
    ch = d["choices"][0]
    m = ch.get("message") or {}
    return {"content": m.get("content") or "", "reasoning_content": m.get("reasoning_content") or "",
            "finish_reason": ch.get("finish_reason"), "usage": d.get("usage") or {},
            "seconds": round(time.time() - t0, 1)}


class Sandbox(object):
    """Pool of singularity instances started with --containall --no-home --net --network none.
    Only a private, empty work dir (/work) plus explicit read-only binds are visible.
    run(script, stdin_text) executes `script` with bash in a fresh /work/j_<id> dir (removed after)."""

    def __init__(self, sif=None, n=4, binds=None, tag="sb", env=None):
        self.sif = sif or DEFAULT_SIF
        self.binds = binds or []
        self.env = env or {}
        self.n = n
        self.names = []
        self.dirs = []
        self.lock = threading.Lock()
        self.rr = 0
        self.tag = "%s%d_%s" % (tag, os.getpid(), uuid.uuid4().hex[:6])

    def start(self):
        for i in range(self.n):
            name = "%s_%d" % (self.tag, i)
            d = os.path.join(SBX_ROOT, name)
            os.makedirs(os.path.join(d, "home"))
            extra = []
            for b in self.binds:
                extra += ["-B", b]
            r = subprocess.run(["singularity", "instance", "start", "--containall", "--no-home",
                                "--net", "--network", "none", "-B", "%s:/work" % d] + extra +
                               [self.sif, name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True)
            if r.returncode != 0:
                self.stop()
                raise RuntimeError("sandbox start failed: " + r.stdout[-2000:])
            self.names.append(name)
            self.dirs.append(d)
        return self

    def stop(self):
        for name in self.names:
            subprocess.run(["singularity", "instance", "stop", name], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        for d in self.dirs:  # only the private work dirs this object created
            shutil.rmtree(d, ignore_errors=True)
        self.names = []
        self.dirs = []

    def run(self, script, stdin_text="", timeout=600):
        with self.lock:
            k = self.rr % self.n
            self.rr += 1
        name = self.names[k]
        job = uuid.uuid4().hex[:12]
        envs = " ".join("%s=%s" % (a, b) for a, b in sorted(self.env.items()))
        full = ("export HOME=/work/home %s; d=/work/j_%s; mkdir -p $d && cd $d && { %s\n}; rc=$?; cd /; "
                "rm -rf $d; exit $rc" % (envs, job, script))
        try:
            r = subprocess.run(["singularity", "exec", "instance://" + name, "bash", "-c", full],
                               input=stdin_text, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, timeout=timeout)
            return r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            return 124, "TIMEOUT after %ds" % timeout


def run_json(sbx, script, payload, timeout=600):
    """Run a grader inside the sandbox that reads one JSON object on stdin and prints one JSON line
    prefixed with 'RESULT '. Returns (dict or None, raw output)."""
    rc, out = sbx.run(script, json.dumps(payload), timeout=timeout)
    res = None
    for line in out.splitlines():
        if line.startswith("RESULT "):
            try:
                res = json.loads(line[7:])
            except ValueError:
                pass
    return res, out


def stratified_sample(items, key, n, seed):
    """Proportional stratified sample (largest remainder), fixed seed; returns items in input order."""
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    total = len(items)
    quota = {}
    rem = []
    for g, v in groups.items():
        q = n * len(v) / float(total)
        quota[g] = int(q)
        rem.append((q - int(q), g))
    left = n - sum(quota.values())
    for _, g in sorted(rem, key=lambda x: (-x[0], x[1]))[:left]:
        quota[g] += 1
    rng = random.Random(seed)
    chosen = set()
    for g in sorted(groups):
        ids = [it["id"] for it in groups[g]]
        for i in rng.sample(ids, quota[g]):
            chosen.add(i)
    return [it for it in items if it["id"] in chosen]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
