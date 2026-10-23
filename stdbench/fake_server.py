# CPU-only end-to-end test of run.py: a fake /v1/chat/completions that recognises every stdbench
# prompt (by the first user message) and answers like llama-server (content + reasoning_content).
# MODE right: gold answer every turn; wrong: wrong answer; mixed: wrong on turn 1, gold on later
# turns (exercises the multi-turn path, e.g. a second try). Every 9th request reports
# finish_reason=length so truncation accounting is exercised. Python 3.6.
# python3 fake_server.py PORT MODE [BENCHES]
import hashlib, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

MODE = sys.argv[2] if len(sys.argv) > 2 else "right"
BENCHES = sys.argv[3].split(",") if len(sys.argv) > 3 else core.ALL_BENCHES
KEY = {}
MODS = []


def key_of(msgs):
    first = [m for m in msgs if m["role"] == "user"][0]["content"]
    return hashlib.md5((first if isinstance(first, str) else json.dumps(first)).encode()).hexdigest()


for b in BENCHES:
    m = core.bench_module(b)
    MODS.append(m)
    for it in core.load_items(b):
        KEY[(getattr(m, "fake_key", None) or key_of)(m.messages(it))] = (m, it)
N = [0]
LK = threading.Lock()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        with LK:
            N[0] += 1
            n = N[0]
        hit = None
        for m_ in MODS:  # benches with a constant preamble provide their own prompt key / turn
            if hasattr(m_, "fake_key") and m_.fake_key(body["messages"]) in KEY:
                hit = KEY[m_.fake_key(body["messages"])]
        hit = hit or KEY.get(key_of(body["messages"]))
        if hit is not None and hasattr(hit[0], "fake_turn"):
            turn = hit[0].fake_turn(body["messages"])
        else:
            turn = sum(1 for x in body["messages"] if x["role"] == "assistant")
        if hit is None:
            content = "UNKNOWN PROMPT"
        else:
            m, it = hit
            if MODE == "right" or (MODE == "mixed" and turn > 0):
                content = m.gold_reply(it, turn) or "I do not know."
            else:
                w = getattr(m, "wrong_reply", None)
                content = (w(it, turn) if w else None) or "I do not know."
        fin = "length" if n % 9 == 0 else "stop"
        time.sleep(0.05)
        out = {"choices": [{"finish_reason": fin, "message": {"role": "assistant", "content": content,
                                                               "reasoning_content": "thinking..."}}],
               "usage": {"completion_tokens": 100 + len(content) // 4, "prompt_tokens": 50}}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class S(ThreadingMixIn, HTTPServer):
    daemon_threads = True


print("fake server: %d prompts, mode=%s" % (len(KEY), MODE), flush=True)
S(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
