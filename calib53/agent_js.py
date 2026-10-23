#!/usr/bin/env python3
"""Agent session generation on MINED JS instances (NodeBB / mocha), clean room:
   /wrk/base (git history deleted) + /wrk/nm/<image> for node_modules; /app masked, shared file system not mounted.
   Reward: every f2p title in 'passes', zero failures among them, non-empty patch, tests not tampered with."""
import subprocess, json, urllib.request, urllib.error, re, os, time, pathlib, ast
W = "/work/tasks"
SP = os.environ["TASK_DIR"]
INST = os.environ.get("INST", "genjs")
IMGKEY = os.environ["IMGKEY"]
MODEL = os.environ.get("LLM_MODEL", W + "/Qwen3.8-27B")
VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
# --- teacher via API (DeepSeek V4.1-Flash) ------------------------------------
API_KEY = os.environ.get("LLM_API_KEY", "")
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = "Bearer " + API_KEY
USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0, "calls": 0}

TEMP = float(os.environ.get("TEMP", "0.8"))
ATT = os.environ.get("ATTEMPT", "0")
TASK = os.environ.get("TASK", os.path.basename(SP.rstrip("/")))
OUTDIR = pathlib.Path(os.environ.get("GEN_OUT", W + "/gen_traj_js"))
WID = int(re.sub(r"\D", "", INST) or "0")
RPORT = 6500 + (WID % 40)

meta = json.load(open(SP + "/meta.json"))
def aslist(v):
    try: return ast.literal_eval(v)
    except Exception: return [x for x in v.split("\n") if x.strip()]
f2p_raw = aslist(open(SP + "/f2p.txt").read().strip())
f2p_titles = [x.split("|", 1)[-1].strip() for x in f2p_raw if "|" in x]
test_files = sorted({x.split("|")[0].strip() for x in f2p_raw if "|" in x})
BASE = "cd /wrk/w && export NODE_ENV=test PATH=/usr/local/bin:/usr/bin:/bin:$PATH && "


def exec_raw(cmd, timeout=300, limit=None):
    limit = OUTLIM if limit is None else limit
    try:
        r = subprocess.run(["singularity", "exec", f"instance://{INST}", "bash", "-lc", cmd],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, universal_newlines=True)
        return (r.stdout or "")[:limit]
    except subprocess.TimeoutExpired: return "[TIMEOUT]"
    except Exception as e: return f"[ERR {e}]"


def sh(cmd, timeout=180, limit=None): return exec_raw(BASE + cmd, timeout, limit)


# Thinking models (GLM-5.2) send their reasoning in the separate reasoning_content field,
# but it CONSUMES max_tokens. With a fixed 2500, the actual answer may not fit and content
# comes back empty -> the agent nudges -> empty targets in the training set (we cleaned 407
# such targets on 8 Sep). MAXTOK allows raising the cap without changing the default,
# so the student baseline stays reproducible.
MAXTOK = int(os.environ.get("MAXTOK", "16000"))
# The ceiling it may raise itself to; above it a generation stuck in a loop
# would fill the context and burn the turn.
TOKCEIL = int(os.environ.get("TOKCEIL", "32768"))

# Scaffold caps. The DEFAULTS are the ones the student baseline was produced with
# (30 turns, 3000 characters) and do NOT change - teachers get larger values through the environment.
MAXTURN = int(os.environ.get("MAXTURN", "30"))
OUTLIM = int(os.environ.get("OUTLIM", "3000"))
KEEPFULL = int(os.environ.get("KEEPFULL", "0"))   # how many RECENT observations stay whole (0 = no compression)



LAST_REASONING = ""


def llm(messages, max_tokens=None, retries=12):
    max_tokens = MAXTOK if max_tokens is None else max_tokens
    data = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": max_tokens, "temperature": TEMP}).encode()
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(VLLM, data=data, headers=HEADERS)
            _r = json.load(urllib.request.urlopen(req, timeout=600))
            _u = _r.get("usage") or {}
            USAGE["calls"] += 1
            USAGE["prompt_tokens"] += _u.get("prompt_tokens", 0) or 0
            USAGE["completion_tokens"] += _u.get("completion_tokens", 0) or 0
            USAGE["cache_hit_tokens"] += _u.get("prompt_cache_hit_tokens", 0) or 0
            m = _r["choices"][0]["message"]
            _txt = (m.get("content") or "").strip()
            _cut = (_r["choices"][0].get("finish_reason") == "length") or (
                not _txt and (m.get("reasoning_content") or "").strip())
            if _cut and max_tokens < TOKCEIL:
                # Do not return a truncated answer as if it were complete - request it again
                # with twice the room. This way the starting value can no longer kill a turn.
                nt = min(TOKCEIL, max_tokens * 2)
                print("[maxtok] truncated at %d (finish=%s, %dch) - retrying with %d"
                      % (max_tokens, _r["choices"][0].get("finish_reason"), len(_txt), nt), flush=True)
                return llm(messages, max_tokens=nt, retries=retries)
            if _cut:
                print("[maxtok] truncated even at the maximum ceiling %d" % TOKCEIL, flush=True)
            global LAST_REASONING
            LAST_REASONING = m.get("reasoning_content") or ""
            return m["content"]
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (503, 502, 500, 429): time.sleep(15); continue
            raise
        except urllib.error.URLError as e:
            last = e; time.sleep(15); continue
    raise last if last else RuntimeError("llm failed")


def extract_cmd(out):
    ans = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
    blocks = re.findall(r"```(?:bash|sh)?\s*(.*?)```", ans, re.DOTALL)
    if not blocks: blocks = re.findall(r"```(?:bash|sh)?\s*(.*?)```", out, re.DOTALL)
    for b in blocks:
        if b.strip(): return b.strip()
    return ""


PARSE = ("python3 -c \"import json,sys;o=open('/tmp/mo.txt',errors='replace').read();"
         "i=o.find('\\\"stats\\\"');st=o.rfind('{',0,i) if i>=0 else -1;"
         "d=(json.JSONDecoder().raw_decode(o[st:])[0] if st>=0 else {});"
         "print('PASS:'+'\\t'.join((p.get('fullTitle') or '').strip() for p in d.get('passes',[])));"
         "print('FAILN:'+str(len(d.get('failures',[]))))\"")


def run_tests(timeout=1500):
    cmd = ("npx mocha test/mocks/databasemock.js " + " ".join(test_files) +
           " --reporter json > /tmp/mo.txt 2>&1; " + PARSE)
    return sh(cmd, timeout, limit=30000)


setup = ("[ -d /wrk/base ] || { echo NO_BASE; exit 1; }; "
         "rm -rf /wrk/w && cp -r /wrk/base /wrk/w && "
         "ln -sfn /wrk/nm/" + IMGKEY + "/node_modules /wrk/w/node_modules && cd /wrk/w && "
         "git config --global --add safe.directory '*' >/dev/null 2>&1; "
         "sed -i 's/6379/" + str(RPORT) + "/g' config.json 2>/dev/null; "
         "rm -rf /tmp/rd" + str(RPORT) + " && mkdir -p /tmp/rd" + str(RPORT) + "; "
         "redis-server --port " + str(RPORT) + " --daemonize yes --dir /tmp/rd" + str(RPORT) + " --save '' >/dev/null 2>&1; sleep 2; "
         "redis-cli -p " + str(RPORT) + " ping >/dev/null 2>&1 || { echo REDIS_MORT; exit 1; }; "
         "n=$(git rev-list --count HEAD 2>/dev/null); "
         "[ \"$n\" = 1 ] && echo REVERT_OK || echo REVERT_FAIL_HISTORY_$n")
print("=== SETUP %s f2p=%d files=%s ===" % (TASK, len(f2p_titles), test_files), flush=True)
so = exec_raw(setup, timeout=900)
print(so[-200:], flush=True)
if "REVERT_OK" not in so:
    print("### %s ATT %s SETUP_FAIL ###" % (TASK, ATT), flush=True)
    raise SystemExit(0)

pre = run_tests()
_m = re.search(r"FAILN:(\d+)", pre)
NFAIL_PRE = int(_m.group(1)) if _m else 0
print("preexisting failures (unrelated to the fix): %d" % NFAIL_PRE, flush=True)
SYS = ("You are an expert Node.js engineer. The working copy is your current directory (NodeBB). There is a bug: some tests are failing. "
       "You act by emitting EXACTLY ONE bash command per turn inside a ```bash code block. "
       "Workflow: (1) read the failing test and the source it exercises (use `grep -rn` and `sed -n`), "
       "(2) EDIT the JS source (non-test) to fix the bug (heredoc `cat > file <<'EOF'`, `sed -i`, or `git apply`), "
       "(3) verify with `npx mocha test/mocks/databasemock.js <testfile> --grep '<title>'`. "
       "You MUST make at least one source (.js, non-test) edit. Do NOT modify anything under test/. "
       "Only after you edited a source file AND the tests pass, emit ```bash\nDONE\n```.")
seed = ("Failing test(s):\n" + "\n".join(f2p_titles[:8]) + "\nTest file(s): " + " ".join(test_files) +
        "\nFind why they fail and fix the source. First command:")
messages = [{"role": "system", "content": SYS}, {"role": "user", "content": seed}]
edited = False; empties = 0
for turn in range(MAXTURN):
    try: out = llm(messages)
    except Exception as e:
        print("LLM ERR: %s" % e, flush=True); break
    cmd = extract_cmd(out)
    if cmd == "DONE":
        if not edited:
            messages.append({"role": "assistant", "content": out, "reasoning_content": LAST_REASONING})
            messages.append({"role": "user", "content": "You have not edited any source file yet. Emit a bash command that edits the buggy .js source file."})
            continue
        break
    if not cmd:
        empties += 1
        if empties >= 3: break
        messages.append({"role": "assistant", "content": out, "reasoning_content": LAST_REASONING})
        messages.append({"role": "user", "content": "No bash command detected. Emit EXACTLY ONE ```bash block."})
        continue
    empties = 0
    if re.search(r"(>|>>|tee |<<|git apply|sed -i|patch )", cmd) and "test/" not in cmd: edited = True
    res = sh(cmd)
    # Old observations are compressed to head+tail. Without this a large turn budget fills
    # the context, and the prompt gets truncated exactly at the fresh part, which matters most.
    if KEEPFULL:
        _u = [i for i, m in enumerate(messages) if m["role"] == "user"][1:]   # skip the seed message
        for i in (_u[:-KEEPFULL] if len(_u) > KEEPFULL else []):
            c = messages[i]["content"]
            if len(c) > 700 and not c.startswith("[compressed]"):
                messages[i]["content"] = "[compressed] " + c[:350] + "\n...\n" + c[-350:]
    messages.append({"role": "assistant", "content": out, "reasoning_content": LAST_REASONING})
    nxt = "Output:\n" + res + ("\n\nYou have explored enough and NOT edited. Make your best edit to the buggy .js source NOW. Next command:"
                               if (not edited and turn >= 14) else "\n\nNext command:")
    messages.append({"role": "user", "content": nxt})

names = sh("git diff --name-only HEAD", limit=20000).split()
tampered = [f for f in names if f.startswith("test/")]
model_patch = sh("git diff HEAD -- . ':(exclude)test/*'", limit=400000)
rout = run_tests()
mp = re.search(r"PASS:(.*)", rout)
passed = set(x.strip() for x in (mp.group(1).split("\t") if mp else []) if x.strip())
mf = re.search(r"FAILN:(\d+)", rout)
nfail = int(mf.group(1)) if mf else 1
covered = [t for t in f2p_titles if t in passed]
resolved = (len(covered) == len(f2p_titles) and nfail <= NFAIL_PRE and not tampered
            and "[TIMEOUT]" not in rout and bool(model_patch.strip())
            and not model_patch.startswith("["))
print("\n=== REWARD === pass=%d/%d fail=%d (preexisting %d) patch_len=%d TESTS_MODIFIED=%s" % (
    len(covered), len(f2p_titles), nfail, NFAIL_PRE, len(model_patch), tampered), flush=True)
print("### %s ATT %s RESOLVED: %s (patch=%s) ###" % (TASK, ATT, resolved, bool(model_patch.strip())), flush=True)

OUTDIR.mkdir(parents=True, exist_ok=True)
traj = {"task": TASK, "attempt": ATT, "temp": TEMP, "resolved": resolved, "lang": "js", "mined": True,
        "repo": meta.get("repo", ""), "f2p": f2p_titles, "tampered_tests": tampered,
        "turns": len([m for m in messages if m["role"] == "assistant"]),
        "model_patch": model_patch if not model_patch.startswith("[") else "",
        "messages": messages}
traj["scaffold"] = os.path.basename(__file__)
traj["model"] = "GLM-5.3-Flash Q8_0 (full)"
traj["usage"] = dict(USAGE)
(OUTDIR / ("%s_att%s.json" % (TASK, ATT))).write_text(json.dumps(traj))
print("[traj %s_att%s resolved=%s]" % (TASK, ATT, resolved), flush=True)
