#!/usr/bin/env python3
"""RFT generation on MINED Python instances (ansible/openlibrary/qutebrowser).
   WORK in a writable overlay (/wrk/w) => 0 inodes on scratch. Deps come from the image's site-packages (no drift).
   Setup: reset fix_commit + revert gold => failing state. Reward: all f2p node-ids PASSED, zero FAILED."""
import subprocess, json, urllib.request, urllib.error, re, os, time, pathlib
W = "/work/tasks"
SP = os.environ["TASK_DIR"]
INST = os.environ.get("INST", "genpy")
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
OUTDIR = pathlib.Path(os.environ.get("GEN_OUT", W + "/gen_traj_py"))

meta = json.load(open(SP + "/meta.json"))
FIXC = meta["fix_commit"]
PP = meta.get("pythonpath", ".")
f2p_list = [l.strip() for l in open(SP + "/f2p.txt") if l.strip()]
TESTFILES = sorted(set(t.split("::")[0] for t in f2p_list))

PYENV = "cd /wrk/w && export PYTHONNOUSERSITE=1 PYTHONPATH=/wrk/w/%s && " % PP   # see the note above


OUTLIM  = int(os.environ.get("OUTLIM",  "3000"))   # how many characters of a command's output the agent sees
MAXTURN = int(os.environ.get("MAXTURN", "200"))    # effectively no cap (was 30: 12 of the full model's 16 failures were stopped at turn 30)
KEEPFULL= int(os.environ.get("KEEPFULL", "0"))     # how many RECENT outputs stay whole (0 = no compression)


def exec_raw(cmd, timeout=300, limit=None):
    limit = OUTLIM if limit is None else limit
    try:
        r = subprocess.run(["singularity", "exec", f"instance://{INST}", "bash", "-lc", cmd],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, universal_newlines=True)
        return (r.stdout or "")[:limit]
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERR {e}]"


# 32 GB of virtual memory per command: a runaway script (seen 12 Oct: a 365 GB peak on the evaluation node,
# then OOM for the whole Cf144 job) fails on its own with MemoryError instead of killing the node with 16 agents.
ULIMIT_KB = int(os.environ.get("ULIMIT_KB", "33554432"))


def sh(cmd, timeout=180, limit=None):
    return exec_raw("ulimit -v %d; " % ULIMIT_KB + PYENV + cmd, timeout, limit)


# Thinking models (GLM-5.2) send their reasoning in the separate reasoning_content field,
# but it CONSUMES max_tokens. With a fixed 2500, the actual answer may not fit and content
# comes back empty -> the agent nudges -> empty targets in the training set (we cleaned 407
# such targets on 8 Sep). MAXTOK allows raising the cap without changing the default,
# so the student baseline stays reproducible.
MAXTOK = int(os.environ.get("MAXTOK", "16000"))
# The ceiling it may raise itself to; above it a generation stuck in a loop
# would fill the context and burn the turn.
TOKCEIL = int(os.environ.get("TOKCEIL", "32768"))


LAST_REASONING = ""


def llm(messages, max_tokens=None, retries=12):
    max_tokens = MAXTOK if max_tokens is None else max_tokens
    data = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": max_tokens, "temperature": TEMP}).encode()
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(VLLM, data=data, headers=HEADERS)
            _r = json.load(urllib.request.urlopen(req, timeout=3600))
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
            if e.code in (503, 502, 500, 429):
                time.sleep(15); continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e; time.sleep(15); continue
    raise last if last else RuntimeError("llm failed")


def extract_cmd(out):
    ans = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
    blocks = re.findall(r"```(?:bash|sh)?\s*(.*?)```", ans, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```(?:bash|sh)?\s*(.*?)```", out, re.DOTALL)
    for b in blocks:
        if b.strip():
            return b.strip()
    # Pruned/quantized models sometimes slip into GLM's native tool-call format: the right command, but
    # closed with </arg_value> instead of ``` (12 Oct: 0 cases on the full model, 2-7 failures on a pruned model).
    # An agent with native tool calling would have executed it, so we accept it too.
    for pat in (r"```(?:bash|sh)?\s*(.*?)</arg_value>", r"<arg_value>(.*?)</arg_value>", r"```(?:bash|sh)\s+(.*)$"):
        for src in (ans, out):
            m = re.search(pat, src, re.DOTALL)
            if m and m.group(1).strip():
                return m.group(1).strip()
    return ""


def run_tests(timeout=900):
    return sh("python3 -m pytest " + " ".join(TESTFILES) + " -p no:cacheprovider -v 2>&1 | tail -120", timeout)


def score(timeout=1200):
    # pytest ids may contain spaces -> match per line, not per field.
    # f2p names (from mining) contain no spaces, so keep only names without spaces.
    cmd = ("python3 -m pytest " + " ".join(TESTFILES) + " -p no:cacheprovider -v > /wrk/res.txt 2>&1; "
           "echo PASSNAMES:$(grep -E ' PASSED' /wrk/res.txt | sed -E 's/ PASSED.*$//' | grep -v ' ' | sort -u | tr '\\n' ' ');"
           "echo NFAIL:$(grep -cE ' (FAILED|ERROR)([ []|$)' /wrk/res.txt);"
           "echo TAIL:; tail -8 /wrk/res.txt")
    return sh(cmd, timeout, limit=20000)


setup = ("[ -d /wrk/base ] || { echo NO_BASE; exit 1; }; "
         "rm -rf /wrk/w && cp -r /wrk/base /wrk/w && cd /wrk/w && "
         "git config --global --add safe.directory '*' >/dev/null 2>&1; "
         "git reset --hard >/dev/null 2>&1; git clean -fdq 2>/dev/null; "
         "n=$(git rev-list --count HEAD 2>/dev/null); "
         "[ \"$n\" = 1 ] && echo REVERT_OK || echo REVERT_FAIL_HISTORY_$n")
print("=== SETUP %s (reset %s, revert gold) f2p=%d ===" % (TASK, FIXC[:8], len(f2p_list)), flush=True)
so = exec_raw(setup, timeout=900)
print(so[-200:], flush=True)
if "REVERT_OK" not in so:
    print("### %s ATT %s SETUP_FAIL ###" % (TASK, ATT), flush=True)
    raise SystemExit(0)

# GOLD_ONLY: instead of letting the agent search for the fix, apply the GOLD PATCH and score.
# Answers: is the instance solvable AT ALL? If the tests do not pass even with the correct solution,
# the instance is broken and does not belong in the benchmark - not even as a target for expensive teachers.
# Reuses EXACTLY the score() used in real runs, so the verdict uses the same criterion.
if os.environ.get("GOLD_ONLY") == "1":
    pre_g = score()
    nf_pre = int((re.search(r"NFAIL:(\d+)", pre_g) or re.match("(0)","0")).group(1))
    ap = sh("cd /wrk/w && git apply /wrk/gold.patch && echo GOLD_APPLIED || echo GOLD_APPLY_FAIL", 300)   # /wrk is in the overlay, visible in the clean room
    if "GOLD_APPLIED" not in ap:
        print("### %s GOLD_APPLY_FAIL ###" % TASK, flush=True); raise SystemExit(0)
    post = score()
    names = set((re.search(r"PASSNAMES:(.*)", post) or re.match("(?P<g>)","")).group(1).split()) if "PASSNAMES:" in post else set()
    nf_post = int((re.search(r"NFAIL:(\d+)", post) or re.match("(0)","0")).group(1))
    missing = [t for t in f2p_list if t not in names]
    verdict = "VALID" if not missing else "BROKEN"
    print("### GOLD %s: %s | f2p=%d passed=%d missing=%d | failures before=%d after=%d"
          % (TASK, verdict, len(f2p_list), len(f2p_list)-len(missing), len(missing), nf_pre, nf_post), flush=True)
    if missing: print("    fail even with gold: %s" % " ".join(missing[:5]), flush=True)
    raise SystemExit(0)

pre = score()
_m = re.search(r"NFAIL:(\d+)", pre)
NFAIL_PRE = int(_m.group(1)) if _m else 0
print("preexisting failures (unrelated to the fix): %d" % NFAIL_PRE, flush=True)
SYS = ("You are an expert Python engineer. The working copy is your current directory. There is a bug: some tests are failing. "
       "You act by emitting EXACTLY ONE bash command per turn inside a ```bash code block. "
       "Workflow: (1) read the failing test and the source it exercises (use `grep -rn` and `sed -n`), "
       "(2) EDIT the Python source (non-test) to fix the bug (heredoc `cat > file <<'EOF'`, `python3 - <<'EOF'`, or `git apply`), "
       "(3) verify with `python3 -m pytest <testfile> -p no:cacheprovider -q`. "
       "You MUST make at least one source (non-test .py) edit. Do NOT modify test files. "
       "Only after you edited a source file AND the tests pass, emit ```bash\nDONE\n```.")
seed = ("Failing tests:\n" + "\n".join(f2p_list[:20]) + "\nTest file(s): " + " ".join(TESTFILES) +
        "\nCurrent output (tail):\n" + pre[-int(os.environ.get("SEEDLIM","1200")):] + "\nFind why they fail and fix the source. First command:")
messages = [{"role": "system", "content": SYS}, {"role": "user", "content": seed}]
edited = False
empties = 0
for turn in range(MAXTURN):
    try:
        out = llm(messages)
    except Exception as e:
        print("LLM ERR: %s" % e, flush=True); break
    cmd = extract_cmd(out)
    if cmd == "DONE":
        if not edited:
            messages.append({"role": "assistant", "content": out, "reasoning_content": LAST_REASONING})
            messages.append({"role": "user", "content": "You have not edited any source file yet. Emit a bash command that edits the buggy source file."})
            continue
        break
    if not cmd:
        empties += 1
        if empties >= int(os.environ.get("MAXEMPTY", "3")):
            break
        messages.append({"role": "assistant", "content": out, "reasoning_content": LAST_REASONING})
        messages.append({"role": "user", "content": "No bash command detected. Emit EXACTLY ONE ```bash block."})
        continue
    empties = 0
    if re.search(r"(>|>>|tee |<<|git apply|sed -i|patch )", cmd) and not re.search(r"test_\w*\.py|_test\.py", cmd):
        edited = True
    res = sh(cmd)
    if KEEPFULL:
        _u = [i for i, m in enumerate(messages) if m["role"] == "user"][1:]   # skip the seed
        for i in _u[:-KEEPFULL] if len(_u) > KEEPFULL else []:
            c = messages[i]["content"]
            if len(c) > 700 and not c.startswith("[compressed]"):
                messages[i]["content"] = "[compressed] " + c[:350] + "\n...\n" + c[-350:]
    messages.append({"role": "assistant", "content": out, "reasoning_content": LAST_REASONING})
    nxt = "Output:\n" + res + ("\n\nYou have explored enough and NOT edited. Make your best edit to the buggy source NOW. Next command:"
                               if (not edited and turn >= MAXTURN // 2) else "\n\nNext command:")
    messages.append({"role": "user", "content": nxt})

# NEW files created by the model must be part of the patch: `git diff HEAD` ignores untracked files (12 Oct:
# openlibrary-8b68 passed the tests with a new directory and was counted as a failure, for every model)
sh("git add -N -A -- . ':(exclude)*__pycache__*' ':(exclude)*.pyc' 2>/dev/null; true", limit=2000)
names = sh("git diff --name-only HEAD", limit=20000).split()
tampered = [f for f in names if re.search(r"(^|/)tests?/|test_[^/]*\.py$|_test\.py$|conftest\.py$", f)]
model_patch = sh("git diff HEAD -- . ':(exclude)*test*'", limit=400000)
rout = score()
passed = set(x for x in re.search(r"PASSNAMES:(.*)", rout).group(1).split() if x) if "PASSNAMES:" in rout else set()
nfailed = int(re.search(r"NFAIL:(\d+)", rout).group(1)) if "NFAIL:" in rout else 1
covered = [t for t in f2p_list if t in passed]
resolved = (len(covered) == len(f2p_list) and nfailed <= NFAIL_PRE
            and "[TIMEOUT]" not in rout and not tampered and bool(model_patch.strip())
            and not model_patch.startswith("["))
print("\n=== REWARD === pass=%d/%d failed/err=%d (preexisting %d) patch_len=%d TESTS_MODIFIED=%s" % (
    len(covered), len(f2p_list), nfailed, NFAIL_PRE, len(model_patch), tampered), flush=True)
print(rout[-800:], flush=True)
print("### %s ATT %s RESOLVED: %s (patch=%s) ###" % (TASK, ATT, resolved, bool(model_patch.strip())), flush=True)

OUTDIR.mkdir(parents=True, exist_ok=True)
traj = {"task": TASK, "attempt": ATT, "temp": TEMP, "resolved": resolved, "lang": "python", "mined": True,
        "tampered_tests": tampered,
        "repo": meta.get("repo", ""), "f2p": f2p_list,
        "turns": len([m for m in messages if m["role"] == "assistant"]),
        "model_patch": model_patch if not model_patch.startswith("[") else "",
        "messages": messages}
traj["scaffold"] = os.path.basename(__file__)
traj["model"] = "GLM-5.3-Flash Q8_0 (full)"
traj["usage"] = dict(USAGE)
(OUTDIR / ("%s_att%s.json" % (TASK, ATT))).write_text(json.dumps(traj))
print("[traj %s_att%s resolved=%s]" % (TASK, ATT, resolved), flush=True)
