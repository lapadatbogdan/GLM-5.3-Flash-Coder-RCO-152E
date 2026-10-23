import json, re, subprocess, sys, urllib.request, tempfile
URL = sys.argv[1]; TAG = sys.argv[2]
def ask(msg, tools=None, max_tokens=12000):
    b = {"model": "q", "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95}
    if tools: b["tools"] = tools
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600))
    return r["choices"][0]["message"], r.get("usage", {})
def num(s): return re.sub(r"[^\d]", "", (s or "").strip().split("\n")[-1])
res = []
for name, q, want in [("mul_17x23", "What is 17*23? Reply with only the number.", "391"),
                      ("mul_4digit", "What is 1234*5678? Reply with only the number.", "7006652"),
                      ("fib20", "What is the 20th Fibonacci number, with F1=1 and F2=1? Reply with only the number.", "6765"),
                      ("capital", "What is the capital of Australia? Reply with one word.", "canberra")]:
    m, u = ask(q); c = (m.get("content") or "")
    ok = (want in c.lower()) if name == "capital" else (num(c) == want)
    res.append((name, ok, c.strip()[:60], u.get("completion_tokens")))
m, u = ask("Write a Python function is_prime(n) that returns True if n is prime. Reply with only one python code block.")
code = re.findall(r"```(?:python)?\n(.*?)```", m.get("content") or "", re.S)
ok = False
if code:
    t = code[0] + "\nassert [x for x in range(30) if is_prime(x)]==[2,3,5,7,11,13,17,19,23,29]\nassert not is_prime(1) and is_prime(7919)\nprint(\"OK\")\n"
    open("/work/glm53/smoke/code_%s.py" % TAG, "w").write(t)
    try:
        pr = subprocess.run(["singularity", "exec", "/work/glm53/vllm-rocm.sif", "python3", "-c", t], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        ok = pr.stdout.strip().endswith(b"OK")
        if not ok: print(TAG, "code stderr:", pr.stderr.decode()[-300:].replace(chr(10), " | "))
    except Exception as e: ok = False; print(TAG, "code exc", e)
res.append(("code_is_prime", ok, (code[0][:60] if code else "NO CODE"), u.get("completion_tokens")))
tools = [{"type": "function", "function": {"name": "bash", "description": "Execute a bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
m, u = ask("Use the bash tool to list the files in /tmp.", tools=tools, max_tokens=4000)
tc = m.get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "bash" and "ls" in tc[0]["function"]["arguments"]
res.append(("tool_call", ok, json.dumps(tc)[:60], u.get("completion_tokens")))
for r in res: print("%s %-14s %-5s tok=%-6s %s" % (TAG, r[0], "PASS" if r[1] else "FAIL", r[3], r[2].replace("\n", " ")))
print("%s SUMMARY %d/%d" % (TAG, sum(1 for r in res if r[1]), len(res)))
