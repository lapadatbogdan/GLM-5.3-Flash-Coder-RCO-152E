#!/usr/bin/env python3
"""One attempt on a Harbor task (SWE-bench Pro V2 / converted v1 / DeepSWE), in two separate containers:

 AGENT PHASE (clean room): a user-owned copy of /app in a sparse ext3 overlay, .git deleted and recreated with
             a single commit, container --containall --no-home --net --network none. The LLM loop = mini-swe-agent
             (DefaultAgent, the official mini.yaml config with tool calls, or mini_textbased.yaml) runs on the
             host; only the bash commands go into the container (like host_mini_swe.py in the official Pro V2 tooling).
 GRADE PHASE (clean container): a NEW copy of /app from the untouched image, the model's patch applied, the task's
             official verifier (tests/test.sh) unchanged. The agent never sees the hidden tests.

Modes: agent (default) | gold (patch = the reference solution) | nop (empty patch: the verifier must fail).
The result is written atomically to OUT/<id>/att<k>.json; the agent phase leaves agent_att<k>.json, so a crash between
the phases only reruns grading. Infrastructure errors (dead LLM server, container that does not start) do NOT write
a final result, so a rerun redoes them and they are not counted as model failures.
"""
import argparse, json, os, pathlib, platform, re, shutil, subprocess, sys, time, uuid, traceback

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True, choices=["sbp2", "sbp1", "deepswe"])
ap.add_argument("--id", required=True)
ap.add_argument("--task-dir", required=True)
ap.add_argument("--sif", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--attempt", type=int, default=0)
ap.add_argument("--mode", default="agent", choices=["agent", "gold", "nop"])
ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
ap.add_argument("--wid", default="0")
ap.add_argument("--meta", default="{}", help="json with repo/lang/calibrated, copied into the result")
a = ap.parse_args()

SCR = pathlib.Path(os.environ.get("SWEPRO_SCRATCH", "/work/swe_scratch"))
TEMP = float(os.environ.get("TEMP", "0.95"))
TOPP = float(os.environ.get("TOPP", "1.0"))
WALL = int(os.environ.get("WALL", "21600"))           # 6 h per attempt (the Z.ai setting for DeepSWE)
CMD_TIMEOUT = int(os.environ.get("CMD_TIMEOUT", "600"))  # one agent command
STEP_LIMIT = int(os.environ.get("STEP_LIMIT", "0"))     # 0 = no turn cap (lesson learned: caps hide points)
AGENT_CFG = os.environ.get("AGENT_CFG", "mini.yaml")    # mini.yaml (tool calls, as in Pier/DeepSWE) | mini_textbased.yaml
MAXTOK = int(os.environ.get("MAXTOK", "0"))             # 0 = do not send max_tokens (the server decides)
CLEAN_ROOM = os.environ.get("CLEAN_ROOM", "1") == "1"
GRADE_TIMEOUT = int(os.environ.get("GRADE_TIMEOUT", "3600"))
OVL_MB = int(os.environ.get("OVL_MB", "81920"))

out_dir = pathlib.Path(a.out) / a.id
out_dir.mkdir(parents=True, exist_ok=True)
tag = "att%d" % a.attempt
final_f = out_dir / ("%s.json" % tag)
agent_f = out_dir / ("agent_%s.json" % tag)
task_dir = pathlib.Path(a.task_dir)
meta = json.loads(a.meta)
if final_f.exists():
    print("[skip] %s exists" % final_f); sys.exit(0)


def log(*x):
    print("[%s %s w%s] " % (time.strftime("%H:%M:%S"), a.id[:60], a.wid) + " ".join(str(y) for y in x), flush=True)


def run(cmd, timeout=None, check=False):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                       timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError("cmd failed rc=%d: %s\n%s" % (r.returncode, " ".join(cmd)[:300], r.stdout[-2000:]))
    return r


def write_json_atomic(path, obj):
    tmp = path.with_suffix(".tmp%s" % uuid.uuid4().hex[:6])
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


class InfraError(Exception):
    pass


class Box:
    """One work container: its own sparse ext3 overlay, /app = a user-owned copy of the image."""

    def __init__(self, kind, network):
        self.name = "swp_%s_%s_%s" % (kind, a.wid, uuid.uuid4().hex[:6])
        self.ovl = SCR / "ovl" / (self.name + ".img")
        self.wd = SCR / "wd" / self.name
        self.network = network
        self.repo = "/app"
        self.binds = []

    def create(self):
        self.ovl.parent.mkdir(parents=True, exist_ok=True)
        self.wd.mkdir(parents=True, exist_ok=True)
        run(["singularity", "overlay", "create", "--sparse", "--size", str(OVL_MB), str(self.ovl)], check=True, timeout=600)
        # The images are "opened" (chmod -R a+rwX when pulled), so through the overlay we can write anywhere as a
        # regular user, including /app and /root (the image's HOME, where go/cargo/nvm/pyenv live).
        r = run(["singularity", "exec", "--containall", "--no-home", "--workdir", str(self.wd), "--overlay", str(self.ovl), a.sif,
                 "bash", "-c", "src=/app; [ -d /app ] || src=/testbed; touch $src/.swp_w 2>/dev/null && touch /root/.swp_w 2>/dev/null "
                 "&& rm -f $src/.swp_w /root/.swp_w && echo SRC=$src"], timeout=1800)
        m = re.search(r"SRC=(\S+)", r.stdout)
        if r.returncode != 0 or not m:
            raise InfraError("image not writable (not an opened image?): " + r.stdout[-800:])
        self.repo = m.group(1)

    def start(self):
        # --no-mount /etc/localtime: otherwise the container inherits the host's time zone instead of the image's UTC,
        # and time zone tests (e.g. dateutil tzlocal) fail even with the correct solution (seen 15 Oct in the gold check).
        cmd = ["singularity", "instance", "start", "--containall", "--no-home", "--no-mount", "/etc/localtime",
               "--workdir", str(self.wd), "--overlay", str(self.ovl)]
        if not self.network:
            cmd += ["--net", "--network", "none"]
        for src, dst, opt in self.binds:
            cmd += ["-B", "%s:%s%s" % (src, dst, (":" + opt) if opt else "")]
        cmd += [a.sif, self.name]
        r = run(cmd, timeout=600)
        if r.returncode != 0:
            raise InfraError("instance start failed: " + r.stdout[-800:])

    def sh(self, command, timeout=600, env=None):
        # HOME cannot be overridden through --env (singularity refuses); we export it in the command. HOME=/root as in Docker.
        pre = "export HOME=/root; " + "".join("export %s=%s; " % (k, shq(str(v))) for k, v in (env or {}).items())
        cmd = ["singularity", "exec", "--pwd", self.repo, "instance://" + self.name,
               "bash", "-c", pre + "timeout -k 10 %d bash -c %s" % (timeout, shq(command))]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                               timeout=timeout + 60)
            return r.stdout, r.returncode
        except subprocess.TimeoutExpired as e:
            o = e.output.decode("utf-8", "replace") if isinstance(e.output, bytes) else (e.output or "")
            return o + "\n[timeout after %ds]" % timeout, -1

    def destroy(self):
        run(["singularity", "instance", "stop", self.name], timeout=300)
        try:
            self.ovl.unlink()
        except FileNotFoundError:
            pass
        shutil.rmtree(self.wd, ignore_errors=True)


def shq(s):
    return "'" + s.replace("'", "'\\''") + "'"


# ======================= AGENT PHASE =======================
def agent_phase():
    box = Box("ag", network=False)
    res = {"exit_status": None, "n_calls": 0, "clean_room": None}
    t0 = time.time()
    try:
        box.create(); box.start()
        # git identity + safe.directory for the agent (DeepSWE asks for a commit on a new branch from main)
        box.sh("git config --global user.email agent@example.com; git config --global user.name agent; "
               "git config --global --add safe.directory '*'; git config --global init.defaultBranch main", 60)
        if CLEAN_ROOM:
            o, rc = box.sh("rm -rf .git && git init -q && (git checkout -q -b main 2>/dev/null || true) && git add -A >/dev/null 2>&1 "
                           "&& git commit -q --no-verify -m base >/dev/null 2>&1; "
                           "echo BASE=$(git rev-parse HEAD) N=$(git rev-list --all --count) "
                           "BR=$(git branch --format='%(refname:short)' | tr '\\n' ,) TAGS=$(git tag | wc -l)", 1800)
        else:
            o, rc = box.sh("echo BASE=$(git rev-parse HEAD) N=$(git rev-list --all --count) TAGS=$(git tag | wc -l)", 600)
        m = re.search(r"BASE=([0-9a-f]{40}) N=(\d+)", o)
        if not m:
            raise InfraError("clean room failed: " + o[-500:])
        base, ncommits = m.group(1), int(m.group(2))
        res["clean_room"] = {"enabled": CLEAN_ROOM, "commits_visible": ncommits, "raw": o.strip()[-200:]}
        if CLEAN_ROOM and ncommits != 1:
            raise InfraError("clean room leak: %d commits visible" % ncommits)
        # network check: there must be no way out of the clean room
        o, _ = box.sh("(curl -s -m 5 -o /dev/null -w '%{http_code}' https://pypi.org || wget -q -T 5 -O /dev/null https://pypi.org && echo WGET_OK) 2>&1 | tail -1", 60)
        res["net_probe"] = o.strip()[-60:]
        if re.search(r"\b200\b|WGET_OK", o):
            raise InfraError("network reachable from agent sandbox: " + o[-100:])

        os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import yaml
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.config import get_config_path
        from minisweagent.exceptions import Submitted
        cfg = yaml.safe_load(get_config_path(AGENT_CFG).read_text())
        agent_cfg = dict(cfg.get("agent", {}))
        agent_cfg.update(step_limit=STEP_LIMIT, cost_limit=0.0, wall_time_limit_seconds=WALL,
                         output_path=out_dir / ("traj_%s.json" % tag))
        agent_cfg.pop("mode", None)
        env_vars = dict(cfg.get("environment", {}).get("env", {}))

        class SandboxEnv:
            """mini-swe-agent environment: every command = a new subshell in the container (as in the official one)."""
            def __init__(self):
                self.cwd, self.env, self.timeout = box.repo, env_vars, CMD_TIMEOUT

            def execute(self, action, cwd="", *, timeout=None):
                out, rc = box.sh(action.get("command", ""), timeout or self.timeout, self.env)
                output = {"output": out, "returncode": rc, "exception_info": ""}
                lines = out.lstrip().splitlines(keepends=True)
                if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and rc == 0:
                    sub = "".join(lines[1:])
                    raise Submitted({"role": "exit", "content": sub,
                                     "extra": {"exit_status": "Submitted", "submission": sub}})
                return output

            def get_template_vars(self, **kw):
                u = platform.uname()._asdict()
                return {"cwd": self.cwd, "env": self.env, "timeout": self.timeout, **u, **kw}

            def serialize(self):
                return {"info": {"config": {"environment": {"cwd": self.cwd, "timeout": self.timeout, "sandbox": "singularity-instance",
                                                            "network": "none", "clean_room": CLEAN_ROOM},
                                            "environment_type": "swe_attempt.SandboxEnv"}}}

        mk = dict(cfg.get("model", {}).get("model_kwargs", {}))
        mk.pop("reasoning_effort", None)   # the GLM template defaults to 'max' (the Z.ai recommendation for reproduction)
        mk.update(api_base=a.url, api_key="none", temperature=TEMP, top_p=TOPP, timeout=int(os.environ.get("LLM_TIMEOUT", "14400")))
        if MAXTOK:
            mk["max_tokens"] = MAXTOK
        if os.environ.get("REASONING_EFFORT"):   # test 21 Oct: explicit reasoning effort through the chat template (default xhigh)
            mk["extra_body"] = {"chat_template_kwargs": {"reasoning_effort": os.environ["REASONING_EFFORT"]}}
        mcfg = {k: v for k, v in cfg.get("model", {}).items() if k not in ("model_kwargs", "model_class", "model_name")}
        if "textbased" in AGENT_CFG:
            from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel as M
        else:
            from minisweagent.models.litellm_model import LitellmModel as M
        model = M(model_name="openai/glm", model_kwargs=mk, cost_tracking="ignore_errors", **mcfg)
        agent = DefaultAgent(model, SandboxEnv(), **agent_cfg)
        instruction = (task_dir / "instruction.md").read_text()
        try:
            r = agent.run(instruction)
            res["exit_status"] = r.get("exit_status")
        except Exception as e:  # ContextWindowExceeded, RepeatedFormatError etc.: the patch so far is graded anyway
            res["exit_status"] = type(e).__name__
            res["exception"] = str(e)[-1500:]
            infra = type(e).__name__ in ("APIConnectionError", "ServiceUnavailableError", "InternalServerError",
                                         "Timeout", "APIError", "BadGatewayError")
            if infra:
                raise InfraError("LLM server: %s %s" % (type(e).__name__, str(e)[-300:]))
        res["n_calls"] = agent.n_calls
        res["n_steps_assistant"] = sum(1 for m_ in agent.messages if m_.get("role") == "assistant")
        u = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        for m_ in agent.messages:
            rr = (m_.get("extra") or {}).get("response") or {}
            us = rr.get("usage") or {} if isinstance(rr, dict) else {}
            u["prompt_tokens"] += us.get("prompt_tokens") or 0
            u["completion_tokens"] += us.get("completion_tokens") or 0
            u["reasoning_tokens"] += ((us.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        res["usage"] = u
        res["last_prompt_tokens"] = max([((m_.get("extra") or {}).get("response") or {}).get("usage", {}).get("prompt_tokens", 0) or 0
                                         for m_ in agent.messages if isinstance((m_.get("extra") or {}).get("response"), dict)] or [0])
        # Patch collection. Pro: everything changed vs base (commits + uncommitted, like locked_mini_swe.py).
        # DeepSWE: officially only what is COMMITTED (git diff base HEAD, from [[verifier.collect]]); uncommitted changes are kept separately.
        box.sh("rm -f .git/index.lock", 60)
        if a.bench == "deepswe":
            p, _ = box.sh("git diff --binary %s HEAD" % base, 600)
            pu, _ = box.sh("git add -A >/dev/null 2>&1; git diff --cached --binary %s" % base, 600)
            (out_dir / ("model_uncommitted_%s.patch" % tag)).write_text(pu)
        else:
            p, _ = box.sh("git add -A >/dev/null 2>&1; git diff --cached --binary %s" % base, 600)
        res["patch"] = p
    finally:
        box.destroy()
    res["wall_agent_s"] = round(time.time() - t0)
    return res


# ======================= GRADE PHASE =======================
def grade_phase(patch):
    vdir = out_dir / ("verifier_%s" % tag); adir = out_dir / ("artifacts_%s" % tag)
    shutil.rmtree(vdir, ignore_errors=True); shutil.rmtree(adir, ignore_errors=True)
    vdir.mkdir(); adir.mkdir()
    (adir / "model.patch").write_text(patch)
    # Pro V2: the verifier has network (a few Go tests fetch modules); DeepSWE: no network (task.toml); v1: as V2.
    box = Box("gr", network=(a.bench != "deepswe"))
    box.binds = [(str(task_dir / "tests"), "/tests", "ro"), (str(vdir), "/logs/verifier", ""), (str(adir), "/logs/artifacts", "")]
    t0 = time.time()
    try:
        box.create(); box.start()
        box.sh("git config --global --add safe.directory '*'", 60)
        if a.bench == "sbp2":
            # like the official patch_replay.py: apply, then --3way, then patch --fuzz=3
            cmd = ("if [ -s /logs/artifacts/model.patch ]; then git apply --verbose /logs/artifacts/model.patch "
                   "|| git apply --3way /logs/artifacts/model.patch || patch --fuzz=3 -p1 -i /logs/artifacts/model.patch "
                   "|| echo APPLY_FAILED; fi; bash /tests/test.sh")
        else:
            cmd = "bash /tests/test.sh"
        o, rc = box.sh(cmd, GRADE_TIMEOUT)
    finally:
        box.destroy()
    (out_dir / ("grade_%s.log" % tag)).write_text(o[-200000:])
    reward = None
    if (vdir / "reward.json").exists():
        try:
            j = json.loads((vdir / "reward.json").read_text())
            reward = float(j.get("reward", j.get("score", 0)))
        except Exception:
            reward = None
    if reward is None and (vdir / "reward.txt").exists():
        try:
            reward = float((vdir / "reward.txt").read_text().strip() or "0")
        except Exception:
            reward = None
    return {"reward": reward, "resolved": bool(reward is not None and reward >= 1.0), "grade_rc": rc,
            "apply_failed": "APPLY_FAILED" in o or "APPLY_FAIL" in o, "wall_grade_s": round(time.time() - t0),
            "grade_tail": o[-1500:]}


def main():
    res = {"id": a.id, "bench": a.bench, "attempt": a.attempt, "mode": a.mode, **meta,
           "temp": TEMP, "top_p": TOPP, "agent_cfg": AGENT_CFG, "wall_limit_s": WALL, "step_limit": STEP_LIMIT,
           "model": os.environ.get("MODEL_TAG", ""), "job": os.environ.get("SLURM_JOB_ID", ""),
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        if a.mode == "agent":
            if agent_f.exists():
                ag = json.loads(agent_f.read_text()); log("rerunning grading only")
            else:
                ag = agent_phase()
                write_json_atomic(agent_f, ag)
                log("agent done: exit=%s calls=%s patch=%dB %ds" % (ag["exit_status"], ag["n_calls"], len(ag["patch"]), ag["wall_agent_s"]))
            patch = ag["patch"]
            res.update({k: v for k, v in ag.items() if k != "patch"})
        elif a.mode == "gold":
            sol = task_dir / "solution"
            patch = next((p.read_text() for p in [sol / "gold_patch.diff", sol / "solution.patch"] if p.exists()), "")
        else:
            patch = ""
        (out_dir / ("model_%s.patch" % tag)).write_text(patch)
        res["patch_bytes"] = len(patch)
        g = grade_phase(patch)
        res.update(g)
        res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json_atomic(final_f, res)
        log("RESULT resolved=%s reward=%s" % (res["resolved"], res["reward"]))
    except InfraError as e:
        log("INFRA_ERROR (no result written, will be retried):", str(e)[-600:])

        (out_dir / ("infra_%s.log" % tag)).write_text(str(e) + "\n" + traceback.format_exc())
        sys.exit(3)


main()
