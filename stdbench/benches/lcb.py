# LiveCodeBench code generation (release_v6, contest window in data/SOURCES_lcb.json), pass@1.
# Prompt: lcb_runner/prompts/code_generation.py (commit 28fef95e) for chat models (LMStyle.OpenAIChat /
# DeepSeekAPI): system = SYSTEM_MESSAGE_GENERIC, user = get_generic_question_template_answer().
# Extraction: lcb_runner/utils/extraction_utils.py extract_code (lines between the LAST two ``` lines of the
# final answer; empty if fewer than two fence lines). Grading: vendored lcb_runner evaluation in the
# no-network sandbox, 6 s per test, all public+private tests must pass. Python 3.6 (host).
import os, uuid

import core

NAME = "lcb"
MAX_TOKENS = 65536
SAMPLES = 1
LCB_DIR = os.path.join(core.TOOLS, "lcb")
TESTS_DIR = os.path.join(core.DATA, "lcb_tests")
GOLD_DIR = os.path.join(LCB_DIR, "gold")
SANDBOX = {"sif": None, "n": 8, "binds": ["%s:/tools/lcb:ro" % LCB_DIR, "%s:/lcbdata:ro" % TESTS_DIR]}
# 32 GB address-space cap per grading process tree (lcb_runner itself sets none; this only protects the node)
GRADE_SCRIPT = ("export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -v 33554432; "
                "python3 /tools/lcb/lcb_grade.py")

SYSTEM_MESSAGE_GENERIC = ("You are an expert Python programmer. You will be given a question (problem specification) "
                          "and will generate a correct Python program that matches the specification and passes all tests.")
FORMATTING_MESSAGE_WITH_STARTER_CODE = ("You will use the following starter code to write the solution to the problem "
                                        "and enclose your code within delimiters.")
FORMATTING_WITHOUT_STARTER_CODE = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not "
                                   "directly test on the sample inputs). Enclose your code within delimiters as follows. "
                                   "Ensure that when the python program runs, it reads the inputs, runs the algorithm and "
                                   "writes output to STDOUT.")


def user_prompt(item):
    # verbatim get_generic_question_template_answer
    prompt = "### Question:\n%s\n\n" % item["question_content"]
    if item["starter_code"]:
        prompt += "### Format: %s\n" % FORMATTING_MESSAGE_WITH_STARTER_CODE
        prompt += "```python\n%s\n```\n\n" % item["starter_code"]
    else:
        prompt += "### Format: %s\n" % FORMATTING_WITHOUT_STARTER_CODE
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def messages(item):
    return [{"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": user_prompt(item)}]


def extract_code(model_output):
    # verbatim lcb_runner extract_code, default (non CodeLLaMa / non base) branch
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1:indexlines[-1]])


def score(item, reply, sbx):
    code = extract_code(reply.get("content") or "")
    if not code.strip():
        # lcb_runner would still execute "" and fail every test; skip the sandbox round trip
        return {"pass": False, "extracted": "", "exec": "NO_CODE"}
    nonce = uuid.uuid4().hex
    res, out = core.run_json(sbx, GRADE_SCRIPT, {"qid": item["qid"], "code": code, "timeout": 6, "nonce": nonce},
                             timeout=max(900, 8 * item["n_tests"] + 120))
    if not res or res.get("nonce") != nonce:
        return {"pass": False, "extracted": code[-4000:], "exec": ("GRADER_NO_RESULT " + out[-1500:])}
    return {"pass": bool(res["pass"]), "extracted": code[-4000:], "n_run": res.get("n_run"),
            "n_tests": res.get("n_tests"), "grade_seconds": res.get("grade_seconds"),
            "exec": str(res.get("metadata"))[-1500:], "empty_results": res.get("empty_results")}


def _gold_code(item):
    p = os.path.join(GOLD_DIR, "%s.py" % item["qid"])
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return f.read()


def gold_reply(item, turn):
    # LCB ships no reference solutions; gold exists only for the hand-written subset in tools/lcb/gold/
    c = _gold_code(item)
    if c is None:
        return None
    return "Here is the solution.\n\n```python\n%s\n```\n" % c.rstrip("\n")


def wrong_reply(item, turn):
    if item["starter_code"]:
        body = item["starter_code"].rstrip() + "\n        return None\n"
        return "```python\n%s```\n" % body
    return "```python\nprint(-123456789)\n```\n"
