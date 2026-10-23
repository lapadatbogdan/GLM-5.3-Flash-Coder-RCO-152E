# Shared by math500 and aime25: Qwen2.5-Math / DeepSeek-R1 style prompt, Qwen2.5-Math grader in the sandbox.
import os
import core

PROMPT = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
SANDBOX = {"sif": None, "n": 8, "binds": ["%s:/tools:ro" % core.TOOLS]}
SCRIPT = "PYTHONPATH=/tools/pylib310:/tools/qwenmath python3 /tools/graders/math_grade.py"


def messages(item):
    return [{"role": "user", "content": PROMPT.format(problem=item["problem"])}]


def score(item, reply, sbx, data_name):
    res, out = core.run_json(sbx, SCRIPT, {"text": reply["content"], "gold": item["answer"], "data_name": data_name},
                             timeout=300)
    if res is None:
        return {"pass": False, "grader_error": out[-1500:]}
    return {"pass": bool(res["pass"]), "extracted": res["pred"], "gold_norm": res["gold_norm"],
            "math_verify": res["math_verify"], "metrics": {"math_verify": float(res["math_verify"] is True)}}


def gold_reply(item, turn):
    return "Some reasoning.\n\nThe final answer is $\\boxed{%s}$." % item["answer"]


def wrong_reply(item, turn):
    return "The final answer is $\\boxed{-98765}$."
