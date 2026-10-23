# IFEval (541 prompts, google-research instruction_following_eval input_data.jsonl, pinned 3ef5211).
# The response (final content, reasoning excluded) is checked by the official evaluation_lib strict and
# loose functions inside the sandbox. pass = prompt-level strict; metrics: prompt-level strict/loose
# (per item) and instruction-level strict/loose (summed over instructions).
import core

NAME = "ifeval"
MAX_TOKENS = 32768
SAMPLES = 1
SANDBOX = {"sif": None, "n": 8, "binds": ["%s:/tools:ro" % core.TOOLS]}
SCRIPT = "NLTK_DATA=/tools/nltk_data PYTHONPATH=/tools/pylib310:/tools/ifeval python3 /tools/graders/ifeval_grade.py"


def messages(item):
    return [{"role": "user", "content": item["prompt"]}]


def score(item, reply, sbx):
    res, out = core.run_json(sbx, SCRIPT, {"key": item["key"], "prompt": item["prompt"],
                                           "instruction_id_list": item["instruction_id_list"],
                                           "kwargs": item["kwargs"], "response": reply["content"]}, timeout=300)
    if res is None:
        return {"pass": False, "grader_error": out[-1500:]}
    n = len(res["strict_list"])
    return {"pass": bool(res["strict_all"]), "strict_list": res["strict_list"], "loose_list": res["loose_list"],
            "metrics": {"prompt_strict": float(res["strict_all"]), "prompt_loose": float(res["loose_all"])},
            "metrics_w": {"inst_strict": [sum(res["strict_list"]), n], "inst_loose": [sum(res["loose_list"]), n]}}


_GPT4 = {}


def gold_reply(item, turn):
    """No gold responses exist for IFEval. For the harness check we return the GPT-4 response
    shipped with the official repo (data/input_response_data_gpt4_20231107_145030.jsonl); the check
    is that our per-item scores equal the official evaluation_main.py output on that file
    (tests/ifeval_official_check.sh), not that every item passes."""
    import json, os
    if not _GPT4:
        p = os.path.join(core.HERE, "raw", "gh", "ifeval", "data", "input_response_data_gpt4_20231107_145030.jsonl")
        for l in open(p):
            r = json.loads(l)
            _GPT4[r["prompt"]] = r["response"]
    return _GPT4.get(item["prompt"])


def wrong_reply(item, turn):
    return ""  # empty response fails every instruction in the official checker
