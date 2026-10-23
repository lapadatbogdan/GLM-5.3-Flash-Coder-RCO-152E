# MATH-500 (HuggingFaceH4/MATH-500, 500 problems). Prompt + grading: see _mathcommon.py.
from benches import _mathcommon as mc

NAME = "math500"
MAX_TOKENS = 32768
SAMPLES = 1
SANDBOX = mc.SANDBOX
messages = mc.messages
gold_reply = mc.gold_reply
wrong_reply = mc.wrong_reply


def score(item, reply, sbx):
    return mc.score(item, reply, sbx, NAME)


def gold_reply(item, turn):  # noqa: F811  the dataset's full reference solution (ends in \boxed{...})
    return item["solution"]
