# AIME 2025 I+II (MathArena/aime_2025, answers cross-checked with opencompass/AIME2025), 30 problems.
# Reported as avg@k over SAMPLES independent samples (temperature 1.0, top_p 0.95).
from benches import _mathcommon as mc

NAME = "aime25"
MAX_TOKENS = 65536
SAMPLES = 4
SANDBOX = mc.SANDBOX
messages = mc.messages
gold_reply = mc.gold_reply
wrong_reply = mc.wrong_reply


def score(item, reply, sbx):
    return mc.score(item, reply, sbx, NAME)
