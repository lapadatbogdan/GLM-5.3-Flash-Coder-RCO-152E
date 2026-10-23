#!/usr/bin/env python3
"""reap_score.py with the DSA fix (heal/dsa_fix.py) installed; reap_score.py unchanged.
Run with --calib-dir calib53/v2 (the corrected v2 reference) and --keeps 144,120,96."""
import os, sys, runpy
P = "/work/glm53"
for p in (P + "/heal", P + "/rco", P + "/rco/src"):
    sys.path.insert(0, p)
import dsa_fix
dsa_fix.install_long_context_patches(256, 8)
sys.argv = [P + "/rco/reap_score.py"] + sys.argv[1:]
runpy.run_path(P + "/rco/reap_score.py", run_name="__main__")
