#!/bin/bash
# PREPARATION (runs in an instance with access to /app and the shared file system):
# builds /wrk/base = the FAILING state, with the git history DELETED, so the model cannot read the fix.
set -u
SP=$1               # mined_go/<repo>/instance_<id>
FIXC=$(python3 -c "import json;print(json.load(open('$SP/meta.json'))['fix_commit'])")
# Clean EVERYTHING that accumulates, not only base. The overlay has finite space AND inodes;
# the Go cache and the agent's working copies exhaust them after a few dozen instances,
# and from then on the worker is dead for the rest of the run.
rm -rf /wrk/base /wrk/w /wrk/gocache /wrk/gopath 2>/dev/null
mkdir -p /wrk/base /wrk/gocache /wrk/gopath
_err=$(cp -r /app/. /wrk/base/ 2>&1)
if [ $? -ne 0 ]; then
  # The real error, not a generic label: without it space cannot be told apart from inodes.
  echo "PREP_COPY_FAIL reason=$(echo "$_err" | head -1) space=$(df -h /wrk | awk 'NR==2{print $4}') inodes=$(df -i /wrk | awk 'NR==2{print $4}')"
  exit 1
fi
cd /wrk/base && git config --global --add safe.directory '*' >/dev/null 2>&1
git reset --hard "$FIXC" >/dev/null 2>&1 || { echo PREP_RESET_FAIL; exit 1; }
git clean -fdq 2>/dev/null
cp "$SP/gold.patch" /tmp/_g.patch && git apply -R /tmp/_g.patch 2>/dev/null || { echo PREP_REVERT_FAIL; exit 1; }
rm -f /tmp/_g.patch
# ONLY for validation: leave the patch in /wrk, which is in the overlay and therefore visible to the agent too.
# In real runs it is NOT copied - the clean room must stay clean, otherwise the model
# could read the solution and we would get a 93%% success rate that means nothing.
[ "${GOLD_ONLY:-0}" = "1" ] && cp "$SP/gold.patch" /wrk/gold.patch

# DELETE the history: the fix can no longer be recovered from the repo
rm -rf .git
git init -q 2>/dev/null && git add -A >/dev/null 2>&1 \
  && git -c user.email=a@b -c user.name=rft commit -q -m base >/dev/null 2>&1 || { echo PREP_GITINIT_FAIL; exit 1; }
# check: the fix no longer appears anywhere in the history
echo "PREP_OK commits=$(git rev-list --count HEAD) log=$(git log --oneline | tr '\n' ' ')"
