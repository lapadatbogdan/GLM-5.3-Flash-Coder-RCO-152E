#!/bin/bash
# JS PREPARATION (instance with /app + the shared file system): builds /wrk/base = FAILING state with the git history DELETED,
# and copies node_modules into the overlay at /wrk/nm/<image key> (reused between tasks on the same image),
# because in the clean room /app is masked and a symlink to it would be dead.
set -u
SP=$1; IMGKEY=$2
FIXC=$(python3 -c "import json;print(json.load(open('$SP/meta.json'))['fix_commit'])")
# Node resolves modules ONLY through directories literally named "node_modules" -> keep the name
NM=/wrk/nm/$IMGKEY/node_modules
if [ ! -d "$NM" ]; then
  mkdir -p /wrk/nm/$IMGKEY
  cp -r /app/node_modules "$NM" || { echo PREP_NM_FAIL; exit 1; }
fi
echo "  node_modules: $(ls $NM | wc -l) packages"

rm -rf /wrk/base && mkdir -p /wrk/base
shopt -s dotglob 2>/dev/null
for d in /app/*; do b=$(basename "$d"); [ "$b" = node_modules ] && continue; cp -r "$d" /wrk/base/ 2>/dev/null; done
cd /wrk/base && git config --global --add safe.directory '*' >/dev/null 2>&1
git reset --hard "$FIXC" >/dev/null 2>&1 || { echo PREP_RESET_FAIL; exit 1; }
git clean -fdq 2>/dev/null
cp "$SP/gold.patch" /tmp/_gj.patch && git apply -R /tmp/_gj.patch 2>/dev/null || { echo PREP_REVERT_FAIL; exit 1; }
rm -f /tmp/_gj.patch
rm -rf .git
git init -q 2>/dev/null && git add -A >/dev/null 2>&1 \
  && git -c user.email=a@b -c user.name=rft commit -q -m base >/dev/null 2>&1 || { echo PREP_GITINIT_FAIL; exit 1; }
echo "PREP_OK commits=$(git rev-list --count HEAD)"
