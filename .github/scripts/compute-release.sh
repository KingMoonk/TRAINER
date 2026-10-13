#!/usr/bin/env bash
# Next version from the Conventional Commits since the last v* tag; writes release/next/bump/breaking to $GITHUB_OUTPUT.
#   feat -> minor, fix/perf/revert/refactor/style -> patch, anything else -> no release.
#   `type!:` or a BREAKING CHANGE: footer -> major, but flagged so the workflow waits for approval.
# FORCE_BUMP=patch|minor|major overrides the commits (manual dispatch counts as the approval).
set -euo pipefail
last=$(git describe --tags --abbrev=0 --match 'v[0-9]*' 2>/dev/null || true)
range=${last:+$last..}HEAD
cur=${last#v}; cur=${cur:-0.0.0}
bump=none breaking=false
while IFS= read -r subject; do
  [[ $subject =~ ^([a-z]+)(\([^\)]*\))?(!)?:\  ]] || continue
  [[ -z ${BASH_REMATCH[3]} ]] || breaking=true
  case ${BASH_REMATCH[1]} in
    feat) bump=minor ;;
    fix|perf|revert|refactor|style) [[ $bump != none ]] || bump=patch ;;
  esac
done < <(git log --no-merges --format=%s $range)
[[ -z $(git log --no-merges --format=%B $range | grep -E '^BREAKING[ -]CHANGE:' || true) ]] || breaking=true
[[ $breaking == false ]] || bump=major
if [[ ${FORCE_BUMP:-auto} != auto ]]; then bump=$FORCE_BUMP breaking=false; fi

IFS=. read -r major minor patch <<<"$cur"
case $bump in
  major) next=$((major + 1)).0.0 ;;
  minor) next=$major.$((minor + 1)).0 ;;
  patch) next=$major.$minor.$((patch + 1)) ;;
  none) next= ;;
esac
{ echo "release=$([[ $bump == none ]] && echo false || echo true)"; echo "next=$next"; echo "bump=$bump"; echo "breaking=$breaking"; echo "current=$cur"; } | tee -a "${GITHUB_OUTPUT:-/dev/null}"
