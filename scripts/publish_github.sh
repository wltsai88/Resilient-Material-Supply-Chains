#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v gh >/dev/null || { echo 'Install GitHub CLI (gh), then run gh auth login.' >&2; exit 1; }
account="$(gh api user --jq .login)"
if [ "$account" != 'wltsai88' ]; then
    echo "Expected authenticated GitHub account wltsai88, got $account. Switch accounts before publishing." >&2
    exit 1
fi
if gh repo view wltsai88/Resilient-Material-Supply-Chains >/dev/null 2>&1; then
    echo 'Repository already exists; inspect it before choosing a push target. Nothing overwritten.' >&2
    exit 1
fi
if [ ! -d .git ]; then
    git init -b main
    git add .
    git commit -m 'Implement synthetic material policy compliance benchmark'
fi
# Research-in-progress repository starts private. Change visibility explicitly when ready.
gh repo create wltsai88/Resilient-Material-Supply-Chains --private \
    --description 'Resilient Material Supply Chains: synthetic policy-version compliance benchmark and numerical models' \
    --source . --remote origin --push
gh repo view wltsai88/Resilient-Material-Supply-Chains --json url,isPrivate
