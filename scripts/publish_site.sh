#!/usr/bin/env bash
# Publish experiments/site/ to the gh-pages branch (GitHub Pages).
# Each publish is a fresh orphan commit force-pushed — site history lives
# in main (results JSON), not in gh-pages.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

.venv/bin/python scripts/exp_summary.py
.venv/bin/python scripts/build_dashboard.py

BRANCH=gh-pages
TMP=$(mktemp -d)
trap 'git worktree remove --force "$TMP" 2>/dev/null || true; git branch -D $BRANCH 2>/dev/null || true' EXIT
git branch -D $BRANCH 2>/dev/null || true
git worktree add --orphan -b $BRANCH "$TMP" > /dev/null
cp -r experiments/site/. "$TMP"/
touch "$TMP/.nojekyll"
git -C "$TMP" add -A
git -C "$TMP" commit -q -m "Publish benchmark site $(date -u +%Y-%m-%d-%H%M)"
git -C "$TMP" push -qf origin $BRANCH
echo "Published to gh-pages: https://sungsuhyun.github.io/arc-prize-2026-arc-agi-3/"
