#!/usr/bin/env bash
# Submit a kernel version to the leaderboard as soon as the daily allowance resets (00:00 UTC). Usage: submit_when_allowed.sh <version> "<message>"
set -u
cd "$(dirname "$0")/.."
VER=${1:?version}; MSG=${2:-"rulebook kernel v$VER"}
LOG=${SUBMIT_LOG:-/tmp/submit_when_allowed.log}
export KAGGLE_API_TOKEN=$(cat .kaggle/access_token)
while true; do
    now=$(date -u +%s); next=$(date -u -d "tomorrow 00:00:30" +%s)
    allowed=$(.venv/bin/python - <<'PY'
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
try:
    print(api.competition_get_submission_limits("arc-prize-2026-arc-agi-3").numAllowedNow)
except Exception as e:
    print(0)
PY
)
    echo "$(date -u) allowed_now=$allowed" >> "$LOG"
    if [ "$allowed" != "0" ]; then
        out=$(.venv/bin/kaggle competitions submit -c arc-prize-2026-arc-agi-3 -k sungsuhyun/arc3-rulebook -v "$VER" -f submission.parquet -m "$MSG" 2>&1)
        echo "$(date -u) submit v$VER: $out" >> "$LOG"
        case "$out" in *"400"*|*Error*) sleep 600; continue;; esac
        exit 0
    fi
    sleep $(( next - now > 60 ? next - now : 60 ))
done
