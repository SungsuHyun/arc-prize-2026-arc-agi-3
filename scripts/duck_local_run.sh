#!/usr/bin/env bash
# Run the Duck harness (harness/duck/ARC3-Inference) against the local vLLM server.
# Usage: scripts/duck_local_run.sh RUN_NAME [GAME=ls20-9607627b|"" for official 25] [MINUTES=20] [CONCURRENT=1]
# Advisor ensemble is enabled by exporting ARC3_ADVISORS (see inference/agent/advisors.py), e.g.
#   ARC3_ADVISORS='[{"name":"q9b-mech","model":"qwen3.5:9b","role":"mechanics"},
#                   {"name":"q9b-goal","model":"qwen3.5:9b","role":"goal"}]' scripts/duck_local_run.sh ens27-ls20 ls20-9607627b
# The run is over only when a line starting with "diagnostics:" appears in the log.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_NAME="${1:?run name}"; GAME="${2:-${GAME:-}}"
MINUTES="${MINUTES:-20}"; CONCURRENT="${CONCURRENT:-1}"; PASSES="${PASSES:-1}"
LOG="${LOG:-$ROOT/vendor/duck-runs/${RUN_NAME}.log}"
mkdir -p "$ROOT/vendor/duck-runs"
cd "$ROOT/harness/duck/ARC3-Inference"
if [ -n "$GAME" ]; then SEL=(--game "$GAME"); else SEL=(--include-tags official); fi
echo "run=$RUN_NAME games=${GAME:-official25} minutes=$MINUTES concurrent=$CONCURRENT advisors=${ARC3_ADVISORS:-off} log=$LOG"
exec uv run --no-sync inference-taaf-run "${SEL[@]}" \
  --environments-dir "$ROOT/environment_files" --run-name "$RUN_NAME" \
  --experiments-dir "$ROOT/vendor/duck-runs" --max-runtime-minutes "$MINUTES" \
  --n-passes "$PASSES" --concurrent-jobs "$CONCURRENT" --agent inference --model local-qwen \
  --analyzer-timeout 120 --deployment-target inline --deployment-wait --no-save-request-logs >"$LOG" 2>&1
