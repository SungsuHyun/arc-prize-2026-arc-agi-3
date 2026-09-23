#!/usr/bin/env bash
# Run the Duck harness (harness/duck/ARC3-Inference) against the local vLLM server via its Makefile,
# so every analyzer env var (LOCAL_ANALYZER_*, MULTIMODAL_*) comes from configs/local5090.json.
# Usage: scripts/duck_local_run.sh RUN_NAME [GAME_ID]      (no GAME_ID = the official 25 games)
#   MINUTES=20 CONCURRENT=1 PASSES=1 CONFIG=configs/local5090.json LOG=vendor/duck-runs/<RUN_NAME>.log
# Advisor ensemble: export ARC3_ADVISORS (see inference/agent/advisors.py), e.g.
#   ARC3_ADVISORS='[{"name":"q9b-mech","model":"qwen3.5:9b","role":"mechanics"}]' scripts/duck_local_run.sh ens27-ls20 ls20-9607627b
# The run is over only when a line starting with "diagnostics:" appears in the log.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_NAME="${1:?run name}"; GAME_ID="${2:-}"
MINUTES="${MINUTES:-20}"; CONCURRENT="${CONCURRENT:-1}"; PASSES="${PASSES:-1}"
CONFIG="${CONFIG:-configs/local5090.json}"
LOG="${LOG:-$ROOT/vendor/duck-runs/${RUN_NAME}.log}"
mkdir -p "$ROOT/vendor/duck-runs"
cd "$ROOT/harness/duck/ARC3-Inference"
if [ -n "$GAME_ID" ]; then SEL=(GAME="$GAME_ID" GAME_TAGS=); else SEL=(GAME= GAME_TAGS=official); fi
echo "run=$RUN_NAME games=${GAME_ID:-official25} minutes=$MINUTES concurrent=$CONCURRENT advisors=${ARC3_ADVISORS:-off} log=$LOG"
exec make run CONFIG_PATH="$CONFIG" "${SEL[@]}" RUN_NAME="$RUN_NAME" \
  ENVIRONMENTS_DIR="$ROOT/environment_files" EXPERIMENTS_DIR="$ROOT/vendor/duck-runs" \
  MAX_RUNTIME_MINUTES="$MINUTES" CONCURRENT_JOBS="$CONCURRENT" N_PASSES="$PASSES" >"$LOG" 2>&1
