#!/usr/bin/env bash
# Start the local vLLM server used by the Duck harness (OpenAI-compatible, :1234).
# Usage: scripts/duck_vllm_serve.sh [UTIL=0.90] [MAX_NUM_SEQS=4] [LOG=path]
#   UTIL          --gpu-memory-utilization (lower it to co-host an ollama advisor model)
#   MAX_NUM_SEQS  --max-num-seqs
#   MODEL         model dir (default ~/models/qwen3.6-27b-awq)
# Prints the PID; waits until /health answers or 20 min pass. Kill it with `kill <pid>`
# (never `pkill -f "vllm serve"` from a shell whose own cmdline contains that string).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UTIL="${UTIL:-0.90}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"; PORT="${PORT:-1234}"
MODEL="${MODEL:-$HOME/models/qwen3.6-27b-awq}"
LOG="${LOG:-$ROOT/vendor/vllm_serve27.log}"
mkdir -p "$(dirname "$LOG")"
if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "a server already answers on :${PORT}; stop it first" >&2; exit 1
fi
cd "$ROOT"
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 nohup .venv-llm/bin/vllm serve "$MODEL" \
  --served-model-name local-qwen --port "$PORT" --max-model-len 32768 \
  --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAX_NUM_SEQS" --enable-prefix-caching \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice --reasoning-parser qwen3 \
  --additional-config '{"gdn_prefill_backend":"triton"}' >"$LOG" 2>&1 &
PID=$!
echo "vllm pid=$PID log=$LOG util=$UTIL max_num_seqs=$MAX_NUM_SEQS"
for i in $(seq 1 240); do
  if ! kill -0 "$PID" 2>/dev/null; then echo "server exited early; tail of log:"; tail -n 30 "$LOG"; exit 1; fi
  if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    grep -h "Available KV cache memory\|GPU KV cache size" "$LOG" | tail -2; echo "ready"; exit 0
  fi
  sleep 5
done
echo "timed out waiting for server" >&2; exit 1
