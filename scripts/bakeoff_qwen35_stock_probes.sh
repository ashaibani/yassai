#!/usr/bin/env bash
# Sequential stock assist probes for Qwen3.5 bake-off + MiniCPM5 base control.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/config/.env"
export YZMA_LIB="${YZMA_LIB:-$HOME/opt/llama}"
export PATH="$YZMA_LIB:$PATH"
mkdir -p eval-results

run_probe() {
  local model="$1" out="$2" port="$3" label="$4"
  echo "===== START $label $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  # kill any leftover on this port
  lsof -ti tcp:"$port" | xargs kill -9 2>/dev/null || true
  go run ./cmd/localprobe \
    -assist \
    -model "$model" \
    -tasks testdata/variant_tasks_golden.json \
    -threads 2 \
    -ngl 99 \
    -port "$port" \
    -reasoning off \
    -out "$out"
  echo "===== END $label $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
}

run_probe \
  models/qwen35/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf \
  eval-results/localprobe-qwen35-0.8b-stock-variants.json \
  18101 \
  qwen35-0.8b

run_probe \
  models/qwen35/Qwen_Qwen3.5-2B-Q4_K_M.gguf \
  eval-results/localprobe-qwen35-2b-stock-variants.json \
  18102 \
  qwen35-2b

run_probe \
  models/minicpm5/MiniCPM5-1B-base-Q4_K_M.gguf \
  eval-results/localprobe-minicpm5-1b-stock-assist-variants.json \
  18103 \
  minicpm5-1b-base-assist

echo "ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
