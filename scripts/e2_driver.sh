#!/bin/bash
# E2 驱动（在 M3 跑）：远程切 M1 的模型 -> 隧道轮询就绪 -> 本地开压
# 用法: bash e2_driver.sh <model_path> <name> <tag> [serve extra args]
set -u
M1="ssh -p 40010 root@223.92.35.244"
PATH_=$1; NAME=$2; TAG=$3; shift 3
source /data/llm-infra-lab/scripts/env.sh
OUT=/data/llm-infra-lab/results/e2; mkdir -p "$OUT"

SRV_EXTRA=""; [ $# -gt 0 ] && SRV_EXTRA=$(printf '%q ' "$@")
$M1 'for p in $(pgrep -f "bin[/]vllm serve"); do kill $p 2>/dev/null; done
for i in $(seq 1 60); do pgrep -f "bin[/]vllm serve" >/dev/null || break; sleep 3; done
for p in $(pgrep -f "bin[/]vllm serve"); do kill -9 $p; done; sleep 8
cd /data/llm-infra-lab && LOG=/data/llm-infra-lab/results/serve_'$TAG'.log bash scripts/serve.sh '"$PATH_ $NAME"' 8000 --max-model-len 8192 '"$SRV_EXTRA"

for i in $(seq 1 240); do
  curl -s --max-time 2 http://localhost:8001/v1/models 2>/dev/null | grep -q "$NAME" && break
  sleep 5
done
curl -s --max-time 3 http://localhost:8001/v1/models | grep -q "$NAME" || { echo "[$TAG] M1 serve FAILED"; exit 1; }
echo "[$TAG] M1 up -> bench"

for C in 8 32 64; do
  NP=$((C * 4)); [ $NP -gt 256 ] && NP=256
  vllm bench serve --endpoint /v1/completions --base-url http://localhost:8001 \
    --model "$NAME" --tokenizer /model/Qwen3.5-4B \
    --dataset-name random --random-input-len 512 --random-output-len 128 --random-range-ratio 0.2 \
    --num-prompts $NP --max-concurrency $C --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --save-detailed --result-dir "$OUT" --result-filename "e2_${TAG}_c${C}.json" \
    2>&1 | tail -25
done
echo "[$TAG] sweep done"
