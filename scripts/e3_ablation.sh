#!/bin/bash
# E3 消融驱动：重启服务 -> 等待就绪 -> 跑 bench -> 归档
# 用法: bash e3_ablation.sh <tag> <bench_args> -- <extra serve args>
# 例:  bash e3_ablation.sh pc_on "--dataset-name prefix-repetition" -- --no-enable-prefix-caching
set -u
source /data/llm-infra-lab/scripts/env.sh
TAG=$1; BENCH=$2; shift 2
[ "${1:-}" = "--" ] && shift
EXTRA="$*"

MODEL=/model/Qwen3.5-4B; NAME=qwen35-4b; PORT=8000
OUT=/data/llm-infra-lab/results/e3; mkdir -p "$OUT"

echo "=== [$TAG] stopping old server ==="
for p in $(pgrep -f "bin/vllm serve"); do kill "$p" 2>/dev/null; done
sleep 8

echo "=== [$TAG] starting server: $EXTRA ==="
nohup vllm serve "$MODEL" --served-model-name "$NAME" --port $PORT \
  --max-model-len 8192 $EXTRA > "$OUT/serve_${TAG}.log" 2>&1 &
SVPID=$!

for i in $(seq 1 120); do
  curl -s --max-time 2 http://localhost:$PORT/v1/models 2>/dev/null | grep -q "$NAME" && break
  sleep 5
done
if ! curl -s --max-time 2 http://localhost:$PORT/v1/models | grep -q "$NAME"; then
  echo "[$TAG] SERVER FAILED, see $OUT/serve_${TAG}.log"; exit 1
fi
echo "=== [$TAG] server up, benching ==="

vllm bench serve --endpoint /v1/completions --base-url http://localhost:$PORT \
  --model "$NAME" --tokenizer "$MODEL" \
  $BENCH --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
  --save-result --save-detailed --result-dir "$OUT" --result-filename "e3_${TAG}.json" \
  2>&1 | tail -30
echo "=== [$TAG] done ==="
