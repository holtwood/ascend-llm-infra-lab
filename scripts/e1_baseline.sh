#!/bin/bash
# E1 基线压测：固定输入/输出长度，扫描并发档位
# 用法: bash e1_baseline.sh <host:port> <model_name> <tag>
set -u
source /data/llm-infra-lab/scripts/env.sh

BASE=${1:-localhost:8000}
MODEL=${2:-qwen35-4b}
TAG=${3:-e1_$(date +%m%d_%H%M)}
OUT=/data/llm-infra-lab/results/e1
mkdir -p "$OUT"

IN_LEN=512
OUT_LEN=128
LEVELS="1 4 8 16 32 64"

for C in $LEVELS; do
  NP=$((C * 4)); [ $NP -lt 16 ] && NP=16; [ $NP -gt 256 ] && NP=256
  echo "=== concurrency=$C num_prompts=$NP ==="
  vllm bench serve \
    --endpoint /v1/chat/completions \
    --base-url "http://$BASE" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len $IN_LEN --random-output-len $OUT_LEN \
    --random-range-ratio 0.2 \
    --num-prompts $NP \
    --max-concurrency $C \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --save-detailed \
    --result-dir "$OUT" --result-filename "${TAG}_c${C}.json" \
    2>&1 | tail -40
done
echo "ALL DONE -> $OUT"
