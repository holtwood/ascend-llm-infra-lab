#!/bin/bash
# E3b chunked-prefill 消融（在 M3 本机跑）：扫 max-num-batched-tokens 档位
# 主臂: 长prefill负载 in2048/out128 c32，6档 {256,512,1024,2048,4096,8192}
# 对照臂: 短prefill decode重负载 in256/out512 c32，极端2档 {256,8192}
# 收尾恢复默认 serve（M3 是 E4 upstream 副本之一）
# 用法: bash e3b_chunked.sh
set -u
source /data/llm-infra-lab/scripts/env.sh
MODEL=/model/Qwen3.5-4B; NAME=qwen35-4b; PORT=8000
OUT=/data/llm-infra-lab/results/e3; mkdir -p "$OUT"

restart_serve() {  # $1=mnb 数值或 "default"  $2=log_tag
  local EXTRA="--max-num-batched-tokens $1"
  [ "$1" = "default" ] && EXTRA=""
  for p in $(pgrep -f "bin[/]vllm serve"); do kill "$p" 2>/dev/null; done
  for i in $(seq 1 80); do pgrep -f "bin[/]vllm serve" >/dev/null || break; sleep 3; done
  for p in $(pgrep -f "bin[/]vllm serve"); do kill -9 "$p" 2>/dev/null; done
  sleep 8
  nohup vllm serve "$MODEL" --served-model-name "$NAME" --port $PORT \
    --max-model-len 8192 $EXTRA > "$OUT/serve_e3b_$2.log" 2>&1 &
  for i in $(seq 1 150); do
    curl -s --max-time 2 http://localhost:$PORT/v1/models 2>/dev/null | grep -q "$NAME" && return 0
    sleep 4
  done
  return 1
}

run_bench() {  # $1=tag，余下为 vllm bench serve 参数
  local TAG=$1; shift
  vllm bench serve --endpoint /v1/completions --base-url http://localhost:$PORT \
    --model "$NAME" --tokenizer "$MODEL" "$@" --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --save-detailed --result-dir "$OUT" --result-filename "e3b_${TAG}.json" \
    2>&1 | grep -aE "Successful|throughput|Mean TTFT|Mean TPOT|P99 TTFT|P99 TPOT"
}

echo "===== E3b 主臂：长prefill in2048/out128 c32 ====="
for MNB in 256 512 1024 2048 4096 8192; do
  echo "--- mnb=$MNB ---"
  restart_serve $MNB mnb$MNB || { echo "SERVE FAIL mnb$MNB"; continue; }
  run_bench mnb$MNB --dataset-name random --random-input-len 2048 \
    --random-output-len 128 --random-range-ratio 0.2 \
    --num-prompts 128 --max-concurrency 32
done

echo "===== E3b 对照臂：短prefill in256/out512 c32 ====="
for MNB in 256 8192; do
  echo "--- ctl mnb=$MNB ---"
  restart_serve $MNB ctl$MNB || { echo "SERVE FAIL ctl$MNB"; continue; }
  run_bench ctl$MNB --dataset-name random --random-input-len 256 \
    --random-output-len 512 --random-range-ratio 0.2 \
    --num-prompts 128 --max-concurrency 32
done

echo "===== 恢复默认 serve ====="
restart_serve default restore && echo "default serve restored" || echo "RESTORE FAILED"
echo "E3b ALL DONE -> $OUT"
