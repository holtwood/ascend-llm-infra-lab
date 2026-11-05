#!/bin/bash
# E2 流水线（在 M1 上跑）：依次起不同量化变体并压测
# 由 M3 远程驱动: ssh M1 'bash /data/llm-infra-lab/scripts/e2_pipeline.sh'
set -u
cd /data/llm-infra-lab

wait_down() {  # 等旧服务完全退出（释放 HBM）
  for i in $(seq 1 60); do pgrep -f "vllm serve" >/dev/null || return 0; sleep 3; done
  for p in $(pgrep -f "vllm serve"); do kill -9 $p 2>/dev/null; done; sleep 10
}
wait_up() {
  for i in $(seq 1 200); do
    curl -s --max-time 2 localhost:8000/v1/models 2>/dev/null | grep -q "$1" && return 0
    sleep 5
  done
  return 1
}
run() {  # run <model_path> <name> <tag> [extra...]
  local path=$1 name=$2 tag=$3; shift 3
  wait_down
  LOG=/data/llm-infra-lab/results/serve_${tag}.log bash scripts/serve.sh "$path" "$name" 8000 --max-model-len 8192 "$@"
  wait_up "$name" || { echo "[$tag] FAILED"; return 1; }
  echo "[$tag] up, sweep via M3 tunnel..."
  sleep 5
}

run /model/Qwen3.8-27B-W8A16 qwen27b-w8a16 w8a16 --compilation-config '{"cudagraph_capture_sizes":[1,4,8,16,32,64]}'
run /model/Qwen3-8B qwen3-8b-bf16 bf16_8b
echo "E2 pipeline done"
