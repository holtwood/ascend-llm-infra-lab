#!/bin/bash
# 通用 vLLM 服务启动（NPU 节点）
# 用法: bash serve.sh <model_path> <served_name> <port> [extra vllm args...]
# 例:  bash serve.sh /model/Qwen3.5-4B qwen35-4b 8000 --max-model-len 8192
set -u
MODEL_PATH=$1; NAME=$2; PORT=$3; shift 3
source /data/llm-infra-lab/scripts/env.sh
LOG=${LOG:-/data/llm-infra-lab/results/serve_${NAME}.log}
mkdir -p "$(dirname "$LOG")"
nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$NAME" \
  --port "$PORT" "$@" \
  > "$LOG" 2>&1 &
echo "serve $NAME pid $! -> $LOG"
