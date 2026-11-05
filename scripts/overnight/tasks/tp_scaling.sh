#!/bin/bash
# TP scaling bench — first multi-card serving measurement on our side.
# Serves Qwen3-0.6B bf16 at tp=1,2,4, runs bench_min c=16, records JSON.
set -u
MODEL=${BF16_MODEL:-/data1/pytroch-7/models/Qwen3-0.6B}
OUT=${OUT_DIR:-/tmp/overnight}/tp_scaling
TASKS=${TASKS_DIR:-/data1/pytroch-7/overnight/tasks}
mkdir -p "$OUT"
cd /data1/pytroch-7
for tp in 1 2 4; do
    port=$((8100 + tp))
    echo "=== tp=$tp port=$port ==="
    ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((tp-1))) \
    python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" \
        --served-model-name m --port $port --max-model-len 4096 \
        --gpu-memory-utilization 0.85 --enforce-eager -tp $tp \
        > "$OUT/server_tp$tp.log" 2>&1 &
    SPID=$!
    ready=0
    for i in $(seq 1 240); do
        sleep 5
        code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/v1/models || true)
        if [ "$code" = "200" ]; then ready=1; break; fi
        kill -0 $SPID 2>/dev/null || { echo "tp$tp server died"; tail -20 "$OUT/server_tp$tp.log"; break; }
    done
    if [ "$ready" = "1" ]; then
        python3 "$TASKS/bench_min.py" --port $port --c 16 --n 64 --max-tokens 64 \
            --out "$OUT/bench_tp$tp.json"
    fi
    kill -INT $SPID 2>/dev/null; sleep 15; kill -9 $SPID 2>/dev/null; sleep 15
done
echo "tp_scaling done"
