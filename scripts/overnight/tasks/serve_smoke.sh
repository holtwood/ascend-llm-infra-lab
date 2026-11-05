#!/bin/bash
# W8A16 serve smoke — the last gap in the #21112 chain.
# Serves the int8 weight-only checkpoint, checks /v1/models + one generate.
set -u
MODEL=${W8A16_MODEL:-/data1/pytroch-7/models/Qwen3-0.6B-w8a16}
PORT=${PORT:-8002}
OUT=${OUT_DIR:-/tmp/overnight}/serve_smoke
mkdir -p "$OUT"
cd /data1/pytroch-7
python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" \
    --served-model-name m --port $PORT --max-model-len 4096 \
    --gpu-memory-utilization 0.85 --enforce-eager \
    > "$OUT/server.log" 2>&1 &
SPID=$!
echo "server pid=$SPID"
for i in $(seq 1 240); do
    sleep 5
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models || true)
    [ "$code" = "200" ] && break
    kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 "$OUT/server.log"; exit 1; }
done
echo "models_code=$code" | tee "$OUT/models.code"
curl -s http://127.0.0.1:$PORT/v1/models > "$OUT/models.json"
curl -s http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"m","prompt":"The capital of France is","max_tokens":32,"temperature":0}' \
    > "$OUT/gen.json"
python3 - "$OUT/gen.json" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
t=d.get("choices",[{}])[0].get("text","")
print("GEN_TEXT:",repr(t[:120]))
print("NONEMPTY:",bool(t.strip()))
EOF
kill -INT $SPID 2>/dev/null; sleep 10; kill -9 $SPID 2>/dev/null
echo "serve_smoke done"
