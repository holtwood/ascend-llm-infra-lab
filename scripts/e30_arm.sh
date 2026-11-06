#!/bin/bash
# E30 arm: restart vLLM with a capture-size budget, measure startup + 3 loads.
# Usage: e30_arm.sh PORT MODEL BUDGET TAG
set -u
PORT=$1; MODEL=$2; BUDGET=$3; TAG=$4
RES=/data/llm-infra-lab/results/e30
mkdir -p "$RES"
source /data/llm-infra-lab/scripts/env.sh 2>/dev/null
cd /data/llm-infra-lab || exit 1
PYBIN=/usr/local/python3.12.13/bin/python3

for p in $(pgrep -f "bin[/]vllm serve"); do kill "$p" 2>/dev/null; done
for i in $(seq 1 40); do pgrep -f "bin[/]vllm serve" >/dev/null || break; sleep 3; done
for p in $(pgrep -f "bin[/]vllm serve"); do kill -9 "$p" 2>/dev/null; done
sleep 5

T0=$(date +%s)
if [ "$BUDGET" = "0" ]; then
  nohup vllm serve "$MODEL" --served-model-name qwen35-4b --port "$PORT" \
    --max-model-len 8192 > "$RES/serve_${TAG}.log" 2>&1 &
else
  VLLM_ASCEND_CAPTURE_BUDGET=$BUDGET nohup vllm serve "$MODEL" \
    --served-model-name qwen35-4b --port "$PORT" --max-model-len 8192 \
    > "$RES/serve_${TAG}.log" 2>&1 &
fi

code=""
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 3
done
READY=$(( $(date +%s) - T0 ))
[ "$code" = "200" ] || { echo "{\"tag\":\"$TAG\",\"error\":\"serve not ready\"}"; exit 1; }

CAP=$(grep -oE "E30 capture budget=[0-9]+: pruned [0-9]+ -> [0-9]+ sizes \[[^]]*\]" "$RES/serve_${TAG}.log" | tail -1)
[ -z "$CAP" ] && CAP=$(grep -oE "Capturing CUDA graphs \(decode, [A-Z]+\): [0-9]+%" "$RES/serve_${TAG}.log" | tail -1)
echo "{\"tag\":\"$TAG\",\"budget\":$BUDGET,\"startup_s\":$READY,\"capture\":\"${CAP:-none}\"}" >> "$RES/startup.jsonl"

for C in 8 32 128; do
  N=$(( C >= 128 ? 128 : 64 ))
  $PYBIN scripts/bench_openai.py --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/chat/completions --model qwen35-4b --concurrency "$C" \
    --num-requests "$N" --input-tokens 256 --output-tokens 96 \
    --out "$RES/${TAG}_c${C}.json" >/dev/null 2>&1
  $PYBIN - "$RES/${TAG}_c${C}.json" "$TAG" "$C" <<'PY' >> "$RES/bench.jsonl"
import json,sys
d=json.load(open(sys.argv[1]))["summary"]
print(json.dumps({"tag":sys.argv[2],"c":int(sys.argv[3]),"tok_s":round(d["output_tok_per_s"],1),
 "ttft":round(d["ttft_mean_ms"],1),"tpot":round(d["tpot_mean_ms"],1),
 "ttft_p99":round(d["ttft_p99_ms"],1),"ok":d["completed"],"err":d["errors"]}))
PY
done
echo "arm $TAG done (startup ${READY}s)"
