#!/bin/bash
# Per-node experiment runner: restart vLLM with extra args from a FILE
# (avoids quoting hell through ssh layers), wait ready, run bench, write JSON.
#
# Usage: node_serve_bench.sh PORT EXTRA_ARGS_FILE OUT_JSON CONC N IN OUT [MAXLEN] [MODEL]
set -u
PORT=$1; EXTRA_FILE=$2; OUT=$3; C=$4; N=$5; IN=$6; O=$7
MAXLEN=${8:-8192}
MODEL=${9:-/model/Qwen3.5-4B}
source /data/llm-infra-lab/scripts/env.sh 2>/dev/null
cd /data/llm-infra-lab || exit 1

EXTRA=""
[ -f "$EXTRA_FILE" ] && EXTRA=$(cat "$EXTRA_FILE")

for p in $(pgrep -f "bin[/]vllm serve"); do kill "$p" 2>/dev/null; done
for i in $(seq 1 40); do pgrep -f "bin[/]vllm serve" >/dev/null || break; sleep 3; done
for p in $(pgrep -f "bin[/]vllm serve"); do kill -9 "$p" 2>/dev/null; done
sleep 5

# shellcheck disable=SC2086
nohup vllm serve "$MODEL" --served-model-name qwen35-4b --port "$PORT" \
  --max-model-len "$MAXLEN" $EXTRA > "${OUT%.json}.serve.log" 2>&1 &
code=""
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 5
done
[ "$code" = "200" ] || { echo "{\"error\":\"serve not ready\"}"; exit 1; }

/usr/local/python3.12.13/bin/python3 scripts/bench_openai.py \
  --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions \
  --model qwen35-4b --concurrency "$C" --num-requests "$N" \
  --input-tokens "$IN" --output-tokens "$O" --out "$OUT" 2>&1 | tail -3
