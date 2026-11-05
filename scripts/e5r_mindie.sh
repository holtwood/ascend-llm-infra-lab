#!/bin/bash
# E5R: MindIE 2.3.0 vs vLLM-Ascend, same model (Qwen3-8B bf16), same load shape.
# Runs INSIDE the mindie container (host 117.131.156.71:2507 via jump).
# Usage: bash e5r_mindie.sh   (expects bench_openai.py in same dir)
set -u
DIR=/data1/mindie-7
RES=$DIR/results_e5r
mkdir -p "$RES"
BASE=http://127.0.0.1:1025

echo "[e5r] waiting for mindie readiness..."
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 5
done
[ "$code" = "200" ] || { echo "[e5r] service never became ready"; exit 1; }
curl -s "$BASE/v1/models" | head -c 300; echo

for C in 8 16 32 64; do
  N=$((C * 8))
  echo "[e5r] c=$C n=$N"
  python3 "$DIR/bench_openai.py" --base-url "$BASE" --model qwen3 \
    --concurrency "$C" --num-requests "$N" \
    --input-tokens 512 --output-tokens 128 \
    --out "$RES/mindie_c${C}.json"
done
echo "[e5r] done -> $RES"
