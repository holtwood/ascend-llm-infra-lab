#!/bin/bash
# Night line D — quantization-vs-size matrix + ~7.5h soak (~8h total).
# Phase A: 27B-w8a8 / 8B-bf16 / 4B-bf16 x c{8,32,64} (in256/out96)
# Phase B: 4B default soak c32/n512 (~45s/wave) x 600
set -u
export NODE_TAG=d
source /data/llm-infra-lab/scripts/night_common.sh
PORT=8010
M4B=/data/models/Qwen3.5-4B
WAVES=${WAVES:-420}

log "D phase A: quant-vs-size matrix"
for M in "/model/Qwen3.8-27B-w8a8|w8a8_27b" "/model/Qwen3-8B|bf16_8b" "$M4B|bf16_4b" "/model/Qwen3-0.6B|bf16_0p6b"; do
  MP=${M%%|*}; TAG=${M##*|}
  [ -d "$MP" ] || { log "skip $TAG (no $MP)"; continue; }
  serve $PORT "$MP" "quant_$TAG" || { log "serve failed $TAG"; continue; }
  for C in 8 32 64; do
    bench $PORT "q_${TAG}_c${C}" "$C" 96 256 96
    sample "q_${TAG}_c${C}" $PORT
  done
done

log "D phase B: soak ($WAVES waves)"
serve $PORT "$M4B" soak_base || exit 1
T0=$(date +%s)
for i in $(seq 1 "$WAVES"); do
  bench $PORT "dsoak_w$(printf %04d "$i")" 32 512 256 96
  sample "dsoak_w$(printf %04d "$i")" $PORT
  [ $((i % 50)) -eq 0 ] && log "d soak $i/$WAVES elapsed $(( ($(date +%s)-T0)/60 ))min"
done

log "D all done"
restore_default $PORT "$M4B"
log "D restore done"
