#!/bin/bash
# Night line M1 — long-run stability soak (~8h).
# Wave = c16/n512/in256/out96 (~70s at ~750 tok/s) + sampling.
# 400 waves ≈ 7.8h. Detects HBM/KV drift, latency creep, preemptions.
set -u
export NODE_TAG=m1
source /data/llm-infra-lab/scripts/night_common.sh
MODEL=/model/Qwen3.5-4B
PORT=8000
WAVES=${WAVES:-300}

log "M1 soak start (waves=$WAVES)"
serve $PORT "$MODEL" soak_base || exit 1
sample soak_t0 $PORT
T0=$(date +%s)
for i in $(seq 1 "$WAVES"); do
  bench $PORT "soak_w$(printf %04d "$i")" 16 512 256 96
  sample "soak_w$(printf %04d "$i")" $PORT
  [ $((i % 50)) -eq 0 ] && log "progress $i/$WAVES elapsed $(( ($(date +%s)-T0)/60 ))min"
done
log "M1 soak done: $WAVES waves in $(( ($(date +%s)-T0)/60 ))min"
restore_default $PORT "$MODEL"
log "M1 restore done"
