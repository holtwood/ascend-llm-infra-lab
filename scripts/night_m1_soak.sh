#!/bin/bash
# Night line M1 — long-run stability soak (~8h).
# Wave = c16/n512/in256/out96 (~70s at ~750 tok/s) + sampling.
# 400 waves ≈ 7.8h. Detects HBM/KV drift, latency creep, preemptions.
set -u
export NODE_TAG=m1
source /data/llm-infra-lab/scripts/night_common.sh
MODEL=/model/Qwen3.5-4B
PORT=8000
LINE_MIN=${LINE_MIN:-360}   # 实测波次 59s+7s -> 330 波/6h

log "M1 soak start (waves=$WAVES)"
serve $PORT "$MODEL" soak_base || exit 1
sample soak_t0 $PORT
T0=$(date +%s); END=$(( T0 + LINE_MIN * 60 )); i=0
while [ "$(date +%s)" -lt "$END" ]; do
  i=$((i + 1))
  bench $PORT "soak_w$(printf %04d "$i")" 16 512 256 96
  sample "soak_w$(printf %04d "$i")" $PORT
  [ $((i % 50)) -eq 0 ] && log "progress $i waves elapsed $(( ($(date +%s)-T0)/60 ))/${LINE_MIN}min"
done
log "M1 soak done: $i waves in $(( ($(date +%s)-T0)/60 ))min"
restore_default $PORT "$MODEL"
log "M1 restore done"
