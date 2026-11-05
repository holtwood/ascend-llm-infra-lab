#!/bin/bash
# Night line M2 — performance grids + mixed-length soak (~8h).
# Phase A: 3D grid c{8,16,32,64,128} x in{256,1024,4096} x out{64,256} (30 pts)
# Phase B: max_num_batched_tokens {512,2048,8192} x c{16,64}
# Phase C: output-length sweep out{32,128,512,1024} x c{16,64}
# Phase D: mixed-length soak (long/short mix) ~5h
set -u
export NODE_TAG=m2
source /data/llm-infra-lab/scripts/night_common.sh
MODEL=/data/models/Qwen3.5-4B
PORT=8000
LINE_MIN=${LINE_MIN:-360}

log "M2 phase A: 3D grid"
serve $PORT "$MODEL" grid_base || exit 1
for C in 8 16 32 64 128; do
  for IN in 256 1024 4096; do
    for OUT in 64 256; do
      N=$(( C >= 64 ? 128 : 64 ))
      bench $PORT "g_c${C}_i${IN}_o${OUT}" "$C" "$N" "$IN" "$OUT"
    done
  done
done

log "M2 phase B: mnb sweep"
for MNB in 512 2048 8192; do
  serve $PORT "$MODEL" "mnb${MNB}" --max-num-batched-tokens "$MNB" || continue
  for C in 16 64; do bench $PORT "mnb${MNB}_c${C}" "$C" 128 256 96; done
done

log "M2 phase C: output-length sweep"
serve $PORT "$MODEL" outsweep_base || { restore_default $PORT "$MODEL"; exit 0; }
for C in 16 64; do
  for OUT in 32 128 512 1024; do
    bench $PORT "out${OUT}_c${C}" "$C" 64 256 "$OUT"
  done
done

log "M2 phase D: mixed-length soak (until +${LINE_MIN}min)"
T0=$(date +%s); END=$(( T0 + LINE_MIN * 60 )); i=0
while [ "$(date +%s)" -lt "$END" ]; do
  i=$((i + 1))
  $PYBIN scripts/e13_mixedjob.py >/dev/null 2>&1
  echo "{\"tag\":\"mix_$(printf %04d "$i")\",\"ts\":$(date +%s)}" >> "$RES/mix_runs.jsonl"
  [ $((i % 100)) -eq 0 ] && log "mix soak $i elapsed $(( ($(date +%s)-T0)/60 ))/${LINE_MIN}min"
done

log "M2 all done"
restore_default $PORT "$MODEL"
log "M2 restore done"
