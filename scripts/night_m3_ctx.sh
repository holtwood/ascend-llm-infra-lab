#!/bin/bash
# Night line M3 — long-context grids + sleep/wake fatigue + ctx soak (~8h).
# Phase A: ctx{4k,16k,32k,63k} x c{1,2,4,8} (maxlen 81920)
# Phase B: 300 sleep2/wake cycles (latency drift)
# Phase C: admission wall at ctx16k, c{8,16,32}
# Phase D: long-ctx soak ctx16k c4 (~4.5h)
set -u
export NODE_TAG=m3
source /data/llm-infra-lab/scripts/night_common.sh
MODEL=/model/Qwen3.5-4B
PORT=8000
CYCLES=${CYCLES:-300}
SOAK_WAVES=${SOAK_WAVES:-700}

log "M3 phase A: ctx grid"
MAXLEN=81920 serve $PORT "$MODEL" ctx_base || exit 1
for CTX in 4000 16000 32000 63000; do
  for C in 1 2 4 8; do
    bench $PORT "ctx${CTX}_c${C}" "$C" 8 "$CTX" 64
    sample "ctx${CTX}_c${C}" $PORT
  done
done

log "M3 phase B: sleep/wake fatigue ($CYCLES cycles)"
kill_vllm
nohup env VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" --served-model-name qwen35-4b \
  --port $PORT --max-model-len 8192 --enable-sleep-mode --gpu-memory-utilization 0.5 \
  --additional-config '{"rl_config":{"enabled":true}}' > "$RES/serve_sleep.log" 2>&1 &
for i in $(seq 1 100); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/v1/models)" = "200" ] && break
  sleep 5
done
for i in $(seq 1 "$CYCLES"); do
  t0=$(date +%s%N)
  curl -s -m 30 -X POST "http://127.0.0.1:$PORT/sleep?level=2" >/dev/null
  t1=$(date +%s%N)
  curl -s -m 30 -X POST "http://127.0.0.1:$PORT/wake_up" >/dev/null
  t2=$(date +%s%N)
  r0=$(date +%s%N)
  curl -s -m 30 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen35-4b","prompt":"ping","max_tokens":4}' >/dev/null 2>&1
  r1=$(date +%s%N)
  python3 - "$i" "$t0" "$t1" "$t2" "$r0" "$r1" <<'PY' >> "$RES/fatigue.jsonl"
import json,sys
i,t0,t1,t2,r0,r1=[int(x) for x in sys.argv[1:]]
print(json.dumps({"cycle":i,"sleep_ms":(t1-t0)//1000000,"wake_ms":(t2-t1)//1000000,
                  "req_ms":(r1-r0)//1000000}))
PY
  [ $((i % 50)) -eq 0 ] && { log "fatigue $i/$CYCLES"; sample "fatigue_$i" $PORT; }
done

log "M3 phase C: admission wall ctx16k"
MAXLEN=81920 serve $PORT "$MODEL" wall_base || { restore_default $PORT "$MODEL"; exit 0; }
for C in 8 16 32; do
  bench $PORT "wall_ctx16k_c${C}" "$C" 32 16000 64
  sample "wall_ctx16k_c${C}" $PORT
done

log "M3 phase D: long-ctx soak ctx16k c4 ($SOAK_WAVES waves)"
for i in $(seq 1 "$SOAK_WAVES"); do
  bench $PORT "ctxsoak_w$(printf %04d "$i")" 4 32 16000 64
  sample "ctxsoak_w$(printf %04d "$i")" $PORT
  [ $((i % 50)) -eq 0 ] && log "ctx soak $i/$SOAK_WAVES"
done

log "M3 all done"
restore_default $PORT "$MODEL"
log "M3 restore done"
