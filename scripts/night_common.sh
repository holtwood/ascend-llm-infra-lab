#!/bin/bash
# Shared helpers for overnight experiment lines. Source from night_*.sh.
# All results go to $RES as both raw JSON (bench) and JSONL (sampled lines).
RES=${RES:-/data/llm-infra-lab/results/night}
mkdir -p "$RES"
source /data/llm-infra-lab/scripts/env.sh 2>/dev/null
cd /data/llm-infra-lab || exit 1
PYBIN=/usr/local/python3.12.13/bin/python3

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RES/${NODE_TAG:-node}.log"; }

kill_vllm() {
  for p in $(pgrep -f "bin[/]vllm serve"); do kill "$p" 2>/dev/null; done
  for i in $(seq 1 40); do pgrep -f "bin[/]vllm serve" >/dev/null || break; sleep 3; done
  for p in $(pgrep -f "bin[/]vllm serve"); do kill -9 "$p" 2>/dev/null; done
  sleep 5
}

serve() {  # $1 port, $2 model, $3 tag, rest = extra args
  local port=$1 model=$2 tag=$3; shift 3
  kill_vllm
  nohup vllm serve "$model" --served-model-name qwen35-4b --port "$port" \
    --max-model-len "${MAXLEN:-8192}" "$@" > "$RES/serve_${tag}.log" 2>&1 &
  local code=""
  for i in $(seq 1 100); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null)
    [ "$code" = "200" ] && { log "serve[$tag] ready"; return 0; }
    sleep 5
  done
  log "serve[$tag] FAILED"; return 1
}

bench() {  # $1 port, $2 tag, $3 c, $4 n, $5 in, $6 out
  local port=$1 tag=$2 c=$3 n=$4 in=$5 out=$6
  $PYBIN scripts/bench_openai.py --base-url "http://127.0.0.1:$port" \
    --endpoint /v1/chat/completions --model qwen35-4b \
    --concurrency "$c" --num-requests "$n" --input-tokens "$in" \
    --output-tokens "$out" --out "$RES/$tag.json" >/dev/null 2>&1
  $PYBIN - "$RES/$tag.json" "$tag" <<'PY' >> "$RES/summary.jsonl"
import json,sys,time
try:
    d=json.load(open(sys.argv[1]))["summary"]
    print(json.dumps({"tag":sys.argv[2],"ts":int(time.time()),
        "tok_s":round(d["output_tok_per_s"],1),"ttft":round(d["ttft_mean_ms"],1),
        "tpot":round(d["tpot_mean_ms"],1),"ttft_p99":round(d["ttft_p99_ms"],1),
        "tpot_p99":round(d["tpot_p99_ms"],1),"ok":d["completed"],"err":d["errors"]}))
except Exception as e:
    print(json.dumps({"tag":sys.argv[2],"error":str(e)[:120]}))
PY
  log "bench[$tag] c=$c n=$n in=$in out=$out done"
}

sample() {  # $1 tag — AICore/HBM + server KV metrics
  local tag=$1 port=${2:-8000}
  local sm npu kv run wait pre
  npu=$(npu-smi info 2>/dev/null | grep -A1 "910B2" | grep -oE "[0-9]+ +[0-9]+ +/ +[0-9]+ +[0-9]+/ *[0-9]+" | head -1)
  kv=$(curl -s -m 3 "http://127.0.0.1:$port/metrics" 2>/dev/null | grep "^vllm:kv_cache_usage_perc" | awk '{print $2}')
  run=$(curl -s -m 3 "http://127.0.0.1:$port/metrics" 2>/dev/null | grep "^vllm:num_requests_running" | awk '{print $2}')
  wait=$(curl -s -m 3 "http://127.0.0.1:$port/metrics" 2>/dev/null | grep "^vllm:num_requests_waiting" | awk '{print $2}')
  pre=$(curl -s -m 3 "http://127.0.0.1:$port/metrics" 2>/dev/null | grep "^vllm:num_preemptions_total" | awk '{print $2}')
  npu=$(echo "$npu" | tr '\n' ';' | tr -s ' ')
  echo "{\"tag\":\"$tag\",\"ts\":$(date +%s),\"npu\":\"$npu\",\"kv\":\"$kv\",\"run\":\"$run\",\"wait\":\"$wait\",\"pre\":\"$pre\"}" >> "$RES/samples.jsonl"
}

restore_default() {  # $1 port, $2 model
  MAXLEN=8192 serve "$1" "$2" restore_default
}
