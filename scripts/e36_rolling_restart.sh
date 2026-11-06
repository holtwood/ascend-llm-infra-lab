#!/bin/bash
# E36 — rolling restart with vs without gateway drain.
# Run from the lab controller host (has ssh keys to all nodes).
# Arm A "naive": kill M3 replica mid-load, restart immediately -> client errors.
# Arm B "drain": mark nginx upstream down + reload, restart, re-add -> errors.
# Load runs on D against 127.0.0.1:8080 (c24, n600).
set -u
source ~/ascend-lab/hosts.env
SSH_D="ssh -i $KEY -o StrictHostKeyChecking=no -p $D_PORT root@$D_HOST"
SSH_M3="ssh -i $KEY -o StrictHostKeyChecking=no -p $M3_PORT root@$M3_HOST"
CONF=/etc/nginx/conf.d/llblb.conf
OUT=/data/llm-infra-lab/results/e36

kill_restart_m3() {
  $SSH_M3 'source /data/llm-infra-lab/scripts/env.sh 2>/dev/null
    for p in $(pgrep -f "bin[/]vllm serve"); do kill "$p" 2>/dev/null; done
    sleep 8
    for p in $(pgrep -f "bin[/]vllm serve"); do kill -9 "$p" 2>/dev/null; done
    nohup vllm serve /model/Qwen3.5-4B --served-model-name qwen35-4b --port 8000 --max-model-len 8192 > /data/llm-infra-lab/results/e36/serve_restart.log 2>&1 &'
}
wait_m3() {
  for i in $(seq 1 120); do
    $SSH_M3 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/models' 2>/dev/null | grep -q 200 && return 0
    sleep 3
  done
  return 1
}
run_load() {  # $1 tag — launches background bench on D
  $SSH_D "/usr/local/python3.12.13/bin/python3 /data/llm-infra-lab/scripts/bench_openai.py \
    --base-url http://127.0.0.1:8080 --endpoint /v1/chat/completions --model qwen35-4b \
    --concurrency 24 --num-requests 600 --input-tokens 256 --output-tokens 96 \
    --out $OUT/$1.json >/dev/null 2>&1" &
  LOAD_SSH=$!
}

$SSH_D "mkdir -p $OUT" 
echo "== arm A: naive restart =="
run_load naive
sleep 15
kill_restart_m3 &
wait $LOAD_SSH
wait_m3 && echo "M3 back healthy"
sleep 20
echo "== arm B: drain-first =="
run_load drain
sleep 15
$SSH_D "sed -i 's/ server 127.0.0.1:8001;/ server 127.0.0.1:8001 down;/' $CONF && nginx -s reload" && sleep 3
kill_restart_m3 &
wait $LOAD_SSH
wait_m3 && echo "M3 back healthy"
$SSH_D "sed -i 's/ server 127.0.0.1:8001 down;/ server 127.0.0.1:8001;/' $CONF && nginx -s reload" && echo "upstream restored"
echo ALL_DONE
