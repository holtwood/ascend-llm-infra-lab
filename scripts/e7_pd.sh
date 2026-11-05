#!/bin/bash
# E7 PD分离实验（M1 单davinci卡共置 1P1D）
# Arm A mono: 单实例 serve, eager, gpu-mem 0.70（资源对齐：0.35P+0.35D=0.70）
# Arm B pd:   P:23700(kv_producer:31000)+D:23701(kv_consumer:31100) 各0.35 + proxy:28081
# bench: random in512/out128 c8,c32 + in2048/out128 c16（长prefill臂放大KV传输占比）
# 收尾恢复 M1 常规 serve（E4 副本角色）
# 用法: bash e7_pd.sh
set -u
source /data/llm-infra-lab/scripts/env.sh
MODEL=/model/Qwen3.5-4B; NAME=qwen35-4b
OUT=/data/llm-infra-lab/results/e7; mkdir -p "$OUT"
PROXY=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
PYBIN=/usr/local/python3.12.13/bin/python3

kill_all() {
  for p in $(pgrep -f "bin[/]vllm serve"); do kill "$p" 2>/dev/null; done
  for p in $(pgrep -f "load_balance_proxy"); do kill "$p" 2>/dev/null; done
  for i in $(seq 1 60); do pgrep -f "bin[/]vllm serve" >/dev/null || break; sleep 3; done
  for p in $(pgrep -f "bin[/]vllm serve"); do kill -9 "$p" 2>/dev/null; done
  sleep 8
}

wait_ready() {  # $1=port
  for i in $(seq 1 200); do
    curl -s --max-time 2 "http://localhost:$1/v1/models" 2>/dev/null | grep -q "$NAME" && return 0
    sleep 4
  done
  return 1
}

COMMON="--served-model-name $NAME --enforce-eager --no-enable-prefix-caching \
  --max-model-len 8192 --max-num-batched-tokens 4096 --block-size 128"

bench() {  # $1=port $2=tag $3=in $4=out $5=c $6=np
  echo "=== bench $2 port=$1 in=$3 out=$4 c=$5 np=$6 ==="
  vllm bench serve --endpoint /v1/completions --base-url "http://localhost:$1" \
    --model "$NAME" --tokenizer "$MODEL" \
    --dataset-name random --random-input-len "$3" --random-output-len "$4" --random-range-ratio 0.2 \
    --num-prompts "$6" --max-concurrency "$5" --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --save-detailed --result-dir "$OUT" --result-filename "e7_$2.json" \
    2>&1 | grep -aE "Successful|throughput|Mean TTFT|Mean TPOT|P99 TTFT|P99 TPOT"
}

run_arm_bench() {  # $1=port $2=tag_prefix
  bench "$1" "${2}_c8"  512  128 8  64
  bench "$1" "${2}_c32" 512  128 32 128
  bench "$1" "${2}_l16" 2048 128 16 64
}

if [ "${SKIP_MONO:-0}" = "1" ]; then
  echo "===== [1/3] monolithic arm: SKIPPED (SKIP_MONO=1) ====="
else
echo "===== [1/3] monolithic arm ====="
kill_all
nohup vllm serve "$MODEL" $COMMON --port 8000 --gpu-memory-utilization 0.70 \
  > "$OUT/serve_mono.log" 2>&1 &
wait_ready 8000 || { echo "MONO SERVE FAIL"; tail -5 "$OUT/serve_mono.log"; exit 1; }
run_arm_bench 8000 mono
fi

echo "===== [2/3] PD arm: P:23700 + D:23701 + proxy:28081 ====="
kill_all
nohup vllm serve "$MODEL" $COMMON --port 23700 --gpu-memory-utilization 0.35 \
  --max-model-len 4096 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":31000,"engine_id":"e7-prefill","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}' \
  > "$OUT/serve_p.log" 2>&1 &
P_PID=$!
nohup vllm serve "$MODEL" $COMMON --port 23701 --gpu-memory-utilization 0.35 \
  --max-model-len 4096 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":31100,"engine_id":"e7-decode","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}' \
  > "$OUT/serve_d.log" 2>&1 &
D_PID=$!
wait_ready 23700 && wait_ready 23701 || {
  echo "PD SERVE FAIL"; tail -8 "$OUT/serve_p.log"; tail -8 "$OUT/serve_d.log"; exit 1; }
nohup $PYBIN "$PROXY" --host 127.0.0.1 --port 28081 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 23700 \
  --decoder-hosts 127.0.0.1 --decoder-ports 23701 \
  > "$OUT/proxy.log" 2>&1 &
sleep 6
curl -s --max-time 3 http://localhost:28081/healthcheck || { echo "PROXY FAIL"; tail -10 "$OUT/proxy.log"; exit 1; }
echo
run_arm_bench 28081 pd

echo "===== [3/3] restore plain serve ====="
kill_all
LOG=/data/llm-infra-lab/results/serve_m1_restore.log bash /data/llm-infra-lab/scripts/serve.sh \
  "$MODEL" "$NAME" 8000 --max-model-len 8192
echo "E7 ALL DONE -> $OUT"
