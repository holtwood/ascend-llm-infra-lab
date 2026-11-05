#!/bin/bash
# E4b 驱动（M3 跑）：4 副本（M3+M2+D本机+M1）补测，与 e4_driver.sh 同工况
# 前置: M1 已 serve qwen35-4b :8000；M3/M2/D 副本在跑
# 用法: bash e4b_driver.sh
set -u
D="ssh -p 40083 root@223.111.149.165"
OUT=/data/llm-infra-lab/results/e4; mkdir -p "$OUT"
TOK=/data/models/Qwen3.5-4B

bash /data/llm-infra-lab/scripts/e4_cluster.sh 4 || { echo "LB setup fail N=4"; exit 1; }
sleep 2
for C in 32 128; do
  $D "source /data/llm-infra-lab/scripts/env.sh >/dev/null 2>&1
  vllm bench serve --endpoint /v1/completions --base-url http://localhost:8080 \
    --model qwen35-4b --tokenizer $TOK \
    --dataset-name random --random-input-len 512 --random-output-len 128 --random-range-ratio 0.2 \
    --num-prompts $([ $C -eq 32 ] && echo 128 || echo 256) --max-concurrency $C --ignore-eos \
    --percentile-metrics ttft,tpot,e2el \
    --save-result --save-detailed --result-dir /data/llm-infra-lab/results/e4 \
    --result-filename e4_n4_c${C}.json 2>&1 | grep -aE 'Successful|throughput|Mean TTFT|Mean TPOT'"
done
scp -P 40083 'root@223.111.149.165:/data/llm-infra-lab/results/e4/e4_n4_*.json' "$OUT/" 2>/dev/null
echo "E4b done -> $OUT"
