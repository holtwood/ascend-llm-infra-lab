#!/bin/bash
# 统一环境：修复 editable vllm 被 namespace package 遮蔽的问题
# 用法: source /data/llm-infra-lab/scripts/env.sh
export PYTHONPATH=/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:/usr/local/Ascend/cann-9.2.0/python/site-packages:/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe
export PATH=/usr/local/python3.12.13/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PY=/usr/local/python3.12.13/bin/python3
