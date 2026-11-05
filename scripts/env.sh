#!/bin/bash
# 统一环境：修复 editable vllm 被 namespace package 遮蔽的问题
# 用法: source /data/llm-infra-lab/scripts/env.sh
# 先加载 CANN 官方环境（LD_LIBRARY_PATH/driver/toolkit 全套）
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
# 再盖 PYTHONPATH：必须指向包目录而不是 /vllm-workspace（否则 import vllm 变 namespace package）
export PYTHONPATH=/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:/usr/local/Ascend/cann-9.2.0/python/site-packages:/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe
export PATH=/usr/local/python3.12.13/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PY=/usr/local/python3.12.13/bin/python3
