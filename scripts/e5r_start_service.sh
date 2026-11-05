#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/mindie/latest/mindie-llm/lib:/usr/local/Ascend/mindie/latest/mindie-llm/lib/grpc:/usr/local/lib64/python3.11/site-packages/torch/lib:/usr/local/lib64/python3.11/site-packages/torch_npu/lib:/usr/local/Ascend/atb-models/lib:/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/atb-models:/usr/local/Ascend/mindie/latest/mindie-llm/lib:$PYTHONPATH
export ATB_SPEED_HOME_PATH=/usr/local/Ascend/atb-models
export ATB_LLM_BENCHMARK_FILEPATH=/data1/mindie-7/atb_benchmark.csv
export MINDIE_LLM_BENCHMARK_FILEPATH=/data1/mindie-7/benchmark_out.csv
export MINDIE_LOG_TO_STDOUT=0 MINDIE_LOG_TO_FILE=1
cd /usr/local/Ascend/mindie/latest/mindie-service
exec ./bin/mindieservice_daemon >> /data1/mindie-7/mindie_service.log 2>&1
