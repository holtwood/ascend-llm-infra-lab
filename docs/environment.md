# 环境盘点

## 节点清单

| 节点 | SSH | 容器 hostname | NPU | 镜像 | 本地模型资源 |
|---|---|---|---|---|---|
| 本机 | 223.111.149.177:40019 | 9364f16c1458 | davinci7 | vLLM | Qwen3.5-4B(8.8G), Qwen3.8-27B-w8a8(30G), mineru |
| M1 | 223.92.35.244:40010 | ae8eeca26fe8 | davinci4 | vLLM | Qwen3-8B, Qwen3.5-4B, 27B bf16/w8a8/W8A16, qwen3-vl-8b/30b, MindIE, mooncake 工作区, DeepSeek-V4-Flash(151G) |
| M2 | 223.92.35.253:40019 | 222cead0c523 | davinci6 | vLLM | 空（已从 M3 拷入 Qwen3.5-4B） |
| D | 223.111.149.165:40083 | ad8f855a6dc4 | davinci3 | vLLM | **/model 宿主共享缓存 73 项**：DeepSeek-R1-Distill-32B-W8A8, Qwen2.5-32B-FP16, Qwen3-0.6B, Qwen3-Coder-Next, VL 系列等 |

密码见会话记录（不入库）。

## 已验证事实

- `torch.npu.is_available()=True`，2048² matmul 在 910B2 上通过（本机）
- huggingface.co 不通；hf-mirror.com、PyPI 通 → `HF_ENDPOINT=https://hf-mirror.com`
- 容器间仅能通过"公网 IP + 映射 SSH 端口"互通；无自由端口映射 → 跨机服务需 SSH 隧道
- `hccn_tool -i 0 -ip -g` 无 IP → 无跨机 RDMA，跨机 HCCL 不可行；多卡实验仅限单容器内（本平台无多卡容器）
- M1/M2 CANN 9.2；D 仅 driver，无 toolkit

## 关键路径

| 用途 | 路径 |
|---|---|
| Python 3.12（含全部 AI 栈） | /usr/local/python3.12.13/bin/ |
| vLLM/vllm-ascend 源码（editable） | /vllm-workspace/ |
| CANN 环境 | source /usr/local/Ascend/ascend-toolkit/set_env.sh |
| 模型缓存 | /model/ |
| 实验仓库 | /data/llm-infra-lab/（本机） |

## 限制与对策

| 限制 | 对策 |
|---|---|
| 单容器单卡，无 TP/PP | 改做多副本扩展 + 单机调度特性深挖 |
| 27B bf16 (74G) 超单卡 HBM | 用 w8a8/W8A16 版做量化主题 |
| 跨机仅 SSH 端口 | 服务互通走 `ssh -L/-R` 隧道（HTTP 流量可行） |
| 根盘 50G | 一切产物写 /data 或 /model 侧 |
