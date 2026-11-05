# 迁移 / 接手指南

写给后续在 ubuntu 任务机上启动的 agent（或任何接手的机器）。

## 仓库结构即上下文

- `docs/environment.md` — 4 台 NPU 容器清单、已验证事实、限制
- `docs/experiment-plan.md` — E1–E6 实验设计（每个实验对应面试问题）
- `docs/status.md` — **当前进度快照，先读这个**
- `scripts/env.sh` — NPU 节点统一环境（PYTHONPATH 修复 + python3.12 + hf-mirror）
- `scripts/*.sh` — serve/bench 脚本；`results/` — 全部产出

## NPU 机器接入方式

4 台容器仅暴露 SSH 端口（公网 IP:port），密码询问用户获取，
**不要把密码提交进仓库**。接入步骤：

```bash
# 1. 在新 agent 宿主机生成密钥
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519

# 2. 用 sshpass 把公钥推到每台 NPU 机（密码由用户提供）
sshpass -p '<PWD>' ssh-copy-id -p <PORT> -o StrictHostKeyChecking=no root@<IP>

# 3. 之后免密操作
ssh -p <PORT> root@<IP>
```

## 关键坑（已踩过，勿重踩）

1. **`import vllm` 必须带修复后的 PYTHONPATH**（见 scripts/env.sh）——
   裸环境变量里 `/vllm-workspace` 会让 PathFinder 把仓库根目录当
   namespace package 抢走 `vllm` 名字，editable finder 排在其后失效。
2. 非交互 SSH 下 `npu-smi`/`hccn_tool` 需要
   `export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH`
3. huggingface.co 不通 → `HF_ENDPOINT=https://hf-mirror.com`
4. 服务/模型/产物一律写 /data 或 /model 侧，根盘只有 50G
5. 压测客户端与被测服务**不同机**（WAN RTT 会污染 TTFT/TPOT 数据）

## 持续同步约定

每个实验节点完成后：`git add -A && git commit && git push gitee main`。
原始日志体量大（`results/logs/`）按需入库或仅留汇总。
