# 交接文档（给腾讯云上的下一个 agent）

> 你在腾讯云 mission control（debian@106.52.120.182:2243）上。
> 先读 `docs/environment.md`（集群详情）+ 本文（当前状态 + 下一步）+ `docs/experiment-plan.md`（设计）。

## 机器接入

`~/ascend-lab/hosts.env`（mode 600）含全部节点 IP/端口/密码 + key 路径。
NPU 节点已信任 `~/ascend-lab/keys/npu_nodes` 公钥，直接：

```bash
ssh -i ~/ascend-lab/keys/npu_nodes -p <PORT> root@<IP>
```

| 别名 | IP:PORT | 角色 |
|---|---|---|
| M3 | 223.111.149.177:40019 | 主实验机（本仓库在 /data/llm-infra-lab） |
| M1 | 223.92.35.244:40010 | 量化/MindIE 实验机（/model 最丰富） |
| M2 | 223.92.35.253:40019 | 副本 3（模型在 /data/models/） |
| D  | 223.111.149.165:40083 | 副本 4 / LB+客户端（/model 73 项共享） |

## 当前状态快照（2026-11-05 ~05:30）

### 已完成
- E1 基线（M3, Qwen3.5-4B）：results/e1/README.md + 6 档 json
- E3 prefix caching 消融：results/e3/README.md（OFF 组 TTFT+28%/吞吐−34%）
- 环境/踩坑/env.sh 全固化在仓库

### 在飞任务（登录各节点查看）
1. **E3 spec-ngram 消融**（M3）：driver `results/e3/spec_driver.log`；
   压测结果落 `results/e3/e3_spec_ngram_c8.json`。**注意：random 数据集对
   ngram 不公平（无 n-gram 可复用 → 接受率~1-8%）。重做时换
   `--dataset-name sonnet`（真实文本有可复现 n-gram），并把 serve 端日志里的
   `Mean acceptance length / Avg Draft acceptance rate` 抄进报告。**
2. **E2 W8A16 sweep**（M1）：driver `results/e2/w8a16_driver.log`@M3，
   服务日志 `/data/llm-infra-lab/results/serve_w8a16.log`@M1。
   27B 模型必须带 `--compilation-config '{"cudagraph_capture_sizes":[1,4,8,16,32,64]}'`
   （默认 40+ 档抓图会耗尽 NPU 流资源，error 207005）。
3. **M2 副本服务已就绪**：qwen35-4b @ M2:8000

### 下一步队列（按优先级）
1. E2 收尾：W8A16 sweep 完后跑 bf16 参照（`e2_driver.sh /model/Qwen3-8B qwen3-8b-bf16 bf16_8b`），汇总三变体表 → results/e2/README.md
2. E3 补完：spec-ngram 用 sonnet 数据集重测（c=8 和 c=32 交叉点）；可选 chunked-prefill 消融
3. E4 多副本：`scripts/e4_cluster.sh <N>` 一键建 D 上 nginx LB+隧道；M3/M1/M2 各起 qwen35-4b 副本，压 1→2→3 副本吞吐
4. E5 MindIE：M1 上 `/model/mindie/` 有部署包（start_mindie.sh + venv），同工况对比
5. 全程：每完成一项就 `git add -A && git commit && git push gitee main`

## 操作速查

```bash
# M3 上统一环境
source /data/llm-infra-lab/scripts/env.sh
# 起服务（任意 vLLM 节点）
bash scripts/serve.sh <model_path> <name> 8000 --max-model-len 8192 [extra]
# 压测（客户端与被测机不同机；经 SSH 隧道: ssh -N -L <localport>:localhost:8000 <节点>）
bash scripts/e1_baseline.sh <host:port> <model_name> <tag> <tokenizer_path>
# M1 换模型+压测一条龙（M3 驱动）
bash scripts/e2_driver.sh <model_path> <name> <tag> [serve args]
# E3 消融（重启+压测，M3 本机）
bash scripts/e3_ablation.sh <tag> "<bench args>" -- <serve extra args>
# 汇总 json → 表格
$PY scripts/parse_bench.py 'results/eN/*.json'
```

## 已知坑（勿重踩，详见 status.md）

1. `import vllm` 前必须 source env.sh（PYTHONPATH 修复 namespace 遮蔽）
2. 非交互 SSH 跑 NPU 任务：env.sh 已含 set_env.sh + LD_LIBRARY_PATH + ATB
3. `vllm bench` 必须 `--tokenizer <本地路径>` + `--endpoint /v1/completions`
4. 27B 模型必须压 cudagraph_capture_sizes（error 207005）
5. `/model` 只读（宿主共享）；自有模型放 `/data/models/`
6. `pkill -f` 会匹配自身——按 PID 杀进程
7. HF 不通→hf-mirror；容器间只能公网IP+SSH端口互通→服务互连用 SSH 隧道

## 若 M3 容器死了

仓库在 gitee 有全量历史；M3 的 /data/llm-infra-lab 与工作区在
`/data/models/`（Qwen3-0.6B/VL-7B/R1-32B 部分）。重建后 clone 仓库继续。
