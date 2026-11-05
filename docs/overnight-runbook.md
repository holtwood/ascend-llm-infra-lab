# 通宵任务队列（远端 npt · 8×910B1 · 独占）

> 目标：硬件窗口内不空转。一键部署 → master.sh 串行跑 8.5h 队列 → 全部产物落 JSON。
> 位置：`scripts/overnight/`。执行机仅限 **npt**（遵守三容器独占规则，不碰 nmindie/nmindspore/本机卡）。

## 一键执行（用户侧）

```bash
# 本地容器执行：推脚本+模型→校验远端状态→nohup 启动 master
bash scripts/overnight/deploy.sh

# 随时看进度（不伤任务）
bash scripts/overnight/observe.sh

# 早上收结果（拉回 results/overnight_<ts>/）
bash scripts/overnight/observe.sh --fetch
```

`deploy.sh` 幂等：已传的模型/源码自动跳过；已在跑的 master 不会重复启动。
SSH 断连不影响：master 以 nohup 跑在远端，所有输出在 `/data1/pytroch-7/overnight/run_<ts>/`。

## 队列设计（顺序串行，每个任务独立 timeout + 孤儿回收）

| # | 任务 | 上限 | 依赖 | 产物 | 面试价值 |
|---|------|------|------|------|----------|
| 0 | env_probe | 5min | — | env.txt | 环境事实存证 |
| 1 | finish_install | 60min | — | install.log | 补完 vllm wheel + vllm-ascend 可编辑安装（含 W8A16 patch 树） |
| 2 | hccl_pairs | 50min | torch | hccl_pairs.json | **8×8 全对 p2p 矩阵 → HCCS 拓扑推断**（谁和谁直连快、谁绕路慢）——全新数据 |
| 3 | kv_quant_matrix | 90min | torch | kv_quant_matrix.json | E8k 扩成 S{1k-16k}×B{1-64} 网格：int8-KV 误差曲线 + naive 税随规模变化 |
| 4 | paged_gather | 45min | torch | paged_gather.json | E8j 扩成 block{16-128}×S{4k-32k}：页粒度对 gather 税的敏感度 |
| 5 | copy_bw | 40min | torch | copy_bw.json | DRAM↔HBM pinned/pageable×双向×64M-2G + **计算并发下的拷贝衰减**（KV-offload 设计常数） |
| 6 | serve_smoke | 35min | vllm | serve_smoke/ | **#21112 最后一环**：W8A16 checkpoint → /v1/models 200 + 非空 generate |
| 7 | tp_scaling | 90min | vllm | tp_scaling/ | **首次多卡 serving**：0.6B @ tp=1/2/4 + c16 bench → TP 扩展效率实测 |
| 8 | hccl_soak | 填满剩余 | torch | soak.jsonl | 256MB allreduce 持续数小时 → 降频/漂移检测（首/末十分位对比） |

预算：固定项 ≈ 6h；soak 自适应填到 8.5h（`BUDGET_HOURS` 可调）。
vllm 门控：`finish_install` 后仍 `import vllm_ascend` 失败 → 自动跳过 6/7 号任务，soak 顶满不空转。
纯 torch 路径（vllm 彻底失败的最坏情形）≈ 5.5h——建议睡前 `observe.sh` 一眼确认安装进度。

## 失败语义

- 任务 rc 记进 `MANIFEST.jsonl`（124=超时杀）；队列继续下一个，**单点失败不炸整夜**
- `timeout --signal=INT --kill-after=60` + 每任务后按名回收孤儿 worker（修 e8i 的父死子活坑）
- HCCL 任务各占独立 MASTER_PORT（29711/29721），避免端口残留竞争
- 每任务日志 `run_<ts>/<task>.log`；JSON 产物按上表命名

## 数据→文档对接（次日）

- `hccl_pairs.json` → results/e8i/README.md 追加"拓扑矩阵"节（快/慢 pair 归类 → HCCS 结构推断）
- `kv_quant_matrix.json` / `paged_gather.json` → e8k/e8j README 扩网格 + kernel 层三诫更新
- `copy_bw.json`（含 under-load）→ docs/design/kv-tiering.md 常数修正
- `serve_smoke/` → #21112 comment 补终验 + issue 文档"Fix status"改 verified
- `tp_scaling/` → 新 results/e26_tp/（或并入 E4 扩展效率线）
- `soak.jsonl` → 稳定性小节（有漂移才有故事；无漂移=正面结论"8h 无衰减"）

## 前置状态备忘（部署时部署脚本会自动处理）

- vllm wheel (310MB aarch64) 已在远端下载；**注意 vllm 0.30.0 wheel 硬钉
  `torch==2.13.0`，直接装会毁掉 torch_npu 2.10.0 配对**——finish_install
  走 `--no-deps` + 从 wheel METADATA 抽取依赖清单剔除 torch* 后镜像源安装
- /vllm-workspace tar 流式拷贝进行中（含 W8A16 patch 的可编辑树）；
- 若部署时未完，finish_install 步骤会等待/补装；tar 未完则 vllm-ascend 步骤判 MISSING → 跳过 serving 任务。

## 远端环境坑（已写进 master.sh，勿踩第二次）

- 非登录 ssh **不 source .bashrc** → `import torch_npu` 报 `libhccl.so` 缺失。
  master.sh 开头 source `ascend-toolkit/cann-9.0.0/nnal-atb` 三个 set_env.sh
- `.bashrc` 默认 `ASCEND_RT_VISIBLE_DEVICES="6"` 只露一卡！
  master.sh 显式 export `0,1,2,3,4,5,6,7`；tp_scaling 按 tp 数重新切片
- 跳转链路 SSH 偶发掉线：所有远端长活必须 nohup + 日志落盘（master 已保证）
