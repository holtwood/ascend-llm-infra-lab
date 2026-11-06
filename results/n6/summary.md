# N6: vllm-ascend 部署线（TP=1 / MTP / TP=2）— 2026-11-06

硬件: 3x Ascend 910B2 (64GB HBM)。本实验用容器内逻辑卡 2 (=物理卡7, TP=1/MTP) 与逻辑 1,2 (=物理卡1+7, 跨NUMA, TP=2)。
CANN 8.5.1 / Python 3.10 / venv /data/venv-vllm。版本见 versions.txt。

## 部署命令（最终可用）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /data/venv-vllm/bin/activate

# TP=1 baseline (物理卡7)
ASCEND_RT_VISIBLE_DEVICES=2 vllm serve /model/Qwen3.8-27B-w8a8 \
  --served-model-name qwen38 --quantization ascend \
  --max-model-len 8192 --max-num-seqs 32 --gpu-memory-utilization 0.9 --port 8100

# MTP (实验C)
ASCEND_RT_VISIBLE_DEVICES=2 vllm serve ...同上... \
  --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3}"

# TP=2 (实验J, 物理卡1+7 跨NUMA)
ASCEND_RT_VISIBLE_DEVICES=1,2 vllm serve ...同上... --tensor-parallel-size 2
```

启动时长: 权重 29.6GB 加载 ~1min + torchair 图编译 4-6min (冷)。
MTP: 自动规范化 method=qwen3_5_mtp -> mtp, Resolved architecture Qwen3_5MTP,
drafter 共享 target embedding, 总权重 32.4GB, cudagraph capture 扩到 128。

## 冒烟（TP=1 baseline, greedy）

- 地月距离: "地球到月球的平均距离约为38.44万千米" (正确; 默认带 <think> 段)
- 自我介绍: 通义千问, 连贯流畅
- MTP 下同 prompt 输出正常; TP=2 下 6 条抽查 (3 greedy + 3 sampled) 全部正常,
  1 条早期 greedy 冒烟 ("一句话说明什么是 all-reduce") 输出退化为 "!!!"
  ——孤立发散, 非系统性 (同 prompt 类型其余请求正常)。

## 简测口径

16 请求/档, max_tokens=100, temperature=0, enable_thinking=false,
stream + stream_options.include_usage (用 usage.completion_tokens 精确计数),
TTFT=首 content chunk, TPOT=(total-TTFT)/(ntok-1)。aiohttp 并发信号量控制。
v1 轮数据 (按网络包误当行解析 + 带思考) 口径失真已作废, 只保留 v2。

## 结果表 (TPOT/吞吐, P50)

| 配置 | c | TTFT P50 (ms) | TPOT P50 (ms) | 单流 tok/s | 聚合 tok/s |
|---|---|---|---|---|---|
| TP1 base | 1 | 380 | 73.6 | 13.0 | 12.9 |
| TP1 base | 4 | 574 | 84.8 | 11.1 | 41.4 |
| TP1 base | 16 | 904 | 81.2 | 11.2 | 166.9 |
| TP1 MTP (k=3) | 1 | 477 | 45.1 | 20.6 | 20.1 |
| TP1 MTP (k=3) | 4 | 925 | 51.7 | 16.6 | 62.9 |
| TP1 MTP (k=3) | 16 | 1096 | 58.9 | 13.9 | 142.2 |
| TP2 base (卡1+7 跨NUMA) | 1 | 351 | 82.3 | 11.8 | 11.8 |
| TP2 base | 4 | 604 | 90.8 | 10.4 | 38.8 |
| TP2 base | 16 | 1288 | 125.4 | 7.3 | 108.9 |

## 结论

### 实验 C: MTP 负载敏感性 — "低负载帮、高负载伤"复现, 且 TPOT 与吞吐背离
- c=1: TPOT 73.6→45.1ms (-39%), 单流 13.0→20.6 tok/s (+58%) — 显著正收益
- c=4: TPOT -39%, 聚合吞吐 41.4→62.9 (+52%) — 仍正
- c=16: TPOT 仍更低 (58.9 vs 81.2, -27%) 但聚合吞吐净负 15% (166.9→142.2), TTFT +21%
- 机制: MTP 每步 verify = batch x(1+k) token 前向 + draft 开销, 算力消耗增大;
  低负载时算力空闲, 接受的 token 直接转化为时延收益; 高并发 decode 时每步
  计算拉长, 系统 token 生产率反而下降。**TPOT 改善 ≠ 吞吐改善** (c=16 反例)
  ——面试讲 MTP 的关键反直觉点。

### 实验 J: TP=2 跨 NUMA — 全面净负, 验证 N4 推论
- TP=2 (物理1+7, SYS 跨NUMA) vs TP=1: c=1 TPOT +12% (82.3 vs 73.6ms),
  c=4 +7%, c=16 +54% (125.4 vs 81.2ms); 聚合吞吐 c=16 -35% (108.9 vs 166.9)
- 单卡 27B 权重本来就放得下 (29.6GB/64GB), TP=2 只有通信没有容量收益
- 机制: N4 实测跨NUMA all-reduce 裸延迟 288us (barrier 主导, 与消息大小无关
  4KB-256KB 平台), 每层 2 次 all-reduce, decode 每步 64 层 = 通信税固定;
  c=16 时 all-reduce 消息随 batch 变大, 16.3GB/s 带宽限制开始显形 -> TPOT +54%
- 结论: 慢互联上小模型 TP decode 净负, 与 N4 微基准推论一致。TP=2 的唯一
  理由是放不下的权重, 本模型不适用。

## 已知坑 (排障记录)

1. pip 依赖冲突: vllm 0.18.0 钉 torch 2.10.0 vs vllm-ascend 0.18.0 钉 torch 2.9.0
   -> ResolutionImpossible。解法见 versions.txt (官方 Dockerfile 路径的 pip 等价)。
2. ModuleNotFoundError: pkg_resources — venv 默认 setuptools 84.2.1 已移除
   pkg_resources, torchair (torch_npu.dynamo) import 时崩 -> pip install "setuptools<81"。
3. pypi.org 直连 ~50KB/s, vllm 385MB wheel 不可行 -> 清华镜像 1-4MB/s。
4. vllm-ascend 0.18.0 官方要求 dev 版 torch_npu/triton-ascend (release notes),
   从 OBS wheel 手动覆盖; pip check 报 triton-ascend 版本不匹配是预期内的。
5. Triton kernel import 警告 (triton.language.target_info) 是 gpt_oss MoE 路径,
   Qwen3.8 不受影响。
6. pkill -f "vllm serve" 会连带杀掉同会话 ssh 命令 (exit 255) — 用 ssh -f +
   远端 nohup 起服务更稳。

## 文件清单

- versions.txt — 版本组合与安装路径
- serve_tp1_base.log / serve_tp1_mtp.log / serve_tp2.log — 三次启动全日志
- bench_tp1_base_v2.jsonl / bench_tp1_mtp_v2.jsonl / bench_tp2_base_v2.jsonl — v2 口径原始数据
- bench_tp1_*.jsonl (无 v2 后缀) — v1 口径作废数据, 仅存档
- n6_pip_install.log / n6_pip_torch.log — 安装日志
