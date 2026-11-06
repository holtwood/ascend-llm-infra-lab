# E30 — 引擎源码级优化：capture-size 预算规划器（启动 −10%，吞吐持平）

> 面试题：**"你改过推理引擎源码吗？改了之后怎么证明有效？"**
> 答：给 vllm-ascend 加了 ACL graph capture-size 预算规划器（env 开关、默认关闭、
> 不改上游行为），四卡并行矩阵 + 同节点背靠背 A/B：**capture 尺寸 35→9，
> 启动 180s→162s（−10%），c32/c128 吞吐 +0.7%/+1.7%（持平）**；
> 再收紧到 4 个尺寸则吞吐掉 5-9% —— 找到甜点并解释边界。

## 1. 动机（来自本仓库自己的 issue）

- vLLM 按 `max_num_seqs` 推导 `cudagraph_capture_sizes`，本机 4B 模型默认生成 **35 个**尺寸
- 每个尺寸独立占用一组 ACL stream / SQ-CQ 对 → 尺寸过密时 capture 报
  `EE1023 Too many streams are created`（torch_npu 层 `207005`），即 **issue #21113**
- 运行时真正命中的 batch size 很少：未命中的步会被 pad 到最近的已捕获尺寸（或退 eager）

## 2. 改动（`patches/e30_capture_budget/`）

| 文件 | 作用 |
|---|---|
| `capture_budget.py` | 纯函数 `plan_capture_sizes(sizes, budget)`：保留 2 的幂 + 最大值，超预算则等比抽稀；无副作用、可单测 |
| `deploy.py` | 幂等部署：拷模块 + 在 `platform.py` 的 `vllm_config._set_cudagraph_sizes()` 后插入**env 门控钩子**（`VLLM_ASCEND_CAPTURE_BUDGET`），自动备份 `platform.py.e30bak`，`--revert` 可回滚 |
| `test_capture_budget.py` | 6 个断言：预算 32/16/8/4/1、无需裁剪、空输入、最大值必留 |

钩子默认 **不生效**（无 env 时行为与上游一致），异常也只 warning 不阻断启动——
这是"调优旋钮"该有的边界。

## 3. 数据一：四卡并行矩阵（每臂一节点，跨节点有 11% 方差，仅看趋势）

| 臂 | 尺寸数 | 启动 | c8 | c32 | c128 |
|---|---|---|---|---|---|
| default | 35 | 181s | 496.8 | 1224.9 | 1828.2 |
| b10 | 9 | 163s | 479.9 | 1236.0 | 1903.0 |
| b4 | 4 | 154s | 483.7 | 1246.0 | 1811.1 |
| b2 | 2 | 151s | **313.6** | **904.9** | 1689.0 |

b2 已经明显塌陷（c8 −37%）——尺寸太少，多数 batch 只能走 eager。

## 4. 数据二：同节点背靠背 A/B（M1，消掉节点方差，这才是结论）

| 臂 | 启动 | Δ启动 | c32 | Δc32 | c128 | Δc128 |
|---|---|---|---|---|---|---|
| default（35） | 180s | — | 1218.1 | — | 1865.7 | — |
| **b10（9）** | **162s** | **−10.0%** | 1226.9 | **+0.7%** | 1898.2 | **+1.7%** |
| b4（4） | 169s | −6.1% | 1158.9 | −4.9% | 1703.7 | −8.7% |

**甜点 = 9 个尺寸（2 的幂阶梯）**：启动省 18 秒、吞吐不降反微升；
再往下压（4 个）就开始为未捕获的 batch 付 eager 代价。

## 5. 机制解释

1. **为什么省启动**：capture 是逐个尺寸建图 + 建 stream 的，35→9 直接砍掉 26 次建图；
   同时把 SQ/CQ 占用压到 EE1023 阈值以下（这正是 issue #21113 的修法方向）。
2. **为什么 b4 掉吞吐**：捕获集合只剩 `[1,8,32,256]`，运行中 c32 档的实际 batch 会在
   8~32 之间抖动；没有匹配尺寸时 vLLM 退 eager（或 pad 到 32 造成浪费），
   TPOT 从 20.4 → 21.4ms、c128 掉 8.7%。
3. **为什么 b10 能持平**：2 的幂阶梯（1,2,4,…,256）覆盖了 decode batch 的主要落点，
   pad 浪费 < 一个 2 的幂步长。

## 6. 面试口径

- "我改的是**配置生成层**而不是 kernel：把 capture 尺寸从 35 压到 9，启动 −10%，
  吞吐持平；再压就掉——边界是'2 的幂阶梯够不够覆盖 batch 落点'"
- "验证方法：先四卡并行扫矩阵找趋势，再**同节点背靠背**确认，因为跨节点有 11% 方差"（E21）
- "这个改动直接对应我提的 issue #21113（EE1023 流耗尽），是'报告问题→给方案'的闭环"
- PR-ready，但**未提交上游**（需确认目标仓库/分支）

## 7. 局限（主动说）

- 启动 180s 里大头是权重加载 + torch.compile（~150s），capture 只占 ~30s，
  所以上限就是 −10% 量级；纯 capture 时间未单独打点
- 只测了 Qwen3.5-4B / 910B2 / 单卡；多卡 TP、更大 max_num_seqs（512）未测
- 未验证长稳（capture 少了是否影响长跑）——可作为后续

## 8. 复现 / 回滚

```bash
# 部署（幂等，自动备份）
python3 patches/e30_capture_budget/deploy.py
# 单测
python3 patches/e30_capture_budget/test_capture_budget.py
# A/B（同节点两臂）
bash scripts/e30_arm.sh 8000 /model/Qwen3.5-4B 10 b10
bash scripts/e30_arm.sh 8000 /model/Qwen3.5-4B 0  default
# 回滚
python3 patches/e30_capture_budget/deploy.py --revert
```
