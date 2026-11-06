# E33 — 把 E32 算子接进推理路径：命中计数证伪了一个"假收益"

> 面试题：**"你怎么确认你的改动真的生效了？"**
> 答：我在分发点埋了命中计数器。结果**命中数 = 0**——decode 步在
> `FULL_AND_PIECEWISE` 模式下走 ACL graph 回放，Python 算子不在热路径上；
> 第一轮测到的"+14.7%"是热身/顺序噪声，第二轮反序复测方向直接反转（−12%）。
> 结论：E32 的算子级收益成立，但**对本模型的稳态 decode 没有端到端影响**。

## 1. 做了什么

- `patches/e33_conv1d_in_model/deploy.py`：在 `vllm_ascend/ops/causal_conv1d.py`
  顶部插入环境变量门控分发（`VLLM_ASCEND_CONV1D_NOSYNC=1`），decode 形状时
  走 `conv1d_nosync`，否则回退原实现；分发处埋 `_HITS` 计数器打日志
- 同节点 A/B：`off → on → off → on` 交替（吸取 E31 顺序污染教训）

## 2. 数据（M3 / Qwen3.5-4B / c32 主档）

| 轮次 | 臂 | tok/s | tpot |
|---|---|---|---|
| R1 | off | 1105.3 | 23.7 |
| R1 | on | 1183.6 | 21.9（看似 +7%） |
| R2 | r1_off | 1201.7 | 21.2 |
| R2 | r2_on | 1055.0 | 24.8（**方向反转 −12%**） |

- 两个 off 臂自身相差 **8.7%**（1105 vs 1201）——同条件运行间漂移
- R1 的"on 更快"与 R2 的"on 更慢"互相矛盾 → 效应不存在，全是噪声

## 3. 为什么补丁没生效（命中计数 = 0）

1. **decode 走 FULL graph 回放**：日志确认 `Capturing CUDA graphs
   (decode, FULL): 35/35` 成功；回放阶段不执行 Python 代码，
   `causal_conv1d_update` 只在 capture warmup 时被调用
2. 分发条件第二版改成 `x.shape[0] == query_start_loc.numel() - 1`
   （纯形状判断、无同步），在 capture 期理论上应命中，但实际零命中——
   说明该模型的 decode capture 走的是**别的入口**（编译期 fused/splitting
   op，如 `qwen_gdn_attention_core` 拆分路径），而非这个 Python 函数
3. 第一版条件 `query_start_loc.numel() <= 1` 也是错的（decode 时 qsl 有
   B+1 个条目）——两个版本都没生效，只是错误方式不同

## 4. 对 E32 / issue #18978 的机制修正

上游警告说 per-request 同步 "stalls ACL graph capture at decode-FULL and
taxes steady-state ITL"。本实验证伪了后半句在本环境的形式：

- capture 在**本 build 上成功**（35/35），sync 让它变慢但没堵死
- 稳态 decode 走回放，Python 侧同步**不产生稳态 ITL 代价**
- 真正受影响的场景：capture 失败的配置（如 E17 的流耗尽、`--enforce-eager`、
  oversize batch 超出捕获表）、以及其他没有原生算子的 hybrid 模型

## 5. 这轮实验的价值（虽然是负结果）

- **命中计数是确认改动生效的最便宜手段**——没有它就会把噪声当收益
- **交替 A/B 是必要设计**：如果只做 off→on 一轮，+14.7% 会被写进报告
- 修正了对上游警告的过度外推：sync 税的是 capture 期与 eager 路径，
  不是回放期

## 6. 状态

- M3 上的补丁已回滚（`deploy.py --revert`），`conv1d_nosync.py` 已从
  vllm_ascend/ops 移除，副本恢复默认启动
- E32 算子本身保留：数值正确 + 算子级加速真实存在，只是这个模型的
  热路径不在那

## 7. 如果要让它真正生效（遗留方案）

- 找 decode capture 实际走的 fused 入口（`qwen_gdn_attention_core` /
  `mamba_mixer2` splitting op），在那层做无同步化
- 或在 `--enforce-eager` 部署（E17 场景）下启用，那里 Python 路径是真热路径
