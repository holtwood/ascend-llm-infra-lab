# E32 — 算子级：无 host 同步的 conv1d 状态更新（B=64 快 31×）

> 面试题：**"你写过算子/改过 kernel 吗？host 同步为什么贵？"**
> 答：给 GDN 的 `causal_conv1d_update` 写了一个**纯张量、零 host 同步**的
> decode 实现（正是 issue #18978 的根因），数值与参考实现一致（fp32 7e-6、
> state 完全相等、pad 槽正确），算子上 **B=64 快 31×、B=128 快 61×**；
> B=1 时反而慢 2×（多 5 个 kernel launch > 一次同步），交叉点在 B≈2。

## 1. 问题（来自本仓库自己的 issue）

vllm-ascend 的 PyTorch fallback 在 `_select_state()` 里对每个序列做
`conv_state_indices[i].item()`——**每请求每层一次 host 读回**：

- 打断 stream，阻塞 ACL graph 在 decode-FULL 的捕获（issue #18978 的原文症状）
- 稳态 decode 也按 O(B) 付出同步代价（B = decode batch）

## 2. 实现（`patches/e32_conv1d_nosync/conv1d_nosync.py`）

decode 路径（每序列 1 个新 token）不需要任何数据相关分支：

```
idx    = conv_state_indices (long)              # 不读回
valid  = idx != pad_slot_id                     # 设备侧掩码
state  = conv_state.index_select(0, safe_idx)   # 一次 gather
seq    = cat([state, x.unsqueeze(-1)], -1)      # [B, dim, width]
out    = (seq * weight.unsqueeze(0)).sum(-1)    # 深度可分离 width-tap 卷积
state  = seq[..., 1:]                           # 移位写回（index_copy_）
```

5 个算子全部在设备上完成，**没有任何 `.item()` / Python 逐序列循环**。
varlen（prefill 多 token）显式拒绝并回退到原实现，边界清楚。

## 3. 数值验证（M3 / 910B2）

| 检查 | 结果 |
|---|---|
| fp32 输出 vs `causal_conv1d_ref` | max\|diff\| **6.98e-06** |
| fp32 状态写回 vs 参考 | max\|diff\| **0.0**（完全一致） |
| pad 槽（`pad_slot_id=-1`） | 输出为 0、状态未被改写 ✅ |
| 第二步递推（真实滑动窗口语义） | max\|diff\| **8.07e-06** ✅ |
| bf16 输出 | 4.88e-04（算子顺序不同，合理） |

## 4. 算子级基准（M3，iters=100，dim=128，width=4，bf16）

| B | fallback ms/call | nosync ms/call | 加速 |
|---|---|---|---|
| 1 | 0.644 | 1.305 | **0.49×**（反而慢） |
| 8 | 5.000 | 1.295 | 3.86× |
| 32 | 19.203 | 1.256 | **15.3×** |
| 64 | 36.542 | 1.174 | **31.1×** |
| 128 | 70.381 | 1.161 | **60.6×** |

- fallback 严格 **O(B)**（≈0.55ms/序列，就是那次 `.item()` 同步）
- nosync 恒定 **~1.2ms**（5 个 kernel launch 的固定成本）
- 交叉点 **B≈2**：B=1 时同步比 5 次 launch 便宜——**所以不能无脑替换**，
  要按 decode batch 分派

## 5. 机制解释

1. **为什么同步这么贵**：`.item()` 要等设备完成并拷回标量，等于一次全流同步；
   B 个序列就是 B 次，且每次都会让后续 kernel 重新排队。
2. **为什么无同步版是 O(1)**：把"逐序列取状态"换成"一次 gather 整个索引张量"，
   索引本身留在设备上，Python 侧只处理形状（host 已知）。
3. **为什么 B=1 会输**：`cat + mul + sum + index_select + index_copy` 是 5 次
   launch，约 1.2ms；一次 `.item()` 约 0.6ms。**小 batch 下 launch 成本占优**。

## 6. 与本仓库其他证据的呼应

- E24：MTP/hybrid 4B 在高并发下反而输给 dense 8B——其中一个原因就是这类
  per-request 同步随 batch 放大
- issue #18978：patch 绑定已删除符号 → 每请求同步 fallback；本实现给出可落地的替代路径
- E30：同属"引擎/算子层的可测改动"，两者一起构成"我改过引擎"的证据链

## 7. 未做（诚实边界）

- **接进推理路径的尝试（E33）证伪了收益预期**：命中计数显示本模型 decode 走
  FULL graph 回放 / fused 入口，该 Python 函数不在热路径；A/B 差异为噪声
- 未测 varlen prefill 路径（显式回退）
- 未在 TP>1 / 大 dim（如 2048）下测

## 8. 复现

```bash
# 数值 + pad + 递推（NPU 上跑）
python3 patches/e32_conv1d_nosync/test_conv1d_nosync.py
# 算子基准
python3 patches/e32_conv1d_nosync/bench_conv1d_nosync.py --iters 100
```
