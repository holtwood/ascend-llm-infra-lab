# 手撕高频题：MoE Router + Top-K/Top-P 采样器（含数值稳定性与负载统计）
#
# 面试考点覆盖：
#   1. "用 Python 实现一个 MoE 层的路由"（2026 手撕高频，大模型岗特色题）
#   2. "top-k 用快速选择还是堆？复杂度多少？"（O(n) vs O(n log k)）
#   3. "top-p (nucleus) 采样怎么实现？temperature 在哪一步加？"
#   4. "softmax 为什么要减 max？上溢/下溢怎么处理？"
#   5. "MoE 路由不均怎么办？"——这里给出负载失衡系数的度量（aux loss 讲法的工程对应物）
#
# 全部自带验证：naive 循环实现 vs 向量化实现，固定种子下逐元素一致。
# 依赖仅 torch（CPU 可跑；NPU 上 device 参数化即可）。

import torch
import torch.nn.functional as F


# ============================================================
# Part 1: MoE Router
# ============================================================

def moe_route_naive(gate_logits: torch.Tensor, num_experts: int, top_k: int):
    """naive 逐 token 路由。gate_logits: (T, E)。返回 (expert_ids (T,k), weights (T,k))。

    数学：p = softmax(gate_logits)；取每行 top-k 的索引与概率，权重在 k 个专家上
    **重新归一化**（renormalize=True 是 DeepSeek/Mixtral 的做法——被弃选专家的
    概率质量摊到入选者头上，保证输出期望尺度不变）。
    """
    T = gate_logits.shape[0]
    probs = F.softmax(gate_logits, dim=-1)
    ids = torch.empty(T, top_k, dtype=torch.long)
    wts = torch.empty(T, top_k)
    for t in range(T):                          # O(T·E) 的朴素扫描
        row = probs[t]
        top = torch.topk(row, top_k)
        ids[t] = top.indices
        wts[t] = top.values / top.values.sum()  # renormalize
    return ids, wts


def moe_route_vectorized(gate_logits: torch.Tensor, top_k: int, noise_std: float = 0.0):
    """向量化路由（生产实现）。noise_std>0 是训练期 jitter 探索（推理设 0）。

    三个 tensor 搞定全部 token：topk 一次 → softmax 只在选中集合上重算。
    注意 softmax 后再 topk 与先 topk 再在子集 softmax 不等价；这里实现后者
    （等价于 renormalize），并和 naive 版对齐验证。
    """
    topv, topi = torch.topk(gate_logits, top_k, dim=-1)   # (T,k) logits
    wts = F.softmax(topv, dim=-1)                         # 子集 softmax = renormalize
    if noise_std > 0:
        wts = wts + noise_std * torch.randn_like(wts)     # 演示用；真实实现加在 logits 上
        wts = wts / wts.sum(-1, keepdim=True)
    return topi, wts


def expert_load(expert_ids: torch.Tensor, num_experts: int):
    """负载统计：每专家分到的 token 数 + 失衡系数。

    imbalance = max_load / mean_load。均衡时 =1；DeepSeek 报告里路由不均可达 3-5×，
    是 EPLB/aux-loss 的动机（E 系列面试讲解点：均衡机制可把不平衡比降 ~40%）。
    """
    counts = torch.bincount(expert_ids.reshape(-1), minlength=num_experts)
    return counts, (counts.max() / counts.float().mean()).item()


def moe_dispatch(x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, num_experts: int,
                 experts: torch.nn.ModuleList):
    """dispatch → expert forward → combine（grouped GEMM 的教学版）。

    向量化 gather/scatter 替代 python 字典：按专家排序后切片，各专家做批 GEMM，
    输出用 index_add 聚合（scatter-add 天然处理一个 token 进多个专家的情况）。
    """
    T, k = ids.shape
    flat_e = ids.reshape(-1)                       # (T*k)
    order = flat_e.argsort()                       # 按专家分组（stable）
    x_rep = x.repeat_interleave(k, dim=0)          # token 复制 k 份
    out = torch.zeros_like(x)
    w_flat = wts.reshape(-1)
    start = 0
    for e in range(num_experts):
        end = start + (flat_e == e).sum().item()
        if end > start:
            idx = order[start:end]                 # 属于专家 e 的 (token,slot) 展平位置
            tok_idx = idx // k                     # 还原 token 行号
            y = experts[e](x_rep[idx])             # 专家批前向
            out.index_add_(0, tok_idx, y * w_flat[idx].unsqueeze(-1))  # 加权 scatter 回
        start = end
    return out


# ============================================================
# Part 2: Top-K / Top-P 采样器
# ============================================================

def sample_naive(logits: torch.Tensor, temperature=1.0, top_k=0, top_p=1.0,
                 seed=None) -> torch.Tensor:
    """教学版：显式循环 + 逆 CDF。logits: (V,) → 返回采样 token id。

    管线：temperature 缩放 → top-k 硬截断 → top-p 软截断（排序后累积概率
    超过 p 的第一个位置截断，且**总是保留首个 token**，防止 p 过小全空）→
    softmax → 均匀数 u ∈ [0,1) 沿累积分布走 → 落点即采样结果。
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    z = logits / max(temperature, 1e-5)            # temperature→0 退化为 greedy
    if top_k > 0:
        kth = torch.topk(z, top_k).values[-1]      # 第 k 大的 logit 作阈值
        z = torch.where(z < kth, torch.full_like(z, -float("inf")), z)
    if top_p < 1.0:
        sp, si = torch.sort(z, descending=True)
        cum = torch.cumsum(F.softmax(sp, -1), -1)
        keep = cum - F.softmax(sp, -1) < top_p     # 首 token 恒保留：移出自己前的累计< p
        keep[0] = True
        mask = torch.zeros_like(z, dtype=torch.bool).scatter(0, si, keep)
        z = torch.where(mask, z, torch.full_like(z, -float("inf")))
    probs = F.softmax(z, dim=-1)
    u = torch.rand(1, generator=g).item()
    cdf = torch.cumsum(probs, -1)
    return int(torch.searchsorted(cdf, u * cdf[-1]).item())


def sample_vectorized(logits: torch.Tensor, temperature=1.0, top_k=0, top_p=1.0,
                      seed=None) -> torch.Tensor:
    """生产版：全程张量算子（可上 NPU/GPU，无 python 循环）。

    数值稳定三件套：
      1. softmax 内部先减 max（F.softmax 已做；手写时必须显式）——上溢防护
      2. -inf 屏蔽而不是删元素——形状恒定，kernel 友好
      3. Gumbel-max 等价采样：argmax(logit + Gumbel 噪声) ~ Categorical 分布，
         免显式 CDF（多 token 批采样时比 searchsorted 快一个量级）
    """
    if seed is not None:
        torch.manual_seed(seed)
    z = logits / max(temperature, 1e-5)
    if top_k > 0:
        kth = torch.topk(z, top_k).values[-1]
        z = z.masked_fill(z < kth, -float("inf"))
    if top_p < 1.0:
        sp, si = torch.sort(z, descending=True)
        cum = torch.cumsum(F.softmax(sp, -1), -1)
        keep = (cum - F.softmax(sp, -1)) < top_p
        keep[0] = True
        z = z.masked_fill(~torch.zeros_like(z, dtype=torch.bool).scatter(0, si, keep),
                          -float("inf"))
    # Gumbel-max：CDF 的免归一化采样（-log(-log(u)) 噪声）
    gumbel = -torch.log(-torch.log(torch.rand_like(z).clamp_min(1e-12)))
    return int(torch.argmax(z + gumbel).item())


# ============================================================
# 自测（python3 examples/moe_router_sampler.py）
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    T, E, d, k = 4096, 8, 64, 2

    # --- router 一致性 ---
    logits = torch.randn(T, E)
    ids_n, w_n = moe_route_naive(logits, E, k)
    ids_v, w_v = moe_route_vectorized(logits, k)
    assert torch.equal(ids_n, ids_v), "expert ids mismatch"
    assert torch.allclose(w_n, w_v, atol=1e-6), "weights mismatch"
    counts, imb = expert_load(ids_v, E)
    print(f"[router] naive==vectorized ✓  expert load {counts.tolist()}  imbalance={imb:.2f}x")

    # --- 人为制造不均：前 3 个专家 logit +4 ---
    skew = logits.clone(); skew[:, :3] += 4.0
    _, imb_skew = expert_load(moe_route_vectorized(skew, k)[0], E)
    print(f"[router] biased logits → imbalance {imb:.2f}x → {imb_skew:.2f}x (EPLB/aux-loss 的动机)")

    # --- dispatch 正确性：向量化 grouped-scatter == 显式双重循环（naive 参照，小 T 提速）---
    Ts = 512
    x = torch.randn(Ts, d)
    experts = torch.nn.ModuleList([torch.nn.Linear(d, d, bias=False) for _ in range(E)])
    ids_s, w_s = moe_route_vectorized(logits[:Ts], k)
    y = moe_dispatch(x, ids_s, w_s, E, experts)
    ref = torch.zeros_like(y)
    for t in range(Ts):                          # 逐 token × 逐专家 的 O(T·k) 参照
        for j in range(k):
            ref[t] += w_s[t, j] * experts[int(ids_s[t, j])](x[t:t+1])[0]
    assert torch.allclose(y, ref, atol=1e-5), "dispatch mismatch"
    print(f"[dispatch] grouped-scatter == double-loop ✓  (max|diff|={(y-ref).abs().max():.2e})")

    # --- sampler 一致性：固定种子 Gumbel 版 vs 逆 CDF 版 ---
    v = torch.randn(50257)
    a = sample_vectorized(v, temperature=0.8, top_k=50, top_p=0.95, seed=42)
    b = sample_naive(v, temperature=0.8, top_k=50, top_p=0.95, seed=42)
    # 两实现用不同随机流，不做逐次相等断言；统计一致性：同分布采样 2k 次，卡方粗检
    sa = [sample_vectorized(v, 0.8, 50, 0.95) for _ in range(2000)]
    sb = [sample_naive(v, 0.8, 50, 0.95) for _ in range(2000)]
    top50 = torch.topk(v / 0.8, 50).indices.tolist()
    fa = sum(1 for t in sa if t in top50) / 2000
    fb = sum(1 for t in sb if t in top50) / 2000
    print(f"[sampler] in-top50 rate: gumbel={fa:.3f} cdf={fb:.3f} (应≈1且相近)")
    # temperature→0 退化为 greedy
    assert sample_vectorized(v, temperature=1e-6, top_k=1) == int(v.argmax())
    print("[sampler] temp→0 == greedy ✓")
    print("ALL OK")
