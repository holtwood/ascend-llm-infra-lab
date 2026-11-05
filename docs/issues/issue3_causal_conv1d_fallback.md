# [Issue Draft] causal_conv1d_update NPU Triton kernel missing → per-request sync stalls decode

**Repo**: vllm-project/vllm-ascend
**Component**: ops / triton mamba kernels

## Environment

- vllm-ascend 0.30.0rc2, vLLM 0.30.0, CANN 9.2.0, Atlas 910B2, Ubuntu 22.04
- Model: `Qwen3.5-4B` — hybrid architecture (linear_attention/GDN layers + periodic full attention)

## Reproduce

```bash
vllm serve /model/Qwen3.5-4B --served-model-name qwen35-4b \
  --port 8000 --max-model-len 8192   # default compile mode, cudagraph enabled
```

## Actual

Every startup logs:

```
WARNING [patch_triton.py:332] NPU Triton causal_conv1d_update is unavailable
(cannot import name 'causal_conv1d_update_npu' from
'vllm_ascend.ops.triton.mamba.causal_conv1d'
(/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/mamba/causal_conv1d.py));
falling back to the PyTorch implementation, which syncs per request and
therefore stalls ACL graph capture at decode-FULL.
```

The module `vllm_ascend/ops/triton/mamba/causal_conv1d.py` exists but does not
export `causal_conv1d_update_npu`, so the patch silently falls back.

## Impact

The PyTorch fallback synchronizes per request during decode. In our benchmark
data (Qwen3.5-4B, in512/out128, c32) ITL p99 shows 160–250ms outliers against a
~16ms median decode step — consistent with a periodic host sync every time a
request's conv state updates outside the captured graph.

## Suggested fix

1. Implement `causal_conv1d_update_npu` in the Triton kernel module (the conv
   state update is a small in-place op; a native Triton/Ascend C kernel should
   be straightforward and keeps decode inside the ACL graph).
2. If it cannot land soon, downgrade the stall: batch the per-request sync, or
   run the conv update on a side stream so decode iterations don't serialize on it.
3. Document that hybrid linear-attention models (Qwen3.5 family, gated-deltanet
   layers) currently pay this penalty on NPU — it's non-obvious that "serve works
   but ITL p99 is 10× the median" traces back to a missing kernel symbol.

## 更新（2026-11-05 复查上游历史）

上游已解决此问题，**不建议再提 PR**：

- PR #5984（已合并）：Triton 版 `causal_conv1d_update_npu` 首次实现
- commit `8e5a61a`（#8842）：用 **AscendC 实现替换 Triton 版**
- commit `f0a938956`（#14620）：`Remove causal_conv1d_update_npu` ——符号整体移除
- `432743d83`/`ce9da92ce`："Move GDN conv1d metadata to device tensors" ——
  上游消灭 host sync 的正解是把元数据放 device 侧

本仓库 pin 的 v0.30.0rc2 中 `patch_triton.py` 仍 try-import 该已删除符号，
属版本残留绑定。正确行动：**升级 vllm-ascend**，而非补函数。

面试启示：给上游提 PR 前先查 mainline 历史——目标符号可能已被移除/重构。
