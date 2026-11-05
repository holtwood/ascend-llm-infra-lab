# [Comment — NOT a new issue] Post as a comment on vllm-project/vllm-ascend#18978

**Why a comment, not an issue**: #18978 already documents the exact defect —
`patch_triton.py` re-references `causal_conv1d_update_npu` after #14620 removed
it (re-introduced by #15127), forcing the per-request-syncing PyTorch fallback.
Filed there for GLM-5.3-Flash KDA; our observation is the same root cause on a
different model family and version. Fix PR #13489 (open) drops the stale
rebind. A new issue would be a duplicate; this comment adds an independent
reproduction + measured decode-time impact.

---

## Comment body

Confirming the same root cause on a different model family and an older
release — **Qwen3.5-4B** (GDN `linear_attention` layers + periodic full
attention) on **vllm-ascend 0.30.0rc2** (vLLM 0.30.0, CANN 9.2.0, Atlas 910B2
64 GiB, Ubuntu 22.04 aarch64):

```text
WARNING [patch_triton.py:332] NPU Triton causal_conv1d_update is unavailable
(cannot import name 'causal_conv1d_update_npu' from
'vllm_ascend.ops.triton.mamba.causal_conv1d');
falling back to the PyTorch implementation, which syncs per request and
therefore stalls ACL graph capture at decode-FULL.
```

`vllm_ascend/ops/triton/mamba/causal_conv1d.py` exists but no longer exports
`causal_conv1d_update_npu`, so the fallback is unconditional on this version.

**Measured impact beyond the capture stall** — the fallback also taxes steady
state decode. Benchmark `in512/out128 @ concurrency 32` on this model:

| Metric | Value |
|---|---|
| median decode step (ITL p50) | ~16 ms |
| ITL p99 | 160–250 ms |

The p99 outliers are consistent with a periodic host sync each time a
request's conv state updates outside the captured graph — i.e., the penalty
isn't only at startup, it recurs per conv-state write during decode.

So even where decode-FULL capture happens to succeed, users of Qwen3.5/GDN
hybrids pay ~10× median ITL in the tail. Given #14620 intentionally removed
the Triton kernel, is the plan (a) an AscendC/310P conv-update path for the
affected models, or (b) batching/side-streaming the PyTorch sync so decode
iterations stop serializing on it? Happy to test a patch on this hardware.

*(Comment prepared with AI assistance; measurements taken on a real
deployment.)*
