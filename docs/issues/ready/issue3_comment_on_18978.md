# Posted as a comment on https://github.com/vllm-project/vllm-ascend/issues/18978

Same root cause on a different model family and an older release: **Qwen3.5-4B** (GDN `linear_attention` layers + periodic full attention), vllm-ascend **0.30.0rc2**, CANN 9.2.0, 910B2 64 GiB:

```text
WARNING [patch_triton.py:332] NPU Triton causal_conv1d_update is unavailable
(cannot import name 'causal_conv1d_update_npu' from
'vllm_ascend.ops.triton.mamba.causal_conv1d');
falling back to the PyTorch implementation, which syncs per request and
therefore stalls ACL graph capture at decode-FULL.
```

Beyond the capture stall, the fallback also taxes steady-state decode. `in512/out128 @ c32` on this model: median decode step ~16 ms vs ITL p99 160–250 ms — consistent with a host sync on every conv-state write, not just a startup cost.

Since #14620 removed the Triton kernel intentionally, what's the planned path — an AscendC conv-update for the affected models, or batching/side-streaming the PyTorch sync? Happy to test a patch on this hardware.
