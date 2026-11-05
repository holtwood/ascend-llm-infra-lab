# [Bug] Default `cudagraph_capture_sizes` still exhausts NPU SQ/CQ stream resources on dense models (AclmdlRICaptureBegin 207005)

**Repository**: vllm-project/vllm-ascend
**Component**: compilation / ACL graph capture
**Duplicate check**: this is **not** a re-report of closed #2629 (`Alloc sq cq fail` on Qwen3 MoE, fixed by #2511 — that fix addressed the MoE/EP capture path). Our case triggers on a **dense** model with the **default** capture-size list, i.e. the residual gap the RFC #9691 calls "capture-size pruning — the highest-priority problem" and proposes to solve via profiling-based sizing. This issue supplies a reproduction + verified workaround for that residual case. Searches run: `cudagraph_capture_sizes`, `207005 capture`, `AclmdlRICaptureBegin`, `SQ CQ`.

## Environment

| Item | Version |
|---|---|
| vllm-ascend | 0.30.0rc2 |
| vLLM | 0.30.0 |
| CANN | 9.2.0 |
| torch / torch_npu | 2.10 |
| Hardware | Atlas 910B2, 64 GiB HBM, aarch64 |
| OS | Ubuntu 22.04 |

**Model**: `Qwen3.8-27B-w8a8` (msmodelslim `W8A8_DYNAMIC`, ~29.6 GB weights),
`--max-model-len 8192`.

## Reproduction

```bash
vllm serve /model/Qwen3.8-27B-w8a8 \
  --served-model-name qwen27b-w8a8 \
  --port 8000 --max-model-len 8192
# i.e. DEFAULT compilation_config: cudagraph_capture_sizes auto-expands
# (35 entries in this run: 1..256 step 8; up to ~64 entries when
#  MTP/speculative decoding adds padded decode shapes)
```

## Expected behavior

Graph capture completes, or the platform layer caps `cudagraph_capture_sizes`
to a budget the NPU SQ/CQ hardware can sustain and tells the user what it did.

## Actual behavior

Capture aborts part-way through the size list during engine init:

```text
RuntimeError: capture_begin:../torch_npu/csrc/core/npu/NPUGraph.cpp:212 NPU function error:
  c10_npu::acl::AclmdlRICaptureBegin(capture_stream_, capture_mode), error code is 207005
[ERROR] ... PTA call acl api failed
[Error]: Failed to apply for memory.
  Check the remaining storage space in the hardware environment.
```

Driver plog reveals the actual resource:

```text
[INFO] RUNTIME ... StreamEndCapture: capture end success, device_id=0, ... capture model_id=1945.
[INFO] RUNTIME ... NormalSqCqAllocate: halSqCqAllocate no resource: device_id=0, drvRetCode=14.
[INFO] RUNTIME ... PeekLastErr: level=THREAD_LEVEL(0) err=207005.
```

## Analysis

Each ACL graph capture allocates SQ/CQ (stream-command-queue) pairs on the
device. The default `cudagraph_capture_sizes` list — 35 entries here, ~64 with
speculative decoding — needs that many concurrent capture streams and exceeds
the SQ/CQ budget on 910B2, particularly with a ~30 GB model already resident.
The surface message (`Failed to apply for memory ... check the remaining
storage space`) is misleading: the exhausted resource is **hardware queues,
not memory** — the plog `NormalSqCqAllocate ... no resource` line is the real
signature.

This is the same failure family as #2629, but the trigger is different:
#2511 addressed the Qwen3-MoE capture path; here the failure arises purely
from the *count* of default capture sizes on a dense model — exactly the
sizing-policy problem #9691 proposes to replace with profiling.

## Workaround (verified on this hardware)

```bash
vllm serve /model/Qwen3.8-27B-w8a8 --max-model-len 8192 \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,16,32,64]}'
```

6 capture sizes → engine comes up and benches normally (throughput within a
few % of the eager-mode baseline at the covered batch sizes).

## Suggested fix

1. **Short term**: on `AclmdlRICaptureBegin` returning 207005, retry capture
   with a progressively truncated `cudagraph_capture_sizes` list (graceful
   degradation) and surface a message like
   `"ACL graph capture aborted: NPU stream/SQ-CQ resources exhausted —
   reduce cudagraph_capture_sizes"` instead of the opaque memory error.
2. **Aligned with #9691**: bound the default capture-size count by an
   estimated SQ/CQ budget for the device generation (e.g. cap ≪ 48 entries
   for large models), pending the profiling-based sizing the RFC proposes.

---

*This report was prepared with AI assistance (Cognition Devin); reproduction,
plog signature, and workaround were verified against a real NPU deployment.*
