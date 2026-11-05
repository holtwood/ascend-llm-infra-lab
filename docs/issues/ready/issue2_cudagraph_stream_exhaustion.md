## Environment

- vllm-ascend 0.30.0rc2, vLLM 0.30.0, CANN 9.2.0, torch_npu 2.10
- Atlas 910B2 64 GiB, Ubuntu 22.04 aarch64
- Model: `Qwen3.8-27B-w8a8` (msmodelslim `W8A8_DYNAMIC`, ~29.6 GB weights), `--max-model-len 8192`

## Reproduce

```bash
vllm serve /model/Qwen3.8-27B-w8a8 --max-model-len 8192
# default compilation_config -> cudagraph_capture_sizes auto-expands
# (~35 entries in this run; ~64 when MTP pads decode shapes)
```

## Expected

Capture completes, or the platform caps `cudagraph_capture_sizes` to a budget the SQ/CQ hardware can sustain.

## Actual

Capture aborts part-way through the size list during engine init:

```text
RuntimeError: capture_begin:../torch_npu/csrc/core/npu/NPUGraph.cpp:212 NPU function error:
  c10_npu::acl::AclmdlRICaptureBegin(capture_stream_, capture_mode), error code is 207005
[Error]: Failed to apply for memory.
```

Driver plog shows the actual exhausted resource:

```text
[INFO] RUNTIME ... NormalSqCqAllocate: halSqCqAllocate no resource: device_id=0, drvRetCode=14.
[INFO] RUNTIME ... PeekLastErr: level=THREAD_LEVEL(0) err=207005.
```

## Cause

Each ACL graph capture holds SQ/CQ stream-queue pairs on the device. The default capture-size list needs that many concurrent capture streams, and on a ~30 GB resident model the SQ/CQ budget runs out. `Failed to apply for memory` is misleading — the plog `halSqCqAllocate no resource` line shows it's hardware queues, not memory.

Same resource family as #9564 (stream exhaustion under graph mode, different signature — capture-event wait vs SQ/CQ alloc) and #2629 (MoE path, fixed by #2511). This instance needs no exotic config — it's the plain dense-model default-list case, i.e. the capture-size-pruning problem #9691 ranks as top priority.

## Verified workaround

```bash
--compilation-config '{"cudagraph_capture_sizes":[1,4,8,16,32,64]}'
```

6 entries → engine starts and benches normally.

## Suggested fix

1. On `AclmdlRICaptureBegin` 207005, degrade gracefully (retry with a truncated capture list) or at least map it to "NPU stream/SQ-CQ resources exhausted — reduce cudagraph_capture_sizes". Same treatment open PR #17491 applies to 507015 for the same reason.
2. Bound the default capture-size count by device SQ/CQ budget until #9691's profiling-based sizing lands.

---

*Prepared with AI assistance; reproduction, plog signature, and workaround verified on real hardware.*
