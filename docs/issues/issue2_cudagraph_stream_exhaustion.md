# [Issue Draft] Default cudagraph_capture_sizes exhausts NPU stream resources: AclmdlRICaptureBegin error 207005

**Repo**: vllm-project/vllm-ascend
**Component**: compilation / ACL graph capture

## Environment

- vllm-ascend 0.30.0rc2, vLLM 0.30.0, CANN 9.2.0, Atlas 910B2 64GB, Ubuntu 22.04
- Model: `Qwen3.8-27B-w8a8` (msmodelslim W8A8_DYNAMIC, ~29.6GB weights), `--max-model-len 8192`

## Reproduce

```bash
vllm serve /model/Qwen3.8-27B-w8a8 \
  --served-model-name qwen27b-w8a8 --port 8000 --max-model-len 8192
# i.e. DEFAULT compilation_config -> cudagraph_capture_sizes auto-expands to ~48 entries (1..512)
```

## Expected

Graph capture completes, or the platform layer caps `cudagraph_capture_sizes` to a
budget the NPU stream/queue hardware can actually sustain, with a clear message.

## Actual

Capture aborts part-way through the size list during engine init:

```
RuntimeError: capture_begin:../torch_npu/csrc/core/npu/NPUGraph.cpp:212 NPU function error:
  c10_npu::acl::AclmdlRICaptureBegin(capture_stream_, capture_mode), error code is 207005
[ERROR] ... PTA call acl api failed
[Error]: Failed to apply for memory.
  Check the remaining storage space in the hardware environment.
```

Driver plog shows the true resource exhaustion:

```
[INFO] RUNTIME ... StreamEndCapture: capture end success, device_id=0, ... capture model_id=1945.
[INFO] RUNTIME ... NormalSqCqAllocate: halSqCqAllocate no resource: device_id=0, drvRetCode=14.
[INFO] RUNTIME ... PeekLastErr: level=THREAD_LEVEL(0) err=207005.
```

## Analysis

Each ACL graph capture allocates SQ/CQ (stream-queue) pairs on the device. The
default `cudagraph_capture_sizes` list (~48 entries) requires ~48 concurrent capture
streams, which exceeds the SQ/CQ budget on 910B2 — especially with a 30GB model
already holding device memory. The surface error (`Failed to apply for memory`)
is misleading: the exhausted resource is hardware queues, not DRAM.

## Workaround (verified)

```bash
vllm serve /model/Qwen3.8-27B-w8a8 --max-model-len 8192 \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,16,32,64]}'
```

6 capture sizes → serve comes up and benches normally.

## Suggested fix

1. On NPU, derive the default capture-size list from an estimated SQ/CQ budget
   (e.g. cap count ≪ 48 for large models) instead of copying the CUDA default.
2. On `AclmdlRICaptureBegin` failure, retry with a progressively smaller capture
   list (graceful degradation) or at minimum surface "stream/queue resource
   exhausted — reduce cudagraph_capture_sizes" rather than the opaque 207005.
