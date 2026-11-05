# E5R: MindIE 2.3.0 vs vLLM-Ascend — Bring-up Record (blocked at warmup)

**Goal**: real MindIE-vs-vLLM comparison on identical model/load
(Qwen3-8B bf16 vs the vLLM reference ~1318 tok/s @ c64).

**Environment**: mindie container (117.131.156.71:2507 via jump
42.194.206.190), MindIE 2.3.0 + CANN 8.5.0 + glibc 2.38 + torch 2.1.0 +
torch_npu — i.e. **exactly the stack the old environment lacked** (the original
E5 died on glibc 2.35 vs MindIE 3.0 binaries). This run therefore isolates
MindIE runtime behavior, not ABI mismatch.

**Status**: ❌ blocked — reproducible warmup hang inside
`acl_encoder_operation.set_weight`. Documented as a failure-mode finding.

## Fix chain to reach model-load (9 steps)

| # | Symptom | Fix |
|---|---|---|
| 1 | `libboost_system.so.1.87.0` missing | `LD_LIBRARY_PATH += mindie-llm/lib` |
| 2 | `libabsl_bad_any_cast_impl.so.2308.0.0` | `+= mindie-llm/lib/grpc` |
| 3 | `libtorch.so` | `+= torch/lib`, `torch_npu/lib` |
| 4 | `MINDIE_LLM_BENCHMARK_FILEPATH` needs absolute file | set to `…/benchmark_out.csv` |
| 5 | `No module named atb_llm` | `PYTHONPATH += /usr/local/Ascend/atb-models` |
| 6 | `libhccl.so` | source CANN `set_env.sh` (HCCL libdir) |
| 7 | `ATB_SPEED_HOME_PATH` unset → `NoneType.join` | `export ATB_SPEED_HOME_PATH=/usr/local/Ascend/atb-models` |
| 8 | `_libatb_torch` import | `LD_LIBRARY_PATH += nnal/atb/8.5.0/atb/cxx_abi_1/lib`, `atb-models/lib` |
| 9 | `_cpu_logits_handler` not found | `PYTHONPATH += mindie-llm/lib` (py ext, not just libdir) |

Reproducible launcher: `scripts/e5r_start_service.sh`
(must `exec` the daemon directly — wrapping it in `setsid` breaks MindIE's own
process-group management: `MIE04E01011C Failed to set process group`).

## Progress reached

- Qwen3-8B and Qwen3-0.6B both: config OK → 36/28 layers loaded →
  effective EOS/PAD ids → `Allocating npu_cache` → NPU0 HBM 41.4/65.5 GB
  (model+KV resident) → warmup forward begins.

## The block: `init_ascend_weight` hang

- py-spy stack (stable across dumps, both models):

```
FlashQwen2ForCausalLM.init_ascend_weight (flash_causal_qwen2.py:289)
→ acl_encoder_operation.set_weight(self.ascend_weight)
FlashForCausalLM.forward → ModelRunner.forward → (warmup request)
```

- Single "inference" thread pegged at ~100% CPU; 483 threads idle.
- **No compile output**: zero writes under `kernel_meta`/`.cache`/CANN opp
  dirs during the spin.
- Kernel stack sampled with `do_mem_abort` + `__sys_rt_sigreturn` frames —
  consistent with a signal-fault retry loop in native code rather than
  productive TBE compilation (inconclusive: unwinding partial).
- Same frame for 8B (60+ min) and 0.6B (60+ min) → not weight-size-scaled;
  shrinking warmup (maxPrefillTokens 2048→256, maxIterTimes 64→8,
  maxSeqLen 2560→512, maxInputTokenLen 2048→256) did not unblock within ~10 min.
- Daemon watchdog restarts the cycle: warmup never returns before kill.

Config traps found along the way: `maxInputTokenLen` exists in BOTH
`ModelDeployConfig` and `ScheduleConfig` and both must satisfy
`maxInputTokenLen ≤ maxPrefillTokens`; `maxSeqLen`/`maxInputLen` are in
`ModelDeployConfig` top-level. See `mindie_config_smallwarmup.json`.

## Isolation: not qwen3-specific — ATB `set_weight` is broken stack-wide

Repointed the service at TinyLlama-1.1B (`modelName=llama`) → hangs at the
**same call site in a different model file**:
`FlashLlamaForCausalLM.init_ascend_weight` (flash_causal_llama.py:285).

With `ASCEND_GLOBAL_LOG_LEVEL=0`, the debug plog shows the smoking gun:
during `set_weight`, weight uploads complete (22 × 46MB/23MB/8MB copies)
and then the runtime degenerates into a **1 Hz poll loop** of the identical
32-byte `_drvMemcpyInner(dst=0xfffc0c000be0, src=0x12c041000000)` — a status
word being re-read indefinitely, i.e. the host waiting on a device-side ack
that never arrives. Combined with zero `kernel_meta`/opp output and the
mem-abort/sigreturn kernel-stack frames, the picture is: **the fused-op
build submitted to the device never completes; the host spins on a
completion word.**

Summary of evidence that this is a hang, not slow compilation:

| Test | Outcome |
|---|---|
| Qwen3-8B, default warmup | same frame, >60 min |
| Qwen3-0.6B, default warmup | same frame, >60 min |
| Qwen3-0.6B, shrunk warmup (256 tok/8 iter/512 seq) | same frame, >15 min |
| TinyLlama-1.1B (llama class, different graph file) | same frame at `flash_causal_llama.py:285` |
| Compile-cache growth during spin | zero writes |
| CANN debug plog during spin | only a repeating identical 32-byte poll memcpy |
| Thread attribution | 1/483 threads at ~100%, rest idle |

## Interview value (why a failed bring-up still counts)

- The 9-step chain is the real content: dynamic-lib archaeology → PYTHONPATH
  vs LD_LIBRARY_PATH split for .so extensions → env var discovery through
  `NoneType.join` traceback → config schema traps (two same-named
  `maxInputTokenLen` keys).
- Diagnosis discipline: distinguish "slow compile" from "hang" via artifact
  growth (zero kernel_meta writes), thread CPU attribution (1/483 threads),
  py-spy stable frame, and kernel-stack sampling.
- Takeaway for slides: vendor serving stacks (MindIE) do compile-at-first-
  forward via `set_weight`; when it wedges there is no cache file to watch —
  only CPU + stack evidence.

## What would close it

- ~~Try a different ATB model class~~ → done: llama hits the identical hang —
  the fault is in the ATB op-build path itself, not a model graph.
- Patch out warmup in `mindie_llm` to reach steady-state service directly
  (the hang is triggered by the first forward regardless, so this needs the
  op-build fault itself fixed).
- Device/firmware angle: the never-arriving device ack + fault-retry frames
  hint at an aicpu/driver-level issue on this host — checking driver version
  vs CANN 8.5.0 compatibility matrix is the next vendor-side step.
- Vendor ticket material: MindIE 2.3.0 + CANN 8.5.0 + 910B1, py-spy frame,
  1 Hz 32-byte poll signature, config bundle (attached above).
```
