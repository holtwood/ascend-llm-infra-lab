# E8d: HCCL Allreduce on 2× 910B1 — the cost model behind tensor parallelism

**Environment**: pytorch container (new env), `torch_npu` 2.10 + CANN 9.0,
`torch.distributed` backend=`hccl`, 2 workers pinned to physical NPU2/NPU3
(`ASCEND_RT_VISIBLE_DEVICES=2,3` → logical 0/1).
Script: `scripts/e8d_hccl.py`. Raw data: `hccl_allreduce.json` (19 points, 8KB–2GB).

## Measured curve

| Message | latency | effective bw |
|---|---|---|
| 8–262 KB | **0.16 ms (floor)** | 0.1–1.6 GB/s |
| 1–2 MB | 0.15 ms | 7–14 GB/s |
| 4 MB | 0.26 ms | 16.1 GB/s |
| 16 MB | 0.98 ms | 17.2 GB/s |
| ≥256 MB | linear | **18.5 GB/s (asymptote)** |

Two-regime model: `t(m) ≈ 160 μs + m / 18.5 GB/s` — the floor is sync+launch
latency, the slope is the HCCS/inter-die link.

## Why this matters: TP decode-step cost

Per transformer layer, TP-N decode does 2 allreduces (attn out + MLP out) of
`bs × hidden` bf16. With the measured model, for Qwen3-8B (hidden 4096,
36 layers):

| bs | msg per allreduce | per-layer cost | per-step (×2×36) | context |
|---|---|---|---|---|
| 1 | 8 KB | ~0.32 ms | **~11.5 ms** | vs bs=1 TPOT ~13 ms → TP-2 net-negative |
| 32 | 256 KB | ~0.33 ms | ~12 ms | vs c32 TPOT ~25 ms → ~45% overhead |
| 128 | 1 MB | ~0.15–0.3 ms | ~11–22 ms | vs c128 TPOT ~70 ms → ~15–30% |

So intra-host TP-2 on this link **only pays once decode is already
compute-heavy**: the allreduce term is latency-bound at realistic batch sizes
(the messages never leave the flat latency region), so TP cannot buy back the
~160 μs floor. This is the communication-side explanation of why replicas beat
TP for small models: TP splits 16 GB weights (halves HBM per copy → more
KV room) but taxes every decode step with a fixed latency cost, while N
replicas scale throughput linearly without any comm (E4 measured 1.63×–2.10×).

Rule of thumb extracted: TP only wins when `memory/KV-capacity gain × reuse`
outweighs `n_allreduce × 160 μs` per decode step — i.e. large models that
don't fit otherwise, or very large batches where the latency term amortizes.

## Cross-checks

- 18.5 GB/s asymptote is far below HBM (~1600 GB/s measured in E8b) — HCCL
  traffic goes over the inter-die HCCS link, not HBM. Contrast with decode's
  ~1.5 TB/s weight reads: on-device matvec is ~80× faster than cross-chip
  streaming — the physical reason TP layers can't hide comm for small msgs.
- The 160 μs floor > the entire bs=1 decode step budget in E1-class TPOT
  targets → TP-N for latency-critical small-batch serving is structurally
  wrong; replicas/pipeline are the right axis.

## Interview narrative

> "I measured HCCL allreduce on 2×910B1: latency floor 160 μs, asymptote
> 18.5 GB/s. Plugged into the TP cost model — 2 allreduces/layer/step of
> bs×hidden — TP-2 costs ~11–12 ms fixed per decode step regardless of batch,
> so for an 8B model at realistic batch it eats 15–45% of TPOT while only
> freeing KV capacity. That's why we chose replicas in E4 (linear scaling,
> zero comm) and why TP is reserved for models that don't fit."

## Limits

- 2 devices, same host (HCCS); cross-node HCCL (RoCE) would have a higher
  floor (~500 μs+) and lower bandwidth — directionally the same conclusion.
- allreduce only; production TP also uses allgather/reduce-scatter
  (similar cost structure, ~half the bytes).
