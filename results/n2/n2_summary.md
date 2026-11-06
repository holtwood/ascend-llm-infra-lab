# N2: prefill vs decode 算子级剖析（Qwen3.5-4B bf16, 910B2, eager）

采集: prefill=4096 tokens 1 次 (wall 1481 ms); decode=1 token×10 步 (wall ~114 ms/步)


## PREFILL 4096tk — device 算子总耗时 857 ms, 算子调用 57329 次 (1 次 forward)

| Core Type | 耗时(ms) | 占比 |
|---|---:|---:|
| AI_VECTOR_CORE | 591.0 | 68.9% |
| AI_CORE | 175.3 | 20.4% |
| MIX_AIV | 82.9 | 9.7% |
| MIX_AIC | 7.8 | 0.9% |
| AI_CPU | 0.5 | 0.1% |

Top-12 算子:

| OP | Core | Count | Total(ms) | Avg(us) | 占比 |
|---|---|---:|---:|---:|---:|
| Mul | AI_VECTOR_CORE | 8205 | 183.0 | 22.3 | 21.348% |
| Slice | AI_VECTOR_CORE | 19946 | 148.0 | 7.4 | 17.267% |
| MatMulV3 | AI_CORE | 201 | 110.8 | 551.3 | 12.925% |
| ReduceSum | MIX_AIV | 1560 | 78.9 | 50.6 | 9.199% |
| BatchMatMulV2 | AI_CORE | 7752 | 61.1 | 7.9 | 7.125% |
| ViewCopy | AI_VECTOR_CORE | 3048 | 59.7 | 19.6 | 6.966% |
| Add | AI_VECTOR_CORE | 4947 | 41.7 | 8.4 | 4.867% |
| Transpose | AI_VECTOR_CORE | 1828 | 36.7 | 20.1 | 4.276% |
| Tril | AI_VECTOR_CORE | 48 | 33.0 | 688.5 | 3.855% |
| Sub | AI_VECTOR_CORE | 3096 | 28.1 | 9.1 | 3.274% |
| Cast | AI_VECTOR_CORE | 510 | 12.0 | 23.6 | 1.404% |
| TensorMove | AI_VECTOR_CORE | 176 | 9.8 | 55.4 | 1.138% |

## DECODE 1tk×10步 — device 算子总耗时 241 ms, 算子调用 29350 次 (10 步 forward)

| Core Type | 耗时(ms) | 占比 |
|---|---:|---:|
| AI_VECTOR_CORE | 132.3 | 54.8% |
| AI_CORE | 89.8 | 37.2% |
| MIX_AIV | 15.0 | 6.2% |
| MIX_AIC | 3.1 | 1.3% |
| AI_CPU | 1.1 | 0.4% |

Top-12 算子:

| OP | Core | Count | Total(ms) | Avg(us) | 占比 |
|---|---|---:|---:|---:|---:|
| MatMulV2 | AI_CORE | 2490 | 83.7 | 33.6 | 34.687% |
| TransData | AI_VECTOR_CORE | 960 | 53.4 | 55.6 | 22.133% |
| Mul | AI_VECTOR_CORE | 5730 | 31.0 | 5.4 | 12.834% |
| ReduceSum | MIX_AIV | 960 | 9.5 | 9.9 | 3.94% |
| Cast | AI_VECTOR_CORE | 5110 | 7.7 | 1.5 | 3.19% |
| Add | AI_VECTOR_CORE | 3630 | 7.6 | 2.1 | 3.135% |
| Conv2D | AI_CORE | 240 | 6.1 | 25.4 | 2.531% |
| Slice | AI_VECTOR_CORE | 1300 | 5.8 | 4.5 | 2.4% |
| Pows | AI_VECTOR_CORE | 1050 | 5.7 | 5.5 | 2.372% |
| ReduceMean | MIX_AIV | 1050 | 5.5 | 5.2 | 2.266% |
| RepeatInterleave | AI_VECTOR_CORE | 480 | 4.1 | 8.5 | 1.682% |
| MemSet | AI_VECTOR_CORE | 480 | 4.0 | 8.3 | 1.645% |

## 派生指标

- prefill 算术强度佐证: AIC 占比 20.4% vs decode AIC 占比 37.2%
- decode 每步: 算子调用 2935 次, device 合计 24.1 ms, wall ~114 ms → host/gap 开销 ~89.9 ms (79%)
- prefill device 合计 857 ms vs wall 1481 ms → gap 624 ms
- decode AI_CPU 回退占比 0.4%

## 面试讲法草稿

1. prefill 4096tk: 大矩阵进 Cube, 但 Top 算子是海量小 AIV 算子(Mul/Slice), 说明 eager 未融合下逐元素/切片开销可观 → 框架级融合的空间
2. decode 1tk: 每步 ~2935 次算子下发, device 只忙 24 ms, 其余是 host 调度/launch → 小 batch decode 在 NPU eager 下是 launch-bound, 这正是 cudagraph/算子融合要解决的问题
3. decode 里 MatMul(GEMV) 占 1/3+ 是权重读取(memory-bound 主证据), TransData(格式转换) 22% 是 bf16→计算格式的搬运税