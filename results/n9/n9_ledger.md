# N9: KV offload crossover 账本 — 换出 host / 重算 / 卡间迁移

- 常数（实测）：KV 32 KB/token（N3）、prefill 2765 tok/s（N2）、卡间 16.3 GB/s（N4）
- host 带宽（4 GB 档 p50）：pinned D2H 28.59 / H2D 24.98 GB/s；pageable D2H 1.26 / H2D 1.27 GB/s
- NUMA（1GB D2H pinned）：卡1域核 [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] → 28.52 GB/s；卡0域核 [155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167] → 19.35 GB/s

**等效重算带宽** = prefill 2765 tok/s × 32 KB/tok = **91 MB/s**。换出/换入是线性搬运、重算也是线性（对固定吞吐模型），故分界不随 X 变化：只要换入通道带宽 > 91 MB/s，搬回就永远比重算快。实测 pinned 往返带宽 26.66 GB/s（调和平均）是分界线的 294 倍。

| X | tokens | 重算 prefill | 换出 D2H(pinned) | 换入 H2D(pinned) | host 往返 | 换出(pageable 往返) | 卡间迁移 | host 往返 vs 重算 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 256MB | 8K | 2963 ms | 9 ms | 11 ms | 20 ms | 425 ms | 16 ms | offload 赢（147×）|
| 512MB | 16K | 5925 ms | 19 ms | 21 ms | 40 ms | 849 ms | 33 ms | offload 赢（147×）|
| 1GB | 33K | 11851 ms | 38 ms | 43 ms | 81 ms | 1699 ms | 66 ms | offload 赢（147×）|
| 2GB | 66K | 23702 ms | 75 ms | 86 ms | 161 ms | 3397 ms | 132 ms | offload 赢（147×）|
| 4GB | 131K | 47404 ms | 150 ms | 172 ms | 322 ms | 6795 ms | 263 ms | offload 赢（147×）|
| 8GB | 262K | 94808 ms | 300 ms | 344 ms | 644 ms | 13589 ms | 527 ms | offload 赢（147×）|

三方结论（restore 视角，X=1GB）：
- 重算 11851 ms ｜ host pinned 往返 81 ms ｜ host pageable 往返 1699 ms ｜ 卡间 66 ms
- 跨 NUMA 税：绑错域（[155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167]）1GB D2H 掉到 19.35 GB/s（绑对 28.52），慢 1.5×——offload 线程必须绑卡所属 NUMA 域。
- offload 的隐藏收益不在本次账本内：释放 HBM 后可容纳更多并发请求（本机 2TB DRAM vs 每卡 64GB HBM），以及 H2D 是 DMA 不占 AI Core，重算则排挤 decode 算力。
