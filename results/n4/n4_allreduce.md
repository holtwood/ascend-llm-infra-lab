# N4: HCCL all-reduce 微基准(2x Ascend 910B2,跨 SYS/PCIe 互联)

日期:2026-11-06
环境:3x Ascend 910B2(npu-smi topo:三卡两两 SYS,无 HCCS 直连),CANN 8.5.1,torch 2.9.0+cpu / torch_npu 2.9.0,Python 3.10。
配置:ASCEND_RT_VISIBLE_DEVICES=1,2(容器内 ID 1/2 = host 物理卡 1/7;卡 0 留给主线实验),torchrun --nproc_per_node=2,backend=hccl。
方法:bf16 tensor,同步 all_reduce,每档 warmup 10 + 计时 50 次,torch.npu.synchronize() 前后 perf_counter 计时,报告 p50/p99(两 rank 取最差)。等效带宽:N=2 时 busbw 系数 2*(N-1)/N = 1,故 algbw = busbw = bytes/latency。
脚本:scripts/n4_allreduce_bench.py;原始数据:results/n4/n4_allreduce_raw_w2_sweep.json(4KB-16MB)、n4_allreduce_raw_w2_big.json(32-256MB)。

## 主表:消息大小 vs 延迟 vs 等效带宽(两次独立 run 取值一致)

| bytes | p50 (us) | p99 (us) | busbw = algbw (GB/s) |
|------:|---------:|---------:|---------------------:|
| 2 (1 elem) | **285-289** | 356-861 | - |
| 4096 | 293-296 | 325-338 | 0.014 |
| 16384 | 293-294 | 327-373 | 0.056 |
| 65536 | 292-298 | 323-336 | 0.22 |
| 262144 | 299-302 | 317-333 | 0.87 |
| 1048576 | 352-360 | 399-481 | 2.9-3.0 |
| 4194304 | 542-552 | 576-595 | 7.6-7.7 |
| 16777216 | 1300-1313 | 5548-6119 | 12.8-12.9 |
| 33554432 | 2299 | 2335 | 14.6 |
| 67108864 | 4318 | 4454 | 15.5 |
| 134217728 | 8354 | 8508 | 16.1 |
| 268435456 | 16479 | 18863 | **16.3(饱和)** |

两次 run 对比(2B: 289->285 us;16MB p50: 1300->1313 us)表明数字可复现。
延迟模型拟合:latency(us) ~= 285 + bytes / 16.6 GB/s(16MB 预测 1296 vs 实测 1313;256MB 预测 16450 vs 实测 16479)。

## 结论

1. **裸 all_reduce(barrier/握手)延迟 ~285-290 us。** 1 个元素(2 字节)的 all_reduce p50 约 288 us,这是每次调用必付的固定税,比 NVLink 互联(H200 16KB all-reduce 6.9 us)高出约 40 倍。
2. **4KB-256KB 的延迟与消息大小无关(~292-302 us),完全由握手/barrier 主导。** 64KB all-reduce 中固定开销占 >97%;直到 4MB(542 us)传输时间才与固定开销相当(285/542 = 52%)。典型 TP decode all-reduce(4B 模型每层 hidden 全量,几十 KB 到几 MB)正落在这一延迟平台区,即 **TP decode 的通信开销是 barrier/延迟主导而非带宽主导**,且慢互联把这一项放大了约 40 倍(16KB: 293 us vs NVLink 的 6.9 us)。
3. **带宽饱和点 ~16.3 GB/s(busbw)。** 256MB 才到 16.29 GB/s,128->256MB 增益仅 1.4%;约为 PCIe Gen4 x16 理论单向 32 GB/s 的 51%,与跨 NUMA/SYS 两跳 + 协议开销的预期一致。
4. **16MB 的 p99 出现 4-5 倍尾延迟(5.5-6.1 ms vs p50 1.3 ms),且在两次 run 中复现。** 大消息偶发抖动明显,小消息(<1MB)p99/p50 < 1.2 倍较稳。TP 训练/推理若用 MB 级通信需按 p99 预算。
5. **TP=2 decode 的量化推论:** 若每层 2 次 all-reduce(attn 输出 + MLP 输出),每次至少 ~288 us 固定开销,每层纯通信 >= ~576 us;对 decode 单层计算 ~150-300 us 的 4B 级模型,通信开销是计算的 2-4 倍,慢互联下 TP decode 的扩展比接近失效,应优先考虑流水并行/单卡或通信-计算重叠。

## 坑记录

- **ASCEND_RT_VISIBLE_DEVICES 的取值是容器内逻辑 ID,不是 host 物理 ID。** 本容器内 0/1/2 = host 物理 0/1/7;写 1,7(host 语义)会得到 device_count=1 且 rank1 的 set_device 报 EE1001 107001。正确写法:ASCEND_RT_VISIBLE_DEVICES=1,2。验证方法:probe 进程占 1GB 显存 + npu-smi info 对照占用卡号。
- 探针期间曾短暂(约 12 秒,1GB)占用 host 物理 0:vis=0,1 的 set_device(0) 映射到物理卡 0。已确认结束且卡 0 恢复无进程。这是映射语义未知的代价,之后所有 run 均避开 ID 0。
- 3 卡(N=3)all-reduce 需要 host 卡 0(容器内 ID 0),被主线实验占用,按约束跳过;N=2 下 busbw 系数 2*(N-1)/N=1,algbw 与 busbw 相同。
- hccl 初始化(torchrun 起 2 进程 + 组网)实测约 30-60 秒,属正常,不算卡死。
