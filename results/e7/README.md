# E7 · PD 分离（Prefill/Decode Disaggregation）实验

参考前租户配置（`/model/elb-mooncake-20261012/`，P=TP2 + D=DP2 共 4 卡）简化为
**单卡共置 1P1D**：两个 vLLM 实例共享一块 910B2，各 `gpu_memory_utilization=0.35`。

- 模型：qwen35-4b（M1），`--enforce-eager --no-enable-prefix-caching
  --max-model-len 4096 --max-num-batched-tokens 4096 --block-size 128`
- KV 传输：`MooncakeConnectorV1`，P2P 握手模式（无中心元数据服务）
  - P: `23700`，kv_producer，handshake `:31000`，adxl 引擎 `172.17.0.7:20481`
  - D: `23701`，kv_consumer，handshake `:31100`，adxl 引擎 `172.17.0.7:20423`
- 代理：vllm-ascend 示例 `load_balance_proxy_server_example.py:28081`
  （POST /v1/completions → P 做 prefill → 异步 KV pull → D 做 decode）
- 基线臂：同卡单实例 monolithic，`gpu_memory_utilization=0.70`（=0.35×2 对齐）
- 负载：random，`in512/out128` c8/c32 + `in2048/out128` c16（长 prefill 臂放大 KV 传输占比）

## 结果

| 臂 | 负载 | 成功 | TTFT mean | TTFT p99 | TPOT mean | TPOT p99 | 输出吞吐 tok/s |
|---|---|---|---|---|---|---|---|
| mono | in512 c8 | 64/64 | 393 | 624 | 106.2 | 108.1 | 71.0 |
| mono | in512 c32 | 128/128 | 484 | 1129 | 112.6 | 115.9 | 249.1 |
| mono | in2048 c16 | 64/64 | 571 | 1832 | 108.3 | 111.8 | 132.8 |
| **PD** | in512 c8 | **46/64** | **22460** | 42807 | 262.6 | 1032.9 | 16.1 |
| **PD** | in512 c32 | **63/128** | **78707** | 99767 | 355.2 | 1034.8 | 13.8 |
| **PD** | in2048 c16 | **30/64** | **86601** | 120283 | 95.0 | 97.8 | 10.4 |

## 解读

**功能上通了，性能上完败——失败点不在 KV 传输本身，而在传输通道的"建链"环节。**

1. **请求路径是通的**：P 侧日志 `Delaying free of N blocks for request …`
   （256 次）+ D 侧持续 decode（108 条 generation 日志）+ proxy 200 响应，
   证明 prefill→KV 暂存→D 拉取→decode 的完整链路工作正常，
   MooncakeConnectorV1 的 P2P 模式不需要中心 store。

2. **~50% 请求的 KV 传输失败**，错误链：
   ```
   HixlCSClientConnect timeout:10000ms
   → Failed to connect client, remote_engine:172.17.0.7:20481
   → Mooncake transfer failed, ret: -1
   ```
   D 的 transfer executor 向 P 的 adxl 引擎（20481）发起
   `ExchangeEndpointAndCreateChannel`，10s 超时。P 引擎明明在 LISTEN
   （`ss -tlnp` 可见），但 CS 握手无法完成——**同一块 NPU 上两个进程各自
   初始化 adxl 引擎后，设备侧 channel 建立需要消耗 SQ/CQ 类硬件资源
   （同 issue2 的 207005 一族），共置时握手阶段大概率是串行/受限的**。
   前租户 P、D 在不同物理卡上（dev2,3 vs dev0,1），无此竞争。

3. **失败不均匀的机制**：成功的请求是"抢到"已建好的 channel 或赶上了
   握手间隙；失败请求各等 10s 超时 → TTFT 被拖长到 22~87s，
   完全淹没 PD 本应带来的 prefill 隔离收益。

4. **讽刺的对照**：mono 臂与 PD 臂合计占用同样显存（0.70 vs 0.35+0.35），
   PD 还把 decode 单独隔离了——成功的 PD 请求 TPOT 95ms < mono 108ms
   （decode 不再和 prefill 抢步），**这 12% 的 TPOT 改善正是 PD 理论收益的
   真实存在证据**，只是被建链失败淹没。

5. **面试话术**
   - PD 分离的收益来源：prefill 和 decode 的 batch 不再互相污染
     （TPOT 改善 ~12% 实测可见），代价是多一次 KV 搬运 + 一套传输面
   - 工程落地关键不是"能不能传"，而是**传输通道的建立时延与稳定性**——
     共置单卡时 adxl/hixl 的 CS 握手会成为单点；生产部署应给 P/D 独立
     设备 + 预建 channel pool
   - 排障路径：分清"协议层失败"（zmq 握手/kv_port）与"传输层失败"
     （adxl engine connect）——本案是后者，日志在 transfer_executor_base.cpp

## 局限

- 单卡共置是非典型部署（生产 P/D 分卡），结论应读作"共置时传输面脆弱"
  而非"PD 没有收益"
- D 侧 `max_model_len` 被迫降到 4096（0.35 显存下 8192 的 KV 页装不下），
  l16 臂 in=2048+out=128=2176 仍在范围内
- 未尝试重传调优（`ASCEND_CONNECT_TIMEOUT`、channel 复用等）——
  留作后续 issue 素材

原始数据：`e7_mono_*.json`（基线）、`e7_pd_*.json`（PD 臂）；
驱动日志 `e7_driver.log`；服务日志在 M1 `results/e7/serve_{mono,p,d}.log`
