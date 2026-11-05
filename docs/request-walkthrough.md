# 一个请求在 vLLM(-ascend) 里的完整一生

源码基线：`/vllm-workspace/vllm`（V1 引擎）+ `/vllm-workspace/vllm-ascend`
（Ascend 插件，v0.30.0rc2）。实测基线：本仓库 E1-E16 在 910B1/910B2 上的
数据（每条标注出处）。用法：面试被问 "walk a request through vLLM" 时，
按本文顺序讲，每层都能 file:line 落地 + 实测数字佐证。

## 0. 全景图

```
HTTP POST /v1/chat/completions
 └─ api_router.create_chat_completion          # FastAPI 路由
     └─ OpenAIServingChat._create_chat_completion
         └─ OnlineRenderer.render_chat          # ★ chat 模板+tokenize 在这里
             └─ AsyncLLM.generate → add_request
                 └─ InputProcessor.process_inputs → EngineCoreRequest
                     └─ AsyncMPClient ──ZMQ(ROUTER→DEALER,msgpack)──▶ EngineCore 进程
                                                                        │
                        input_socket → Request(status=WAITING) → scheduler.add_request → waiting 队列
                                                                        │
run_busy_loop 每步：                                                      ▼
   scheduler.schedule()          ←── 预算+prefix命中+分配KV块+抢占决策 ──┐
   executor.execute_model()      ←── SchedulerOutput ──────────────────┤
      └ NPUModelRunner.execute_model                                   │
         └ _update_states → _prepare_inputs → attn metadata            │
             └ model.forward → FIA attention → sampler                 │
   scheduler.update_from_output() ←── 新token/停止判定/释放              │
   output_socket ──ZMQ(PUSH→PULL)──▶ AsyncLLM.output_handler            │
       └ OutputProcessor → IncrementalDetokenizer.update                │
           └ RequestOutputCollector → stream_generator → SSE chunk
```

关键架构事实：**API server 和引擎核心是两个进程**，ZMQ+msgpack 通信；
tokenize 在 API 侧做，引擎只吃 token ids。这条分界决定了 TTFT 的组成：
`渲染+tokenize + ZMQ 排队 + 调度等待 + prefill`，decode 阶段每步重复
`schedule→forward→sample→detok` 循环。

---

## 1. HTTP 入口与 tokenize（API-server 进程）

**路由**：`vllm/entrypoints/openai/chat_completion/api_router.py:41-80`
`@router.post("/v1/chat/completions")` → `create_chat_completion`（L54）→
`handler.create_chat_completion`（L62）。流式分支返回
`StreamingResponse(with_sse_keep_alive(generator))`（L77-80）。
`api_server.py` 本身只剩 59 行废弃 shim（L24-30 deprecation）。

**服务层**：`vllm/entrypoints/openai/chat_completion/serving.py`
`_create_chat_completion`（L258-415）：
- `await self.render_chat_request(request)`（L275）→
  `online_renderer.render_chat`（L240）
- `request_id = "chatcmpl-" + ...`（L281-283）
- `self.engine_client.generate(engine_input, sampling_params, ...)`（L368-385）

**tokenize 的真实位置**：`vllm/renderers/online_renderer.py`
`render_chat`（L170-271）→ `preprocess_chat`（L648-745）→
`renderers/base.py:_tokenize_prompt`（L582）。chat 模板渲染和 tokenize
**在 API 进程完成**——引擎进程从不解析原始文本（chat 路径）。

> 实测关联：这条 API 侧链路解释了 E10 为什么 `wake` 只需 84ms——
> 服务进程常驻，只有权重/KV 被换出；也解释了 E11 async scheduler
> 的收益来自哪里（下一节）。

## 2. AsyncLLM：请求进 ZMQ 前最后一站

`vllm/v1/engine/async_llm.py`:
- `__init__`（L143-177）：`input_processor`、`output_processor`、
  `engine_core = EngineCoreClient.make_async_mp_client(...)`（L169-177）——
  默认 `vllm serve` 路径**多进程 + ZMQ**（注释 L164）。
- `add_request`（L361-505）：已 tokenize 的 EngineInput 走
  `process_inputs`（L437-451）→ `assign_request_id`（L475）→
  `RequestOutputCollector` 队列（L483）→ `_add_request`（L489）。

**request_id 改写坑**：`input_processor.assign_request_id`
（`vllm/v1/engine/input_processor.py:262-279`）把
`chatcmpl-x` 改写成 `chatcmpl-x-{uuid8}`（内部 id），原始 id 存
`external_req_id`——回包时再映射回去（output_processor L325-345）。
n>1 时在 add_request L492-505 扇出成 ParentRequest 子请求。

**传输层**：`vllm/v1/engine/core_client.py` `AsyncMPClient`（L1046）——
input `zmq.ROUTER`（L598）/ output `zmq.PULL`（L605），msgpack 编解码。
`add_request_async`（L1219）→ `_send_input(EngineCoreRequestType.ADD)`
（L1178-1197）。引擎进程由 `vllm/v1/engine/utils.py:launch_core_engines`
（L1104）`Process(target=EngineCoreProc.run_engine_core)`（L190-196）拉起。

## 3. EngineCore：接收、入队、忙循环

`vllm/v1/engine/core.py`:
- IO 线程：`process_input_sockets`（L1686-1787）ZMQ DEALER 收包 →
  `preprocess_add_request`（L980-1002）→ `Request.from_engine_core_request`
  → `input_queue.put_nowait`（L1787）。
- `Request`（`vllm/v1/request.py:59`）：初始
  `status=WAITING`（L98）、`num_computed_tokens=0`（L182）。
  `RequestStatus` 全状态机 L364-405（含 `WAITING_FOR_REMOTE_KVS`——
  PD 分离时 D 侧等 KV 的状态）。
- 忙循环 `run_busy_loop`（L1403-1414）：
  `_process_input_queue` → `_process_engine_step` 每 tick。
- `step`（L589-619）——顺序即答案：
  `scheduler.schedule()`（L600）→
  `model_executor.execute_model(scheduler_output, non_block=True)`（L601）→
  `get_grammar_bitmask`（L602，structured output 时才算）→
  `future.result()`（L607）→ 若为 None 则 `sample_tokens`（L609）→
  `scheduler.update_from_output`（L614）。

> 实测关联：E11 里 `--async-scheduling` 让 TPOT 从 48.4ms 降到 29.4ms
> （-39%）——原理就是把 schedule() 的 CPU 工作和 GPU forward 重叠，
> 这个 step() 顺序图直接解释了为什么有优化空间（L600 和 L601 本可以并行）。

## 4. Scheduler：预算、命中、分配、抢占

`vllm/v1/core/sched/scheduler.py` `schedule()`（L561+）：
- 头部注释 L563-572 是理解 V1 的钥匙："There's no 'decoding phase' nor
  'prefill phase'... each request just has num_computed_tokens"——
  **连续批处理本质：每步所有人一起跑，只是有的算 1 个 token 有的算几千**。
- 双预算：`token_budget = max_num_scheduled_tokens`（L581，
  单步总 token 上限）、`input_budget = max_num_batched_tokens`（L584，
  新 token 输入上限——**chunked prefill 的阀门**）。
- Phase 1 running 队列（L612-808）：每请求 `num_new_tokens =
  num_tokens_with_spec - num_computed_tokens`（L656-660），超
  `long_prefill_token_threshold` 截断（L661-662）。
- 抢占：分配失败时挑 victim（L748-785，按 priority 否则队尾），
  `_preempt_request`（L1477-1518）→ 释放块、`status=PREEMPTED`、
  `num_computed_tokens=0`、塞回 waiting 队首（L1496-1517）。
- Phase 2 waiting 队列（~L850-1297）：
  - `kv_cache_manager.get_computed_blocks(request)`（L980）——
    **prefix caching 命中查询在这里**；
  - `enable_chunked_prefill` 关闭且 `num_new_tokens > budget` 直接 break
    （L1072-1078）——chunked prefill 的开关语义；
  - `allocate_slots`（L1156-1171）返回 None → 整条队停止尝试（队头阻塞）。
- spec-decode：`scheduled_spec_decode_tokens`（L812-819）、
  `pad_spec_decode` 补齐到 `1+k` 对齐 aclgraph 形状（L1024-1064）。

> 实测关联：
> - **E3b** mnb 扫描就是直接调 L584 的 `input_budget`：in=2048 时
>   mnb=256 → 每请求 8 个 chunk → 队头阻塞下 mean TTFT 13.1s；
>   mnb=4096 → TTFT 1.06s（12×）。完全对应"排队深度∝chunk 数"的模型。
> - **E3** APC 实验：prefix_repetition 数据集（4×2048 共享前缀）下
>   `get_computed_blocks` 命中 → TTFT 省 28%——就是 L980 这一行。
> - **E14** MTP：k=3 时每步 `scheduled_spec_decode_tokens` 把 4 位置
>   塞进同一个 forward——接受率 65.6% 但 3 次草稿前向+验证前向的
>   固定开销仍负收益（results/e14）。

## 5. KV cache 子系统：PagedAttention 的工程面

`vllm/v1/core/kv_cache_manager.py` `KVCacheManager.allocate_slots`
（L370-601）：算差量→问 `block_pool` 够不够（L559-563，不够 → 调度器
触发抢占）→ `allocate_new_blocks`（L578-583）→ `cache_blocks`（L599，
把满块哈希注册进 prefix cache——**写入侧在这里**）。

`vllm/v1/core/kv_cache_coordinator.py`：按 KV group 扇出到
`single_type_managers`（L136-156）；`find_longest_cache_hit`（L461-476）。

`vllm/v1/core/single_type_kv_cache_manager.py` `FullAttentionManager.
find_longest_cache_hit`（L741-835）：**按 block 哈希链顺序探测**
（L792-793 `block_pool.get_cached_block(block_hash)`），**首次 miss 即停**
——prefix cache 是纯前缀语义，不是任意子串。

`vllm/v1/core/block_pool.py`：
- `get_new_blocks`（L664-688）：空闲队列取块，**顺带驱逐旧的缓存映射**
  （`_maybe_evict_cached_block` L686）；
- `free_blocks`（L770-798）：非缓存块 **prepend**（LIFO，蹭热度），
  缓存块 **append**（FIFO/LRU 淘汰）——回收队列本身编码了替换策略。
- 哈希：`kv_cache_utils.hash_block_tokens`（L638-665）——
  `hash(parent_hash, token_ids, extra_keys)` **链式**（所以命中只能是从
  prompt 头开始的连续前缀）。

> 实测关联：E8 KV 容量账本（results/e8）验证的 `kv_cache_size_tokens`
> 和 E16 长上下文 KV 占用曲线（ctx 42k 峰值 KV 5.1%）就是这套
> block 计数在服务端的表现；910B 上 block_size=128（见 §6）。

## 6. Ascend 执行路径：SchedulerOutput → NPU 前向

**入口**：`NPUWorker.execute_model`（`vllm_ascend/worker/worker.py:757-791`）
→ `NPUModelRunner.execute_model`
（`vllm_ascend/worker/model_runner_v1.py:2229`，继承 GPUModelRunner）。

执行流水线（行号即真相）：

| 阶段 | 位置 | 干什么 |
|---|---|---|
| KV 传输抢占 | 2279-2284 | `handle_preemptions` 先于 `_update_states`——新块复用物理页前先把被抢占请求的 KV 存走 |
| 批状态更新 | 2309-2311 | `_update_states`：新建/续跑请求的 block table、token 状态 |
| 空转保护 | 2323-2338 | `num_scheduled_tokens==0` → 跳过 forward（connector 仍可跑 load/store） |
| 输入组装 | `_prepare_inputs` 1281-1466 | block_table 先异步 H2D（1304 注释：与后续 CPU 重叠）；input_ids **在 CPU pinned buffer 里 index_select 收集**（1379-1390）再整块 copy_to_gpu（1814） |
| attn 状态机 | `_build_attn_state` 1708-1735 | 五种状态：PrefillNoCache（全 0 计算）、PrefillCacheHit、DecodeOnly（全 1-token）、ChunkedPrefill、SpecDecoding（MTP） |
| metadata | 2495-2509 | `_build_attention_metadata`（3449）：`query_start_loc`/`seq_lens` 有 **CPU+GPU 双副本**（attention_v1.py:205-206 注释：aclgraph replay 只重放 device 部分） |
| rope | 2529-2530 | `update_cos_sin`——全局 cos/sin 表（`ops/rotary_embedding.py:174`），非每层独立 rope |
| forward ctx | 2552-2573 | `set_ascend_forward_context`：塞 MoE comm 方法、mc2_mask、aclgraph_runtime_mode |
| 前向 | 2596-2598 | `_model_forward`（3192-3251）→ `self.model(**model_inputs)` |
| 采样 | 2764+ `sample_tokens` | **与 execute_model 分离**（deferred sampling）；grammar bitmask **在 CPU**（2805 注释：Ascend 不支持 torch.compile 路径） |

### attention kernel 选路

`vllm_ascend/attention/attention_v1.py`：
- `AscendMetadata.attn_state`（162-233）就是 §4 那个状态机的产物，
  `block_tables`/`slot_mapping` 是 paged KV 寻址表。
- `reshape_and_cache`（1170-1204）：当前步 K/V 写进 paged cache →
  `torch_npu.npu_scatter_pa_kv_cache`（device_op.py:166-173）。
- `forward_impl` 分支（1248-1269）：
  - **DecodeOnly + 无滑窗 + FIA 形状白名单** → `forward_paged_attention`
    （1102-1121）→ `torch_npu._npu_paged_attention`——纯 decode 走
    paged-attention kernel；
  - 否则 → `forward_fused_infer_attention`（875）→
    `torch_npu.npu_fused_infer_attention_score`（TND layout + block_table +
    `sparse_mode`，滑窗 sparse_mode=4 L956-972）——**FIA 是 Ascend 的
    FlashAttention 对应物**，单 kernel 吃 prefill/chunked/decode 三种态。
- KV cache 形状：`get_kv_cache_shape`（108-118）默认
  `[2, num_blocks, block_size, num_kv_heads, head_size]`（BSHD）；
  **kernel block_size=128**（`get_supported_kernel_block_sizes` L147-151）。
  分配时是原始 int8 字节 buffer + 2MiB 对齐（model_runner_v1.py:4946+，
  为 PD 分离的 RDMA 注册），`_reshape_kv_cache_tensors`（5368-5786）
  重解释成各 backend 视图。

### ACLGraph（Ascend 版 CUDA graph）

`vllm_ascend/compilation/acl_graph.py` `ACLGraphWrapper`（66-302）：
- capture：`torch.npu.graph` 在**专用 stream**（307-335，与后台 kernel
  隔离）录制 `self.runnable`，产物按 BatchDescriptor 存
  `concrete_aclgraph_entries`（124）。
- replay：FULL 模式走 `_updatable_graph_replay`（297-319）——
  **UpdatableGraph 可以在不重录的情况下改 FIA 参数**（seq_lens/
  block_table），`use_updatable_graph`（utils.py:1896-1901）仅
  AscendAttentionBackend 类为真。这就是"图模式下 seq 变了不用重录"
  的机制。
- `_torch_cuda_wrapper`（model_runner_v1.py:6511-6562）把
  `torch.cuda.CUDAGraph` 等整套 API monkey-patch 成 `torch.npu` 等价物——
  上游 graph 代码零改动跑在 NPU 上。
- 选尺寸：`_check_and_update_cudagraph_mode`（6362-6428）按
  `capture_sizes` 集合；运行时 `_determine_batch_execution_and_padding`
  （3325-3432）把实际 batch 映射到最近图形状，不够格就 eager。

> 实测关联：issue2 草稿（docs/issues/）记录的 "cudagraph 207005
> SQ-CQ stream 耗尽" 就是这套 stream/graph 机制在量化模型上的边界；
> E14 MTP 提到 spec-decode 下 graph 形状对齐是 `pad_spec_decode`
> （scheduler L1024-1064）在买单。

### Sampler

`vllm_ascend/sample/sampler.py` `AscendSampler`（45）：
- greedy：`logits.argmax`（82-84）纯 device；
- `random_sample`（18-42）：**故意不用 `torch.multinomial`**——注释
  L24-25 "causes CPU-NPU synchronization"，改用 Gumbel-max
  （`exponential_` 噪声 + argmax）在 device 上闭环；
- top-k/top-p：`_apply_top_k_top_p_pytorch`（114-147）自实现 sort+mask——
  `npu_top_k_top_p` 在大 k/混合 k 下慢 5-28ms（注释 L156-157），
  **NPU 原生算子反而被 Python 实现反超**——面试可讲的真实取舍。

## 7. 回程：detok → SSE

- `scheduler.update_from_output`（scheduler.py:1874-2290）：
  `check_stop`（utils.py:94-136：EOS→STOPPED、stop_token_ids、
  长度 cap、min_tokens 守卫、重复检测）→ 完成的请求 `_free_request`
  → `kv_cache_manager.free`（L2615）。
- EngineCoreOutputs → ZMQ PUSH→PULL → `AsyncMPClient.process_outputs_socket`
  （core_client.py:1111-1165）→ `async_llm.output_handler`（L785-846）→
  `OutputProcessor.process_outputs`（output_processor.py:629-758）。
- `IncrementalDetokenizer.update`（detokenizer.py:96-143）：
  增量 detok + `check_stop_strings`（带 `stop_buffer_length` 回退窗口，
  stop string 不会泄漏进输出）。stop-string 命中时反向通知引擎 abort
  （output_processor L734-746）。
- `RequestOutputCollector` → `chat_completion_stream_generator`
  （serving.py:450-910）→ `data: {...}\n\n` × N → `data: [DONE]\n\n`（L910）。

> 实测关联：E16 长上下文（ctx 42k）下 TPOT 19→62ms——decode 单步的
> attention 随 KV 长线性增长（E8b 实测 127→999 GB/s 访存爬升），
> 这就是 §6 FIA decode 分支 `seq_lens` 那批数据在时延上的表现。

## 8. 面试速答卡（30 秒版）

> "请求先在 API 进程做完 chat 模板和 tokenize（renderer），包成
> EngineCoreRequest 走 ZMQ+msgpack 进独立的 EngineCore 进程；每个引擎
> tick 里 scheduler 先跑 running 队列扣 token budget，KV block 分配失败
> 就按优先级抢占，然后 waiting 队列做 prefix-cache 最长命中和 chunked
> 分配——没有 prefill/decode 阶段之分，只有每请求 num_computed_tokens
> 的差量。Executor 把 SchedulerOutput 交给 NPUModelRunner：CPU 侧拼
> input_ids、异步 H2D block table、按 attn_state（五种）选 FIA 或
> paged-attention kernel，decode 形状走 ACLGraph replay、其余 eager；
> sampler 在 device 上闭环（greedy argmax / Gumbel-max，特意绕开会同步
> 的 multinomial）。输出 token 走 ZMQ 回 API 侧做增量 detok、stop-string
> 判定，最后 SSE 流出。我在 910B 上实测过这条链上每一段的开销：
> chunked-prefill 预算从 256 调到 4096 时 TTFT 13.1s→1.06s，async
> scheduler 把 TPOT 砍了 39%，prefix caching 共享前缀下省 28% TTFT。"
