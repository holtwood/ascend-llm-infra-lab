# 设计题：Prefill/Decode 分离什么时候值得做？

> 设计题不是"PD 是什么"而是"什么条件下 PD 净赚"。用实测常数算盈亏。

## 一、PD 的收益项和成本项（全部实测锚点）

**收益**：prefill 是 compute-bound、decode 是 memory-bound（E8b 实测：
单流 matvec 1475GB/s ≈ HBM 峰值 92%），同机混跑时两类请求抢同一资源——
实测 c=64 时 TPOT 12→71ms（E1）、TTFT 161→797ms；分离后各用各的机型/
并行度，尾部延迟不再被 prefill 尖峰拖着走。

**成本**（E18 实测三层）：
1. KV 要搬家：prefill 产出的整段 KV 传给 decode 节点
   - 每 token KV ≈ 73.7KB（4B GQA 模型账本）
   - 8k prompt → ~0.6G、64k → ~4.7G
2. 传输税 = 延迟底 + 字节/带宽：
   - 同机 HCCS：160μs + 字节/18.5GB/s（E8d）
   - 跨节点网络：>ms 底 + 视链路（E18 环境直接证伪 device-mem 路径）
3. 控制面复杂度：请求路由、KV 生命周期、失败重试（E18 的
   200-垃圾-token 陷阱：传输失败仍可返回正常状态码）

## 二、盈亏公式（可代数字用）

```
单请求 PD 净收益 ≈ (混跑时的排队+争抢代价) − (KV传输税 + 额外调度延迟)

混跑代价 ∝ prefill 突发 × decode 敏感度 —— E1: c64 时 TPOT 恶化 6×
KV 传输税 = 160μs + prompt_len×73.7KB / 18.5GB/s（同机）
          = >1ms  + prompt_len×73.7KB / net_bw  （跨节点）
```

**算个例**（8k prompt，同机 HCCS）：
- 传输税 ≈ 0.16ms + 0.6G/18.5GB/s ≈ **33ms**
- 混跑省下的排队：E3b 显示 mnb 限 256 时长 prompt TTFT 13.1s → 拆开各跑
  可省掉大部分 prefill 阻塞 → **长 prompt 场景净赚秒级**
- 短 prompt（512 tok）：KV 仅 ~38MB → 税 ~2ms，混跑代价也小 → **净收益≈0 或为负**
  （还多付一层控制面）

## 三、决策表（能背下来的版本）

| 负载形态 | PD 建议 | 依据 |
|---|---|---|
| 短请求、高并发、延迟敏感 | 不分 | 传输税虽小但收益也小，控制面成本>收益（E9 goodput 视角） |
| 长 prompt（≥8k）、prefill 尖峰 | **分** | E3b：排队代价秒级 >> 传输税 ~33ms |
| 长 decode 为主（agentic/长 CoT） | 分且 decode 侧多卡 | decode 访存 bound，实例数按 TPOT-SLO 反推 |
| 跨节点环境、网络一般 | 慎分 | E18：device-mem 不可行时走 host 拷贝，税翻倍 |
| 同机多卡 | 优选项 | HCCS 税最低；E18 数据面在同机形态有戏 |

## 四、面试速答

> "PD 分离的账：收益是 prefill/decode 不再互相抢资源（混跑时 c64 TPOT
> 恶化 6 倍），成本是 KV 传输税——同机 HCCS 下 8k prompt 约 33ms，
> 所以**长 prompt 高收益负载分、短请求不分**；跨节点要看数据面是否可行，
> 我实测过跨节点 device-mem 传输不可行、走 host 拷贝税翻倍，而且
> 传输失败会返回 200+垃圾 token——监控要盯输出质量。"

## 五、监控设计（E18 教训）

- PD 系统的健康检查不能只查状态码：抽样对输出做 perplexity/格式校验
- `num_requests_waiting_by_reason` + KV 传输耗时直方图是核心指标
- 传输层失败要有显式失败码（不要静默 fallback 成垃圾输出）
