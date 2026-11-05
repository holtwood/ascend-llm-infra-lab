# E17 — sleep/wake 多租户换租：冷启动 169s vs 唤醒 0.14s

> 面试题：**"多租户推理怎么快速换租？冷启动两分钟的模型怎么弹性伸缩？"**
> 数据回答：同卡两租户共存实测——A 睡出空间（1.8s），B 冷启动占用（114s），
> **B 在线期间唤醒 A 只要 0.14s**——换租成本比重启低 **1200×**。

## 场景设计（真实多租户换租，而非单 sleep 演示）

```
t0      租户A (tenant-a, :8000, util=0.5) 冷启动
t1      A 服务中
t2      POST /sleep?level=2  → A 卸载 weights+KV 到 CPU，HBM 4480/65536MB
t3      租户B (tenant-b, :8005, util=0.3, eager) 在腾出空间冷启动
t4      B 服务中（A 仍睡）  ← 两引擎共享一张 910B2
t5      POST /wake_up      → A 0.14s 复活，B 不受影响
t6      A、B 同时在线
```

`level=2` 睡眠 = CaMem 同时卸载 weights 和 kv_cache 到 CPU 内存；
`level=1` 只卸载 weights 保留 KV。两个坑见下。

## 实测数据（910B2, Qwen3.5-4B）

| 事件 | 耗时 | 备注 |
|---|---|---|
| A 冷启动 → ready | **169.4s** | 加载+编译+graph capture 全量 |
| A sleep level=2 | **1.77s** | CaMem offload 172/172 allocations |
| A 睡眠驻留显存 | **~1.1GB** | 31.2GB → 进程表 1004+99MB |
| B 冷启动（在腾出空间） | **114.2s** | util=0.3 + eager（无 graph capture） |
| **A wake（B 在线时）** | **0.139s** | restore 172/172 allocations |
| A 唤醒后首请求 | **0.129s** | 无 warm-up 惩罚，直接全速 |
| 对比：换租 vs 重启 | **~1200×** | 0.14s vs 169s |

## 机制解释

1. **为什么 wake 只要 0.14s**：sleep 不是 kill——CaMem 把显存块按 tag
   （weights/kv_cache）搬到 CPU 但保留 allocator 元数据和 ACL graph 资源；
   wake 只是回灌权重 + 恢复 KV 池指针，不重新编译、不重新 capture。
2. **为什么 sleep2 要 1.77s**：32GB→CPU 的 PCIe 拷贝（实测带宽 ~18GB/s）。
   这是换租的固定成本，但换来 B 能占用 ~20GB。
3. **唤醒后无 ramp**：权重按原地址回灌，prefill/decode 路径全量保留——
   对比冷启动的 169s（含 torch.compile + graph capture）是质的区别。
4. **为什么 B 必须 eager**：A 虽然睡掉显存，但 **stream/CQ 资源不释放**——
   B 的 ACL graph capture 报 `EE1023 too many streams`（即 E10/issue2 的
   207005 SQ/CQ 耗尽同一根因）。eager 模式跳过 capture，B 才能起。
   **这是 sleep 机制的实测边界：显存可共享，调度资源不行。**

## 面试口径

- "sleep mode 不是把进程停掉——是 CaMem 按 tag 搬显存，allocator 元数据留着，
  所以 wake 是 O(权重拷贝) 不是 O(初始化)"——有 0.139s 实测
- "换租场景的真实成本是 sleep 的 1.8s（PCIe 拷贝），唤醒端几乎免费"——有数据
- "但显存能共享不代表资源能共享：睡眠方仍占 stream/CQ，共卡引擎只能 eager"——
  踩坑一手经验，直接对应上游 issue2

## 坑位

- `/sleep` 路由需 `VLLM_SERVER_DEV_MODE=1`
- FRACTAL_NZ 权重无法回灌 → 需 `--additional-config '{"rl_config":{"enabled":true}}'`
- 共卡第二引擎 capture 失败（207005/EE1023）→ 用 `--enforce-eager`
- B eager 模式吞吐会低于 A（无 graph capture），生产上给睡眠方用 eager、
  活跃方用 graph 是合理分工

## 复现

```bash
# 租户 A（睡眠模式 + rl_config 必须）
VLLM_SERVER_DEV_MODE=1 vllm serve /model/Qwen3.5-4B \
  --served-model-name tenant-a --port 8000 --max-model-len 8192 \
  --enable-sleep-mode --gpu-memory-utilization 0.5 \
  --additional-config '{"rl_config":{"enabled":true}}'

curl -X POST "http://:8000/sleep?level=2"   # 1.77s, HBM→4480MB

# 租户 B（eager，因 stream 资源被 A 持有）
vllm serve /model/Qwen3.5-4B --served-model-name tenant-b --port 8005 \
  --max-model-len 8192 --enable-sleep-mode --gpu-memory-utilization 0.3 \
  --enforce-eager --additional-config '{"rl_config":{"enabled":true}}'

curl -X POST http://:8000/wake_up          # 0.139s，B 在线不受影响
```
