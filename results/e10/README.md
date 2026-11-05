# E10 — Sleep/Wake 模式：多租户场景的快速换租能力

> 面试原题："模型频繁换租/冷启动怎么优化？" Ascend sleep mode 实测：
> 释放几乎全部 NPU 显存、恢复 <1s——比重新 serve（~2.5min）快 ~150×。

## 前提（两个开关缺一不可）

```
vllm serve ... --enable-sleep-mode              # 模型侧开关
VLLM_SERVER_DEV_MODE=1 vllm serve ...           # HTTP 路由开关（/sleep /wake_up /is_sleeping）
--additional-config '{"rl_config":{"enabled":true}}'   # Ascend 必需，见坑 #2
```

## 实测（qwen35-4b @ util0.5，910B2）

| 操作 | 耗时 | 进程显存（npu-smi） |
|---|---|---|
| baseline | — | 31,994 MB |
| `POST /sleep?level=1` | **2.99 s** | —（level1 仅释放 KV+激活） |
| `POST /sleep?level=2` | **1.90 s** | **1,004 MB（释放 96.9%）** |
| `POST /wake_up` | **84 ms** | 31,826 MB |
| 唤醒后首个请求 | **0.20 s** | — |
| 参考：冷启动 serve | ~150 s | — |

- level=2 把**权重也卸载到 host**，显存几乎清零（31.2G→1.0G）
- wake_up 返回仅 84ms——权重回灌极快（NZ 格式 + host pinned，DMA 直回）
- 唤醒后推理立即恢复（首请求 0.20s，与 warm 持平）

## 面试可用的机制解释

- 用途：**在线权重热切换 / 多租户时分复用单卡 / RL 训练中
  rollout-train 交替**（vLLM 官方动机）。比进程重启省掉：
  权重加载 + torch.compile + ACL graph 捕获（我们测 ~2.5min）。
- 为什么 Ascend 需要 `rl_config`：权重以 **FRACTAL_NZ** 布局存显存
  （昇腾计算友好格式），sleep 回灌时普通格式路径不认 → 报错 500。
  `rl_config.enabled=true` 自动禁用 NZ → 可正确回灌。
- level 语义：1=保留权重（只释放 KV/激活，换租但模型还在卡上）、
  2=权重也走（彻底腾空，适合模型轮换）。

## 踩坑实录

1. `/sleep` 404：路由需 `VLLM_SERVER_DEV_MODE=1`（dev router）。
2. wake_up 500：`FRACTAL_NZ mode is enabled...` → rl_config 或
   `weight_nz_mode=0`。
3. `npu-smi info` 显存列在进程表里（Process memory MB），
   芯片行显示的是温度/利用率。

数据：`serve_sleep.log`（404 复现）、`serve_sleep2.log`（NZ 报错复现）、
`serve_sleep3.log`（成功链路）、`serve_restore.log`。
