# 路线图 v3 —— AI Infra Serving 面试加分的开发/实验差集

截至 E14 后的差集分析。已完成：E1–E14、E3b、E4b、E6、E7、E10、E11、E13，
加上并行会话的 E8b（算子级手撕）、E8c（手写 draft/verify）。
基线/调度/容量/量化精度/PD/sleep/spec 均有数据；剩余价值集中在
**真实工程贡献**和**少数未覆盖的系统面**。

## 环境资产盘点（别人没有的）

- 4 节点 910B 真集群 + nginx 4 副本拓扑（可 A/B 路由策略）
- `/vllm-workspace/vllm-ascend` 源码树在机，可改可提 PR
- hybrid 模型（Qwen3.5-4B：24×linear + 8×full-attn + **自带 MTP 头**）
- 27B 三变体（bf16/w8a8/w8a16）、Mooncake、全套未探调度器
- 三份已写好的上游 issue 草稿（docs/issues/）

## P0 —— 开发类（面试区分度最高）

| 项 | 面试原题 | 成本 | 依据 |
|---|---|---|---|
| **D1. 修 conv1d fallback 并提 PR** | "给开源提过什么？" | 1–2 天 | `vllm_ascend/ops/triton/mamba/causal_conv1d.py` 只有 `extract_last_width`，`causal_conv1d_update_npu` 真缺；补齐后消掉每请求同步（E14 已实测其放大 MTP 开销）。issue 草稿已有，做成修复 PR 是最强信号 |
| **D2. 提交 3 个 issue 草稿** | 开源参与度 | ~1h | docs/issues/ 已就绪，只需英文校对后提交 vllm-ascend |
| **D3. prefix-aware 路由** | "多副本怎么路由？sticky session？" | 半天 | 替换 nginx least_conn → 按 prompt 前缀 hash 选副本，复用 E4 拓扑；测 prefix-cache 命中率与 TTFT 变化，对比 least_conn 基线（数据已有） |

## P1 —— 实验类（补系统面空白）

| 项 | 面试原题 | 成本 | 依据 |
|---|---|---|---|
| **E14 ✅ 已做** MTP spec decode | "spec decode 何时有用" | 已完成 | 65.6% 接受率但 TPOT 2.3× 负收益；接受率≠净收益 |
| **E16 长上下文扫描** | "长上下文 serving 成本？" | ~1h | 模型支持 262k；扫 ctx∈{4k,16k,32k,64k} 测 TPOT/TTFT/KV 占用曲线，串起 E8（容量公式）+ E8b（访存墙）成完整故事 |
| **E17 sleep/wake 多租户换租 demo** | "多租户/弹性伸缩？" | 半天 | E10 已证机制；写 driver：流量 → sleep 空闲副本 → 请求到来唤醒 → 对比冷启动 ~150s vs 唤醒 0.3s，产出可演示叙事 |
| **E18 PD 跨节点重试** | "PD 分离什么时候有用？" | 1–2h | E7 单卡共置失败在建链；P=M3、D=M1 跨节点重试——成功即正面数据，失败即"跨节点 hixl 通道"新坑，双向都有故事 |
| **E19 其余调度器画像** | "调度策略 trade-off" | 每个 ~20min | `recompute_scheduler`、`dyntra_lb`、`profiling_chunk`、`short_request_first` 按 E13 模式各跑一次 A/B，凑齐 Ascend 调度器全景表 |

## P2 —— 锦上添花

- **E12 block-size**：hybrid 模型 block 被统一页锁到 640（E8 已证实），
  只值一节"为什么不能调"的说明，不值得单独实验
- **结构化输出/引导解码 bench**、observability 仪表盘：常规内容，
  有富余再做
- **E15 Ascend C 算子**：若选 D1（conv1d 修复）则部分重叠；纯 Ascend C
  融合算子是 1–3 天成本的最高稀缺信号，时间充裕才做

## 叙事层（半天，面试前必做）

- `docs/serving-walkthrough.md`：一个请求穿过 api_server → processor →
  (Batch)Scheduler → GPU model_runner → gdn/attention kernel 的源码级
  走查，标注我们改过的、踩过的点
- `docs/interview-notes.md` 收尾：每个实验一句"被问到 X 就讲 Y 数字"
- hero 叙事主线建议：**"同一台 910B2 上把 vLLM 的每个主流serving优化
  都量化了一遍：有效的是哪些、多少、为什么；无效的是哪些、为什么"**

## 建议顺序

D2（1h，issue 落袋）→ E16（1h，长上下文故事闭环）→ D3（半天，路由开发）
→ E17（半天，sleep 叙事闭环）→ D1（1–2 天，conv1d PR）→ E19/E18 视余量
→ 叙事层文档。
