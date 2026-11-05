# E6 · 量化精度评测（W8A8 vs BF16）

两条臂：

- **PPL/NLL 臂（同模型配对对比）**：`/model/Qwen3.8-27B-w8a8`（vLLM-NPU serve,
  `prompt_logprobs`）vs `/model/Qwen3.8-27B` bf16（transformers 纯 CPU forward,
  192 核 2TB 内存——bf16-27B 装不进 64G HBM，CPU 是唯一能跑的参照路径）。
  同一 `ppl_corpus.txt`（27 段 ~2409 token），逐 chunk 算 mean NLL。
- **QA 臂**：自构造 `qa_set.jsonl`（105 题：cn_knowledge 30 / math 25 /
  en_knowledge 20 / cs_basics 15 / logic 15），两种打分法：
  - 生成式（`e6_qa_eval.py`）：completions API 生成 → 去 `<think>` 提取答案
  - 选项 logprob 打分（`e6_qa_score.py`）：比较 P(letter|prompt) 取 argmax，
    serve/CPU 两侧同方法学，专为"量化前后"配对对比设计

## PPL 臂：逐 chunk 配对 NLL

| chunk | n_tok | w8a8 nll | bf16 nll | Δnll(w8a8−bf16) |
|---|---|---|---|---|
| 0 | 124 | 1.7774 | 1.8030 | −0.026 |
| 1 | 84 | 2.8314 | 2.7085 | +0.123 |
| 2 | 95 | 1.2254 | 1.2366 | −0.011 |
| 3 | 94 | 2.0577 | 2.0744 | −0.017 |
| …（CPU 臂以 ~13min/chunk 持续推进，见 ppl_bf16_cpu.log / ppl_bf16-27b.json） |

w8a8-27b 全语料汇总：**mean_nll=1.6086, ppl=5.00**（2382 token 命中，
coverage 2382/2409=98.9%）

**Top-1 argmax 一致性（393 个配对位置）：93.1%**
（chunk 级 87.1%~100%；w8a8 greedy 预测与 bf16 在 ~7% 的 token 上分叉——
注意分歧同样包含"量化+实现路径"混合差异）

**初步结论：Δnll 在 ±0.13 内且方向不一致、top1 一致性 93.1%——
W8A8 的精度代价处于"实现差异"噪声边缘（vLLM-NPU 量化 kernel vs
transformers-CPU bf16 路径的数值差）。** 即：在该语料/样本量下测不出
统计显著的精度损失，这本身就是"动态 W8A8 精度代价很小"的证据；
更精细的区分需要更大语料或同实现路径的对照。

## QA 臂

### 生成式打分（105 题，含数学数值题）

| 模型 | ALL | cn | en | cs | logic | math |
|---|---|---|---|---|---|---|
| qwen35-4b bf16 | 94.3% (99/105) | 100% | 100% | 100% | 86.7% | 84.0% |
| **qwen38-27b w8a8** | **94.3%** (99/105) | — | — | — | 93.3% | 80.0% |
| qwen3-8b bf16 | 84.8% (89/105) | 96.7% | 95.0% | 93.3% | 80.0% | 60.0% |

### 选项 logprob 打分（80 MCQ，`qascore_*`）

| 模型 | MCQ acc |
|---|---|
| qwen38-27b w8a8 | **96.2%** (77/80) |
| qwen3-8b bf16 | 92.5% (74/80) |

### 解读

1. **W8A8-27B 没有精度崩塌**：生成式 94.3% 追平 bf16-4B 的天花板水平，
   MCQ logprob 打分 96.2% 高于 bf16-8B（92.5%）——量化后仍明显高于
   小一档 bf16 模型，"用量化换可部署性"的账成立。
2. **该题库对 27B/4B 已接近天花板**（94-95%），区分度有限；
   logic/math 是主要失分项（W8A8-27B math 80% vs 4B 84%——
   差距在题库噪声量级内）。
3. **方法学教训（本次踩坑记）**：
   - 思考型模型（Qwen3.8-27B 输出 `<think>` 推理块）用 `max_tokens=8`
     会把输出截在 think 里 → 第一版 76.2% 是解析失败假信号；
     修正：max_tokens≥768 + 去 think 块后匹配 → 94.3%
   - `prompt_logprobs` 打分必须按 **token id** 查该字母自己的 logprob；
     取 max(top5) 会让 4 个变体得分趋同 → 曾测出 8.8% 伪随机值
   - CPU 侧跑 qwen3_5 模型必须 `source env.sh`（torch_npu 链式依赖
     libhccl.so/libascend_hal.so），即便 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`

## 局限

- **同模型同方法学的 bf16-27B QA 不可行**：CPU 单 forward ~13min
  （Qwen3.8 hybrid 架构 causal-conv/linear-attn 算子在 CPU 上是逐 token
  串行 fallback），90 题 MCQ 需 ~20h → bf16-27B 侧只保留 NLL 对照
- PPL 语料偏小（~2.4k token），Δnll 置信区间宽
- serve 侧 NLL（NPU kernel）与 CPU 侧 NLL（transformers eager）存在
  实现路径差异，Δnll 解读为"量化+实现"混合差

原始数据：`ppl_w8a8-27b.json`、`ppl_bf16-27b.json`（CPU 臂增量写盘）、
`qa_*-v2.json`、`qascore_*.json`；CPU 日志 `ppl_bf16_cpu.log`（M1）
