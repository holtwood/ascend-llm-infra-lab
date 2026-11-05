# 已提交（2026-11-05，GitHub holtwood）

| 草稿 | 上游实体 | 链接 |
|---|---|---|
| issue1 W8A16 compressed-tensors | **vllm-ascend #21112** (open) | https://github.com/vllm-project/vllm-ascend/issues/21112 |
| issue2 cudagraph SQ/CQ 耗尽 | **vllm-ascend #21113** (open) | https://github.com/vllm-project/vllm-ascend/issues/21113 |
| issue3 conv1d fallback | **#18978 下 comment**（第三方 iwantknow1 先报同缺陷，dedup 后降级） | https://github.com/vllm-project/vllm-ascend/issues/18978 |

时间线核实（GitHub API 未认证查询）：
- #21112/#21113：holtwood 2026-11-05T13:40 提交，内容与 docs/issues/ready/ 草稿逐字一致
- #18978：iwantknow1 2026-10-16 创建；holtwood 2026-11-05T13:41 追加 comment；
  另有 yjyang62 2026-10-17 的评论
- 提交前 dedup：issue1/2 上游无重复；issue3 与 #18978 同缺陷（#14620 删符号 +
  #15127 重新 import）→ 降级 comment 是正确决策

教训：提交前核验 = 查重复（search API）+ 查 mainline 历史（git log --all --grep）
+ 查是否已修（最新 release 行为），三步都过才提。

## 提交后核验（2026-11-05 复核，逐项对源）

| 核验项 | 方法 | 结果 |
|---|---|---|
| issue1 repro 路径真实 | M1 `ls /model/Qwen3.8-27B-W8A16` | ✅ 存在（M3 无，属 M1 资产） |
| issue1 源码行号属实 | M3 `sed -n 326,385p compressed_tensors_config.py` | ✅ `_detect_quant_type` 起于 326、`raise` 在 382，与正文一致 |
| issue2 repro 路径真实 | M3 `ls /model/Qwen3.8-27B-w8a8` | ✅ 存在 |
| issue1/2 无重复 | search API `w8a16` / `pack-quantized` / `207005` / `EE1023` | ✅ 检索结果中仅 #21112/#21113 命中主题（其余为旧 RFC/相邻问题，已在正文 Related 引用） |
| issue3 确为同缺陷 | 读 #18978 正文（#14620 删符号 + #15127 重新 import） | ✅ 同根因，降级 comment 正确 |
| comment 有增量 | 读 holtwood comment | ✅ 不同模型家族（Qwen3.5-4B GDN）+ 稳态 ITL 税数据 + 愿测补丁 |
| 维护者反馈 | 两 issue comments=0，#18978 无新追问 | 暂无争议，无需修正 |

结论：三份提交均无乌龙、无误判、无重复垃圾。issue 正文的 Related 引用
（#2265/#4267/#5350/#13318）在检索中逐一确认存在且为旧 RFC/feature 请求。
