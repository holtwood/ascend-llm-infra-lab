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
