# 上游 issue 提交指南（更新：重复检查后）

## 重复检查结论（2026-11，GitHub API 搜索 + 正文核对）

| 草稿 | 上游判定 | 处置 |
|---|---|---|
| issue1 W8A16 | 无重复 issue；相关 RFC #5350（compressed-tensors 支持）、#13318（量化重构） | **发新 issue**，引用两个 RFC |
| issue2 cudagraph 207005 | 同族 bug #2629 已 closed（MoE 专用修复 #2511）；RFC #9691 把"capture-size 剪枝"列为最高优先 | **发新 issue**——触发条件不同（dense 模型 + 默认 capture 列表），并给 #9691 的 profiling 方案提供实测论据 |
| issue3 causal_conv1d | **已有 open issue #18978 覆盖同一缺陷**（patch_triton 重新引用被 #14620 删除的符号）；修复 PR #13489 open | **不发新 issue**，改为在 #18978 下 comment 补充：Qwen3.5 GDN 模型族 + 0.30.0rc2 + ITL p99 实测数据 |

打磨后文件在 `ready/`：
- `issue1_w8a16_compressed_tensors.md` — 新 issue（quantization/bug）
- `issue2_cudagraph_stream_exhaustion.md` — 新 issue（compilation/bug）
- `issue3_comment_on_18978.md` — **comment**，不是新 issue

## 实际提交（需要本人 GitHub 凭据——停在这里等批准）

```bash
gh auth login   # 需要你的 github.com PAT

# 两个新 issue：
gh issue create --repo vllm-project/vllm-ascend \
  --title "[Bug] Weight-only W8A16 compressed-tensors checkpoints fall through _detect_quant_type and abort with bare NotImplementedError" \
  --body-file docs/issues/ready/issue1_w8a16_compressed_tensors.md

gh issue create --repo vllm-project/vllm-ascend \
  --title "[Bug] Default cudagraph_capture_sizes still exhausts NPU SQ/CQ stream resources on dense models (AclmdlRICaptureBegin 207005)" \
  --body-file docs/issues/ready/issue2_cudagraph_stream_exhaustion.md

# 一个 comment：
gh issue comment 18978 --repo vllm-project/vllm-ascend \
  --body-file docs/issues/ready/issue3_comment_on_18978.md
```

## 提交注意

- 两份文件已按 AGENTS.md 要求在文末附 AI-assistance 披露
- issue2 正文已包含验证过的 workaround（`--compilation-config`），maintainer 更愿意接
- issue3 的 comment 结尾问了 maintainer 后续路线（AscendC kernel vs 批量化同步），并主动提出可在真实硬件上测 patch——提高被响应概率
- 提交后把返回的 issue URL 记进 `docs/status.md`
