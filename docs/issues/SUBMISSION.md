# 上游 issue 提交指南

三份草稿已核对环境/版本/复现路径，质量达到提交标准。
**实际发帖需要本人 GitHub 账号**（目标仓库 vllm-project/vllm-ascend）——
此步留给你手动执行，或给我 gh token 后我来跑。

## 提交方式

```bash
# 有 gh CLI 时：
gh auth login   # 需要你对 github.com 的 PAT
gh issue create --repo vllm-project/vllm-ascend \
  --title "<标题>" --body-file docs/issues/issueN_*.md
```

## 建议标题与标签

| 文件 | Issue 标题 | 建议 labels |
|---|---|---|
| issue1 | `NotImplementedError serving weight-only (W8A16) compressed-tensors model on NPU` | quantization, bug |
| issue2 | `Default cudagraph_capture_sizes exhausts NPU stream resources: AclmdlRICaptureBegin error 207005` | compilation, bug, good first issue |
| issue3 | `causal_conv1d_update NPU Triton kernel missing → per-request sync stalls decode (Qwen3.5 hybrid models)` | ops, performance |

## 发帖注意

- issue2/issue3 建议先说 workaround（我们已验证），maintainer 会更愿意接
- issue3 与 D1 的 conv1d 修复 PR 关联：发 issue 时注明 "we are preparing
  a fix PR"，避免与 maintainer 重复工作
- 若上游已有同类 issue（提交前搜 "causal_conv1d" / "capture_sizes" /
  "compressed-tensors"），改为 comment 补充实测数据
