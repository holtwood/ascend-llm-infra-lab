# E-W8A16 · #21112 落地验证：detector patch + 真权重内核验证

> 上游 issue https://github.com/vllm-project/vllm-ascend/issues/21112 的本地闭环验证。
> 证据已作为 comment 发回 issue（comment-6660784746）。

## 验证链

| 层级 | 方法 | 结果 |
|---|---|---|
| bug 复现 | `AscendCompressedTensorsConfig.from_config` + weight-only INT8 channel config → `_create_scheme_for_layer_type` | `NotImplementedError: No compressed-tensors compatible quantization type was found.` 精确复现 |
| patch | `_detect_quant_type` 加 `_is_w8a16` 分支（镜像上游 `_is_wNa16_group_channel`：channel/group + static + input None + int8） | 检出 → `AscendW8A16LinearMethod` |
| 回归 | w8a8-dynamic mock → `AscendW8A8DynamicLinearMethod` 不变；w4a16-linear 无 scheme 为 pre-existing gap | 无回归 |
| 真权重 | `scripts/make_w8a16_ckpt.py`：Qwen3-0.6B bf16 → per-channel INT8 + bf16 scale + zero offset，weight-only compressed-tensors config | checkpoint 1.0G |
| 权重加载 | `vllm serve` 走完全部加载路径 | `Loading model weights took 0.7309 GB`（int8 大小正确，bf16 应 ~1.3G） |
| 量化内核 | NPU 实测 `npu_weight_quant_batchmatmul`（真实层 vs bf16 反量化参考） | **max rel diff = 0.13%** ✓ |
| HTTP serve | util 0.08 在共卡 5.45G 空闲下起服 | warmup OOM 差 298MB——纯显存争用，非量化路径问题 |

## 文件

- `patch_w8a16_detector.diff`——源码改动（可 `git apply` 到 vllm-ascend）
- `../../scripts/repro_w8a16_detector.py`——单元级复现/验证脚本
- `../../scripts/make_w8a16_ckpt.py`——真 W8A16 checkpoint 构建器

## 坑位记录（共卡实验）

- 本机 PYTHONPATH 含裸 `/vllm-workspace` → `import vllm` 命中仓库根目录变
  namespace 包；serve 需 `env -u PYTHONPATH` 或只留 CANN 路径
- 8001 端口是对端 lane 的 SSH 隧道转发远端引擎——探测端口要先确认监听者身份
- serve 日志里 `Free memory on device` 显示的是**剩余可分配**，共卡时
  gpu_memory_utilization 按总量比例算，容易超限
- sleep-mode 的对端引擎 sleep 时 HBM 释放到 3.5G 但 API/资源元数据仍在；
  共卡起第二引擎要 --enforce-eager 减小 stream/CQ 需求

## 遗留

- [ ] 完整 HTTP serve 冒烟（等单卡空出，util≥0.09 + 模型需 ~5.5G 池）
- [ ] 若 maintainer 认可方向 → PR（按 vllm-ascend AGENTS.md：signed-off
      conventional commit + 单测覆盖 weight-only INT8 检测路径 + 人工逐行 review）
