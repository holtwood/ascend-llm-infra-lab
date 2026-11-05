# 远端 vllm-ascend 从零部署实录（CANN 9.0 · aarch64 · 无 CUDA）

远端 npt 容器（8×910B1）原本只有 torch+torch_npu，本文是把它变成
能跑 vllm serve 的完整踩坑链。每一步都是实测过的。

## 1. vllm wheel：必须 --no-deps

`vllm==0.30.0` 的 aarch64 wheel 存在（310MB），但它的 METADATA **硬钉
`torch==2.13.0`**——直接 `pip install` 会把 torch 换成 2.13，毁掉与
torch_npu 2.10.0 的配对（torch_npu 按 torch 版本编译）。

正解：`pip install vllm-0.30.0-*.whl --no-deps`，然后从 wheel METADATA
抽取 `Requires-Dist`，**剔除 torch/torch-npu/torchvision/torchaudio 后**
生成 requirements 再装（保留 marker，让 pip 自己求值）。

## 2. 三个"CUDA 生态依赖"必须 stub

vllm 无条件 import 的包里有 CUDA 链接的 .so，在 CPU/NPU torch 上加载即炸：

| 包 | 症状 | 处理 |
|---|---|---|
| `torchcodec` | `OSError: libtorchcodec_image.so` 加载失败 | stub 包（文本服务不解码音视频） |
| `torchvision` | `ModuleNotFoundError`（被我们从 deps 剔除） | stub 包 + 假 dist-info（transformers 查版本元数据） |
| `fla_npu` | `ModuleNotFoundError`（华为私有包，PyPI 没有） | stub 包（只有 Qwen3.5/GDN 会真调用） |

stub 要点：
- 模块级 `__getattr__` 返回万能对象，但**类属性访问要过 metaclass**
  （`InterpolationMode.NEAREST_EXACT` 这类被当字典键用，必须给真 Enum，
  否则 `type object '_Any' has no attribute`）
- torchvision 需要 `transforms/functional.py`、`transforms/v2/functional.py`
  两个子模块路径都存在（transformers 直接 import 这两个）
- 带 `torchvision-0.25.0.dist-info/{METADATA,RECORD}` 骗过版本检查

## 3. vllm-ascend 安装：三个开关

| 方式 | 结果 |
|---|---|
| `pip install vllm-ascend==0.30.0rc2` | ✗ 构建依赖装不出来（走 pypi.org 极慢） |
| `pip install -e . --no-build-isolation` | ✗ 缺 `setuptools_scm` |
| 补 setuptools_scm 后 | ✗ `triton-ascend==3.2.2` 不在 PyPI（华为私有） |
| `--no-deps` + 上述 | ✗ 编译 aclnn 算子：拷贝过来的 CMakeCache 指向旧路径 |
| `COMPILE_CUSTOM_KERNELS=0` | ✗ 连核心扩展 `vllm_ascend_C` 也不编（setup.py 里 ext_modules 被一起关掉） |
| **清 `csrc/build`+`build_cache` 后全量编译** | ✓ 唯一可行路径 |

关键认知：`COMPILE_CUSTOM_KERNELS=0` **不是"跳过算子只编核心扩展"**，
而是把 `ext_modules` 整体置空——所以 vllm_ascend_C 也没了。想跳过 aclnn
只能靠编译成功一次，没有半量开关。

## 4. .so 跨机器不通用

从本机（CANN 9.2）拷贝的 `libvllm_ascend_kernels.so` 在远端（CANN 9.0）
报 `undefined symbol: rtFunctionGetMetaInfoSize`——**CANN 版本不同，
编译产物必须重编**，不能跨机器复用。

## 5. 环境变量三件套（非登录 ssh 必踩）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # .bashrc 默认只露 6 号卡
```

不 source 就 `import torch_npu` → `libhccl.so: cannot open shared object file`。
且 vendor 脚本引用未定义变量——**必须在 `set -u` 之外 source**，否则静默退出。

## 验证节点（每步都实测通过）

- `import vllm` → `0.30.0` ✓
- `from vllm.platforms import current_platform` → `PLATFORM npu` ✓（插件激活）
- `import vllm_ascend` → 可编辑树 ✓
- 剩余：`vllm_ascend_C` 编译 + serve 冒烟（本文档续更）
