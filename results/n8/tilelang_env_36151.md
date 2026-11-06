# N8 续作环境：36.151 工作站 tilelang-ascend 探矿结果

日期：2026-11-06（北京时间 ~17:00 完成验证）
机器：36.151.162.7:40045（root / sshpass），2×910B2（NPU 4/5，各 64GB HBM），CANN 9.2.0（npu-smi 25.2.1），Ubuntu 22.04 aarch64，42 核。
验证时两卡均空闲（HBM ~3.7GB/3.4GB 基线占用，无进程）。

## 一、环境定位

所有候选目录都在 `/model` 下（**/model 是只读文件系统**，编译输出不能落在里面）：

| 路径 | 性质 | 结论 |
|---|---|---|
| `/model/tlasc-env` | uv 创建的 python3.11 venv，装好 tilelang 0.1.1.30+npuir / torch 2.10.0 / torch_npu 2.10.0.post4 | **用这个** |
| `/model/tl015` | tilelang 0.1.15 裸包目录（site-packages 内容摊开） | 备查 |
| `/model/py311tl` | 同上（tilelang 0.1.15 + tvm_ffi 0.1.12） | 备查 |
| `/model/tlasc-npuir` | 只有 tilelang 包目录 | 备查 |
| `/model/tlasc-test` | 现成示例（`example_gemm.py`、`npu_utils.so`） | 跑通了 |
| `/model/tlwork` | N8 烂尾时的历史工作目录（npuir_membound.py、npuir_final_bench.py、attn_decompose.py、tl950_run.py 等） | 脚本参考 |
| `/model/tilelang-0.1.1.30+ubuntu.22.4.npuir-cp311-cp311-linux_aarch64.whl` | wheel | 已装进 tlasc-env |

**关键修复（本次探矿的核心发现）**：tlasc-env 完好，但其 base python 是 uv 管理的
`/root/.local/share/uv/python/cpython-3.11-linux-aarch64-gnu/`，该目录曾被删（环境回收清理），
导致 venv 里所有 python symlink 断链，表现为 `bin/python: No such file or directory`。

修复只需一行（uv 0.12.21 在 `/model/vllm-env/bin/uv`，机器对外网通，实测 12.6s 装完）：

```bash
export PATH=/model/vllm-env/bin:$PATH
uv python install 3.11     # cpython-3.11.16 装回原路径，venv 立即复活
```

## 二、环境组合（已验证）

- python：**3.11.16**（cpython-3.11.16-linux-aarch64-gnu，uv 管理）经 `/model/tlasc-env/bin/python`
- tilelang：**0.1.1.030+npuir+4e1bb3e494b0bd70a55ec0b7ddaa78a4b6a93333**（wheel 0.1.1.30+ubuntu.22.4.npuir-cp311）
- torch：**2.10.0+cpu**（venv 内含 CPU 版 torch + torch_npu 2.10.0.post4）
- CANN：**9.2.0**（`/usr/local/Ascend/cann-9.2.0`，`ascend-toolkit/set_env.sh`）
- `torch.npu.is_available() == True`，device_count = 2
- import 时的良性警告：`CUTLASS is not installed` / `Composable Kernel is not installed`（npuir target 不需要，可忽略）

## 三、跑通的 kernel

### 1. 现成示例：GEMM 1024×1024×1024 fp16（`/model/tlasc-test/example_gemm.py`）

- `tilelang.jit(out_idx=[-1], target="npuir")` + `T.Kernel(..., is_npu=True)` + `T.gemm` + `T.Pipelined(num_stages=2)`
- 输出 `All check passed.`（kernel 结果与 `a @ b` 逐元素一致）
- AscendNPU IR compile success，端到端（含首次编译）数十秒内

### 2. 自写 kernel：elementwise add 4096 fp16（本次新写，`/tmp/tlrun/tl_add.py`）

- 最简 `T.Kernel` + `T.copy` + `T.Parallel` 循环，验证"任意新写的 npuir kernel 可编译并运行"
- `max diff: 0.0`，bit-exact（fp16 加法逐位相等），`tl_add PASS`

### 3. 反例（如实记录）：`/model/tlwork/npuir_membound.py` core dump

- N8 烂尾时的历史脚本（rmsnorm+add 内存受限融合基准）在当前组合下运行即 core dump
- 出现在 tensor 创建后、编译前，疑与历史脚本依赖的旧版 tilelang 行为或 CANN 32-padding 告警路径有关
- 未继续追查（限时任务）；N8 续作应从新写 kernel 起步，勿直接复用 tlwork 脚本

## 四、踩坑清单

1. **/model 只读**：tilelang 编译会把 `npu_utils.so` 等中间产物写到 **CWD**，`cd /model/...` 下跑直接
   `Read-only file system` 链接失败。必须 `cd` 到可写目录（`/tmp`、`/root`）再跑。
2. **uv base python 被删**：见上，`uv python install 3.11` 一键修复。
3. **npu-smi 裸跑报错** `libc_sec.so cannot open`：必须先 source set_env.sh。
4. **卡位**：本机 NPU 编号是 **4 和 5**（不是 0/1），单卡实验 `ASCEND_RT_VISIBLE_DEVICES=4`。

## 五、N8 续作启动命令

```bash
sshpass -p '937aed697a7c4973' ssh -p 40045 -o StrictHostKeyChecking=no root@36.151.162.7

# 每次会话固定前奏：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p /tmp/tlrun && cd /tmp/tlrun          # 编译产物落可写目录
PY=/model/tlasc-env/bin/python                # 若报 No such file or directory：
# export PATH=/model/vllm-env/bin:$PATH && uv python install 3.11

# 冒烟（~1 分钟）：
$PY /model/tlasc-test/example_gemm.py         # 期望 All check passed.
$PY /tmp/tlrun/tl_add.py                      # 期望 tl_add PASS（max diff 0.0）

# 跑自己的 kernel：写成脚本放 /tmp/tlrun，模板抄 example_gemm.py
#   @tilelang.jit(out_idx=[-1], target="npuir") + T.Kernel(..., is_npu=True)
#   os.environ["TILELANG_ASCEND_MODE"] = "Developer"

# 看历史 N8 脚本 API 用法（但勿直接跑，membound 会 core dump）：
ls /model/tlwork/
```

注意：**该环境 17:30（北京时间）回收** —— 下次续作若机器已重置，重做 `uv python install 3.11` 即可，
venv 本身在 /model 上持久。
