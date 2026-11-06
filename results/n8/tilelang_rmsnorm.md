# N8 冲刺：TileLang RMSNorm — eager 基线到手，kernel 侧定位到 0.1.30-npuir 系统性 bug 谱系，0.1.15 路线打通到差一步

日期：2026-11-06 17:00–17:35（北京时间），36.151 工作站（2×910B2，CANN 9.2.0）
环境底座见 [tilelang_env_36151.md](tilelang_env_36151.md)（tlasc-env venv 已复活）。

## 一、结果总览

| 侧 | 状态 |
|---|---|
| eager 基线（engine 口径 rms_chain / add_rms_chain） | ✅ 完整拿到，M∈{64,2048}×N=2560 |
| TileLang elementwise 冒烟（tl_add） | ✅ bit-exact PASS（上一轮） |
| TileLang K_a（bf16 copy + fp32 fragment 算术 + cast 出） | ✅ PASS —— 证明底座原语可用 |
| TileLang RMSNorm / fused add+rmsnorm | ❌ 未跑通 —— 定位为 0.1.30-npuir 的 rsqrt/sqrt/reduce 系统性 bug |
| tilelang 0.1.15 overlay 路线 | ✅ import 通、eager 基线复取，仓库自带 fused kernel 距编译只差 lazy-mode 判定一步 |

## 二、eager 基线（新数据）

bf16、eps=1e-6、N=2560（Qwen3.5-4B hidden），10 warmup + 50 iters，`torch.npu.synchronize` 包夹，µs/iter：

| M | eager rms_chain（4 op） | eager add_rms_chain（9-10 op） |
|---|---|---|
| 64 | 266–297 | 297–305 |
| 2048 | 271–286 | 303–308 |

（多轮跑动区间；两 M 差距很小 —— 与 N2/N8 结论一致：该尺寸链完全是 host-launch bound，
device 计算淹没在 ~30µs/op 的调度开销里。这正是融合的价值空间：单 kernel 理论上限
= 1×launch + 2×HBM 往返，M=64 时 HBM 项 <10µs，预测融合后 ~40–90µs，即 3–7×。）

## 三、TileLang 0.1.30-npuir bug 谱系（本轮核心产出，全部有实验证据）

用 add-kernel 已验证原语做拼装式二分（tl_bisect.py）：

| 实验 | 内容 | 结果 |
|---|---|---|
| K_a | bf16 GM→shared copy + fp32 fragment 平方 + (1+w) + T.cast 出 bf16 | **PASS**（编译+运行） |
| K_b | K_a + 逐元素 `T.rsqrt` | 编译过，**运行 core dump** |
| v1 | reduce_sum + 标量 rsqrt | 运行 core dump |
| v4/v5 | 手写 11 步树形归约（去 reduce_sum）+ 标量 rsqrt，bM∈{1,4} | 运行 core dump（排除 UB 容量） |
| build-only sqrt | `1.0/T.sqrt` 仅编译不运行 | **编译期 core dump**（复现两次） |
| reduce-only + 1D copy | `(bM,)` shared→GM 1D copy | MLIR `tensor.expand_shape` 验证失败（干净报错） |
| v*/npuir_* 烟测 | T.vrec/vsqrt/vrsqrt 及完整 rms 变体 | 编译期 DiagnosticError（形参语义不符，非 dump） |
| API 清点 | T.power / T.reciprocal | 不存在于 0.1.30 |
| **exp/log 绕开（17:34 冲刺）** | rsqrt = exp(-0.5·log(v))，exp/log/exp2/log2 | **编译+运行全通，但输出错误值**：smoke 出 nan，完整 RMSNorm max\|diff\|=68.75 —— transcendental 家族在 0.1.30-npuir 是**数值级损坏**（不崩但算错） |
| **完整 RMSNorm 里程碑（17:34）** | 树归约 + exp/log 组合的完整 fused RMSNorm | **首次编译+运行不崩**（`scripts/n8_rescue/tl_explog.py`）——证明除 transcendental 数值外全链路（copy/归约/fp32 数学/bf16 cast/融合写回）可用 |

**0.1.15 overlay 路线的最终真相（17:30–17:32 验证）**：`py311tl+tl015` 这份 0.1.15 的
ascend dialect 是 **Ascend 950 专用 build**（目标 dav-3510，`T.Kernel(N)` 1-D grid +
`T.SimtVF`/`T.SimdVF`，无 `is_npu`）；其 README 官方 GEMM 例子在 910B2 上**编译成功**
（生成 dav-3510 AscendC 代码）但**运行即 ERR99999**。烂尾脚本所用的 API
（`is_npu=True` 双层 grid、`PassConfigKey.TL_ASCEND_*`）对应的是**另一个 0.1.15 build**
（910 系支持）——机器上没有。overlay 路线对 910B2 判死刑。

## 六点五、续作两条现实路径（按可信度排序）

1. **0.1.30 上替换 transcendental 实现**（推荐，现有环境即可跑）：K_a 已证明纯算术
   （copy/乘加/cast）全安全。rsqrt 用分段查表 + 1–2 次牛顿修正
   （`y ← y·(1.5 − 0.5·v·y²)`，纯乘加）或多项式逼近实现，精度对 bf16 出
   （2⁻⁸）绰绰有余。骨架直接用 `tl_explog.py` 的完整 RMSNorm（已跑通，只换 inv 一行）。
2. **源码 build 910B2 支持的 tilelang-ascend**：机器外网通（github 可达），
   `USE_ASCEND=ON` 源码编译，`ASCEND_NPU_ARCH` 指定 910B2 目标。耗时但能拿到
   烂尾脚本原生 API（`is_npu`/`T.alloc_ub`/`PassConfigKey.TL_ASCEND_*`）。

## 七、环境回收记录

36.151 于北京时间 **17:35** 回收（连接拒绝），比预告的 17:30 晚 5 分钟。最后一发
`tl_diag.py`（log/exp 单独归因诊断）未能跑成——本地存档于 `scripts/n8_rescue/`，
续作时环境复活后可直接续跑。

## 七点五、产出物（续）

结论：**0.1.30-npuir 的 transcendental 家族（rsqrt 运行崩、sqrt 编译崩）与 reduce_sum
全坏**；elementwise/fragment/cast 底座完好。membound 旧脚本（tlwork）当年 core dump
的同源根因即此 —— N8 烂尾的技术死因定位完毕。

RMSNorm 数学上必含 rsqrt（或 sqrt+div），故在 0.1.30 上**不可实现**。

## 四、0.1.15 overlay 路线（续作的正路，已打通到最后一步）

发现：机器上 `/model/py311tl`（含完整 tvm_ffi C++ core）+ `/model/tl015`（tilelang 0.1.15
包体）可以 overlay 到 tlasc-env 上：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /tmp/tlrun   # /model 只读
PYTHONPATH=/model/py311tl:/model/tl015 /model/tlasc-env/bin/python script.py
# → import tilelang == 0.1.15，torch 仍 2.10.0+cpu（来自 venv site-packages）
```

`scripts/n8_rmsnorm_bench.py` 里烂尾时写好的 `make_tl_add_rmsnorm`（T.alloc_ub/T.tile.*
新版 API）正是 0.1.15 风格 —— 烂尾的技术路线其实是对的，只是当时没有能跑的环境。
本轮用 overlay 已跑到：`tilelang.jit(out_idx=[3,4], pass_configs=..., mode="lazy")` ——

- `PassConfigKey` 四个 key（TL_ASCEND_AUTO_CV_COMBINE 等）在该 0.1.15 build 中全部不存在
  （自动过滤后 pass_configs={}），说明**烂尾时针对的 0.1.15 与机器上这份 0.1.15 仍有 API 漂移**；
- 剩最后一步：`out_idx` 需 lazy mode，但 jit 装饰器无公开 `mode=` kwarg，mode 由
  `_is_lazy_style`（AST 检查"函数是否显式 return PrimFunc"）推断 —— 对本脚本的
  双层嵌套 def 判成了 eager。两条修法（续作 5 分钟内可试）：
  1. 把 kernel 改成 `T.empty()` 声明输出的 eager 风格（去掉 out_idx）；
  2. 看 `_is_lazy_style` 的 AST 判定条件，调整 `make_tl_add_rmsnorm` 的 return 形式。

## 五、产出物

- `scripts/n8_rescue/tl_add.py` — 跑通的 elementwise 冒烟（bit-exact）
- `scripts/n8_rescue/tl_bisect.py` — K_a/K_b/K_c 拼装二分（bug 谱系证据）
- `scripts/n8_rescue/tl_vops.py` — v*/npuir_* intrinsics 烟测
- `scripts/n8_rescue/tl_rmsnorm_v4.py` — 完整 RMSNorm（手写树归约，死于 rsqrt）
- `scripts/n8_rescue/tl_rms_sqrt.py` — sqrt 绕开尝试（死于编译期）
- `scripts/n8_rescue/tl_015_rms.py` — 0.1.15 overlay 路线主脚本（差 lazy 一步）
- 远端 `/tmp/tlrun/` 为可写运行目录（/model 只读，tilelang 编译产物落 CWD）

## 六、N8 续作启动命令

```bash
sshpass -p '937aed697a7c4973' ssh -p 40045 -o StrictHostKeyChecking=no root@36.151.162.7
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p /tmp/tlrun && cd /tmp/tlrun
# 若 venv 断链（机器回收后）：export PATH=/model/vllm-env/bin:$PATH && uv python install 3.11

# 0.1.15 overlay（推荐，仓库 kernel 是这个 API）：
PYTHONPATH=/model/py311tl:/model/tl015 /model/tlasc-env/bin/python -u /tmp/tl_015_rms.py
```

预期：修掉 lazy 判定后，fused add+rmsnorm 对 eager add_rms_chain（~303µs @M=64）
应有 3–7× 提升（N2/N8 预测的 launch 9→1 收益）。
