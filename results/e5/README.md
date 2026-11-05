# E5 · MindIE vs vLLM-Ascend：一次被 ABI 阻断的对比（附完整排障链）

## 结论先行

**MindIE 3.0.0 无法在本实验环境运行**——其全部分发二进制
（daemon、`_libatb_torch.so`、grpc/re2/cares 运行库）均按 **glibc 2.38
（Ubuntu 24.04）** 链接，而全部 4 台容器均为 **Ubuntu 22.04 / glibc 2.35**。
这本身是一个完整的面试故事：闭源推理栈的平台锁定 vs vLLM-Ascend 的
Python 源码可移植性。

## 环境拓扑

- 平台 vLLM 镜像：Ubuntu 22.04.5 + CANN 9.2.0 + torch 2.10 + vLLM 0.30
- M1 `/model/mindie/`：前任租户留下的 MindIE 3.0.0 部署包
  （`Ascend-mindie_3.0.0_linux-aarch64_abi1.run` + CANN 8.5.1 配套包 +
  cp311 whl + atb-models 源码）+ 已损坏的 venv/python3.11 残留

## 排障链（每一步都是真实部署坑）

| # | 障碍 | 根因 | 解法 |
|---|---|---|---|
| 1 | 安装器直接拒绝 | 需先 source CANN `set_env.sh` | 补环境 |
| 2 | `mindie-service Install failed` + bash usage dump | `install.sh` 按 `py${ver}` 选子包：只有 py311 构建，系统 python3=3.12 → glob 为空 | `apt install python3.11` + PATH shim |
| 3 | `/model/mindie/python/bin/python3` 不存在 | 前租户自编译 py3.11 残留为断链 | 弃用，改用 apt 3.11 |
| 4 | `libtorch.so` | daemon 需 torch 运行库 | pip 装 torch-2.9.0/torch_npu-2.9.0 cp311 whl |
| 5 | `libmindie_llm_utils.so` | mindie-llm 库路径未进 LD | `mindie/set_env.sh` + 显式 LD 补丁 |
| 6 | `libpython3.11.so.1.0` | apt python3.11 不含共享库 | `apt install libpython3.11` |
| 7 | **`GLIBC_2.38 not found`** | **全部 .so 按 Ubuntu 24.04 工具链构建** | **无解（不能换系统 glibc）** |
| 8 | atb_llm Python 绕行尝试 | `ModelRunner` → `transformers`/`MINDIE_LLM_BENCHMARK_FILEPATH`/`ATB_SPEED_HOME_PATH` 逐个补齐后，`_libatb_torch.so` 同样 GLIBC_2.38 | 同一堵墙 |

## 对比维度（定性，因实测被阻断）

| 维度 | MindIE 3.0.0 | vLLM-Ascend 0.30rc2 |
|---|---|---|
| 分发形态 | 闭源 .so + makeself 安装器 | Python 源码 editable 安装 + pip |
| 平台耦合 | **glibc 版本硬绑定**（2.38 起）+ CANN 大版本绑定 + Python 小版本绑定（仅 py310/311 whl） | 主要依赖 torch_npu/CANN 运行时，Python 3.10-3.13 均可 |
| 安装面 | CANN toolkit + nnal + mindie + mindie-llm + atb-models + 自编译 Python + 证书体系 | `pip install -e vllm + vllm_ascend`，env.sh 一页环境变量 |
| 服务化 | mindieservice_daemon + config.json（TLS/mTLS/管理面/多后端） | `vllm serve` OpenAI API |
| 出问题时的可观测性 | 报错只到"Install failed"，需逐层拆 makeself/子安装器 | Python traceback 直达源码行 |
| 配套生态 | 官方模型清单 + 认证测试报告 | 上游 vLLM 全特性（spec decode/chunked prefill/前缀缓存可逐开关消融） |

## 面试话术

1. "同箱实验"：同一块 910B2 上 vLLM-Ascend 从 clone 到出数据半小时；
   MindIE 部署链 8 步排障最终停在 glibc ABI——**引擎选型的隐性成本是
   平台版本锁定**，这就是为什么大厂内部栈都重新打包发行版对齐基座
2. ABI 排查方法：`ldd --version` 对齐 + `readelf -V`/`objdump -T` 看
   `GLIBC_` 需求符号版本；makeself 包用 `--extract` 拆开看内层 install.sh
3. 若拿到 Ubuntu 24.04 + CANN 8.5.1 的目标环境，本仓库 `docs` 里的
   排障链可复用（py311 + libpython + torch whl + 三个 env 变量）

## 数据

无 benchmark 数据（无法启动）。安装与排障日志：`mindie.log`、
`docs/status.md` E5 条目、M1:/var/log/mindie_log/mindie_install.log。
