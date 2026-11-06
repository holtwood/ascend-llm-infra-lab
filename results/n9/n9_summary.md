# N9: KV offload 分层缓存账本 + 正确性 demo（Ascend 910B2，2TB DRAM 主机）

环境：卡 1（ASCEND_RT_VISIBLE_DEVICES=1），torch 2.9.0 / torch_npu / CANN 8.5.1；容器 cpuset `8-49,155-175`，8 NUMA 节点（node0=0-23 为卡1域，node6=144-167 为卡0域），host DRAM 2011 GB。
脚本：`scripts/n9_copybench.py`（带宽 + NUMA + 账本）、`scripts/n9_offload_demo.py`（换出换回）。
原始产物：`n9_bench.log` / `n9_copybench.json` / `n9_ledger.md` / `n9_demo.log` / `n9_offload_demo.md`。

## 1. D2H/H2D 带宽微基准（50 次 p50）

| 大小 | D2H pageable | D2H pinned | H2D pageable | H2D pinned |
|---:|---:|---:|---:|---:|
| 64 MB | 0.69 | 26.77 | 0.73 | 23.73 |
| 256 MB | 0.74 | 28.13 | 4.66 | 24.78 |
| 1 GB | 4.32 | 28.49 | 4.26 | 24.81 |
| 4 GB | 1.26 | **28.59** | 1.27 | **24.98** |

- **pinned vs pageable = 25.5×**（4GB 档 28.59 vs 1.12 GB/s 平均）。pageable 路径不稳定且尺寸越大越差（1GB 4.3 → 4GB 1.26，页错误/搬运开销），pinned 稳定在 PCIe Gen4 x16 满线速附近。
- **跨 NUMA 税 = 1.47×**：1GB pinned D2H 在卡1域核（8-23）28.52 GB/s，绑到卡0域核（155-167）掉到 19.35 GB/s。offload 线程必须绑卡所属 NUMA 域；且 torch_npu 进程内 `sched_setaffinity` 换 mask 会 EINVAL，须用独立进程（taskset）绑。

## 2. Crossover 账本（offload vs 重算 vs 卡间迁移）

常数（全部本仓库实测）：KV 32 KB/token（N3）、prefill 2765 tok/s（N2）、卡间 16.3 GB/s（N4）、host pinned D2H 28.59 / H2D 24.98 GB/s（本实验）。

**等效重算带宽 = 2765 tok/s × 32 KB/tok = 91 MB/s。**重算与搬运都线性于 X，分界不在容量在带宽：换入通道只要 > 91 MB/s，搬回永远比重算快。pinned 往返（调和平均 26.66 GB/s）是分界线的 **294 倍**——本机硬件下"换出划算吗"的答案是无条件划算。

| X | tokens | 重算 prefill | host pinned 往返 | host pageable 往返 | 卡间迁移 |
|---:|---:|---:|---:|---:|---:|
| 256 MB | 8K | 2963 ms | 20 ms | 425 ms | 16 ms |
| 1 GB | 33K | 11851 ms | 81 ms | 1699 ms | 66 ms |
| 8 GB | 262K | 94808 ms | 644 ms | 13589 ms | 527 ms |

三方排序（1GB restore 视角）：卡间 66 ms < host pinned 81 ms << pageable 1699 ms << 重算 11851 ms。卡间只在对方卡有空闲 HBM 时可用（抢的是稀缺资源）；host 是 2TB 弹性容量，代价仅慢 23%。**即使 pageable 往返（1699 ms）也比重算快 7×**——最差的 offload 实现都赢过重算。隐藏收益不在账本内：H2D 是 DMA 不占 AI Core（重算排挤 decode 算力），且释放 HBM 换并发容量。

## 3. 正确性 demo：DynamicCache 换出→跑 B→换回（Qwen3.5-4B hybrid）

- prompt A 1525 tok（海盗 persona + 事实 + 两问），prompt B 203 tok；greedy
- 换出：64 个张量 **101.9 MB**（attn KV bf16 + GDN conv/recurrent state fp32 一起交接，dtype 混合确认），pageable `.cpu()` 96 ms；HBM 实测释放 114.9 MB（8737→8622 MB）
- 请求 B 在腾出的 HBM 内 prefill+decode 正常完成（输出为正常海盗腔中文）
- 换入：**88 ms**（pageable）/ **10 ms**（pinned 预分配 buffer）≈ 0.8 个 decode 步的停顿
- 正确性：换出→跑 B→换回后的 greedy 32 token 与不换出**逐 token bit-exact 一致**（ref==back True，零分叉）；换出 payload 与 N3 曲线核算吻合（1525 tok × 32KB KV ≈ 49 MB + GDN state ≈ 52 MB ≈ 102 MB）

实现要点：transformers 5.16.1 DynamicCache 支持 `layer.keys/.values/.conv_states/.recurrent_states` 属性级重赋值——张量搬到 CPU 后 HBM 真实释放（allocator 回收），恢复时 `.to('npu')` 原位放回，cache 对象结构不动，无需重建。

## 结论

1. 本机（910B2 + 2TB DRAM + PCIe Gen4）上 KV offload 到 host DRAM 在任何容量档、任何内存类型下都比重算划算（147×~7×），分界线由等效重算带宽 91 MB/s 决定，与本机带宽无关地成立——**只要 PCIe 带宽 > 0.1 GB/s 就该换出而不是丢弃重算**。
2. pinned 不是锦上添花而是 25× 量级的必需品；跨 NUMA 绑错域再打 7 折。offload 线程绑定 = pinned buffer = 卡所属 NUMA 域，三件套缺一不可。
3. hybrid 模型的 GDN recurrent state（fp32，50 MB 恒定）可以与 KV 一起无损换出换回：整层 DynamicCache 张量级交接是 bit-exact 的，"S 矩阵原子交接"实操可行。
