# E8j · PagedAttention 开销实测：分页税到底在哪

> 单步 decode attention：连续 KV vs 页表 gather 物化 KV（910B1，torch_npu）。
> 脚本 `scripts/e8j_paged_kv.py`，数据 `../e8j_paged.json`。

## 结果（q=[1,32,1,128] 单步 decode，bf16）

| KV 长 | 连续 SDPA | 页表 gather+SDPA（税倍数） |
|---|---|---|
| 8k | 0.152ms | bs16 **3.27×** / bs64 5.76× / bs256 15.6× |
| 32k | 0.536ms | bs16 4.27× / bs64 4.49× / bs256 7.3× |
| 64k | 1.044ms | bs16 4.33× / bs64 4.34× / bs256 5.25× |

## 结论

1. **分页间接必须在 kernel 内部做**：张量级 gather 每步把整段 KV 物化一遍
   （等于多读一份全量 KV + permute），税 3-16×——PagedAttention 的精髓是
   "页表间接寻址长在 attention kernel 访存里"，不是省拷贝而是**不拷贝**
2. 长 KV 下税收敛到 ~4.3×（额外全量拷贝的带宽下限）；小页/长 KV 组合
   最接近下限——这与 vLLM 块长 16 的选择方向一致
3. 大页+短 KV 最差（15.6×）：整页物化读出大量 padding 无效字节
4. 对上 walkthrough §5：vLLM block manager 的 `paged` 实现走
   `reshape_and_cache` 原位写页 + attention kernel 按 block_table 寻址——
   两步都零物化，本实验给出了"如果做错会差多少"的量化下界
