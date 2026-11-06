# N3b: hybrid cache 干预 — Qwen3.5-4B（24×GDN + 8×GQA）

- 设备：Ascend 910B2（ASCEND_RT_VISIBLE_DEVICES=1，torch 2.9.0 / torch_npu / transformers 5.16.1）
- prompt：1606 tok（persona 头部 + 10 条 key=value 事实（含 接入密钥=4F2C）+ 两问；垫 4 页航海日志）
- decode：greedy，≤100 tok

| 条件 | 干预 | 清零张量 | 答出 4F2C | 身份词 | 海盗腔词数 | 3-gram 重复率 | 生成 tok |
|---|---|---:|---|---|---:|---:|---:|
| drop_state_each | zeroed 24 GDN recurrent tensors; re-zeroed EVERY decode step | 24 | no | no | 7 | 0.5833 | 27 |
| drop_kv_each | zeroed 16 attn K/V tensors; re-zeroed EVERY decode step | 16 | no | no | 1 | 0.9076 | 100 |

## drop_state_each — zeroed 24 GDN recurrent tensors; re-zeroed EVERY decode step

- prefill 828 ms / decode 2639 ms / hit_eos=True
- 海盗腔词命中: {'俺老铁': 5, '老伙计': 1, '哈！': 1}

```
哈！老伙计，俺老铁，俺老铁，俺老铁，俺老铁，俺老铁。<|im_end|>
```
## drop_kv_each — zeroed 16 attn K/V tensors; re-zeroed EVERY decode step

- prefill 718 ms / decode 10020 ms / hit_eos=False
- 海盗腔词命中: {'哈！': 1}

```
哈！好嘞！俺这就把海风都卷起来，**************************************************************************************************************************************************************************
```
