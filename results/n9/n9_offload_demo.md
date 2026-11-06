# N9 part3: DynamicCache 换出/换回正确性 demo（Qwen3.5-4B hybrid，含 GDN recurrent state）

- 模型 Qwen3.5-4B（24×GDN + 8×attn），prompt A 1525 tok / prompt B 203 tok，greedy
- 换出 payload：64 个张量，101.9 MB，dtype ['torch.bfloat16', 'torch.float32']（attn KV bf16 + GDN conv/recurrent state 一起交接）
- 换出（pageable .cpu()）96 ms；HBM 释放 8737→8622 MB（实释 114.9 MB）
- 请求 B 在腾出的 HBM 里 prefill+decode 正常完成（1601 ms，16 tok）：「老伙计，今晚海况凶险如深海巨浪，俺老」
- 换入（restore .to('npu')）88 ms（≈ 0.9 个 decode 步）；pinned 路径 restore 10 ms（对照）
- 正确性：ref vs back = True，**BIT-EXACT**：换出→跑 B→换回后的 greedy 32 tok 与不换出逐 token 一致
- decode 32 tok：stay 3475 ms vs roundtrip 3262 ms

```
ref : 老伙计，俺老铁这就给你把这份“机密”的宝贝儿掏出来！\n\n第一问：那接入密钥嘛，俺老铁
back: 老伙计，俺老铁这就给你把这份“机密”的宝贝儿掏出来！\n\n第一问：那接入密钥嘛，俺老铁
```
