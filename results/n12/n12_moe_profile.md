# N12 — Qwen3.6-35B-A3B（MoE）结构画像

机器 36.151.162.7（2×910B2 卡4/5，CANN 9.2.0），模型 `/model/Qwen3.6-35B-A3B`（67G bf16，26 分片）/ `-w8a8`（38G，MindIE ascend 量化，quant_model_description.json）。`/model` 为**只读文件系统**（坑：日志须写 /root 或 /tmp）。

## 一句话结论

这不是"普通 Qwen MoE"：是 **Qwen3.5-MoE 多模态 VL + 3:1 hybrid linear-attention + 256/8 MoE + 独立 MTP 层** 四合一，`Qwen3_5MoeForConditionalGeneration`（model_type `qwen3_5_moe`，HF transformers 4.57.1）。本 fleet 唯一真 MoE + 唯一 hybrid + 唯一 VL 权重。

## config.json 全解读（text_config）

| 字段 | 值 | 解读 |
|---|---|---|
| num_hidden_layers | 40 | 3 linear + 1 full 循环（full_attention_interval=4）→ 30 linear + 10 full |
| layer_types | hybrid | 与 N2b 研究的 GDN 结构同族；**本 fleet 小模型之外唯一真 hybrid 权重** |
| hidden_size | 2048 | |
| num_experts | 256 | safetensors 实测 `mlp.gate.weight [256, 2048]` 确认 |
| num_experts_per_tok | 8 | top-8/256（3.1% 激活率） |
| moe_intermediate_size | 512 | 每专家中间维仅 512（窄专家 × 多数量路线） |
| shared_expert_intermediate_size | 512 | 有共享专家，且额外有 `shared_expert_gate.weight [1, 2048]` |
| router_aux_loss_coef | 0.001 | 训练侧负载均衡系数（推理侧 imbalance 仍在） |
| num_attention_heads / num_key_value_heads | 16 / 2 | full 层 GQA 8:1 |
| head_dim | 256 | q_proj 实测 [8192, 2048] = 16×256×**2**（attn_output_gate=true，Qwen3-Next 式输出门控） |
| partial_rotary_factor | 0.25 | 仅 64/256 维旋转；rope_theta=1e7；**mrope interleaved**，section [11,11,10]（文本/高/宽，VL 用） |
| linear_num_key/value_heads | 16 / 32 | GDN：`in_proj_qkv [8192, 2048]`（qkvz 合并投影），`A_log [32]`，conv kernel 4 |
| mamba_ssm_dtype | float32 | **GDN recurrent state 用 fp32**（数值稳定性；bf16 batch 敏感性问题在此模型上 state 侧天然规避） |
| max_position_embeddings | 262144 | 256K；KV 只有 10 个 full 层承担 |
| vocab_size | 248320 | 超大多模态 vocab，untied lm_head [248320, 2048] |
| mtp_num_hidden_layers | 1 | **完整独立 MTP 块**（实测权重：fc [2048,4096] + 1 层 full self_attn + 自己的 256-expert MoE + shared_expert），`mtp_use_dedicated_embeddings=false`（复用主 embedding） |
| vision_config | 27 层 ViT | hidden 1152，patch 16，merge 2×2，merger 出 2048；`model.visual.patch_embed.proj.weight [1152,3,2,16,16]` 实测 |
| dtype | bfloat16 | 权重 67G ≈ 35B params 核对 |

## 专家结构表（safetensors 实测形状）

| 权重 | 形状 | 说明 |
|---|---|---|
| `mlp.gate.weight` | [256, 2048] | router：线性打分，无 bias |
| `mlp.experts.gate_up_proj` | **[256, 1024, 2048]** | **打包 3D**（无专家号后缀）→ grouped-GEMM/bmm 天然形状；1024 = 2×512（gate+up fused） |
| `mlp.experts.down_proj` | **[256, 2048, 512]** | 同上打包 |
| `mlp.shared_expert.{gate,up,down}_proj` | [512,2048]×2 + [2048,512] | 共享专家常开 |
| `mlp.shared_expert_gate.weight` | [1, 2048] | **共享专家标量门控**（Qwen3-Next 特色：shared 输出乘 sigmoid gate，非 DeepSeek 式常开相加） |

参数量核算：专家 805M/层 ×40 = 32.2B；激活 = 8×3.15M + shared 3.15M ≈ 28M/层 ×40 ≈ 1.1B + attn/embed ≈ **总 35B / 每 token 激活 ~3B**，与 "35B-A3B" 命名吻合。MTP 块额外复制一份 256-expert MoE。

## 到达等级

1. ✅ config 解读 + 权重体积（67G / 38G）
2. ✅ safetensors 实测：router [256,2048]、experts 打包 [256,1024,2048]/[256,2048,512]、shared_expert_gate [1,2048]、MTP/vision 权重存在
3. ⚠️ vllm serve：vllm-env（vllm 0.23.0 + vllm_ascend 0.23.0.post1，torch_npu 2.10.0.post4，原生含 `qwen3_5.py`/`qwen3_5_mtp.py`）TP=2 卡4/5 已发起（`/root/n12_serve.log`，`--enforce-eager --max-model-len 8192`）。前置坑实录：`/model` 只读导致首次 nohup 重定向即死；`svc_35b.log` 是 9/29 另一套环境（CANN 9.1 + py3.12）起 w8a8 的崩溃残留，不可抄。结果见文末补充。

## 第 3 级实录（时间到，服务仍在加载）

- 首次起服失败根因：`ImportError: libGL.so.1` —— **VL 模型的视觉链（cv2 import）需要系统 libGL**，纯文本场景也逃不掉 import。修复：`apt-get install -y libgl1`（已装好，`ctypes.CDLL` 验证 OK）。
- 修复后重启：PID 4199，日志 `/root/n12_serve2.log`，已越过原崩溃点进入引擎初始化（剩 paddlex/triton 两处插件噪音，非致命）。TP=2（ASCEND_RT_VISIBLE_DEVICES=4,5），`--enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 8 --port 8321`。
- 复用姿势：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` + `ASCEND_RT_VISIBLE_DEVICES=4,5 HF_HUB_OFFLINE=1`；**/model 只读，日志写 /root**。加载 67G 需数分钟，后续接手者可直接 `curl 127.0.0.1:8321/v1/chat/completions`。
- 旁证：w8a8 版（38G，MindIE ascend 量化）是 9/29 在另一套环境（CANN 9.1 + py3.12）起过的（`svc_35b.log`），最终崩溃残留，不可直接抄。

## 面试考点映射（fleet 缺 MoE 实验数据背景）

1. **EP all-to-all**：256 experts 在 TP=2 下专家按维切、每卡全专家；真正的 EP（如 EP=8/16）专家按 E 维整切，token 路由后需 all-to-all 分发/回传。本模型 experts 是打包 [E,·,·] 张量，EP 切分 = 沿 dim0 slice，工程上最干净——适合作为 EP vs TP 讨论的实例。
2. **Expert imbalance / EPLB**：top-8/256 意味着单 token 激活率 3.1%，hot expert（如代码/中文 token 集中路由）会让单 EP rank 成为瓶颈；`router_aux_loss_coef=0.001` 只管训练，推理侧 imbalance 纯靠调度（EPLB 冗余复制 / 统计重排）。可做实验：采样真实 prompt 统计 40 层 router 的专家命中直方图 → 量化 imbalance，这是 fleet 完全空白的数据。
3. **MoE batching**：token→expert permute/unpermute + grouped GEMM；与 N2b 的 batch-determinism 发现直接衔接——routing tie-break 在 batch 下是否稳定决定输出可复现性。
4. **Hybrid + MoE 双重稀疏**：30/40 层无 KV cache（GDN fp32 state）+ MoE 3.1% 激活 = 256K 上下文可服务的并发杠杆；同时 N2b 已验证的"hybrid 无 attention-mask 概念，left-padding 语义破坏"在此真模型上同样成立且 state 是 fp32（更贵）。
5. **MTP + MoE 叠加**：MTP 层自带 256-expert MoE → speculative decode 的 draft 也走 routing；N6 实测 acceptance 2.44 的延伸考点：验证 batch 里 routing 是否逐 token 确定。
6. **shared_expert_gate**：为什么给 shared expert 加 sigmoid 门（vs DeepSeek 常开）——容量/噪声控制；一道好的结构辨析题。
