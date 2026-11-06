# N3: hybrid 架构显存曲线 —— KV 只在 8/32 层增长, GDN state 恒定 的实测验证
import time, torch, torch_npu, warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM

M = "/model/Qwen3.5-4B"; DEV = "npu:0"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()

def cache_bytes(cache):
    """分类统计 cache 内部 tensor 字节: 返回 (attn_kv_bytes, gdn_state_bytes, 其他)"""
    attn_kv, gdn_state, other = 0, 0, 0
    def classify(name, t):
        nonlocal attn_kv, gdn_state, other
        if not torch.is_tensor(t): return
        b = t.numel() * t.element_size()
        n = name.lower()
        if ("key" in n or "value" in n) and "recurrent" not in n and "conv" not in n:
            attn_kv += b
        elif "recurrent" in n or "state" in n or "conv" in n or "ssm" in n:
            gdn_state += b
        else:
            other += b
    def walk(o, prefix=""):
        if torch.is_tensor(o): classify(prefix, o)
        elif isinstance(o, (list, tuple)):
            for i, x in enumerate(o): walk(x, f"{prefix}[{i}]")
        elif isinstance(o, dict):
            for k, v in o.items(): walk(v, f"{prefix}.{k}")
        elif hasattr(o, "__dict__"):
            for k, v in vars(o).items():
                if k.startswith("_") or callable(v): continue
                walk(v, f"{prefix}.{k}")
    walk(cache)
    return attn_kv, gdn_state, other

# --- 结构探测: 先看 cache 里到底存了什么 ---
ids0 = torch.randint(1000, 50000, (1, 32), device=DEV)
with torch.no_grad():
    out = model(ids0, use_cache=True)
c = out.past_key_values
print("[cache type]", type(c).__name__, flush=True)
try:
    for i, layer in enumerate(c.layers[:3]) if hasattr(c, "layers") else []:
        print(f"  layer{i}:", {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
                               for k, v in vars(layer).items() if not k.startswith("_") and not callable(v)}, flush=True)
except Exception as e:
    print("  layer probe fail:", e, flush=True)

MB = 1024**2
print("\n| ctx_tokens | attn_KV(MB) | GDN_state(MB) | other(MB) | KV_theory(MB) | wall_prefill_last_chunk(ms) |", flush=True)
print("|---:|---:|---:|---:|---:|---:|", flush=True)
rows = []
for L in [512, 2048, 8192, 32768, 65536, 131072]:
    ids = torch.randint(1000, 50000, (1, L), device=DEV)
    with torch.no_grad():
        model(torch.randint(1000, 50000, (1, 8), device=DEV), use_cache=True)  # warmup 编译
        t0 = time.time()
        out = model(ids, use_cache=True, logits_to_keep=1)
        torch.npu.synchronize()
        wall = (time.time() - t0) * 1000
    a, g, o = cache_bytes(out.past_key_values)
    kv_theory = 8 * 2 * L * 4 * 256 * 2 / MB  # 8 attn层 × K,V × L × 4 KV头 × 256 dim × 2B
    rows.append((L, a, g, o, kv_theory, wall))
    print(f"| {L} | {a/MB:.1f} | {g/MB:.1f} | {o/MB:.2f} | {kv_theory:.1f} | {wall:.0f} |", flush=True)

gdn_theory = 24 * 32 * 128 * 128 * 2 / MB
print(f"\n[GDN state theory] 24层×32头×128×128×2B = {gdn_theory:.1f} MB (应恒定)", flush=True)
print(f"[KV slope] 实测 {[(r[1])/r[0]/1024 for r in rows][0]:.0f}..{[(r[1])/r[0]/1024 for r in rows][-1]:.0f} KB/token; 全注意力对照模型应为 4x", flush=True)
print("N3_KV_OK", flush=True)
