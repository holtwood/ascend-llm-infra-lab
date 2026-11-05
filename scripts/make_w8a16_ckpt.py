#!/usr/bin/env python3
"""Build a real weight-only W8A16 (compressed-tensors) checkpoint from
Qwen3-0.6B bf16 — per-output-channel symmetric INT8 on Linear weights,
weight_scale per channel, weight_offset=0, activations stay bf16.

Output dir keeps tokenizer/config; config.json gains a compressed-tensors
quantization_config that exercises vllm-ascend's _detect_quant_type W8A16 path.
"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

SRC = "/data/models/Qwen3-0.6B"
DST = "/data/models/Qwen3-0.6B-w8a16"

os.makedirs(DST, exist_ok=True)

LINEAR_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def is_linear_weight(name):
    return name.endswith(".weight") and any(s in name for s in LINEAR_SUFFIXES)

# quantize all shard files
src_files = [f for f in os.listdir(SRC) if f.endswith(".safetensors")]
assert src_files, "no safetensors in src"
for fn in src_files:
    sd = load_file(os.path.join(SRC, fn))
    out = {}
    n_q = 0
    for k, w in sd.items():
        if is_linear_weight(k) and w.dtype in (torch.bfloat16, torch.float16, torch.float32):
            w32 = w.float()
            scale = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            q = torch.round(w32 / scale).clamp(-127, 127).to(torch.int8)
            base = k[: -len(".weight")]
            out[base + ".weight"] = q
            out[base + ".weight_scale"] = scale.to(torch.bfloat16)
            out[base + ".weight_offset"] = torch.zeros_like(scale, dtype=torch.bfloat16)
            n_q += 1
        else:
            out[k] = w
    save_file(out, os.path.join(DST, fn), metadata={"format": "pt"})
    print(f"{fn}: {n_q} linear weights quantized")

# index file if present
idx = os.path.join(SRC, "model.safetensors.index.json")
if os.path.exists(idx):
    shutil.copy(idx, DST)

# config: add compressed-tensors weight-only quant section
cfg = json.load(open(os.path.join(SRC, "config.json")))
cfg["quantization_config"] = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "global_compression_ratio": 1.0,
    "ignore": ["lm_head"],
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "type": "int",
                "num_bits": 8,
                "strategy": "channel",
                "symmetric": True,
                "dynamic": False,
            },
            "input_activations": None,
        }
    },
}
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)

# tokenizer + aux files
for f in os.listdir(SRC):
    if not f.endswith(".safetensors") and f != "config.json" and f != "model.safetensors.index.json":
        shutil.copy(os.path.join(SRC, f), DST)

print("done ->", DST)
