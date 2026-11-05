#!/usr/bin/env python3
"""Repro for vllm-ascend#21112: weight-only W8A16 compressed-tensors falls
through _detect_quant_type to a bare NotImplementedError.

CPU-level: builds AscendCompressedTensorsConfig from a mock quantization_config
(weight INT8 channelwise, no input_activations) and asks for the Linear scheme.
Pre-patch:  NotImplementedError("No compressed-tensors compatible ...")
Post-patch: AscendW8A16LinearMethod
"""
import sys

from vllm_ascend.quantization.configs.compressed_tensors_config import (
    AscendCompressedTensorsConfig,
)

# Shape mirrors a llm-compressor W8A16 checkpoint's quantization_config:
# weight INT8 symmetric channel-wise, no activation quantization.
W8A16_CONFIG = {
    "format": "int-quantized",
    "quant_method": "compressed-tensors",
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
                "group_size": None,
            },
            "input_activations": None,
        }
    },
}

cfg = AscendCompressedTensorsConfig.from_config(W8A16_CONFIG)
print("target_scheme_map:", cfg.target_scheme_map, flush=True)
wq = cfg.target_scheme_map["Linear"]["weights"]
iq = cfg.target_scheme_map["Linear"]["input_activations"]
print(f"weight_quant: type={wq.type} bits={wq.num_bits} strategy={wq.strategy} dyn={wq.dynamic}")
print(f"input_quant : {iq}")

try:
    scheme = cfg._create_scheme_for_layer_type(wq, iq, None, "linear")
    print("RESULT: scheme =", type(scheme).__name__)
except NotImplementedError as e:
    print("RESULT: NotImplementedError:", e)
    sys.exit(1)
