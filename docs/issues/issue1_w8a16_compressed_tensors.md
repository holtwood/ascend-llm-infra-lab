# [Issue Draft] NotImplementedError serving weight-only (W8A16) compressed-tensors model on NPU

**Repo**: vllm-project/vllm-ascend
**Component**: quantization / compressed-tensors

## Environment

- vllm-ascend 0.30.0rc2 (`8fde6d4e3`), vLLM 0.30.0
- CANN 9.2.0, torch 2.10 + torch_npu, Atlas 910B2 (64GB HBM), Ubuntu 22.04 aarch64
- Model: `Qwen3.8-27B-W8A16` — compressed-tensors `pack-quantized`, GPTQ int8 channel-wise symmetric weights, `input_activations: null` (weight-only, no activation quantization). Produced by llm-compressor `GPTQModifier` (recipe.yaml ships in the model dir).

## Reproduce

```bash
vllm serve /model/Qwen3.8-27B-W8A16 \
  --served-model-name qwen27b-w8a16 --port 8000 --max-model-len 8192
```

## Expected

vLLM serves the model using a weight-only int8 scheme (the Ascend equivalent of the
GPU W8A16/marlin-style path), i.e., dequantize-to-compute with no activation quant.

## Actual

Startup aborts while resolving the per-layer quant scheme:

```
WARNING  [compressed_tensors_config.py:251] Acceleration for non-quantized schemes is
         not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
ERROR    File ".../vllm_ascend/quantization/configs/compressed_tensors_config.py", line 382, in _detect_quant_type
           raise NotImplementedError("No compressed-tensors compatible quantization type was found.")
         NotImplementedError: No compressed-tensors compatible quantization type was found.
```

Call chain: `get_quant_method` → `_get_linear_scheme` → `_create_scheme_for_layer_type` → `_detect_quant_type`.

## Analysis

The Ascend compressed-tensors adapter only recognizes schemes that carry activation
quantization (e.g. W8A8_DYNAMIC msmodelslim format works fine — the same backbone
in W8A8_DYNAMIC loads and serves correctly). A GPTQ `pack-quantized` config with
`input_activations: null` falls through every branch of `_detect_quant_type` and
raises a bare `NotImplementedError`. Weight-only quantization is the common
bandwidth-bound format for decode-heavy serving, so the gap is material: on this
platform a 27B bf16 model does not fit a 64GB card and W8A16 would be the natural
deployment format.

## Suggested fix

1. Handle `input_activations is None` in `_detect_quant_type` / `_create_scheme_for_layer_type`
   and map it to a weight-only int8 scheme (ATB W8A16 op or a dequant-to-bf16 linear).
2. Until supported, raise a descriptive error naming the rejected scheme
   (`pack-quantized, weights=int8 channel, input_activations=null`) instead of the
   generic `NotImplementedError`, and document the supported quantization format
   matrix for vllm-ascend.

## Workaround

None found on vllm-ascend 0.30.0rc2 for this format. The same backbone quantized
to `W8A8_DYNAMIC` (msmodelslim `ascendv1_saver`) serves correctly.
