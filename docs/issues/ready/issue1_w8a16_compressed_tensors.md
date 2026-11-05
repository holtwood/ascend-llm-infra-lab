## Environment

- vllm-ascend 0.30.0rc2 (`8fde6d4e3`), vLLM 0.30.0, CANN 9.2.0, torch/torch_npu 2.10
- Atlas 910B2 64 GiB, Ubuntu 22.04 aarch64
- Model: `Qwen3.8-27B-W8A16` — compressed-tensors `pack-quantized`, GPTQ int8 channel-wise symmetric weights, `input_activations: null` (weight-only; produced by llm-compressor `GPTQModifier`, recipe.yaml ships in the model dir)

## Reproduce

```bash
vllm serve /model/Qwen3.8-27B-W8A16 --max-model-len 8192
```

## Expected

Serves with a weight-only int8 scheme (dequantize-to-compute, no activation quant). 27B bf16 doesn't fit a 64 GB card, and W8A8 costs accuracy — weight-only is the standard format for decode-heavy serving.

## Actual

Startup aborts in `_detect_quant_type`:

```text
WARNING  [compressed_tensors_config.py:251] Acceleration for non-quantized schemes is
         not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
ERROR    File ".../vllm_ascend/quantization/configs/compressed_tensors_config.py", line 382
           raise NotImplementedError("No compressed-tensors compatible quantization type was found.")
         NotImplementedError: No compressed-tensors compatible quantization type was found.
```

Call chain: `get_quant_method` → `_get_linear_scheme` → `_create_scheme_for_layer_type` → `_detect_quant_type`.

## Cause

`_detect_quant_type` (`compressed_tensors_config.py:326-382`) only recognizes configs carrying **activation** quantization (`W8A8`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `W8A8FP8_DYNAMIC`, `W8A8_MXFP8`, `W4A4_MXFP4`) plus `W4A16` (line 379-380). A `pack-quantized` config with `input_activations: null` matches none and falls through to the bare `NotImplementedError` at line 382.

A `W8A16` linear scheme already exists on the ModelSlim path — `methods/wna16/w8a16.py` registers `("W8A16", "linear")` on `npu_weight_quant_batchmatmul`. The compressed-tensors detector just never maps weight-only int8 to it (or to any scheme).

The same backbone quantized to `W8A8_DYNAMIC` (msmodelslim `ascendv1_saver`) serves correctly on this setup — the failure is confined to the compressed-tensors weight-only path.

Related: #2265, #4267 (earlier W8A16 asks, pre-dating the wna16 scheme landing), #5350 (compressed-tensors support RFC), #13318 (quantization refactor RFC).

## Suggested fix

1. Handle `input_activations is None` + int8 `pack-quantized` weights in `_detect_quant_type` → `W8A16`; the existing `wna16` scheme may be directly reusable.
2. Until supported, raise a descriptive error naming the rejected config (`pack-quantized weights=int8 channel-wise, input_activations=null`) instead of the bare `NotImplementedError`.

## Workaround

None on 0.30.0rc2 for this format. Requantizing to `W8A8_DYNAMIC` works but costs accuracy and a calibration run.

---

*Prepared with AI assistance; reproduction and code refs verified on real NPU hardware.*
