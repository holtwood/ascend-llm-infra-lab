# [Bug] Weight-only W8A16 compressed-tensors checkpoints fall through `_detect_quant_type` and abort with bare `NotImplementedError`

**Repository**: vllm-project/vllm-ascend
**Component**: quantization / compressed-tensors
**Duplicate check**: related to RFC #5350 (compressed-tensors support for LLM Compressor) and #13318 (quantization module refactoring) — this is a concrete bug report on the current shipped behavior, not an RFC. No open issue/PR found handling `input_activations: null` weight-only checkpoints (searched: `W8A16 compressed-tensors`, `weight-only quantization int8`, `_detect_quant_type`).

## Environment

| Item | Version |
|---|---|
| vllm-ascend | 0.30.0rc2 (`8fde6d4e3`) |
| vLLM | 0.30.0 |
| CANN | 9.2.0 |
| torch / torch_npu | 2.10 |
| Hardware | Atlas 910B2, 64 GiB HBM, aarch64 |
| OS | Ubuntu 22.04 |

**Model**: `Qwen3.8-27B-W8A16` — compressed-tensors `pack-quantized` format, GPTQ
int8 channel-wise symmetric weights, `input_activations: null` (weight-only, no
activation quantization). Produced with llm-compressor `GPTQModifier`
(recipe.yaml included in the model directory).

## Reproduction

```bash
vllm serve /model/Qwen3.8-27B-W8A16 \
  --served-model-name qwen27b-w8a16 \
  --port 8000 --max-model-len 8192
```

## Expected behavior

The engine serves the checkpoint with a weight-only int8 scheme (the Ascend
equivalent of the GPU W8A16 path): dequantize-to-compute, no activation
quantization. Weight-only is the standard deployment format for bandwidth-bound
decode serving — on this platform a 27B bf16 model does not fit in 64 GB HBM,
making W8A16 the natural choice over W8A8.

## Actual behavior

Startup aborts while resolving the per-layer quant scheme:

```text
WARNING  [compressed_tensors_config.py:251] Acceleration for non-quantized schemes is
         not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
ERROR    File ".../vllm_ascend/quantization/configs/compressed_tensors_config.py",
           line 382, in _detect_quant_type
           raise NotImplementedError("No compressed-tensors compatible quantization
           type was found.")
         NotImplementedError: No compressed-tensors compatible quantization type was found.
```

Call chain: `get_quant_method` → `_get_linear_scheme` →
`_create_scheme_for_layer_type` → `_detect_quant_type`.

## Root cause

`_detect_quant_type` (`vllm_ascend/quantization/configs/compressed_tensors_config.py:326-382`)
only recognizes compressed-tensors configs that carry **activation** quantization
— `W8A8`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `W8A8FP8_DYNAMIC`, `W8A8_MXFP8`,
`W4A4_MXFP4` — plus `W4A16` (line 379-380). A GPTQ `pack-quantized` config with
`input_activations: null` matches none of the branches and falls through to the
bare `NotImplementedError` at line 382.

Notably, a `W8A16` linear scheme **already exists** on the ModelSlim path:
`vllm_ascend/quantization/methods/wna16/w8a16.py` registers
`("W8A16", "linear")` and applies `torch_npu.npu_weight_quant_batchmatmul` with
NZ-transposed int8 weights. The gap is specifically that the compressed-tensors
detector never maps a weight-only int8 config to this (or an equivalent) scheme.

Cross-check: the same backbone in `W8A8_DYNAMIC` format (msmodelslim
`ascendv1_saver`) loads and serves correctly on the identical environment —
the failure is confined to the compressed-tensors weight-only path.

## Suggested fix

1. Handle `input_activations is None` + int8 `pack-quantized` weights in
   `_detect_quant_type` (return `W8A16`) and register the corresponding
   scheme — the existing `wna16/w8a16.py` implementation may be directly
   reusable.
2. Until supported, replace the bare `NotImplementedError` with a descriptive
   error naming the rejected scheme (e.g.
   `"pack-quantized weights=int8 channel-wise symmetric, input_activations=null
   is not supported; W8A16 weight-only compressed-tensors is not yet implemented,
   see #<this issue>"`), and publish the supported-quantization matrix so users
   can detect the gap before download.

## Workaround

None for this format on vllm-ascend 0.30.0rc2. Requantizing the model to
`W8A8_DYNAMIC` (msmodelslim) works but costs accuracy (activation quantization)
and requires calibration.

---

*This report was prepared with AI assistance (Cognition Devin); reproduction,
logs, and code references were verified against a real NPU deployment.*
