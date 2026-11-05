#!/bin/bash
# Environment probe — record hardware/software facts before the queue.
set -u
OUT=${OUT_DIR:-/tmp/overnight}
mkdir -p "$OUT"
{
    echo "=== date ==="; date
    echo "=== npu-smi ==="; npu-smi info 2>&1 | head -20
    echo "=== devices ==="; ls /dev/davinci* 2>/dev/null
    echo "=== python ==="; python3 --version
    python3 - <<'EOF'
import importlib
for m in ["torch", "torch_npu", "vllm", "vllm_ascend", "transformers", "compressed_tensors"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m}: {getattr(mod,'__version__','?')} @ {getattr(mod,'__file__','?')}")
    except Exception as e:
        print(f"{m}: MISSING ({type(e).__name__})")
try:
    import torch_npu
    print("npu_count:", torch_npu.npu.device_count())
except Exception as e:
    print("npu_count: ERR", e)
EOF
    echo "=== cann ==="; ls /usr/local/Ascend 2>/dev/null; cat /usr/local/Ascend/*/ascend_toolkit_install.info 2>/dev/null | head -3
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"
