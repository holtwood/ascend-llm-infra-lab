#!/bin/bash
# Finish remote vllm+vllm_ascend bring-up, bounded. Idempotent:
#   - wait for in-flight wheel install (stack_install.log)
#   - install vllm-ascend editable from copied tree (has W8A16 patch)
#   - verify imports
set -u
OUT=${OUT_DIR:-/tmp/overnight}
mkdir -p "$OUT"
LOG="$OUT/install.log"
exec > >(tee -a "$LOG") 2>&1
WS=/data1/pytroch-7/vllm-workspace
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

echo "[1/4] wait in-flight wheel install (max 40min)"
for i in $(seq 1 240); do
    pgrep -f "install_stack.sh" >/dev/null || break
    sleep 10
done
tail -5 /tmp/stack_install.log 2>/dev/null

echo "[2/4] vllm import check"
if python3 -c "import vllm" 2>/dev/null; then
    python3 -c "import vllm; print('vllm', vllm.__version__, vllm.__file__)"
else
    echo "vllm wheel not installed; --no-deps install + dep list minus torch*"
    # vllm 0.30.0 wheel hard-pins torch==2.13.0 which would break torch_npu
    # 2.10.0 pairing -> install wheel without deps, then pull deps minus torch*
    pip install /tmp/vllm_whl/vllm-0.30.0-*.whl --no-deps 2>&1 | tail -4
    python3 - <<'PYEOF' > /tmp/vllm_deps.txt
import glob, email
for w in glob.glob("/tmp/vllm_whl/vllm-0.30.0-*.whl"):
    import zipfile
    z = zipfile.ZipFile(w)
    meta = [n for n in z.namelist() if n.endswith("METADATA")][0]
    m = email.message_from_bytes(z.read(meta))
    for r in m.get_all("Requires-Dist") or []:
        spec, _, marker = r.partition(";")
        base = spec.split("=")[0].split(">")[0].split("<")[0].split("!")[0].split("[")[0].strip().lower()
        if base in ("torch", "torch-npu", "torchvision", "torchaudio"):
            continue
        if "extra ==" in marker:
            continue  # optional extra deps
        print(r.strip())  # keep markers; pip evaluates them natively
PYEOF
    pip install -r /tmp/vllm_deps.txt -i $MIRROR 2>&1 | tail -12
    python3 -c "import vllm; print('vllm', vllm.__version__)" || { echo "VLLM_FAIL"; exit 1; }
fi

echo "[3/4] vllm_ascend editable install"
if [ -d "$WS/vllm-ascend" ]; then
    export SETUPTOOLS_SCM_PRETEND_VERSION=0.30.0rc2
    pip install -e "$WS/vllm-ascend" --no-build-isolation -i $MIRROR 2>&1 | tail -8 \
    || pip install -e "$WS/vllm-ascend" --no-build-isolation --no-deps 2>&1 | tail -8
    # runtime deps that editable --no-deps may have missed
    pip install "compressed-tensors>=0.11.0" msgspec -i $MIRROR 2>&1 | tail -4
else
    echo "vllm-ascend tree not copied yet -> VLLM_ASCEND_MISSING"
    exit 1
fi
python3 -c "import vllm_ascend; print('vllm_ascend', vllm_ascend.__file__)" || { echo "VA_FAIL"; exit 1; }

echo "[4/4] platform plugin check"
python3 - <<'EOF'
import vllm_ascend  # noqa
print("PLUGIN_OK")
EOF
