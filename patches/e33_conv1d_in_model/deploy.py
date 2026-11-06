#!/usr/bin/env python3
"""Deploy E33: dispatch the sync-free conv1d update inside the model path.

* copies conv1d_nosync.py into vllm_ascend/ops/
* inserts an env-gated dispatch at the top of
  vllm_ascend/ops/causal_conv1d.py::causal_conv1d_update
  (backup: causal_conv1d.py.e33bak)

The dispatch only takes the sync-free path when:
  * VLLM_ASCEND_CONV1D_NOSYNC=1
  * x is decode-shaped ([B, dim], one token per sequence)
  * query_start_loc is absent or has a single entry
otherwise the original implementation runs unchanged.

`--revert` restores the backup.
"""
import argparse, os, shutil, sys

ROOT = "/vllm-workspace/vllm-ascend/vllm_ascend"
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv1d_nosync.py")
DST = os.path.join(ROOT, "ops", "conv1d_nosync.py")
TARGET = os.path.join(ROOT, "ops", "causal_conv1d.py")
BACKUP = TARGET + ".e33bak"

SIG = "def causal_conv1d_update("
HOOK = '''    # --- E33 sync-free decode dispatch (env-gated) ---
    try:
        import os as _os

        if (
            _os.environ.get("VLLM_ASCEND_CONV1D_NOSYNC") == "1"
            and x.dim() == 2
            and (
                query_start_loc is None
                or x.shape[0] == query_start_loc.numel() - 1
            )
        ):
            import vllm_ascend.ops.conv1d_nosync as _m33
            _m33._HITS[0] += 1
            if _m33._HITS[0] in (1, 1000, 10000):
                import logging as _log
                _log.getLogger(__name__).warning(
                    "E33 nosync conv1d hits=%d", _m33._HITS[0])
            from vllm_ascend.ops.conv1d_nosync import causal_conv1d_update_nosync

            return causal_conv1d_update_nosync(
                x, conv_state, weight, bias, activation,
                conv_state_indices, num_accepted_tokens, query_start_loc, pad_slot_id,
            )
    except Exception as _e33_err:  # never break the model path for a tuning knob
        import logging as _logging

        _logging.getLogger(__name__).warning("E33 dispatch skipped: %s", _e33_err)
    # --- end E33 dispatch ---
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        if os.path.exists(BACKUP):
            shutil.copy2(BACKUP, TARGET)
            print("reverted", TARGET)
        else:
            print("no backup; nothing to revert")
        return

    if not os.path.exists(BACKUP):
        shutil.copy2(TARGET, BACKUP)
        print("backup ->", BACKUP)

    src = open(TARGET).read()
    if "E33 sync-free decode dispatch" in src:
        print("dispatch already present")
    else:
        if SIG not in src:
            sys.exit("signature not found in causal_conv1d.py")
        # insert right after the def line and its docstring-free body start
        head, rest = src.split(SIG, 1)
        # skip the parameter list up to the closing ") ->" then the docstring
        idx = rest.index("):")
        idx = rest.index("\n", idx) + 1
        # skip a leading docstring if present
        tail = rest[idx:]
        stripped = tail.lstrip()
        if stripped.startswith('"""'):
            end = stripped.index('"""', 3) + 3
            end = stripped.index("\n", end) + 1
            idx += len(tail) - len(stripped) + end
        src = head + SIG + rest[:idx] + HOOK + rest[idx:]
        open(TARGET, "w").write(src)
        print("dispatch inserted")

    shutil.copy2(SRC, DST)
    print("module ->", DST)


if __name__ == "__main__":
    main()
