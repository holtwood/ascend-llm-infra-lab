#!/usr/bin/env python3
"""Deploy the E30 capture-budget patch onto one Ascend node (idempotent).

* copies capture_budget.py into vllm_ascend/compilation/
* inserts a gated hook right after `vllm_config._set_cudagraph_sizes()`
  in vllm_ascend/platform.py (backup: platform.py.e30bak)
* `--revert` restores the backup

The hook is inert unless VLLM_ASCEND_CAPTURE_BUDGET is set, so a patched tree
behaves exactly like upstream by default.
"""
import argparse, os, shutil, sys

ROOT = "/vllm-workspace/vllm-ascend/vllm_ascend"
MODULE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capture_budget.py")
MODULE_DST = os.path.join(ROOT, "compilation", "capture_budget.py")
PLATFORM = os.path.join(ROOT, "platform.py")
BACKUP = PLATFORM + ".e30bak"

ANCHOR = "    vllm_config._set_cudagraph_sizes()\n"
HOOK = '''
    # --- E30 capture-budget hook (inert unless the env var is set) ---
    try:
        import os as _os

        _budget = int(_os.environ.get("VLLM_ASCEND_CAPTURE_BUDGET", "0") or 0)
        if _budget > 0:
            from vllm_ascend.compilation.capture_budget import plan_capture_sizes

            _cc = vllm_config.compilation_config
            _orig = list(_cc.cudagraph_capture_sizes or [])
            _new = plan_capture_sizes(_orig, _budget)
            if _new and _new != _orig:
                _cc.cudagraph_capture_sizes = _new
                _cc.max_cudagraph_capture_size = _new[-1]
                logger.info(
                    "E30 capture budget=%d: pruned %d -> %d sizes %s",
                    _budget, len(_orig), len(_new), _new,
                )
    except Exception as _e30_err:  # never break startup for a tuning knob
        logger.warning("E30 capture-budget hook skipped: %s", _e30_err)
    # --- end E30 hook ---
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        if os.path.exists(BACKUP):
            shutil.copy2(BACKUP, PLATFORM)
            print("reverted platform.py from", BACKUP)
        else:
            print("no backup found; nothing to revert")
        return

    if not os.path.exists(PLATFORM):
        sys.exit(f"not found: {PLATFORM}")

    if not os.path.exists(BACKUP):
        shutil.copy2(PLATFORM, BACKUP)

    src = open(PLATFORM).read()
    if "E30 capture-budget hook" in src:
        print("hook already present")
    else:
        if ANCHOR not in src:
            sys.exit("anchor not found: vllm_config._set_cudagraph_sizes()")
        src = src.replace(ANCHOR, ANCHOR + HOOK, 1)
        open(PLATFORM, "w").write(src)
        print("hook inserted")

    shutil.copy2(MODULE_SRC, MODULE_DST)
    print("module ->", MODULE_DST)


if __name__ == "__main__":
    main()
