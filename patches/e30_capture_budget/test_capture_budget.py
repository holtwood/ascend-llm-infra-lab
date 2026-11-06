#!/usr/bin/env python3
"""Unit tests for the E30 capture-budget planner (no device needed)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_budget import plan_capture_sizes

# vLLM's default ladder for max_num_seqs=256 (35 entries, as seen on 910B2)
DEFAULT = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112,
           120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352,
           384, 416, 448, 480, 512]


def main():
    # 1) no-op when the ladder already fits
    assert plan_capture_sizes([1, 2, 4], 8) == [1, 2, 4]
    # 2) empty / non-positive budget are pass-through
    assert plan_capture_sizes([], 8) == []
    assert plan_capture_sizes(DEFAULT, 0) == DEFAULT
    # 3) budget 32 keeps only powers of two (<= 32 entries)
    out = plan_capture_sizes(DEFAULT, 32)
    assert len(out) <= 32 and all(x & (x - 1) == 0 for x in out), out
    # 4) the max is always retained
    for b in (32, 16, 8, 4, 1):
        assert plan_capture_sizes(DEFAULT, b)[-1] == 512, b
    # 5) tight budgets are honoured exactly
    assert len(plan_capture_sizes(DEFAULT, 4)) <= 4
    assert plan_capture_sizes(DEFAULT, 1) == [512]
    # 6) output is sorted, unique and a subset of the input
    out = plan_capture_sizes(DEFAULT, 8)
    assert out == sorted(set(out)) and set(out) <= set(DEFAULT)
    print("all capture_budget tests passed")


if __name__ == "__main__":
    main()
