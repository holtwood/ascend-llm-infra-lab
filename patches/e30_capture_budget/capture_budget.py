# SPDX-License-Identifier: Apache-2.0
"""Capture-size budget planner for Ascend ACL graph capture.

Motivation (measured on 910B2, see the lab repo's E30):

* vLLM derives ``cudagraph_capture_sizes`` from ``max_num_seqs`` (up to 512 on
  Ascend), which on a 4B model yields 35-64 graph sizes.
* Every captured size allocates its own set of ACL streams / SQ-CQ pairs.
  When the ladder is dense, capture fails with
  ``EE1023 Alloc Stream resource failed: Too many streams are created``
  (``207005`` at the torch_npu layer) — see upstream issue #21113.
* Most of those sizes are never hit at runtime: decode batch sizes cluster on
  a few values, and vLLM pads an uncaptured step to the nearest captured size
  (or falls back to eager).

This module keeps a geometric ladder within a caller-supplied budget: all
powers of two up to the max, plus the max itself, thinned evenly if the budget
is tighter than that. Deterministic and side-effect free so it can be unit
tested without a device.
"""

from __future__ import annotations

__all__ = ["plan_capture_sizes"]


def plan_capture_sizes(sizes: list[int], budget: int) -> list[int]:
    """Return a pruned, sorted subset of ``sizes`` with at most ``budget`` entries.

    Rules:
      * always keep ``max(sizes)`` (the capture bound must cover the largest step)
      * prefer powers of two (cheap padding targets for decode batches)
      * if the power-of-two set alone exceeds the budget, thin it evenly while
        keeping both endpoints

    Returns ``sizes`` unchanged when pruning is unnecessary or the inputs are
    degenerate, so callers can apply the result unconditionally.
    """
    if not sizes or budget <= 0:
        return sizes

    uniq = sorted({int(s) for s in sizes if int(s) > 0})
    if not uniq or len(uniq) <= budget:
        return uniq or sizes

    top = uniq[-1]
    keep = [s for s in uniq if s & (s - 1) == 0]  # powers of two
    if top not in keep:
        keep.append(top)
    keep = sorted(set(keep))

    if len(keep) > budget:
        if budget == 1:
            return [top]
        step = (len(keep) - 1) / (budget - 1)
        idx = sorted({round(i * step) for i in range(budget)})
        keep = [keep[i] for i in idx]

    return keep
