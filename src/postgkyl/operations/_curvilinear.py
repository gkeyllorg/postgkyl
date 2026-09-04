"""Shared "which dimensions form a curvilinear block" lookup.

``differentiate`` and ``integrate`` (its per-axis mode) both need
``ctx["mapped_axes"]`` grouped into contiguous, genuinely curvilinear
(``ndim > 1`` -- a joint ``space="conf"`` ``.map()``) blocks before they can
apply the chain-rule/Jacobian math in ``numerics.curvilinear``, mirroring
the sibling-grouping ``select``'s curvilinear guard already does inline.
Kept separate from that inline grouping (rather than factored together)
because ``select`` also needs its *separable* (``space="vel"``) siblings
grouped for its own purposes, where the callers here index a block by its
own local axis order and so need it filtered to curvilinear blocks and
sorted.
"""

from __future__ import annotations


def curvilinear_blocks(grid: list, mapped_axes: dict) -> dict:
  """``{offset: sorted [absolute dims]}`` for every genuinely curvilinear
  (multi-dimensional grid array) block recorded in ``mapped_axes``."""
  blocks: dict = {}
  for d, off in mapped_axes.items():
    if grid[d].ndim > 1:
      blocks.setdefault(off, []).append(d)
  for dims in blocks.values():
    dims.sort()
  return blocks


def block_for_axis(blocks: dict, axis: int):
  """The ``(offset, dims)`` of the block containing ``axis``, or ``None``."""
  for off, dims in blocks.items():
    if axis in dims:
      return off, dims
  return None
