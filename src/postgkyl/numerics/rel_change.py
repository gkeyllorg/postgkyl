"""Relative change of one dataset's values against a reference."""

from __future__ import annotations

import numpy as np


def rel_change(grid: list[np.ndarray],
               values0: np.ndarray,
               values: np.ndarray,
               comp: int | None = None) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute ``(values - values0) / values0``, component-wise.

  Args:
    grid: Nodal coordinate arrays, one per spatial dimension (returned
      unchanged; the two datasets are assumed to share a grid).
    values0: Reference ("before") data array.
    values: Data array to compare against the reference.
    comp: If given, every component is normalized by this single reference
      component instead of its own (e.g. divide every energy component by
      the total energy component).

  Returns:
    ``(grid, out)`` with ``out`` the same shape as ``values``.
  """
  out = np.zeros(values.shape)
  for i in range(out.shape[-1]):
    denom = values0[..., int(comp)] if comp is not None else values0[..., i]
    out[..., i] = (values[..., i] - values0[..., i]) / denom
  return list(grid), out
