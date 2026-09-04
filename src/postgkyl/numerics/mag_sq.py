"""Magnitude-squared of a (sub-range of) vector-valued field."""

from __future__ import annotations

import numpy as np


def mag_sq(grid: list[np.ndarray],
           values: np.ndarray,
           coords: str = "0:3") -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the magnitude squared of a vector field.

  Args:
    grid: Nodal coordinate arrays, one per spatial dimension.
    values: Data array whose last axis is components.
    coords: ``"start:end"`` slice of the component axis to sum the squares
      of. Defaults to the first three components (the common
      three-component-vector case).

  Returns:
    ``(grid, values)`` where ``values`` has the summed components replaced
    by a single trailing component (magnitude squared).
  """
  lo, hi = coords.split(":")
  comps = values[..., slice(int(lo), int(hi))]
  out = np.sum(comps * comps, axis=-1)[..., np.newaxis]
  return list(grid), out
