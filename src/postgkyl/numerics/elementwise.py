"""Pure-array helpers for element-wise dataset arithmetic."""

from __future__ import annotations

import numpy as np


def grids_compatible(grid_a: list, grid_b: list, rtol: float = 1e-9) -> bool:
  """Whether two nodal grids describe the same mesh (same shapes & nodes)."""
  if len(grid_a) != len(grid_b):
    return False
  return all(a.shape == b.shape and np.allclose(a, b, rtol=rtol)
             for a, b in zip(grid_a, grid_b))


def grid_is_prefix(small: list, big: list, rtol: float = 1e-9) -> bool:
  """Whether ``small`` is exactly the leading dimensions of ``big`` (same
  shapes & nodes) -- the conf-space/phase-space compatibility check for
  cross-basis (conf x phase) operations, where a phase-space grid extends a
  lower-dimensional conf-space grid with extra (velocity-space) dimensions."""
  if not 0 < len(small) < len(big):
    return False
  return all(a.shape == b.shape and np.allclose(a, b, rtol=rtol)
             for a, b in zip(small, big[:len(small)]))
