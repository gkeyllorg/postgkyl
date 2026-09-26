"""Pure-array helpers for element-wise dataset arithmetic."""

from __future__ import annotations

import numpy as np


def grids_compatible(grid_a: list, grid_b: list, rtol: float = 1e-9) -> bool:
  """Compare coordinates relative to their extent, allowing roundoff at offsets.

  A fixed absolute tolerance would merge distinct microscopic domains; a
  tolerance proportional to the coordinate origin would merge shifted meshes
  at large offsets. Use the domain extent plus floating-point roundoff.
  """
  if len(grid_a) != len(grid_b):
    return False
  for a, b in zip(grid_a, grid_b):
    if a.shape != b.shape:
      return False
    if a.size == 0:
      continue
    extent = max(np.ptp(a), np.ptp(b))
    roundoff = 8 * np.finfo(float).eps * np.maximum(np.abs(a), np.abs(b))
    if not np.all(np.abs(a - b) <= rtol * extent + roundoff):
      return False
  return True


def grid_is_prefix(small: list, big: list, rtol: float = 1e-9) -> bool:
  """Whether ``small`` is exactly the leading dimensions of ``big`` (same
  shapes & nodes) -- the conf-space/phase-space compatibility check for
  cross-basis (conf x phase) operations, where a phase-space grid extends a
  lower-dimensional conf-space grid with extra (velocity-space) dimensions."""
  if not 0 < len(small) < len(big):
    return False
  return grids_compatible(small, big[:len(small)], rtol=rtol)
