"""Shared physical collocation contract for multi-dataset operations."""

import numpy as np

from postgkyl import numerics
from postgkyl.gdatastate.guards import require_same_quadrature
from postgkyl.gdatastate.layout import dg_layout


def uniform_cartesian_grid(data) -> dict:
  """Resolve actual cell edges for native kernels with a uniform-grid ABI.

  Bounds come from the coordinates used by the dataset. A cached context
  bound cannot override its geometry, and a nonuniform mesh cannot silently
  enter a kernel whose grid descriptor only stores bounds and cell counts.
  """
  cells = data.num_cells
  message = "native DG operations require uniform Cartesian cell edges"
  if len(data.grid) != len(cells):
    raise ValueError(message)
  for coord, count in zip(data.grid, cells):
    if (count < 1 or coord.ndim != 1 or len(coord) != count + 1
        or not np.all(np.isfinite(coord)) or np.any(np.diff(coord) <= 0)):
      raise ValueError(message)
    # Subtracting large coordinates can make equal mathematical widths differ
    # by several ulps. Compare against reconstructed edges at coordinate
    # precision, rather than requiring those subtractions to agree exactly.
    expected = np.linspace(coord[0], coord[-1], count + 1)
    if not numerics.grids_compatible([coord], [expected], rtol=0):
      raise ValueError(message)
  return {
      "ndim": len(cells),
      "lower": np.array([coord[0] for coord in data.grid]),
      "upper": np.array([coord[-1] for coord in data.grid]),
      "cells": np.array(cells),
  }


def require_same_backend(left, right) -> None:
  if left.backend != right.backend:
    raise ValueError(
        "operands have different backends (gkyl vs numpy); call .interpolate() "
        "on the native operand to combine them.")


def require_collocated_layout(left, right) -> None:
  """Require the same geometry and representation, allowing field counts to differ.

  A scalar weight can apply to multiple fields, but it must describe values at
  the same physical locations with the same basis and point ordering.
  """
  require_same_backend(left, right)
  if not numerics.grids_compatible(left.grid, right.grid):
    raise ValueError("operands live on different grids")
  left_layout, right_layout = dg_layout(left), dg_layout(right)
  left_rep = left_layout.value_form if left_layout else None
  right_rep = right_layout.value_form if right_layout else None
  if left_rep != right_rep:
    raise ValueError(
        f"operands are in different value_forms ({left_rep} vs {right_rep}); "
        "convert one explicitly with .represent(to=...).")
  if left_layout is not None:
    if left_layout.basis_identity != right_layout.basis_identity:
      raise ValueError("operands have different DG bases")
    if left_rep == "quad":
      require_same_quadrature(left, right)
  if tuple(left.num_cells) != tuple(right.num_cells):
    raise ValueError("operands have different cell layouts")
