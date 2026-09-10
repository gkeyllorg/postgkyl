"""The ``grid`` verb -- turn a dataset's grid into a dataset of coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def grid(data: "GDataState",
         *,
         inplace: bool = False,
         tag: str | None = None,
         label: str | None = None):
  """Turn a dataset's grid into a dataset of coordinate values.

  Builds a new dataset whose values, at each grid node, are the physical
  coordinates of ``data``'s grid (one component per dimension). Handles
  uniform meshes, separable (velocity) mappings, and full curvilinear mapped
  grids (produced by the ``map`` verb) alike.

  Args:
    data: the dataset whose grid is converted to coordinate values; must be
      NumPy-backed.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset with one component per dimension holding the physical
    coordinates, on a placeholder index grid (one cell per original node).

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed), or its grid does
      not have one entry per dimension reported by ``num_cells``.
  """
  if data.backend == "gkyl":
    raise ValueError(
        "grid operates on interpolated (NumPy) values; call .interpolate() "
        "first -- raw DG coefficients have no per-node coordinates.")
  grid_in = data.grid
  num_dims = data.num_dims
  num_cells = data.num_cells
  if len(grid_in) != num_dims:
    raise ValueError(
        f"grid: dataset reports {num_dims:d} dimension(s) but its grid has "
        f"{len(grid_in):d} axis (axes); shapes are inconsistent.")

  grid_out = [np.arange(nc + 2) for nc in num_cells]

  shape = np.append(np.copy(num_cells) + 1, num_dims)
  values = np.zeros(shape)
  if num_dims == 1:
    values[..., 0] = grid_in[0]
  elif len(grid_in[0].shape) == 1:  # uniform mesh or separable mapping
    for d, t in enumerate(np.meshgrid(*grid_in, indexing="ij")):
      values[..., d] = t
  else:  # curvilinear mapped grid
    for d, t in enumerate(grid_in):
      values[..., d] = t
  return data._result(grid_out, values, inplace=inplace, tag=tag, label=label)
