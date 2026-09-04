"""Distribution-function frame transform -- shift a particle distribution
function's velocity grid by a bulk velocity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...gdatastate.guards import require_field_domain as _require_field_domain

if TYPE_CHECKING:
  from ...gdatastate.gdatastate import GDataState

_REASON = "shifting the grid of raw DG coefficients has no basis-space meaning"


# --------------------------------------------------------- array-level math
def _transform_frame(
    f_grid: list[np.ndarray],
    f_values: np.ndarray,
    u_values: np.ndarray,
    c_dim: int,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Shift a distribution function to a different frame of reference.

  Shifts the velocity-space grid of a distribution function by a supplied
  bulk velocity (a magnetic-field-direction shift is not yet supported).

  Args:
    f_grid: Nodal coordinate arrays, one per configuration- and
      velocity-space dimension (configuration dimensions first).
    f_values: Particle distribution function values (unchanged by the
      shift; only the velocity grid moves).
    u_values: Bulk velocity array, ``num_dims - c_dim`` components, on the
      configuration-space grid.
    c_dim: Number of configuration-space dimensions.

  Returns:
    ``(grid, values)``: a per-cell-shifted velocity grid (one nodal array
    per dimension, matching the input's dimensionality) and the unchanged
    distribution-function values.
  """
  v_dim = len(f_grid) - c_dim
  out_grid = np.meshgrid(*f_grid, indexing="ij")

  if c_dim == 1:
    for v_idx in range(v_dim):
      nx = f_grid[0].shape[0]

      ext_u = np.zeros(nx)
      ext_u[:-1] += u_values[..., v_idx]
      ext_u[1:] += u_values[..., v_idx]
      ext_u[1:-1] = ext_u[1:-1] / 2

      for i in range(nx):
        out_grid[c_dim + v_idx][i, ...] += ext_u[i]

  elif c_dim == 2:
    for v_idx in range(v_dim):
      nx = f_grid[0].shape[0]
      ny = f_grid[1].shape[0]

      ext_u = np.zeros((nx, ny))
      ext_u[:-1, :-1] += u_values[..., v_idx]
      ext_u[1:, 1:] += u_values[..., v_idx]
      ext_u[1:-1, 1:-1] = ext_u[1:-1, 1:-1] / 2

      for i in range(nx):
        for j in range(ny):
          out_grid[c_dim + v_idx][i, j, ...] += ext_u[i, j]

  else:
    for v_idx in range(v_dim):
      nx = f_grid[0].shape[0]
      ny = f_grid[1].shape[0]
      nz = f_grid[2].shape[0]

      ext_u = np.zeros((nx, ny, nz))
      ext_u[:-1, :-1, :-1] += u_values[..., v_idx]
      ext_u[1:, 1:, 1:] += u_values[..., v_idx]
      ext_u[1:-1, 1:-1, 1:-1] = ext_u[1:-1, 1:-1, 1:-1] / 2

      for i in range(nx):
        for j in range(ny):
          for k in range(nz):
            out_grid[c_dim + v_idx][i, j, k, ...] += ext_u[i, j, k]

  return out_grid, f_values


# ---------------------------------------------------------------- GData verb
def transform_frame(distribution: "GDataState",
                    bulk: "GDataState",
                    *,
                    cdim: int,
                    inplace: bool = False,
                    tag: str | None = None,
                    label: str | None = None) -> "GDataState":
  """Shift a distribution function to a moving frame of reference.

  Shifts the velocity-space grid of ``distribution`` by the local ``bulk``
  velocity so the distribution is expressed in the frame co-moving with
  the bulk flow. The values are unchanged; only the velocity coordinates
  are offset. Supports 1, 2, or 3 configuration-space dimensions.

  Args:
    distribution: The particle distribution function to shift; must be
      NumPy-backed.
    bulk: The bulk (drift) velocity field; one component per velocity
      dimension. Must be NumPy-backed.
    cdim: Number of configuration-space dimensions. The remaining grid
      axes are treated as velocity-space dimensions.
    inplace: mutate and return ``distribution`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset with the same values on a velocity-shifted grid.

  Raises:
    ValueError: if either input is native modal (gkyl-backed).
  """
  _require_field_domain(distribution, "transform_frame", _REASON)
  _require_field_domain(bulk, "transform_frame", _REASON)
  grid, values = _transform_frame(distribution.grid, distribution.values,
                                  bulk.values, cdim)
  return distribution._result(grid,
                              values,
                              inplace=inplace,
                              tag=tag,
                              label=label)
