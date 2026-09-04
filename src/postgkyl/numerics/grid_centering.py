"""Convert a nodal (edge) grid to its cell-centered equivalent."""

from __future__ import annotations

import numpy as np


def nodal_to_cell_centered_grid(grid: list[np.ndarray],
                                cells: np.ndarray,
                                meshgrid: bool = False) -> list[np.ndarray]:
  """Return the cell-centered grid corresponding to a nodal (edge) grid.

  Args:
    grid: List of NumPy arrays giving the nodal grid coordinates.
    cells: Number of cells in each dimension.
    meshgrid: If ``True`` and the coordinates are 1-D, return an
      ij-indexed meshgrid instead of the plain 1-D per-axis arrays.

  Returns:
    List of NumPy arrays giving the cell-centered grid coordinates.

  Raises:
    ValueError: If ``grid`` and ``cells`` disagree on the number of
      dimensions, or an axis is neither nodal nor already cell-centered.
  """
  num_dims = len(grid)
  grid_out = []
  if num_dims != len(cells):
    raise ValueError("Number dimensions for 'grid' and 'values' doesn't match")
  for d in range(num_dims):
    if len(grid[d].shape) == 1:
      if grid[d].shape[0] == cells[d]:
        grid_out.append(grid[d])
      elif grid[d].shape[0] == cells[d] + 1:
        grid_out.append(0.5 * (grid[d][:-1] + grid[d][1:]))
      else:
        raise ValueError("Something is terribly wrong...")
    else:
      if grid[d].shape[d] == cells[d]:
        grid_out.append(grid[d])
      elif grid[d].shape[d] == cells[d] + 1:
        if num_dims == 1:
          grid_out.append(0.5 * (grid[d][:-1] + grid[d][1:]))
        else:
          grid_out.append(0.5 * (grid[d][:-1, :-1] + grid[d][1:, 1:]))
      else:
        raise ValueError("Something is terribly wrong...")

  if meshgrid and num_dims > 1 and all(axis.ndim == 1 for axis in grid_out):
    return list(np.meshgrid(*grid_out, indexing="ij"))

  return grid_out
