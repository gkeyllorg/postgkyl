
from __future__ import annotations

from typing import Tuple, TYPE_CHECKING
import numpy as np

import postgkyl.data.idx_parser as idx_parser
if TYPE_CHECKING:
  from postgkyl import GData
#end


def select(data: GData, comp: int | str | None = None,
      z0: int | float | str | None = None, z1: int | float | str | None = None,
      z2: int | float | str | None = None, z3: int | float | str | None = None,
      z4: int | float | str | None = None, z5: int | float | str | None = None,
      overwrite: bool = False) -> Tuple[list, np.ndarray]:
  """Selects parts of the GData.

  Allows to select only a part of GData (both coordinates and
  components).  Allows for numpy slices, selecting multiple
  components, and using both indicies (integer) and values (float).

  Atributes:
    data (GData)
    z0-5 (index, value, or slice (e.g. '1:5')
    comp (index, slice (e.g. '1:5'), or multiple (e.g. '1,5')
  """
  zs = (z0, z1, z2, z3, z4, z5)
  grid = data.get_grid()
  grid = list(grid)  # copy the grid
  values = data.get_values()
  num_dims = data.get_num_dims()
  bounds = data.get_bounds()
  values_idx = [slice(0, values.shape[d]) for d in range(num_dims + 1)]
  uniform_grid = len(grid[0].shape) == 1
  if not uniform_grid:
    grid_idx = [slice(0, grid[d].shape[d]) for d in range(num_dims)]
  # end
  # Handle gk-rz selection.
  value_coords = None if uniform_grid else data.ctx.get("value_coords")

  # Loop for coordinates
  for d, z in enumerate(zs):
    if d < num_dims and z is not None:
      #dat_range = bounds[1][d] - bounds[0][d]
      #if '.' in z:
      #  if bounds[1][d] + 0.25 * dat_range < float(z) or bounds[0][d] - 0.25 * dat_range > float(z):
      #    raise TypeError("The coordinate select is outside of the data boundaries")
      #  #end
      ##end
      if uniform_grid:
        lookup = grid[d]
      elif value_coords is not None and value_coords[d] is not None:
        lookup = value_coords[d]
      else:
        lookup = grid[d]
      # end
      len_grid = lookup.shape[0] if lookup.ndim == 1 else lookup.shape[d]
      is_matching = values.shape[d] == len_grid
      idx = idx_parser.idx_parser(z, lookup, is_matching)
      if isinstance(idx, int):
        # when 'slice' is used instead of an integer
        # number, numpy array is not squeezed after
        # subselecting
        v_idx = slice(idx, idx + 1)
        g_idx = slice(idx, idx + 2) if not is_matching else slice(idx, idx + 1)
      elif isinstance(idx, slice):
        v_idx = idx
        g_idx = slice(idx.start, idx.stop + 1) if not is_matching else idx
      else:
        raise TypeError("The coordinate select can be only single index (int) or a slice.")
      # end
      if uniform_grid:
        grid[d] = grid[d][g_idx]
      else:
        grid_idx[d] = g_idx
      # end
      values_idx[d] = v_idx
    # end
  # end

  # Select components
  if comp is not None:
    values_idx[-1] = idx_parser.idx_parser(comp)
  # end
  values_out = values[tuple(values_idx)]
  new_value_coords = None
  if not uniform_grid:
    # Handle gk-rz selection.
    if value_coords is not None:
      new_value_coords = [
        vc[grid_idx[d]] if vc is not None else None
        for d, vc in enumerate(value_coords)
      ]
    # end
    for d in range(num_dims):
      grid[d] = grid[d][tuple(grid_idx)]
    # end
  # end

  # Adding a dummy dimension indicies
  if num_dims == len(values_out.shape):
    values_out = values_out[..., np.newaxis]
  # end

  if overwrite:
    data.push(grid, values_out)
    if new_value_coords is not None:
      data.ctx["value_coords"] = new_value_coords
    # end
  #end
  return grid, values_out
