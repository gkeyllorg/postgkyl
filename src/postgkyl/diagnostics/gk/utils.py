"""Small file helpers shared by the gyrokinetic loaders and the
layer-13 program-scale diagnostics.

Ported from ``src_bak/postgkyl/gk/gk_utils.py``. ``read_gfile``/
``read_interpolated_gfile`` are adapted to the new API (``postgkyl.gdata.load``
+ ``.interpolate()``) in place of the retired ``GData``/``GInterpModal`` pair;
``read_gfile_if_present`` drops the old code's ``verb_print(ctx, ...)`` call
(``ctx`` was never a parameter of that function in ``src_bak`` -- an existing
bug -- and printing belongs to the CLI, not a loader) in favor of returning a
plain ``found`` flag. ``read_time_trace_if_present`` and
``set_tick_font_size`` are shared by the three program-scale diagnostics that
build figures directly with matplotlib (``energy_balance``,
``particle_balance``, ``nodes``) rather than each keeping its own private
copy.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from postgkyl import numerics
from postgkyl.gdata import GData

# Maximum number of blocks a multiblock simulation is assumed to have, used
# only to bound an open-ended slice request in get_block_indices.
MAX_NUM_BLOCKS = 10000


def read_gfile(file_name: str) -> tuple[list[np.ndarray], np.ndarray, GData]:
  """Read a Gkeyll file, squeezing singleton axes out of the grid and values.

  Args:
    file_name: Path to the ``.gkyl`` file.

  Returns:
    ``(grid, values, gdata)``: the squeezed grid (a list of squeezed
    per-dimension arrays -- ``GDataState.grid`` never hands back a bare
    ``ndarray``), the squeezed value array, and the loaded dataset itself
    (for further chaining).
  """
  gdata = GData(file_name)
  grid = gdata.get_grid()
  values = gdata.get_values()
  grid_out = [np.squeeze(grid[d]) for d in range(len(grid))]
  return grid_out, np.squeeze(values), gdata


def read_gfile_if_present(
    file_name: str,
) -> tuple[bool, list[np.ndarray] | None, np.ndarray | None, GData | None]:
  """Read a Gkeyll file if it exists.

  Args:
    file_name: Path to the file.

  Returns:
    ``(found, grid, values, gdata)``; ``found`` is False and the remaining
    entries are ``None`` when ``file_name`` does not exist.
  """
  if not os.path.exists(file_name):
    return False, None, None, None
  grid, values, gdata = read_gfile(file_name)
  return True, grid, values, gdata


def read_time_trace_if_present(
    file_name: str,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, GData | None]:
  """Read a 1-D time-trace file if present: ``(found, time, values, gdata)``.

  ``read_gfile_if_present`` always returns the grid as a *list* of
  per-dimension arrays (``GDataState.grid`` never hands back a bare
  ``ndarray``, only a list of one for 1-D data) -- this unwraps that single
  entry into the plain time array every trace in
  :mod:`~postgkyl.diagnostics.gk.energy_balance`/
  :mod:`~postgkyl.diagnostics.gk.particle_balance` is indexed
  against.
  """
  found, grid, values, gdata = read_gfile_if_present(file_name)
  time = grid[0] if found else None
  return found, time, values, gdata


def read_interpolated_gfile(
    file_name: str,
    poly_order: int,
    basis_type: str,
    comp: int | str | None = None,
) -> tuple[list[np.ndarray], np.ndarray, GData]:
  """Read a Gkeyll file and interpolate it onto a uniform mesh.

  Args:
    file_name: Path to the file.
    poly_order: Polynomial order of the DG basis.
    basis_type: Long basis name, e.g. ``"serendipity"``.
    comp: Optional component selector applied *after* interpolation
      (an int index or a ``"start:stop"`` slice string); ``None`` keeps
      every component.

  Returns:
    ``(grid, values, gdata)``: the squeezed interpolated grid (a list of
    squeezed per-dimension arrays) and values, and the interpolated dataset.
  """
  gdata = GData(file_name, basis_type=basis_type, poly_order=poly_order)
  interpolated = gdata.interpolate()
  if comp is not None:
    interpolated = interpolated.select(comp=comp)
  grid = interpolated.get_grid()
  values = interpolated.get_values()
  grid_out = [np.squeeze(grid[d]) for d in range(len(grid))]
  return grid_out, np.squeeze(values), interpolated


def interpolated_grid_values(
    data: GData,
    *,
    comp: int = 0) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
  """Interpolate ``data``'s DG coefficients onto its computational mesh.

  Shared by :mod:`~postgkyl.diagnostics.gk.rz` and
  :mod:`~postgkyl.diagnostics.gk.fluxsurf`, which both need a
  field-aligned dataset's fine computational grid (for sampling the
  simulation's geometry) alongside its interpolated values.

  Args:
    data: The dataset. Normally not yet interpolated; an already-interpolated
      dataset is accepted and used as-is.
    comp: Component to return.

  Returns:
    ``(edges, centers, values)``: the refined edge grid (one 1-D array per
    dimension), its cell-centered equivalent, and component ``comp``'s
    values on that grid.
  """
  # Idempotent on purpose: 'pgkyl ... interp gk_rz' is a natural thing to
  # type, and interpolating twice would run the DG evaluation matrix over
  # values that are already point values -- silently wrong output rather
  # than an error.
  field = data if data.ctx.get("interpolated") else data.interpolate()
  cells = field.values.shape[:-1]
  centers = numerics.nodal_to_cell_centered_grid(field.grid, cells)
  return field.grid, centers, field.values[..., comp]


def set_tick_font_size(ax, size: float) -> None:
  """Set an axes' tick-label and offset-text font size to ``size``."""
  ax.tick_params(axis="both", labelsize=size)
  ax.yaxis.get_offset_text().set_size(size)
  ax.xaxis.get_offset_text().set_size(size)


def dict_get_bool(dict_in: dict, key: str, default: bool) -> bool:
  """Interpret a dict value as a bool, returning ``default`` if absent.

  String values ``'1'``/``'true'`` (case-insensitive) are True, anything
  else False; non-string values are converted with ``bool()``.
  """
  if key not in dict_in:
    return default
  val = dict_in[key]
  if isinstance(val, str):
    return val.strip().lower() in ("1", "true")
  return bool(val)


def parse_slice_string(value: str) -> slice:
  """Parse a ``slice()`` from a ``'start:stop:step'`` string.

  Raises:
    ValueError: if any non-empty part is not an integer.
  """
  parts = value.split(":")
  parsed_parts = []
  for p in parts:
    try:
      parsed_parts.append(int(p) if p else None)
    except ValueError:
      raise ValueError(f"Invalid slice part: {p}")
  return slice(*parsed_parts)


def get_block_indices(multib: str, file_path_name: str) -> list[int]:
  """Return the indices of the blocks to process in a multiblock simulation.

  Args:
    multib: ``"-10"`` for a single block (index 0); ``"-1"`` to discover and
      use every block found by globbing ``file_path_name``; otherwise a
      comma-separated list or a ``'start:stop[:step]'`` slice string of the
      desired block indices.
    file_path_name: Path/filename glob used to discover blocks when
      ``multib == "-1"``, with the block index replaced by ``"*"`` (e.g.
      ``"<sim_name>_b*-<species>_field_0.gkyl"``).

  Returns:
    A list of block indices.

  Raises:
    NameError: if ``multib`` is neither ``"-10"``/``"-1"``, a comma-separated
      list, a slice string, nor a single integer.
  """

  def _is_int(s: str) -> bool:
    try:
      int(s)
      return True
    except ValueError:
      return False

  if multib == "-10":
    return [0]
  if multib == "-1":
    return list(range(len(glob.glob(file_path_name))))
  if "," in multib:
    return [int(b) for b in multib.split(",")]
  if ":" in multib:
    s = parse_slice_string(multib)
    return list(range(*s.indices(MAX_NUM_BLOCKS)))
  if _is_int(multib):
    return [int(multib)]
  raise NameError(
      "Blocks given to --multib -m must be a comma separated list or slice.")
