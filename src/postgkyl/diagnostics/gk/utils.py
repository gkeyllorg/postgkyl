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
import re

import numpy as np
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


def read_moment_if_present(
    file_name: str,
    comp: int) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
  """Read one component of an integrated-moment time trace if present:
  ``(found, time, values)``. Unlike :func:`read_time_trace_if_present`, this
  also handles single-component moments (e.g. ``M0``)."""
  found, time, values, _ = read_time_trace_if_present(file_name)
  if not found:
    return False, None, None
  time = np.atleast_1d(time)
  return True, time, values.reshape(time.size, -1)[:, comp]


def integrated_moment_file(stem: str, moments: tuple[str, ...]) -> str:
  """Path of the first existing integrated-moment file among ``moments``.

  Gkeyll writes each integrated moment to ``<stem>integrated_<moment>.gkyl``,
  e.g. ``<name>-<species>_fdot_integrated_HamiltonianMoments.gkyl``; older
  versions wrote a single ``<stem>integrated_moms.gkyl``, tried last.

  Args:
    stem: File path up to ``integrated_``; may contain a ``*`` glob.
    moments: Moment names, in order of preference.

  Returns:
    The first existing candidate, or the first candidate if none exists.
  """
  names = [f"{stem}integrated_{m}.gkyl" for m in (*moments, "moms")]
  return next((n for n in names if glob.glob(n)), names[0])


def read_balance_files(
    file_names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  """Read Gkeyll balance files, one per block, summing columns over blocks.

  A gyrokinetic run that requests a balance diagnostic writes its terms to
  ``<name>-balance_<type>.gkyl`` (summed over species) and
  ``<name>-<species>_balance_<type>.gkyl``. The columns are keyed by the
  files' ``Columns`` metadata. Each block normalizes ``mom_err_norm`` by its
  own conserved quantity, so normalized errors of several blocks do not add:
  that column is only returned for a single file.

  Args:
    file_names: Balance files of the blocks to sum.

  Returns:
    ``(time, columns)``.

  Raises:
    ValueError: if a file has no ``Columns`` metadata.
  """
  time, columns = None, {}
  for file_name in file_names:
    grid, values, gdata = read_gfile(file_name)
    names = gdata.ctx.get("Columns")
    if names is None:
      raise ValueError(f"No 'Columns' metadata in {file_name}")
    # A single sample loses its time axis to the squeeze in read_gfile.
    time, values = np.atleast_1d(grid[0]), np.atleast_2d(values)
    for comp, column in enumerate(c.strip() for c in names.split(",")):
      columns[column] = columns.get(column, 0.0) + values[:, comp]
  if len(file_names) > 1:
    del columns["mom_err_norm"]
  return time, columns


def balance_file_blocks(multib: str, balance_file: str,
                        moment_files: list[str]) -> list[int]:
  """Return the blocks to read from balance files instead of moment files.

  Args:
    multib: Block selection, as in :func:`get_block_indices`.
    balance_file: Balance-file path, with ``*`` for the block index.
    moment_files: Paths of the files the balance is otherwise computed from,
      with ``*`` for the block index; empty if there are none.

  Returns:
    The selected blocks if every one has ``balance_file`` but some lack one
    of ``moment_files``; otherwise ``[]`` (read the moment files).
  """

  def _present(file_name: str, blocks: list[int]) -> bool:
    return bool(blocks) and all(
        os.path.exists(file_name.replace("*", str(b))) for b in blocks)

  if moment_files:
    moment_blocks = get_block_indices(multib, moment_files[0])
    if all(_present(f, moment_blocks) for f in moment_files):
      return []
  blocks = get_block_indices(multib, balance_file)
  return blocks if _present(balance_file, blocks) else []


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


def sum_label(terms: list[str]) -> str:
  """Mathtext label of the sum of terms given by their own labels.

  The labels are concatenated, with a ``+`` before every term but the first
  whose label does not start with a minus sign, e.g. ``["$a$", "$b$",
  "$-c$"]`` gives ``"$a$$+b$$-c$"`` (rendered ``a+b-c``).
  """
  return "".join(t if i == 0 or t.startswith("$-") else "$+" + t[1:]
                 for i, t in enumerate(terms))


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
    file_path_name: Path/filename glob used to discover blocks for ``-1`` or
      an open-ended slice, with the block index replaced by ``"*"`` (e.g.
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

  def _discovered_blocks() -> list[int]:
    if "*" not in file_path_name:
      return []
    prefix, suffix = file_path_name.split("*", 1)
    pattern = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix))
    return sorted({
        int(match.group(1))
        for path in glob.glob(file_path_name)
        if (match := pattern.fullmatch(path)) is not None
    })

  if multib == "-10":
    return [0]
  if multib == "-1":
    return _discovered_blocks()
  if "," in multib:
    return [int(b) for b in multib.split(",")]
  if ":" in multib:
    s = parse_slice_string(multib)
    if s.stop is None:
      discovered = _discovered_blocks()
      if not discovered:
        return []
      selected = range(*s.indices(discovered[-1] + 1))
      available = set(discovered)
      return [block for block in selected if block in available]
    return list(range(*s.indices(MAX_NUM_BLOCKS)))
  if _is_int(multib):
    return [int(multib)]
  raise NameError(
      "Blocks given to --multib -m must be a comma separated list or slice.")
