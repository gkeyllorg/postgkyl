"""Reader for legacy Gkeyll HDF5 output (predates the native .gkyl binary format).

``tables`` (PyTables) is a hard dependency (see ``pyproject.toml``), so this
reader needs no optional-import guard.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tables

from . import mapping


class GkylH5Reader:
  """Provides a framework to read legacy Gkeyll HDF5 output."""

  def __init__(self, file_name: str, ctx: dict | None = None, **kwargs):
    """Initialize the instance of the legacy Gkeyll HDF5 reader.

    Args:
      file_name: path to the ``.h5`` file.
      ctx: dict passing context/metadata back to the caller.
      **kwargs: unused; keeps the constructor signature uniform across the
        reader registry.
    """
    self._file_name = str(file_name)

    self.is_frame = False
    self.is_diagnostic = False

    self.ctx = ctx if ctx is not None else {}

  def is_compatible(self) -> bool:
    """Checks if the file can be read with the legacy Gkeyll HDF5 reader."""
    try:
      fh = tables.open_file(self._file_name, "r")
    except (tables.exceptions.HDF5ExtError, OSError):
      return False

    if "/DataStruct/data" in fh:
      self.is_diagnostic = True
    if "/StructGridField" in fh:
      self.is_frame = True
    fh.close()
    return self.is_frame or self.is_diagnostic

  def _read_frame(self) -> tuple:
    fh = tables.open_file(self._file_name, "r")

    # Postgkyl conventions require the attributes to be arrays even for 1D data.
    lower = np.atleast_1d(fh.root.StructGrid._v_attrs.vsLowerBounds)
    upper = np.atleast_1d(fh.root.StructGrid._v_attrs.vsUpperBounds)
    cells = np.atleast_1d(fh.root.StructGrid._v_attrs.vsNumCells)
    if "/timeData" in fh:
      self.ctx["time"] = fh.root.timeData._v_attrs.vsTime

    data = fh.root.StructGridField.read()

    fh.close()
    return cells, lower, upper, data

  def _read_diagnostic(self) -> tuple:
    fh = tables.open_file(self._file_name, "r")

    grid = fh.root.DataStruct.timeMesh.read()
    data = fh.root.DataStruct.data.read()

    fh.close()
    return [np.squeeze(grid)], [grid[0]], [grid[-1]], data

  # ---- Exposed functions -----
  def preload(self) -> None:
    """Loads metadata. Nothing to precompute for this format."""

  def load(self) -> Tuple[list, np.ndarray]:
    """Loads data.

    Returns:
      A tuple including a grid list and a data NumPy array.

    Notes:
      Needs to be called after ``preload``.
    """
    if self.is_frame:
      cells, lower, upper, data = self._read_frame()
    else:
      grid, lower, upper, data = self._read_diagnostic()
      cells = grid[0].shape

    self.ctx["cells"] = cells
    self.ctx["lower"] = lower
    self.ctx["upper"] = upper
    self.ctx["num_comps"] = 1
    if len(data.shape) > len(cells):
      self.ctx["num_comps"] = data.shape[-1]

    grid = mapping.uniform_grid(np.asarray(lower, dtype=float),
                                np.asarray(upper, dtype=float),
                                np.asarray(cells))
    self.ctx["grid_type"] = "uniform"

    return grid, data
