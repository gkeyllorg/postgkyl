"""Reader for FLASH code HDF5 output.

FLASH variable names (for reference, not enforced here):

- ``dens``: density [g/cc]
- ``tele``/``tion``: electron/ion temperature [K]
- ``velx``/``vely``: fluid velocity [cm/s]
- ``temp``: overall fluid temperature [K]
- ``pres``: pressure [dyn/cm^2]
- ``ye``/``sumy``: used to recover ion/electron density,
  ``n_ele = ye * Na * dens``, ``n_ion = sumy * Na * dens`` (``Na`` = Avogadro
  number); average ionization ``Z' = ye / sumy``, average atomic mass
  ``A' = 1 / sumy``.

``tables`` (PyTables) is a hard dependency (see ``pyproject.toml``), so this
reader needs no optional-import guard.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import tables

from . import mapping


class FlashH5Reader:
  """Provides a framework to read FLASH HDF5 output."""

  def __init__(self,
               file_name: str,
               ctx: dict | None = None,
               var_name: str | None = None,
               **kwargs):
    """Initialize the instance of the FLASH reader.

    Args:
      file_name: path to the ``.h5`` file.
      ctx: dict passing context/metadata back to the caller.
      var_name: FLASH block variable to read (e.g. ``"dens"``); required by
        :meth:`load` but not by :meth:`is_compatible`.
      **kwargs: unused; keeps the constructor signature uniform across the
        reader registry.
    """
    self._file_name = str(file_name)
    self.var_name = var_name

    self.ctx = ctx if ctx is not None else {}

  def is_compatible(self) -> bool:
    """Checks if the file can be read with the FLASH reader."""
    try:
      fh = tables.open_file(self._file_name, "r")
    except (tables.exceptions.HDF5ExtError, OSError):
      return False
    out = "coordinates" in fh.root
    fh.close()
    return out

  def _read_frame(self) -> tuple:
    fh = tables.open_file(self._file_name, "r")
    coord = fh.root["coordinates"].read().transpose()
    bsize = fh.root["block size"].read().transpose()
    ntype = fh.root["node type"].read().transpose()
    bdata = fh.root[self.var_name].read().transpose()
    fh.close()

    nxb, nyb, _, num_blocks = bdata.shape
    res = bsize.min(axis=1)
    lower = (coord - bsize / 2).min(axis=1)
    upper = (coord + bsize / 2).max(axis=1)

    nxax = math.floor((upper[0] - lower[0]) / (res[0] / nxb))
    nyax = math.floor((upper[1] - lower[1]) / (res[1] / nyb))
    data = np.zeros((nxax, nyax))
    for b in range(num_blocks):
      if ntype[b] == 1:
        mult = np.ceil(bsize[:, b] / res)
        idxx = math.floor(
            (coord[0, b] - bsize[0, b] / 2 - lower[0]) / res[0] * nxb)
        idxy = math.floor(
            (coord[1, b] - bsize[1, b] / 2 - lower[1]) / res[1] * nyb)
        for i in range(nxb):
          for j in range(nyb):
            data[
                idxx + i * int(mult[0]):idxx + (i + 1) * int(mult[0]) + 1,
                idxy + j * int(mult[1]):idxy + (j + 1) * int(mult[1]) + 1,
            ] = bdata[i, j, 0, b]
    return data.shape, lower[:2], upper[:2], data[..., np.newaxis]

  # ---- Exposed functions -----
  def preload(self) -> None:
    """Loads metadata. FLASH block reassembly needs the full field, so there
    is nothing cheaper to precompute here."""

  def load(self) -> Tuple[list, np.ndarray]:
    """Loads data.

    Returns:
      A tuple including a grid list and a data NumPy array.

    Raises:
      ValueError: if ``var_name`` was not given.

    Notes:
      Needs to be called after ``preload``.
    """
    if self.var_name is None:
      raise ValueError(
          "FlashH5Reader requires 'var_name' (the FLASH block variable to "
          "read, e.g. 'dens') to load data.")

    cells, lower, upper, data = self._read_frame()
    self.ctx["cells"] = cells
    self.ctx["lower"] = lower
    self.ctx["upper"] = upper
    self.ctx["num_comps"] = data.shape[-1]
    self.ctx["grid_type"] = "uniform"

    grid = mapping.uniform_grid(np.asarray(lower, dtype=float),
                                np.asarray(upper, dtype=float),
                                np.asarray(cells))
    return grid, data
