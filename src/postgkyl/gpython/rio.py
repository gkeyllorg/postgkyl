"""File loading through Gkeyll's ``gkyl_array_rio`` -- the C read path.

``read_field`` performs the whole read (grid + allocate + fill, including
multi-range stitching for file_type 3) inside Gkeyll; ``read_header`` returns
the grid and the raw msgpack metadata blob without touching the payload.
Decoding the msgpack bytes is left to the caller (``io/``) -- metadata policy
is an io concern, bytes are a floor concern.
"""

from __future__ import annotations

import numpy as np

from . import _lib
from .array import GkylArray

# enum gkyl_file_type ordinals used by gkyl_get_gkyl_file_type
FIELD_FILE_TYPES = (1, 3)  # single-range and multi-range field data


def file_type(file_name: str) -> int:
  """The gkyl file type (1..5), or -1 if not a gkyl file."""
  return int(_lib.require().file_type(file_name))


def read_header(file_name: str):
  """Header-only read: ``(grid_dict, file_type, meta_bytes, esznc, tot_cells)``.

  ``grid_dict`` has ``ndim``/``lower``/``upper``/``cells`` as NumPy values;
  ``meta_bytes`` is the raw msgpack blob (b"" when the file has none).
  """
  grid, ftype, meta, esznc, tot_cells = _lib.require().read_header(file_name)
  return _grid_dict(grid), ftype, meta, esznc, tot_cells


def read_field(file_name: str):
  """Full field read inside Gkeyll: ``(grid_dict, GkylArray)``."""
  grid, cap = _lib.require().read_field(file_name)
  return _grid_dict(grid), GkylArray(cap)


def write_field(file_name: str,
                grid: dict,
                arr: GkylArray,
                *,
                meta: bytes = b"") -> None:
  """Write ``arr`` on a uniform ``grid`` through ``gkyl_grid_sub_array_write``.

  The same C write path Gkeyll itself uses, so a round trip through this
  function and :func:`read_field` is bit-exact by construction. ``meta`` is
  a raw msgpack byte blob (encoding policy belongs to ``io/``, which decodes
  it the same way on read); pass ``b""`` for no metadata.

  Args:
    file_name: destination path.
    grid: a dict with ``lower``/``upper``/``cells`` (as returned by
      :func:`read_header`/:func:`read_field`, or built by the caller).
    arr: the array to write; ``arr.size`` must equal ``prod(grid["cells"])``.
    meta: raw msgpack bytes, or empty for none.

  Raises:
    ValueError: ``grid["cells"]`` does not cover ``arr``.
    OSError: the underlying ``gkyl_array_rio`` write failed.
  """
  lower = np.asarray(grid["lower"], dtype=np.float64)
  upper = np.asarray(grid["upper"], dtype=np.float64)
  cells = np.asarray(grid["cells"], dtype=np.int32)
  if int(np.prod(cells)) != arr.size:
    raise ValueError(f"grid cells {tuple(cells)} do not cover the array "
                     f"({int(np.prod(cells))} vs {arr.size} cells)")
  _lib.require().write_field(file_name, lower, upper, cells,
                             meta if meta else None, arr._cap)


def read_dynvec(file_name: str):
  """Read a time-series (dynvector) file: ``(time (n,), data (n, ncomp))``.

  Args:
    file_name: path to a gkyl dynvector file (``file_type`` 2).

  Returns:
    ``time``: 1-D array of ``n`` timestamps.
    ``data``: ``(n, ncomp)`` array of the recorded values.

  Raises:
    OSError: missing file, non-double dynvector, or a read failure.
  """
  ncomp, tm_cap, data_cap = _lib.require().dynvec_read(file_name)
  tm = GkylArray(tm_cap).to_numpy()[:, 0]
  data = GkylArray(data_cap).to_numpy()
  return tm, data


def write_dynvec(file_name: str, time: np.ndarray, data: np.ndarray) -> None:
  """Write a time-series (dynvector) file via ``gkyl_dynvec_write``.

  Args:
    file_name: destination path.
    time: 1-D array of ``n`` timestamps.
    data: ``(n,)`` or ``(n, ncomp)`` array of values, one row per timestamp.

  Raises:
    ValueError: ``time`` and ``data`` disagree on the number of samples.
    OSError: the underlying ``gkyl_dynvec_write`` failed.
  """
  time = np.ascontiguousarray(time, dtype=np.float64)
  data = np.asarray(data, dtype=np.float64)
  if data.ndim == 1:
    data = data[:, None]
  if data.shape[0] != time.shape[0]:
    raise ValueError(f"time has {time.shape[0]} samples but data has "
                     f"{data.shape[0]}")
  _lib.require().dynvec_write(file_name, time, np.ascontiguousarray(data))


def _grid_dict(grid: tuple) -> dict:
  ndim, lower, upper, cells = grid
  return {
      "ndim": int(ndim),
      "lower": np.asarray(lower),
      "upper": np.asarray(upper),
      "cells": np.asarray(cells, dtype=np.int64),
  }
