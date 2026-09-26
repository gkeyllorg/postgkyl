"""``.gkyl`` reading through Gkeyll itself (the primary read path).

``GkylCReader`` delegates the whole read -- header, grid, allocation, payload,
multi-range stitching -- to ``libg0core.so`` via :mod:`postgkyl.gpython.rio` and
returns the data as a **native** :class:`~postgkyl.gpython.array.GkylArray`, so
modal datasets start life in the modal domain. Python decodes the msgpack
metadata blob into ``ctx`` (same key policy as the pure-Python reader) and
builds the NumPy edge grid. Saved interpolated fields are returned as NumPy
point values, preserving their arithmetic semantics.

It declines (``is_compatible() -> False``) when the FFI is unavailable, the
file is not a field file (types 1/3), or a partial load (``axes=``/``comp=``)
was requested -- those fall through to the pure-Python :class:`GkylReader`.
"""

from __future__ import annotations

import numpy as np

from postgkyl import gpython
from . import mapping
from .metadata import resolve_gkyl_metadata, unpack_gkyl_metadata


class GkylCReader:
  """Reader protocol implementation backed by ``gkyl_array_rio``."""

  def __init__(self,
               file_name: str,
               ctx: dict | None = None,
               value_form: str | None = None,
               basis_type: str | None = None,
               poly_order: int | None = None,
               **kwargs):
    self.file_name = str(file_name)
    self.ctx = ctx if ctx is not None else {}
    self._value_form_override = value_form
    self._basis_type_override = basis_type
    self._poly_order_override = poly_order
    # Any partial-load request (axes=, comp=, ...) -> defer to the Python reader.
    self._partial = any(v is not None for v in kwargs.get("axes") or ()) or \
        kwargs.get("comp") is not None or \
        bool({k for k in kwargs if k not in ("axes", "comp")})

  def is_compatible(self) -> bool:
    if self._partial or not gpython.available():
      return False
    try:
      return gpython.rio.file_type(
          self.file_name) in gpython.rio.FIELD_FILE_TYPES
    except (OSError, RuntimeError):
      return False

  def preload(self) -> None:
    grid, _, meta, esznc, _ = gpython.rio.read_header(self.file_name)
    resolve_gkyl_metadata(self.ctx,
                          unpack_gkyl_metadata(meta),
                          basis_type=self._basis_type_override,
                          poly_order=self._poly_order_override,
                          value_form=self._value_form_override)
    self.ctx["_load_metadata"]["file_header"] = {
        "cells": grid["cells"].copy(),
        "lower": grid["lower"].copy(),
        "upper": grid["upper"].copy(),
        "num_comps": esznc // 8,
    }
    self.ctx["cells"] = grid["cells"]
    self.ctx["lower"] = grid["lower"]
    self.ctx["upper"] = grid["upper"]
    self.ctx["num_comps"] = esznc // 8  # payload is float64

  def load(self):
    grid, arr = gpython.rio.read_field(self.file_name)
    cells = grid["cells"]
    if arr.size != int(np.prod(cells)):
      raise IOError(
          f"'{self.file_name}': stored cells {arr.size} do not match the "
          f"domain {tuple(cells)} (ghost-cell layout?) -- not supported by "
          "the Gkeyll read path yet")
    edges = mapping.uniform_grid(grid["lower"], grid["upper"], cells)
    self.ctx.setdefault("grid_type", "uniform")
    if self.ctx.get("interpolated"):
      # Saved evaluation meshes contain point values, not DG coefficients.
      return edges, np.array(arr.view(cells), copy=True)
    return edges, arr
