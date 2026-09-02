"""Write a dataset back to disk.

A leaf module: it consumes the read-only *surface* of a dataset (the same
properties the readers fill) and never imports ``gdatastate``/``operations``. Supports the
Gkeyll binary ``.gkyl`` format (round-trips with :class:`GkylReader`), plain
ASCII ``.txt``, NumPy ``.npy``, and legacy VTK structured-grid ``.vtk``
(for external 3-D/VR viewers such as ParaView).
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal, Protocol

import msgpack
import numpy as np
import pyvista as pv

from postgkyl.cli_spec import (
    CommandSpec, Execution, ResultPolicy, Section, command,
)
from postgkyl.numerics import nodal_to_cell_centered_grid

# ctx keys that are either structural (already carried by the binary header
# itself, e.g. cells/lower/upper) or postgkyl's own session-only bookkeeping
# (recomputed by the reader from the meta below) -- never part of the
# msgpack meta blob Gkeyll writes.
_INTERNAL_CTX_KEYS = frozenset({
    "cells", "lower", "upper", "num_comps", "num_dims", "grid_type",
    "value_form", "num_quad", "interpolated", "var_names",
    # The source file's parsed identity (see io.naming): derived from the
    # *path*, never stored in the file. Writing it would be actively wrong --
    # reloading the output under a different name would then find a stale
    # sim/block in the header, and GDataState's setdefault lets header
    # metadata win over the parsed name. "frame" is deliberately NOT here:
    # Gkeyll itself writes that one.
    "sim", "block", "quantity",
})

# ctx uses postgkyl's snake_case names; Gkeyll's own meta blob (and anything
# else that reads the file) expects the original camelCase keys.
_CTX_TO_META_KEY = {"poly_order": "polyOrder", "basis_type": "basisType"}


class _WritableDataset(Protocol):
  """Read-only dataset surface consumed by the format writers."""

  num_dims: int
  num_comps: int
  num_cells: np.ndarray
  bounds: tuple[np.ndarray, np.ndarray]
  values: object
  grid: list
  ctx: dict
# end


@command(CommandSpec(Section.UTILITY, Execution.TERMINAL_EACH,
    result=ResultPolicy.VALUE))
def save(data: _WritableDataset, out_name: str = "",
    extension: Literal["gkyl", "txt", "npy", "vtk"] = "gkyl",
    var_name: str = "CartGridField") -> str:
  """Write ``data`` to ``out_name`` in the requested ``extension``.

  Args:
    data: a dataset exposing ``num_dims``/``num_comps``/``num_cells``/
      ``bounds``/``values``/``grid``/``ctx`` (a ``GDataState`` or subclass).
    out_name: output path; when empty a name is derived from the source file.
    extension: one of ``"gkyl"`` (default), ``"txt"``, ``"npy"``, ``"vtk"``.
    var_name: unused placeholder kept for interface symmetry.

  Returns:
    The path actually written.
  """
  if not out_name:
    src = getattr(data, "_file_name", "") or ""
    stem = src.split(".", maxsplit=1)[0].strip("_") if src else "gdata"
    out_name = f"{stem}_mod.{extension}"
  # end
  elif out_name.split(".")[-1] != extension:
    out_name += "." + extension
  # end

  num_dims = data.num_dims
  num_comps = data.num_comps
  num_cells = data.num_cells
  lo, up = data.bounds
  values = data.values

  if extension == "gkyl":
    ctx = getattr(data, "ctx", {}) or {}
    _write_gkyl(out_name, num_dims, num_comps, num_cells, lo, up, values, ctx)
  # end
  elif extension == "npy":
    np.save(out_name, np.asarray(values).squeeze())
  # end
  elif extension == "txt":
    _write_txt(out_name, data, num_dims, num_comps, num_cells, values)
  # end
  elif extension == "vtk":
    _write_vtk(out_name, data, num_dims, num_cells, values)
  # end
  else:
    raise ValueError(f"Unsupported write extension '{extension}'")
  # end
  return out_name
# end


def _build_meta(ctx: dict) -> dict:
  """Translate ``ctx`` back into the msgpack meta blob Gkeyll itself writes
  (poly order, basis type, physical params, time/frame stamps, ...) --
  everything except the structural/session-only keys in
  ``_INTERNAL_CTX_KEYS``."""
  meta = {}
  for key, val in ctx.items():
    if key in _INTERNAL_CTX_KEYS:
      continue
    # end
    meta[_CTX_TO_META_KEY.get(key, key)] = _to_msgpack_safe(val)
  # end
  return meta
# end


def _to_msgpack_safe(val):
  if isinstance(val, np.generic):
    return val.item()
  # end
  if isinstance(val, np.ndarray):
    return val.tolist()
  # end
  return val
# end


def _write_gkyl(out_name, num_dims, num_comps, num_cells, lo, up, values, ctx) -> None:
  dti = np.dtype("i8")
  dtf = np.dtype("f8")
  meta = _build_meta(ctx)
  packed = msgpack.packb(meta, use_bin_type=True) if meta else b""
  with open(out_name, "wb") as fh:
    np.array([103, 107, 121, 108, 48], dtype=np.dtype("b")).tofile(fh, sep="")  # 'gkyl0'
    np.array([1], dtype=dti).tofile(fh, sep="")              # version 1
    np.array([1], dtype=dti).tofile(fh, sep="")              # file type 1 (field)
    np.array([len(packed)], dtype=dti).tofile(fh, sep="")    # meta size
    if packed:
      fh.write(packed)
    # end
    np.array([2], dtype=dti).tofile(fh, sep="")              # real type (f8)
    np.array([num_dims], dtype=dti).tofile(fh, sep="")
    np.array(num_cells, dtype=dti).tofile(fh, sep="")
    np.array(lo, dtype=dtf).tofile(fh, sep="")
    np.array(up, dtype=dtf).tofile(fh, sep="")
    np.array([num_comps * 8], dtype=dti).tofile(fh, sep="")  # elem_sz
    np.array([int(np.prod(num_cells))], dtype=dti).tofile(fh, sep="")  # asize
    np.array(values, dtype=dtf).tofile(fh, sep="")
  # end
# end


def _write_txt(out_name, data, num_dims, num_comps, num_cells, values) -> None:
  grid = [0.5 * (g[1:] + g[:-1]) for g in data.grid]  # cell centers
  num_rows = int(np.prod(num_cells))
  basis = np.full(num_dims, 1.0)
  for d in range(num_dims - 1):
    basis[d] = np.prod(num_cells[(d + 1):])
  # end
  with open(out_name, "w", encoding="utf-8") as fh:
    for i in range(num_rows):
      idx = i
      idxs = np.zeros(num_dims, np.int32)
      for d in range(num_dims):
        idxs[d] = int(idx // basis[d])
        idx = idx % basis[d]
      # end
      cells = [f"{grid[d][idxs[d]]:.15e}" for d in range(num_dims)]
      comps = [f"{values[tuple(idxs)][c]:.15e}" for c in range(num_comps)]
      fh.write(", ".join(cells + comps) + "\n")
  # end
# end
    # end


def _write_vtk(out_name, data, num_dims, num_cells, values) -> None:
  """Write a legacy VTK structured-grid file via PyVista.

  1-D/2-D fields are written as a height-mapped surface (the field value
  becomes the missing coordinate, e.g. z for a 1-D line); 3-D fields are
  written as a volume with the field stored as point data ``"f_raw"``.
  """
  if num_dims not in (1, 2, 3):
    raise ValueError(f"VTK output supports 1-3 dimensions, got {num_dims}")
  # end

  n_grid = nodal_to_cell_centered_grid(data.grid, num_cells, meshgrid=True)
  fval = np.asarray(values).squeeze()
  if num_dims == 1:
    x = n_grid[0]
    y = np.zeros_like(x)
    z = fval
  # end
  elif num_dims == 2:
    x, y = n_grid
    z = fval
  # end
  else:
    x, y, z = n_grid
  # end

  grid3d = pv.StructuredGrid(x, y, z)
  grid3d["f_raw"] = fval.ravel(order="F")
  grid3d.save(out_name)
  _update_vtk_series_file(data, out_name)
# end


def _update_vtk_series_file(data, out_name: str) -> None:
  """Create or update ParaView ``.series`` metadata for VTK file-series
  time playback: each write of a frame-numbered file appends (or refreshes)
  its entry, keyed by the series' shared stem."""
  out_dir = os.path.dirname(out_name)
  out_file = os.path.basename(out_name)
  stem, ext = os.path.splitext(out_file)
  match = re.match(r"^(.*?)(?:[_-]?(\d+))$", stem)
  if match and match.group(1):
    series_stem = match.group(1).rstrip("_-") or stem
  # end
  else:
    series_stem = stem
  # end

  series_path = os.path.join(out_dir, f"{series_stem}{ext}.series")
  time_value = float(data.ctx.get("time", data.ctx.get("frame", 0.0)))
  rel_file = os.path.relpath(out_name, out_dir if out_dir else ".")

  series_data = {"file-series-version": "1.0", "files": []}
  if os.path.exists(series_path):
    try:
      with open(series_path, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)
      # end
      if isinstance(loaded, dict) and isinstance(loaded.get("files"), list):
        series_data = loaded
        series_data.setdefault("file-series-version", "1.0")
    # end
      # end
    except (OSError, json.JSONDecodeError):
      pass
    # end
  # end

  replaced = False
  for entry in series_data["files"]:
    if entry.get("name") == rel_file:
      entry["time"] = time_value
      replaced = True
      break
    # end
  # end
  if not replaced:
    series_data["files"].append({"name": rel_file, "time": time_value})
  # end

  series_data["files"].sort(key=lambda x: (float(x.get("time", 0.0)), x.get("name", "")))
  with open(series_path, "w", encoding="utf-8") as fh:
    json.dump(series_data, fh, indent=2)
    fh.write("\n")
  # end
# end
