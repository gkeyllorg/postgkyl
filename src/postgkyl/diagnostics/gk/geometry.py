"""Resolve and decode Gkeyll's auxiliary geometry files.

This module owns geometry suffixes, representation selection, multiblock
paths, and file-layout interpretation. Filename parsing belongs to io.naming.
"""
from __future__ import annotations

import os
import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.io import parse_output_name
from postgkyl.numerics import nodal_to_cell_centered_grid
from postgkyl.operations import interpolate
from postgkyl.operations.geometry import Geometry, _validate_geometry

# Mirrors ``enum gkyl_geometry_id`` in gkeyll/core/zero/gkyl_eqn_type.h.
# This foreign-format fact is shared with the grid-node diagnostic, which
# imports it from here instead of maintaining a second copy.
GKYL_GEOMETRY_ID = [
    "GKYL_GEOMETRY_NONE",
    "GKYL_GEOMETRY_TOKAMAK",
    "GKYL_GEOMETRY_MIRROR",
    "GKYL_GEOMETRY_MAPC2P",
    "GKYL_GEOMETRY_FROMFILE",
]
_MAPC2P_IDX = GKYL_GEOMETRY_ID.index("GKYL_GEOMETRY_MAPC2P")


def is_geo_mapc2p(ctx: dict) -> bool:
  """Whether ``ctx`` identifies user-supplied Cartesian MAPC2P geometry.

  Files without ``geometry_type`` retain the historical MAPC2P default.
  """
  return ctx.get("geometry_type", _MAPC2P_IDX) == _MAPC2P_IDX


def geometry_prefix(file_name: str | None) -> str | None:
  """Return the per-block simulation prefix for ``file_name``.

  Parsing is delegated to :mod:`postgkyl.io.naming`, the authoritative home
  of Gkeyll's output-name convention.
  """
  name = parse_output_name(file_name)
  return name.prefix if name is not None else None


def per_block_path(path: str | None, block: int | None) -> str | None:
  """Substitute a multiblock index for ``'*'`` in a geometry override."""
  if path is None or block is None or "*" not in path:
    return path
  return path.replace("*", str(block))


def _gauss_nodes(edges: np.ndarray) -> np.ndarray:
  """Physical p1 Gauss-node coordinates for a one-dimensional edge grid."""
  centers = 0.5 * (edges[:-1] + edges[1:])
  offsets = np.diff(edges) / (2.0 * np.sqrt(3.0))
  return np.ravel(np.column_stack([centers - offsets, centers + offsets]))


def _pointwise_file(
    path: str) -> tuple[list[np.ndarray], np.ndarray, GDataState]:
  """Read a point-value geometry file and squeeze singleton dimensions."""
  data = GDataState(path)
  grid = [np.squeeze(axis) for axis in data.grid]
  return grid, np.squeeze(data.values), data


def _geometry_components(
    values: np.ndarray, data: GDataState,
    path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """Interpret one geometry value array as ``R``, ``Z``, and optional phi."""
  required = 3 if is_geo_mapc2p(data.ctx) else 2
  if values.ndim < 2 or values.shape[-1] < required:
    kind = "Cartesian X/Y/Z" if required == 3 else "R/Z"
    raise ValueError(
        f"Geometry file '{path}' must contain at least {required} {kind} components."
    )

  if is_geo_mapc2p(data.ctx):
    x, y, z = values[..., 0], values[..., 1], values[..., 2]
    return np.sqrt(x**2 + y**2), z, np.arctan2(y, x)
  r, z = values[..., 0], values[..., 1]
  phi = values[..., 2] if r.ndim == 3 and values.shape[-1] >= 3 else None
  return r, z, phi


def _read_mapc2p_geometry(path: str):
  """Interpolate a modal geometry file to physical ``R``, ``Z``, and phi."""
  source = GDataState(path)
  field = interpolate(source)
  cells = field.values.shape[:-1]
  coords = nodal_to_cell_centered_grid(field.grid, cells)
  major_r, vert_z, phi = _geometry_components(field.values, field, path)
  return coords, major_r, vert_z, phi


def _read_nodes_geometry(path: str):
  """Read a p1 pointwise nodal geometry file."""
  grid, values, data = _pointwise_file(path)
  coords = []
  for dim, axis in enumerate(grid):
    if axis.ndim != 1 or axis.shape[0] != values.shape[dim] + 1 \
        or values.shape[dim] % 2:
      raise ValueError(f"Unrecognized nodal geometry layout in '{path}'.")
    coords.append(_gauss_nodes(axis[::2]))
  major_r, vert_z, phi = _geometry_components(values, data, path)
  return coords, major_r, vert_z, phi


def _read_corner_rz(path: str):
  """Read R and Z from a pointwise ``'-geo_corn_nodes.gkyl'`` file."""
  grid, values, data = _pointwise_file(path)
  coords = [
      np.linspace(axis[0], axis[-1], n)
      for axis, n in zip(grid, values.shape[:-1])
  ]
  major_r, vert_z, _ = _geometry_components(values, data, path)
  return coords, major_r, vert_z


def resolve_geometry(file_name: str | None,
                     *,
                     mapc2p: str | None = None,
                     nodes_file: str | None = None) -> Geometry:
  """Resolve and load the geometry belonging to ``file_name``.

  The exact pointwise ``'<prefix>-geo_int_nodes.gkyl'`` representation is
  preferred, with ``'<prefix>-geo_int_mapc2p.gkyl'`` as the modal fallback.
  ``nodes_file`` and ``mapc2p`` override that lookup and are mutually
  exclusive.  Passing ``mapc2p=''`` explicitly requests the inferred modal
  filename.

  Raises:
    ValueError: If both overrides are supplied or no geometry can be found.
  """
  if mapc2p is not None and nodes_file is not None:
    raise ValueError("Pass either mapc2p= or nodes_file=, not both.")

  parsed = parse_output_name(file_name)
  prefix = geometry_prefix(file_name)
  block = parsed.block if parsed is not None else None
  nodes_file = per_block_path(nodes_file, block)
  mapc2p = per_block_path(mapc2p, block)
  if nodes_file is not None:
    path, kind = nodes_file, "nodes"
  elif mapc2p is not None:
    path = mapc2p or (f"{prefix}-geo_int_mapc2p.gkyl" if prefix else None)
    kind = "mapc2p"
  elif prefix is not None:
    path, kind = f"{prefix}-geo_int_nodes.gkyl", "nodes"
    if not os.path.exists(path):
      path, kind = f"{prefix}-geo_int_mapc2p.gkyl", "mapc2p"
  else:
    path, kind = None, None

  if path is None or not os.path.exists(path):
    raise ValueError(
        "Could not find a geometry file; pass nodes_file= or mapc2p= explicitly."
    )

  coords, major_r, vert_z, phi = (_read_nodes_geometry(path) if kind == "nodes"
                                  else _read_mapc2p_geometry(path))

  corner = None
  if prefix is not None:
    corner_path = f"{prefix}-geo_corn_nodes.gkyl"
    if os.path.exists(corner_path):
      corner = _read_corner_rz(corner_path)

  geometry = Geometry(coords=coords,
                      major_r=major_r,
                      vert_z=vert_z,
                      phi=phi,
                      corner=corner)
  _validate_geometry(geometry, len(coords))
  return geometry


__all__ = [
    "resolve_geometry", "geometry_prefix", "per_block_path", "GKYL_GEOMETRY_ID",
    "is_geo_mapc2p"
]
