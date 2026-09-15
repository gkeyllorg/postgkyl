"""Gkeyll auxiliary geometry filenames and on-disk coordinate layouts.

These conventions are shared by all equation systems. File resolution returns
paths; decoding consumes arrays and metadata, without importing dataset state.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal
import numpy as np

from .naming import parse_output_name

# Mirrors enum gkyl_geometry_id in gkeyll/core/zero/gkyl_eqn_type.h.
GKYL_GEOMETRY_ID = (
    "GKYL_GEOMETRY_NONE",
    "GKYL_GEOMETRY_TOKAMAK",
    "GKYL_GEOMETRY_MIRROR",
    "GKYL_GEOMETRY_MAPC2P",
    "GKYL_GEOMETRY_FROMFILE",
)
_MAPC2P_IDX = GKYL_GEOMETRY_ID.index("GKYL_GEOMETRY_MAPC2P")


@dataclass(frozen=True)
class GeometryFiles:
  """Resolved interior geometry and optional boundary-node file."""

  interior: str
  kind: Literal["nodes", "mapc2p"]
  corner: str | None


def is_geo_mapc2p(ctx: dict) -> bool:
  """Identify Cartesian MAPC2P output; absent metadata has this legacy default."""
  return ctx.get("geometry_type", _MAPC2P_IDX) == _MAPC2P_IDX


def geometry_prefix(file_name: str | None) -> str | None:
  """Return the simulation prefix, including its block index when present."""
  name = parse_output_name(file_name)
  return name.prefix if name is not None else None


def per_block_path(path: str | None, block: int | None) -> str | None:
  """Substitute a block index for '*' in an explicit geometry path."""
  if path is None or block is None or "*" not in path:
    return path
  return path.replace("*", str(block))


def resolve_geometry_files(file_name: str | None,
                           *,
                           mapc2p: str | None = None,
                           nodes_file: str | None = None) -> GeometryFiles:
  """Find a block's geometry, preferring interior nodes over a modal map.

  Overrides are mutually exclusive. ``mapc2p=''`` requests the inferred modal
  file. A '*' in either override is replaced by the source file's block index.
  """
  if mapc2p is not None and nodes_file is not None:
    raise ValueError("Pass either mapc2p= or nodes_file=, not both.")

  parsed = parse_output_name(file_name)
  prefix = parsed.prefix if parsed is not None else None
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
    path, kind = None, "nodes"

  if path is None or not os.path.exists(path):
    raise ValueError(
        "Could not find a geometry file; pass nodes_file= or mapc2p= explicitly."
    )
  corner = f"{prefix}-geo_corn_nodes.gkyl" if prefix is not None else None
  if corner is not None and not os.path.exists(corner):
    corner = None
  return GeometryFiles(path, kind, corner)


def geometry_components(
    values: np.ndarray, ctx: dict,
    path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """Decode point coordinates as R, Z and optional toroidal angle.

  MAPC2P files hold Cartesian X/Y/Z; other layouts hold R/Z[/phi]. The caller
  supplies point values, evaluating DG coefficients before calling this decoder.
  """
  cartesian = is_geo_mapc2p(ctx)
  required = 3 if cartesian else 2
  if values.ndim < 2 or values.shape[-1] < required:
    kind = "Cartesian X/Y/Z" if cartesian else "R/Z"
    raise ValueError(
        f"Geometry file '{path}' must contain at least {required} {kind} components."
    )
  if cartesian:
    x, y, z = values[..., 0], values[..., 1], values[..., 2]
    return np.sqrt(x**2 + y**2), z, np.arctan2(y, x)
  r, z = values[..., 0], values[..., 1]
  phi = values[..., 2] if r.ndim == 3 and values.shape[-1] >= 3 else None
  return r, z, phi


def _gauss_nodes(edges: np.ndarray) -> np.ndarray:
  """Physical p1 Gauss-node coordinates within each cell."""
  centers = 0.5 * (edges[:-1] + edges[1:])
  offsets = np.diff(edges) / (2.0 * np.sqrt(3.0))
  return np.ravel(np.column_stack([centers - offsets, centers + offsets]))


def geometry_coordinates(grid: list[np.ndarray],
                         values: np.ndarray,
                         path: str,
                         *,
                         corner: bool = False) -> list[np.ndarray]:
  """Decode interior Gauss points or boundary nodes from a point-file grid."""
  if corner:
    return [
        np.linspace(axis[0], axis[-1], n)
        for axis, n in zip(grid, values.shape[:-1])
    ]
  coords = []
  for dim, axis in enumerate(grid):
    if axis.ndim != 1 or axis.shape[0] != values.shape[dim] + 1 \
        or values.shape[dim] % 2:
      raise ValueError(f"Unrecognized nodal geometry layout in '{path}'.")
    coords.append(_gauss_nodes(axis[::2]))
  return coords
