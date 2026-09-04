"""Shared geometry machinery for gyrokinetic data transformations.

This module owns geometry-file discovery and loading plus the grid helpers
used by both the R-Z and flux-surface operations.  It deliberately constructs
the verb-less :class:`~postgkyl.gdatastate.gdatastate.GDataState` and calls the
lower interpolation operation directly; the operation layer never reaches up
through the fluent :class:`postgkyl.gdata.GData` surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from postgkyl.dg import num_basis
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.io import parse_output_name
from postgkyl.numerics import nodal_to_cell_centered_grid

from ..interpolate import interpolate

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


@dataclass(frozen=True)
class Geometry:
  """Physical ``(R, Z[, phi])`` geometry on its own point grid.

  ``corner`` closes the poloidal domain's ``theta = +/-pi`` ends for a 3-D
  R-Z projection.  It is ``None`` when no
  ``'<prefix>-geo_corn_nodes.gkyl'`` file exists alongside the field.
  """

  coords: list[np.ndarray]
  major_r: np.ndarray
  vert_z: np.ndarray
  phi: np.ndarray | None
  corner: tuple[list[np.ndarray], np.ndarray, np.ndarray] | None


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


def _validate_geometry(geometry: Geometry, num_dims: int) -> None:
  """Validate geometry tensor shapes for a data grid of ``num_dims``."""
  if len(geometry.coords) != num_dims:
    raise ValueError(
        f"Geometry has {len(geometry.coords)} dimensions but the data is "
        f"{num_dims}-D.")
  if any(
      np.asarray(axis).ndim != 1 or np.asarray(axis).size < 2
      for axis in geometry.coords):
    raise ValueError(
        "Geometry coordinates must be one-dimensional arrays with at least two points."
    )
  if any(not (np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0))
         for axis in geometry.coords):
    raise ValueError("Geometry coordinate arrays must be strictly monotonic.")
  shape = tuple(np.asarray(axis).size for axis in geometry.coords)
  if geometry.major_r.shape != shape or geometry.vert_z.shape != shape:
    raise ValueError(
        "Geometry coordinate and R/Z array shapes are incompatible: "
        f"expected {shape}, got R{geometry.major_r.shape} and Z{geometry.vert_z.shape}."
    )
  if geometry.phi is not None and geometry.phi.shape != shape:
    raise ValueError(
        f"Geometry toroidal-angle shape {geometry.phi.shape} does not match {shape}."
    )
  if geometry.corner is not None:
    corner_coords, corner_r, corner_z = geometry.corner
    if len(corner_coords) != num_dims:
      raise ValueError(
          f"Corner geometry has {len(corner_coords)} dimensions; expected {num_dims}."
      )
    corner_shape = tuple(np.asarray(axis).size for axis in corner_coords)
    if (any(
        np.asarray(axis).ndim != 1 or np.asarray(axis).size < 2
        for axis in corner_coords) or corner_r.shape != corner_shape
        or corner_z.shape != corner_shape):
      raise ValueError(
          "Corner geometry coordinate and R/Z array shapes are incompatible.")


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


def _validate_positive_int(value: int, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value,
                                               (int, np.integer)) or value <= 0:
    raise ValueError(f"{name} must be a positive integer.")
  return int(value)


def _validate_modal_data(data: GDataState, operation: str,
                         dimensions: tuple[int, ...]) -> None:
  """Enforce the shared raw-DG input contract for GK projections."""
  if data.num_dims not in dimensions:
    expected = " or ".join(f"{dim}-D" for dim in dimensions)
    raise ValueError(
        f"{operation} requires {expected} data; got {data.num_dims}-D.")
  if data.values is None:
    raise ValueError(f"{operation} requires a loaded dataset.")
  if data.ctx.get("interpolated") or data.ctx.get("value_form",
                                                  "modal") != "modal":
    raise ValueError(f"{operation} expects un-interpolated modal DG data.")
  if not data.ctx.get("basis_type"):
    raise ValueError(
        f"{operation} requires 'basis_type' metadata on the input data.")
  poly_order = data.ctx.get("poly_order")
  if isinstance(poly_order, bool) or not isinstance(poly_order, (int, np.integer)) \
      or poly_order < 0:
    raise ValueError(
        f"{operation} requires a nonnegative integer 'poly_order'.")
  if len(data.grid) != data.num_dims or any(
      np.asarray(axis).ndim != 1 or np.asarray(axis).size < 2
      for axis in data.grid):
    raise ValueError(
        f"{operation} requires one one-dimensional edge grid per data dimension."
    )
  if any(not (np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0))
         for axis in data.grid):
    raise ValueError(
        f"{operation} requires strictly monotonic data edge grids.")


def _num_fields(data: GDataState) -> int:
  """Return the number of physical fields stored in raw modal data."""
  basis_count = num_basis(data.num_dims, int(data.ctx["poly_order"]),
                          data.ctx["basis_type"])
  stored = data.values.shape[-1]
  if stored % basis_count:
    raise ValueError(
        f"Data stores {stored} coefficients per cell, which is incompatible "
        f"with a {basis_count}-coefficient basis.")
  return stored // basis_count


def _validate_component(data: GDataState, comp: int) -> int:
  if isinstance(comp, bool) or not isinstance(comp, (int, np.integer)):
    raise ValueError("comp must be an integer component index.")
  comp = int(comp)
  num_fields = _num_fields(data)
  if not 0 <= comp < num_fields:
    raise ValueError(
        f"comp {comp} is out of bounds for data with {num_fields} component(s)."
    )
  return comp


def _interpolation_grid(
    data: GDataState) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Return interpolation edges/centers without evaluating field values."""
  num_interp = int(data.ctx["poly_order"]) + 1
  edges = [
      np.linspace(axis[0], axis[-1],
                  num_interp * (axis.size - 1) + 1) for axis in data.grid
  ]
  centers = nodal_to_cell_centered_grid(
      edges, np.array([axis.size - 1 for axis in edges]))
  return edges, centers


def _interpolate_component(
    data: GDataState,
    comp: int) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
  """Interpolate and return a component already checked by the public API."""
  field = interpolate(data)
  cells = field.values.shape[:-1]
  centers = nodal_to_cell_centered_grid(field.grid, cells)
  return field.grid, centers, field.values[..., comp]


def _resample_grid(values: np.ndarray, src_coords: list[np.ndarray],
                   dst_coords: list[np.ndarray]) -> np.ndarray:
  """Linearly resample ``values`` between tensor-product coordinate grids."""
  mesh = np.meshgrid(*dst_coords, indexing="ij")
  return RegularGridInterpolator(tuple(src_coords),
                                 values,
                                 bounds_error=False,
                                 fill_value=None)(tuple(mesh))


def _same_grid(left: tuple[np.ndarray, ...] | list[np.ndarray],
               right: tuple[np.ndarray, ...] | list[np.ndarray]) -> bool:
  return len(left) == len(right) and all(
      a.shape == b.shape and np.allclose(a, b, rtol=1e-12, atol=1e-14)
      for a, b in zip(left, right))


__all__ = [
    "GKYL_GEOMETRY_ID",
    "Geometry",
    "geometry_prefix",
    "is_geo_mapc2p",
    "per_block_path",
    "resolve_geometry",
]
