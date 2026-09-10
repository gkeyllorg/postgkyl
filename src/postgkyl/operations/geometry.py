"""In-memory coordinate geometry and shared field-mapping contracts.

These records and helpers do not discover or load auxiliary datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from postgkyl.dg import num_basis
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.numerics import nodal_to_cell_centered_grid
from .interpolate import interpolate


@dataclass(frozen=True)
class Geometry:
  """Physical ``(R, Z[, phi])`` geometry on its own point grid.

  ``corner`` closes the poloidal domain's ``theta = +/-pi`` ends for a 3-D
  R-Z projection. It is optional boundary geometry on its own point grid.
  """

  coords: list[np.ndarray]
  major_r: np.ndarray
  vert_z: np.ndarray
  phi: np.ndarray | None
  corner: tuple[list[np.ndarray], np.ndarray, np.ndarray] | None


@dataclass(frozen=True)
class RzProjection:
  """Precomputed R-Z mapping reusable by fields on one computational grid."""

  num_dims: int
  r: np.ndarray
  z: np.ndarray
  computational_grid: tuple[np.ndarray, ...]
  zc: np.ndarray | None = None
  zf: np.ndarray | None = None
  box: float | None = None
  wind: np.ndarray | None = None
  phi0_zf: np.ndarray | None = None


@dataclass(frozen=True)
class FluxSurfaceGrid:
  """Precomputed toroidal sampling grid for one radial flux surface."""

  x_idx: int
  zc: np.ndarray
  zf: np.ndarray
  phi_tor_list: np.ndarray
  phi_2d: np.ndarray
  computational_grid: tuple[np.ndarray, ...]


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


def _validate_positive_int(value: int, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value,
                                               (int, np.integer)) or value <= 0:
    raise ValueError(f"{name} must be a positive integer.")
  return int(value)


def _validate_modal_data(data: GDataState, operation: str,
                         dimensions: tuple[int, ...]) -> None:
  """Enforce the shared raw-DG input contract for coordinate projections."""
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


def _same_grid(left: tuple[np.ndarray, ...] | list[np.ndarray],
               right: tuple[np.ndarray, ...] | list[np.ndarray]) -> bool:
  return len(left) == len(right) and all(
      a.shape == b.shape and np.allclose(a, b, rtol=1e-12, atol=1e-14)
      for a, b in zip(left, right))


def validate_mapping_grid(data: GDataState,
                          computational_grid: tuple[np.ndarray, ...]) -> None:
  """Require the modal field grid used to construct a reusable mapping."""
  _validate_modal_data(data, "coordinate mapping", (len(computational_grid), ))
  edges, _ = _interpolation_grid(data)
  if not _same_grid(computational_grid, edges):
    raise ValueError(
        "Incompatible mapping: data computational grid does not match "
        "the grid used to build the projection.")


__all__ = [
    "Geometry", "RzProjection", "FluxSurfaceGrid", "validate_mapping_grid"
]
