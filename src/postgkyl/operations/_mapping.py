"""Shared modal-field contracts and reuse across geometry mappings."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, TypeVar
import numpy as np

from postgkyl.dg import num_basis
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.io.geometry import geometry_prefix
from postgkyl.numerics import nodal_to_cell_centered_grid
from .geometry import Geometry, resolve_geometry
from .interpolate import interpolate

DEFAULT_NZ_INTERP = 8


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


def _interpolate_component(data: GDataState, comp: int) -> np.ndarray:
  """Interpolate and return a component already checked by the public API."""
  return interpolate(data).values[..., comp]


def _same_grid(left: tuple[np.ndarray, ...] | list[np.ndarray],
               right: tuple[np.ndarray, ...] | list[np.ndarray]) -> bool:
  return len(left) == len(right) and all(
      a.shape == b.shape and np.allclose(a, b, rtol=1e-12, atol=1e-14)
      for a, b in zip(left, right))


def validate_mapping_grid(
    data: GDataState, computational_grid: tuple[np.ndarray,
                                                ...]) -> list[np.ndarray]:
  """Require a compatible modal field and return its interpolation edges."""
  _validate_modal_data(data, "coordinate mapping", (len(computational_grid), ))
  edges, _ = _interpolation_grid(data)
  if not _same_grid(computational_grid, edges):
    raise ValueError(
        "Incompatible mapping: data computational grid does not match "
        "the grid used to build the projection.")
  return edges


class _GridMapping(Protocol):
  computational_grid: tuple[np.ndarray, ...]


_Mapping = TypeVar("_Mapping", bound=_GridMapping)


def mappings_by_geometry(
    datasets: Iterable[GDataState],
    build: Callable[[GDataState, Geometry], _Mapping],
    *,
    mapc2p: str | None = None,
    nodes_file: str | None = None,
) -> dict[str | None, _Mapping]:
  """Build once per block, checking that later fields use the same grid."""
  mappings: dict[str | None, _Mapping] = {}
  for data in datasets:
    key = geometry_prefix(data.file_name)
    if key in mappings:
      validate_mapping_grid(data, mappings[key].computational_grid)
      continue
    geometry = resolve_geometry(data.file_name,
                                mapc2p=mapc2p,
                                nodes_file=nodes_file)
    mappings[key] = build(data, geometry)
  return mappings


def mapping_for(mappings: dict[str | None, _Mapping],
                data: GDataState) -> _Mapping:
  """Return a previously built mapping for this dataset's block."""
  return mappings[geometry_prefix(data.file_name)]
