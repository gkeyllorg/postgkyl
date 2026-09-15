"""In-memory cylindrical geometry and its assembly from Gkeyll output.

I/O owns filenames and on-disk layouts. This module evaluates modal geometry
and assembles point coordinates for equation-agnostic mapping operations.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.io.geometry import (
    resolve_geometry_files,
    geometry_components,
    geometry_coordinates,
)
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


def _pointwise_file(
    path: str) -> tuple[list[np.ndarray], np.ndarray, GDataState]:
  """Load a geometry point file without evaluating its stored coordinates."""
  data = GDataState(path)
  return [np.squeeze(axis) for axis in data.grid], np.squeeze(data.values), data


def _read_nodes_geometry(path: str):
  grid, values, data = _pointwise_file(path)
  coords = geometry_coordinates(grid, values, path)
  return coords, *geometry_components(values, data.ctx, path)


def _read_mapc2p_geometry(path: str):
  field = interpolate(GDataState(path))
  coords = nodal_to_cell_centered_grid(field.grid, field.values.shape[:-1])
  return coords, *geometry_components(field.values, field.ctx, path)


def _read_corner_rz(path: str):
  grid, values, data = _pointwise_file(path)
  coords = geometry_coordinates(grid, values, path, corner=True)
  major_r, vert_z, _ = geometry_components(values, data.ctx, path)
  return coords, major_r, vert_z


def resolve_geometry(file_name: str | None,
                     *,
                     mapc2p: str | None = None,
                     nodes_file: str | None = None) -> Geometry:
  """Load cylindrical geometry for any Gkeyll simulation output.

  Prefer ``<prefix>-geo_int_nodes.gkyl``, falling back to the modal
  ``<prefix>-geo_int_mapc2p.gkyl``. Optional ``geo_corn_nodes`` close the
  poloidal boundary. Filenames and coordinate layouts are owned by I/O.
  ``mapc2p`` and ``nodes_file`` are mutually exclusive overrides; an empty
  ``mapc2p`` requests the inferred modal filename. '*' selects the source block.
  """
  files = resolve_geometry_files(file_name,
                                 mapc2p=mapc2p,
                                 nodes_file=nodes_file)
  coords, major_r, vert_z, phi = (_read_nodes_geometry(files.interior)
                                  if files.kind == "nodes" else
                                  _read_mapc2p_geometry(files.interior))
  corner = _read_corner_rz(files.corner) if files.corner is not None else None
  geometry = Geometry(coords, major_r, vert_z, phi, corner)
  _validate_geometry(geometry, len(coords))
  return geometry


__all__ = ["Geometry", "resolve_geometry"]
