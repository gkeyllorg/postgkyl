"""Map modal fields onto a physical R-Z slice, with reusable projections."""
from __future__ import annotations

from typing import Annotated
from postgkyl.cli_spec import CliHidden

from dataclasses import dataclass
from functools import partial
import numpy as np

from postgkyl.dg import eval_at_points
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.numerics.resample import resample_grid
from postgkyl.numerics.rz import binormal_fraction
from .geometry import (Geometry, on_data_grid, resolve_geometry,
                       _validate_geometry)
from ._mapping import (
    DEFAULT_NZ_INTERP,
    _component_coefficients,
    _interpolate_component,
    _interpolation_grid,
    validate_mapping_grid,
    _validate_component,
    _validate_modal_data,
    _validate_positive_int,
    mappings_by_geometry,
    mapping_for,
)


@dataclass(frozen=True)
class RzProjection:
  """Precomputed R-Z mapping reusable by fields on one computational grid.

  ``r`` and ``z`` hold the plane's cell corners. A 3-D projection samples
  each radial interpolation center at parallel positions ``zf`` and records
  where the binormal domain lies there: its toroidal angle ``phi_lo`` at the
  lower y edge and its angular ``span`` across y.
  """

  num_dims: int
  r: np.ndarray
  z: np.ndarray
  computational_grid: tuple[np.ndarray, ...]
  zf: np.ndarray | None = None
  phi_lo: np.ndarray | None = None
  span: np.ndarray | None = None


def resolve_rz_projection(first: "GDataState",
                          geo: Geometry,
                          *,
                          z_axis: float = 0.0,
                          nz_interp: int = DEFAULT_NZ_INTERP) -> RzProjection:
  """Build an R-Z projection for ``first``'s grid and ``geo``.

  Only the computational grid and DG metadata are read from ``first``;
  projection construction never evaluates or selects its field values.
  ``z_axis`` is a vertical geometry offset in meters. A geometry read from
  file maps on its own grid, onto which ``first``'s grid is placed.
  """
  _validate_modal_data(first, "map_to_torus_rz", (2, 3))
  nz_interp = _validate_positive_int(nz_interp, "nz_interp")
  _validate_geometry(geo, first.num_dims)
  geo = on_data_grid(geo, first.grid)
  edges, centers = _interpolation_grid(first)
  vert_z = geo.vert_z + float(z_axis)

  if first.num_dims == 2:
    r = resample_grid(geo.major_r, geo.coords, edges)
    z = resample_grid(vert_z, geo.coords, edges)
    return RzProjection(num_dims=2,
                        r=r,
                        z=z,
                        computational_grid=tuple(
                            np.array(axis, copy=True) for axis in edges))

  if geo.phi is None:
    raise ValueError(
        "The geometry has no toroidal-angle component; 3-D toroidal mapping "
        "requires one.")

  xn, yn, zn = edges
  zf_edges = np.linspace(zn[0], zn[-1], nz_interp * (zn.size - 1) + 1)
  zf = 0.5 * (zf_edges[:-1] + zf_edges[1:])
  # The angle is linear in y, so its values on both binormal edges locate
  # every angle along y. Edges past the outer geometry nodes extrapolate.
  phi = np.unwrap(np.unwrap(np.unwrap(geo.phi, axis=2), axis=1), axis=0)
  ends = resample_grid(phi, geo.coords, [centers[0], yn[[0, -1]], zf])
  phi_lo = ends[:, 0, :]
  span = ends[:, 1, :] - phi_lo
  if not np.all(np.isfinite(span)) or np.any(np.isclose(span, 0.0)):
    raise ValueError(
        "Toroidal geometry has a zero or non-finite binormal angular span.")

  gx, _, gz = geo.coords
  r2d = geo.major_r[:, 0, :]
  z2d = vert_z[:, 0, :]
  gz_rz = gz
  if geo.corner is not None:
    corner_coords, corner_r, corner_z = geo.corner
    if len(corner_coords) != 3:
      raise ValueError(
          "Corner geometry must be three-dimensional for 3-D map_to_torus_rz.")
    cx, cz = corner_coords[0], corner_coords[2]
    corner_r = corner_r[:, 0, :]
    corner_z = corner_z[:, 0, :] + float(z_axis)
    r2d = np.concatenate([
        np.interp(gx, cx, corner_r[:, 0])[:, None], r2d,
        np.interp(gx, cx, corner_r[:, -1])[:, None]
    ],
                         axis=1)
    z2d = np.concatenate([
        np.interp(gx, cx, corner_z[:, 0])[:, None], z2d,
        np.interp(gx, cx, corner_z[:, -1])[:, None]
    ],
                         axis=1)
    gz_rz = np.concatenate([[cz[0]], gz, [cz[-1]]])

  r = resample_grid(r2d, [gx, gz_rz], [xn, zf_edges])
  z = resample_grid(z2d, [gx, gz_rz], [xn, zf_edges])
  return RzProjection(num_dims=3,
                      r=r,
                      z=z,
                      zf=zf,
                      phi_lo=phi_lo,
                      span=span,
                      computational_grid=tuple(
                          np.array(axis, copy=True) for axis in edges))


def _validate_projection(data: "GDataState", projection: RzProjection) -> None:
  if projection.num_dims not in (2, 3):
    raise ValueError(
        f"R-Z projection has invalid dimensionality {projection.num_dims}; expected 2 or 3."
    )
  if data.num_dims != projection.num_dims:
    raise ValueError(
        "Incompatible R-Z projection: projection dimensionality does not match the data."
    )
  edges = validate_mapping_grid(data, projection.computational_grid)
  if projection.r.shape != projection.z.shape or projection.r.ndim != 2:
    raise ValueError(
        "Incompatible R-Z projection: R and Z grids must be matching 2-D arrays."
    )

  if projection.num_dims == 2:
    expected = (edges[0].size, edges[1].size)
    if projection.r.shape != expected:
      raise ValueError(
          f"Incompatible R-Z projection: expected grid shape {expected}, "
          f"got {projection.r.shape}.")
    return

  required = (projection.zf, projection.phi_lo, projection.span)
  if any(value is None for value in required):
    raise ValueError(
        "Incompatible R-Z projection: 3-D projection metadata is incomplete.")
  if not np.all(np.isfinite(projection.span)) or np.any(
      np.isclose(projection.span, 0.0)):
    raise ValueError(
        "Incompatible R-Z projection: toroidal angular span must be finite and nonzero."
    )
  samples = (edges[0].size - 1, projection.zf.size)
  if (projection.phi_lo.shape != samples or projection.span.shape != samples
      or projection.r.shape != (samples[0] + 1, samples[1] + 1)):
    raise ValueError(
        "Incompatible R-Z projection: projection and data grid shapes differ.")


def _toroidal_samples(data: "GDataState", comp: int, projection: RzProjection,
                      rows, phi_tor: float | np.ndarray) -> np.ndarray:
  """Evaluate a field component on projection ``rows`` at angles ``phi_tor``.

  Each sample evaluates the DG expansion at its radial center, its parallel
  position, and the binormal position of ``phi_tor`` there. ``phi_tor``
  broadcasts against ``(rows, zf)``.
  """
  xn = projection.computational_grid[0]
  x = 0.5 * (xn[:-1] + xn[1:])[rows, np.newaxis]
  y_lo, y_hi = data.grid[1][0], data.grid[1][-1]
  y = y_lo + (y_hi - y_lo) * binormal_fraction(phi_tor, projection.phi_lo[rows],
                                               projection.span[rows])
  points = np.stack(np.broadcast_arrays(x, y, projection.zf), axis=-1)
  return eval_at_points(_component_coefficients(data, comp),
                        [axis[0] for axis in data.grid],
                        [axis[-1] for axis in data.grid],
                        [axis.size - 1 for axis in data.grid],
                        points,
                        basis_type=data.ctx["basis_type"],
                        poly_order=int(data.ctx["poly_order"]))


def map_to_torus_rz(
    data: "GDataState",
    *,
    projection: Annotated[
        RzProjection | None,
        CliHidden("reuse a projection through the Python API")] = None,
    mapc2p: str | None = None,
    nodes_file: str | None = None,
    z_axis: float = 0.0,
    phi_tor: float = 0.0,
    nz_interp: int = DEFAULT_NZ_INTERP,
    comp: int = 0,
    inplace: bool = False,
    tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Map a modal field component onto a physical R-Z slice.

  Two-dimensional input deforms the interpolated grid. Three-dimensional
  input is sliced at toroidal angle ``phi_tor``: every sample evaluates the
  DG field where the geometry places that angle along y. Angles that the
  simulated binormal domain covers are exact. Elsewhere the periodic domain
  repeats; unless it spans a whole fraction of the torus, its copies meet
  along field lines with a jump. A geometry file's own grid defines the
  mapping, even when written by another run on a different computational
  range: the data's grid is placed on it cell for cell. Geometry is shared
  by all equation systems.

  Args:
    data: Un-interpolated 2-D or 3-D modal DG field.
    projection: Reusable Python projection built with resolve_rz_projection;
      its computational grid must match data. Omit to load geometry.
    mapc2p: Explicit modal geometry path, or '' for the inferred modal file.
    nodes_file: Explicit interior-node geometry path; excludes mapc2p.
      When both paths are omitted, infer geometry from data.file_name,
      preferring geo_int_nodes over geo_int_mapc2p. '*' selects the block.
    z_axis: Vertical offset in meters, added to geometry Z.
    phi_tor: Toroidal angle in radians for a 3-D slice.
    nz_interp: Positive parallel-direction up-sampling factor for 3-D input.
    comp: Zero-based physical field component to map.
    inplace: Mutate and return data instead of creating a dataset.
    tag: Optional result tag.
    label: Optional result label.

  Returns:
    The caller's concrete data class with NumPy point values and a 2-D grid.
    Projection geometry and sampling options cannot accompany a reusable
    projection; set them when building that projection instead.
  """
  _validate_modal_data(data, "map_to_torus_rz", (2, 3))
  _validate_component(data, comp)
  if projection is None:
    nz_interp = _validate_positive_int(nz_interp, "nz_interp")
    geometry = resolve_geometry(data.file_name,
                                mapc2p=mapc2p,
                                nodes_file=nodes_file)
    projection = resolve_rz_projection(data,
                                       geometry,
                                       z_axis=z_axis,
                                       nz_interp=nz_interp)
  elif (mapc2p is not None or nodes_file is not None or z_axis != 0.0
        or nz_interp != DEFAULT_NZ_INTERP):
    raise ValueError(
        "projection cannot be combined with geometry or sampling options; "
        "set them in resolve_rz_projection instead.")
  _validate_projection(data, projection)
  if projection.num_dims == 2:
    out = _interpolate_component(data, comp)
  else:
    out = _toroidal_samples(data, comp, projection, slice(None), float(phi_tor))

  return data._result([projection.r, projection.z],
                      out[..., np.newaxis],
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      grid_type="mapped",
                      mapped_axes={
                          0: 0,
                          1: 0
                      })


def rz_projections(
    datasets,
    *,
    mapc2p: str | None = None,
    nodes_file: str | None = None,
    z_axis: float = 0.0,
    nz_interp: int = DEFAULT_NZ_INTERP) -> dict[str | None, RzProjection]:
  """Build one reusable R-Z projection per block across fields or frames."""
  return mappings_by_geometry(datasets,
                              partial(resolve_rz_projection,
                                      z_axis=z_axis,
                                      nz_interp=nz_interp),
                              mapc2p=mapc2p,
                              nodes_file=nodes_file)


projection_for = mapping_for
