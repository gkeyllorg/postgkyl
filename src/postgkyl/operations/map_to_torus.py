"""Map modal fields onto the boundary of a toroidal wedge, with reusable projections."""
from __future__ import annotations

from typing import Annotated
from postgkyl.cli_spec import CliHidden

from dataclasses import dataclass
from functools import partial
import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.numerics import nodal_to_cell_centered_grid
from postgkyl.numerics.rz import wedge_surface
from .geometry import Geometry, resolve_geometry
from ._mapping import (
    DEFAULT_NZ_INTERP,
    _radial_index,
    _validate_component,
    _validate_modal_data,
    _validate_positive_int,
    mappings_by_geometry,
    mapping_for,
)
from .map_to_rz import (RzProjection, resolve_rz_projection, _toroidal_samples,
                        _validate_projection)

_DEFAULT_X_IDX = (2, -3)
_DEFAULT_PHI_TOR = (0.5 * np.pi, 2.0 * np.pi)
_DEFAULT_NPHI = 64


@dataclass(frozen=True)
class TorusProjection:
  """Precomputed toroidal-wedge boundary reusable by fields on one grid.

  The wedge spans interpolated radial ``rows`` and toroidal angles
  ``phi_tor``. Fields on its faces are sampled through the ``rz``
  projection; ``points`` holds the Cartesian X, Y, Z of its closed boundary.
  """

  rz: RzProjection
  rows: np.ndarray
  phi_tor: np.ndarray
  points: tuple[np.ndarray, np.ndarray, np.ndarray]

  @property
  def computational_grid(self) -> tuple[np.ndarray, ...]:
    return self.rz.computational_grid


def _radial_rows(x_idx: tuple[int, int], nx: int) -> np.ndarray:
  """Parse inner and outer radial indices into the rows between them."""
  if len(x_idx) != 2:
    raise ValueError("x_idx must be two radial indices.")
  inner, outer = (_radial_index(index, nx) for index in x_idx)
  if not inner < outer:
    raise ValueError(
        f"x_idx {tuple(x_idx)} must select an inner index below the outer one.")
  return np.arange(inner, outer + 1)


def resolve_torus_projection(
    first: "GDataState",
    geo: Geometry,
    *,
    z_axis: float = 0.0,
    x_idx: tuple[int, int] = _DEFAULT_X_IDX,
    phi_tor: tuple[float, float] = _DEFAULT_PHI_TOR,
    nphi: int = _DEFAULT_NPHI,
    nz_interp: int = DEFAULT_NZ_INTERP) -> TorusProjection:
  """Build a toroidal-wedge boundary for ``first``'s grid and ``geo``.

  Only the computational grid and DG metadata are read from ``first``.
  Surface points sit where ``map_to_rz`` places its values: at the centers
  of its R-Z cells, rotated to each toroidal angle.
  """
  _validate_modal_data(first, "map_to_torus", (3, ))
  nphi = _validate_positive_int(nphi, "nphi")
  if nphi < 2:
    raise ValueError("nphi must be at least 2 to span the toroidal wedge.")
  first_phi, last_phi = (float(angle) for angle in phi_tor)
  if not first_phi < last_phi <= first_phi + 2.0 * np.pi:
    raise ValueError(
        "phi_tor must increase from its first to its last angle by at most "
        "2*pi.")
  rz = resolve_rz_projection(first, geo, z_axis=z_axis, nz_interp=nz_interp)
  rows = _radial_rows(x_idx, rz.computational_grid[0].size - 1)
  angles = np.linspace(first_phi, last_phi, nphi)
  major_r, vert_z = nodal_to_cell_centered_grid([rz.r, rz.z],
                                                np.array(rz.r.shape) - 1)

  def cartesian(rows, angles):
    r = major_r[rows]
    return np.stack(np.broadcast_arrays(r * np.cos(angles), r * np.sin(angles),
                                        vert_z[rows]),
                    axis=-1)

  points = wedge_surface(cartesian, rows, angles)
  return TorusProjection(rz=rz,
                         rows=rows,
                         phi_tor=angles,
                         points=tuple(np.moveaxis(points, -1, 0)))


def map_to_torus(
    data: "GDataState",
    *,
    projection: Annotated[
        TorusProjection | None,
        CliHidden("reuse a projection through the Python API")] = None,
    mapc2p: str | None = None,
    nodes_file: str | None = None,
    z_axis: float = 0.0,
    x_idx: tuple[int, int] = _DEFAULT_X_IDX,
    phi_tor: tuple[float, float] = _DEFAULT_PHI_TOR,
    nphi: int = _DEFAULT_NPHI,
    nz_interp: int = DEFAULT_NZ_INTERP,
    comp: int = 0,
    inplace: bool = False,
    tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Map a modal field component onto the boundary of a toroidal wedge.

  The wedge lies between two flux surfaces and two poloidal planes. Its
  boundary is one closed surface in Cartesian X, Y, Z: the poloidal cap at the
  first angle, the outer flux surface, the cap at the last angle, and the
  inner flux surface. Every face samples the field as map_to_rz does, so the
  caps are its slices and faces agree where they meet. Render it with
  pyvista's surface and no_normalize options.

  Args:
    data: Un-interpolated 3-D field-aligned modal DG field.
    projection: Reusable Python projection built with
      resolve_torus_projection; its computational grid must match data.
      Omit to load geometry.
    mapc2p: Explicit modal geometry path, or '' for the inferred modal file.
    nodes_file: Explicit interior-node geometry path; excludes mapc2p.
      When both paths are omitted, infer geometry from data.file_name,
      preferring geo_int_nodes over geo_int_mapc2p. '*' selects the block.
    z_axis: Vertical offset in meters, added to geometry Z.
    x_idx: Inner and outer flux-surface indices on the interpolated radial
      grid; negative indices count back from the outer edge.
    phi_tor: First and last toroidal angles in radians, at most 2*pi apart.
    nphi: Number of toroidal samples from the first to the last angle.
    nz_interp: Positive parallel-direction up-sampling factor.
    comp: Zero-based physical field component to map.
    inplace: Mutate and return data instead of creating a dataset.
    tag: Optional result tag.
    label: Optional result label.

  Returns:
    The caller's concrete data class with NumPy point values on a mapped
    ``(s, z, 1)`` grid: ``s`` follows the wedge boundary, ``z`` the field
    line, and the size-1 axis is the surface's zero thickness. Projection
    geometry and sampling options cannot accompany a reusable projection;
    set them when building that projection instead.
  """
  _validate_modal_data(data, "map_to_torus", (3, ))
  _validate_component(data, comp)
  if projection is None:
    geometry = resolve_geometry(data.file_name,
                                mapc2p=mapc2p,
                                nodes_file=nodes_file)
    projection = resolve_torus_projection(data,
                                          geometry,
                                          z_axis=z_axis,
                                          x_idx=x_idx,
                                          phi_tor=phi_tor,
                                          nphi=nphi,
                                          nz_interp=nz_interp)
  elif (mapc2p is not None or nodes_file is not None or z_axis != 0.0
        or tuple(x_idx) != _DEFAULT_X_IDX or tuple(phi_tor) != _DEFAULT_PHI_TOR
        or nphi != _DEFAULT_NPHI or nz_interp != DEFAULT_NZ_INTERP):
    raise ValueError(
        "projection cannot be combined with geometry or sampling options; "
        "set them in resolve_torus_projection instead.")
  rz = projection.rz
  _validate_projection(data, rz)
  if (projection.rows.min() < 0 or projection.rows.max() >= rz.phi_lo.shape[0]):
    raise ValueError(
        "Incompatible torus projection: radial rows exceed the data grid.")

  def field(rows, angles):
    return _toroidal_samples(data, comp, rz, rows, angles)

  surface = wedge_surface(field, projection.rows, projection.phi_tor)
  if any(axis.shape != surface.shape for axis in projection.points):
    raise ValueError(
        "Incompatible torus projection: surface points and samples differ.")
  return data._result([axis[..., np.newaxis] for axis in projection.points],
                      surface[..., np.newaxis, np.newaxis],
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      grid_type="mapped",
                      mapped_axes={
                          0: 0,
                          1: 0,
                          2: 0
                      })


def torus_projections(
    datasets,
    *,
    mapc2p: str | None = None,
    nodes_file: str | None = None,
    z_axis: float = 0.0,
    x_idx: tuple[int, int] = _DEFAULT_X_IDX,
    phi_tor: tuple[float, float] = _DEFAULT_PHI_TOR,
    nphi: int = _DEFAULT_NPHI,
    nz_interp: int = DEFAULT_NZ_INTERP) -> dict[str | None, TorusProjection]:
  """Build one reusable torus projection per block across fields or frames."""
  return mappings_by_geometry(datasets,
                              partial(resolve_torus_projection,
                                      z_axis=z_axis,
                                      x_idx=x_idx,
                                      phi_tor=phi_tor,
                                      nphi=nphi,
                                      nz_interp=nz_interp),
                              mapc2p=mapc2p,
                              nodes_file=nodes_file)


projection_for = mapping_for
