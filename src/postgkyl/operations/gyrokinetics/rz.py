"""Project gyrokinetic DG fields onto a physical poloidal R-Z plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

from .geometry import (
    Geometry,
    geometry_prefix,
    _interpolate_component,
    _interpolation_grid,
    _resample_grid,
    _same_grid,
    _validate_component,
    _validate_geometry,
    _validate_modal_data,
    _validate_positive_int,
    per_block_path,
    resolve_geometry,
)

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


@dataclass(frozen=True)
class RzProjection:
  """Precomputed R-Z mapping reusable by fields on one computational grid."""

  num_dims: int
  r: np.ndarray
  z: np.ndarray
  zc: np.ndarray | None = None
  zf: np.ndarray | None = None
  box: float | None = None
  wind: np.ndarray | None = None
  phi0_zf: np.ndarray | None = None
  computational_grid: tuple[np.ndarray, ...] | None = None


def _fft_poloidal_project(values: np.ndarray, zc: np.ndarray, box: float,
    wind: np.ndarray, phi0_zf: np.ndarray, zf: np.ndarray,
    phi_tor: float) -> np.ndarray:
  """FFT twist-and-shift reconstruction at one physical toroidal angle."""
  nx, ny, nz = values.shape
  fk = np.fft.rfft(values, axis=1, norm="forward")
  mode_count = fk.shape[1]

  dz = zc[1] - zc[0]
  z_extended = np.concatenate(([zc[0] - dz / 2], zc, [zc[-1] + dz / 2]))
  fk_extended = np.zeros((nx, mode_count, nz + 2), dtype=complex)
  fk_extended[:, :, 1:-1] = fk
  phase_shift = (2.0 * np.pi / box) * wind
  for mode in range(mode_count):
    phase = np.exp(-1j * mode * phase_shift)
    fk_extended[:, mode, -1] = 0.5 * (
        fk[:, mode, -1] + phase * fk[:, mode, 0])
    fk_extended[:, mode, 0] = 0.5 * (
        fk[:, mode, 0] + np.conj(phase) * fk[:, mode, -1])
  # end

  fk_zf = (PchipInterpolator(z_extended, fk_extended.real, axis=2)(zf)
      + 1j * PchipInterpolator(z_extended, fk_extended.imag, axis=2)(zf))

  fraction = (phi_tor - phi0_zf) / box
  out = np.zeros((nx, len(zf)))
  for mode in range(mode_count):
    weight = 1.0 if (mode == 0 or (ny % 2 == 0 and mode == mode_count - 1)) else 2.0
    out += weight * np.real(
        fk_zf[:, mode, :] * np.exp(-1j * 2.0 * np.pi * mode * fraction))
  # end
  return out
# end


def resolve_rz_projection(first: "GDataState", geo: Geometry, *,
    z_axis: float = 0.0, nz_interp: int = 8) -> RzProjection:
  """Build an R-Z projection for ``first``'s grid and ``geo``.

  Only the computational grid and DG metadata are read from ``first``;
  projection construction never evaluates or selects its field values.
  ``z_axis`` is the magnetic-axis vertical position in meters.
  """
  _validate_modal_data(first, "gk_rz", (2, 3))
  nz_interp = _validate_positive_int(nz_interp, "nz_interp")
  _validate_geometry(geo, first.num_dims)
  edges, centers = _interpolation_grid(first)
  vert_z = geo.vert_z + float(z_axis)

  if first.num_dims == 2:
    r = _resample_grid(geo.major_r, geo.coords, edges)
    z = _resample_grid(vert_z, geo.coords, edges)
    return RzProjection(num_dims=2, r=r, z=z,
        computational_grid=tuple(np.array(axis, copy=True) for axis in edges))
  # end

  if geo.phi is None:
    raise ValueError(
        "The geometry file has no toroidal-angle component; 3-D gk_rz requires one.")
  # end

  xc, yc, zc = centers
  nx, ny, nz = xc.size, yc.size, zc.size
  if ny < 2 or nz < 2:
    raise ValueError("3-D gk_rz requires at least two interpolated y and z points.")
  # end

  phi = np.unwrap(np.unwrap(np.unwrap(geo.phi, axis=2), axis=1), axis=0)
  phi_field = _resample_grid(phi, geo.coords, centers)
  box_estimate = np.mean(np.diff(phi_field[nx // 2, :, nz // 2])) * ny
  if not np.isfinite(box_estimate) or np.isclose(box_estimate, 0.0):
    raise ValueError("Toroidal geometry has a zero or non-finite binormal angular span.")
  # end
  n0 = max(1, int(round(abs(2.0 * np.pi / box_estimate))))
  box = np.sign(box_estimate) * 2.0 * np.pi / n0
  wind = phi_field[:, 0, -1] - phi_field[:, 0, 0]

  xn, _, zn = edges
  zf_edges = np.linspace(zn[0], zn[-1], nz_interp * nz + 1)
  zf = 0.5 * (zf_edges[:-1] + zf_edges[1:])
  phi0_zf = np.array([
      np.interp(zf, zc, phi_field[ix, 0, :]) for ix in range(nx)])

  gx, _, gz = geo.coords
  r2d = geo.major_r[:, 0, :]
  z2d = vert_z[:, 0, :]
  gz_rz = gz
  if geo.corner is not None:
    corner_coords, corner_r, corner_z = geo.corner
    if len(corner_coords) != 3:
      raise ValueError("Corner geometry must be three-dimensional for 3-D gk_rz.")
    # end
    cx, cz = corner_coords[0], corner_coords[2]
    corner_r = corner_r[:, 0, :]
    corner_z = corner_z[:, 0, :] + float(z_axis)
    r2d = np.concatenate([
        np.interp(gx, cx, corner_r[:, 0])[:, None], r2d,
        np.interp(gx, cx, corner_r[:, -1])[:, None]], axis=1)
    z2d = np.concatenate([
        np.interp(gx, cx, corner_z[:, 0])[:, None], z2d,
        np.interp(gx, cx, corner_z[:, -1])[:, None]], axis=1)
    gz_rz = np.concatenate([[cz[0]], gz, [cz[-1]]])
  # end

  r = _resample_grid(r2d, [gx, gz_rz], [xn, zf_edges])
  z = _resample_grid(z2d, [gx, gz_rz], [xn, zf_edges])
  return RzProjection(num_dims=3, r=r, z=z, zc=zc, zf=zf, box=box,
      wind=wind, phi0_zf=phi0_zf,
      computational_grid=tuple(np.array(axis, copy=True) for axis in edges))
# end


def _validate_projection(data: "GDataState", projection: RzProjection) -> None:
  if projection.num_dims not in (2, 3):
    raise ValueError(
        f"R-Z projection has invalid dimensionality {projection.num_dims}; expected 2 or 3.")
  # end
  if data.num_dims != projection.num_dims:
    raise ValueError(
        "Incompatible R-Z projection: projection dimensionality does not match the data.")
  # end
  edges, _ = _interpolation_grid(data)
  if projection.computational_grid is not None \
      and not _same_grid(projection.computational_grid, edges):
    raise ValueError(
        "Incompatible R-Z projection: data computational grid does not match "
        "the grid used to build the projection.")
  # end
  if projection.r.shape != projection.z.shape or projection.r.ndim != 2:
    raise ValueError("Incompatible R-Z projection: R and Z grids must be matching 2-D arrays.")
  # end

  if projection.num_dims == 2:
    expected = (edges[0].size, edges[1].size)
    if projection.r.shape != expected:
      raise ValueError(
          f"Incompatible R-Z projection: expected grid shape {expected}, "
          f"got {projection.r.shape}.")
    # end
    return
  # end

  required = (projection.zc, projection.zf, projection.box,
      projection.wind, projection.phi0_zf)
  if any(value is None for value in required):
    raise ValueError("Incompatible R-Z projection: 3-D projection metadata is incomplete.")
  # end
  if not np.isfinite(projection.box) or np.isclose(projection.box, 0.0):
    raise ValueError("Incompatible R-Z projection: toroidal angular span must be finite and nonzero.")
  # end
  nx, _, nz = (axis.size - 1 for axis in edges)
  if (projection.zc.shape != (nz,) or projection.wind.shape != (nx,)
      or projection.phi0_zf.shape != (nx, projection.zf.size)
      or projection.r.shape != (nx + 1, projection.zf.size + 1)):
    raise ValueError("Incompatible R-Z projection: projection and data grid shapes differ.")
  # end
# end


def map_to_rz(data: "GDataState", projection: RzProjection, *,
    phi_tor: float = 0.0, comp: int = 0, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Map one component of ``data`` with a reusable ``projection``.

  ``phi_tor`` is in radians and is used only for 3-D field-aligned input.
  The source and projection computational grids must be identical.
  """
  _validate_modal_data(data, "gk_rz", (2, 3))
  _validate_component(data, comp)
  _validate_projection(data, projection)
  _, _, values = _interpolate_component(data, comp)

  if projection.num_dims == 2:
    out = values[..., np.newaxis]
  else:
    out = _fft_poloidal_project(values, projection.zc, projection.box,
        projection.wind, projection.phi0_zf, projection.zf,
        float(phi_tor))[..., np.newaxis]
  # end

  return data._result([projection.r, projection.z], out, inplace=inplace,
      tag=tag, label=label, interpolated=True)
# end


def rz_projections(datasets, *, mapc2p: str | None = None,
    nodes_file: str | None = None, z_axis: float = 0.0,
    nz_interp: int = 8) -> dict[str | None, RzProjection]:
  """Build one reusable R-Z projection per block geometry.

  Frames from the same block share a projection; distinct blocks resolve
  their own geometry. A ``'*'`` in an explicit geometry path is replaced by
  the dataset's block index.
  """
  projections: dict[str | None, RzProjection] = {}
  for data in datasets:
    key = geometry_prefix(data.file_name)
    if key in projections:
      continue
    # end
    block = data.ctx.get("block")
    geometry = resolve_geometry(data.file_name,
        mapc2p=per_block_path(mapc2p, block),
        nodes_file=per_block_path(nodes_file, block))
    projections[key] = resolve_rz_projection(data, geometry, z_axis=z_axis,
        nz_interp=nz_interp)
  # end
  return projections
# end


def projection_for(projections: dict[str | None, RzProjection],
    data: "GDataState") -> RzProjection:
  """Return the projection belonging to ``data``'s block."""
  return projections[geometry_prefix(data.file_name)]
# end


def gk_rz(
    data: "GDataState",
    *,
    mapc2p: str | None = None,
    nodes_file: str | None = None,
    z_axis: float = 0.0,
    phi_tor: float = 0.0,
    nz_interp: int = 8,
    comp: int = 0,
    inplace: bool = False,
    tag: str | None = None,
    label: str | None = None,
) -> "GDataState":
  """Interpolate one DG component and project it onto a physical R-Z grid.

  ``data`` must be un-interpolated modal DG data with two computational
  dimensions, or three for a field-aligned reconstruction. Geometry is
  inferred from ``data.file_name``: the pointwise
  ``'<prefix>-geo_int_nodes.gkyl'`` file is preferred, falling back to
  ``'<prefix>-geo_int_mapc2p.gkyl'``. ``nodes_file`` or ``mapc2p`` may
  override that choice, but they are mutually exclusive; ``mapc2p=''``
  forces the inferred modal filename.

  Args:
    data: Un-interpolated 2-D or 3-D modal DG field.
    mapc2p: Optional explicit modal geometry path, or ``''`` for inferred.
    nodes_file: Optional explicit nodal geometry path.
    z_axis: Magnetic-axis vertical position in meters, added to geometry Z.
    phi_tor: Toroidal angle in radians for a 3-D poloidal reconstruction.
    nz_interp: Positive integer z-direction up-sampling factor for 3-D data.
    comp: Zero-based physical field component to map (default first).
    inplace: Replace ``data`` rather than returning a new concrete instance.
    tag: Optional result tag; ``None`` preserves the source tag.
    label: Optional result label; ``None`` preserves the source label.

  Returns:
    The caller's concrete data class, marked ``interpolated=True``. If the
    interpolated input counts are ``(Nx, Nz)``, 2-D grid arrays have shape
    ``(Nx+1, Nz+1)`` and values ``(Nx, Nz, 1)``. For interpolated 3-D counts
    ``(Nx, Ny, Nz)``, grid arrays have shape
    ``(Nx+1, nz_interp*Nz+1)`` and values
    ``(Nx, nz_interp*Nz, 1)``.

  Raises:
    ValueError: For mutually exclusive or missing geometry, input other than
      un-interpolated 2-D/3-D modal data, a missing 3-D toroidal angle,
      invalid ``comp`` or ``nz_interp``, malformed geometry, or a projection
      incompatible with its data grid (when using :func:`map_to_rz`).
  """
  _validate_modal_data(data, "gk_rz", (2, 3))
  _validate_positive_int(nz_interp, "nz_interp")
  _validate_component(data, comp)
  geometry = resolve_geometry(data.file_name, mapc2p=mapc2p,
      nodes_file=nodes_file)
  projection = resolve_rz_projection(data, geometry, z_axis=z_axis,
      nz_interp=nz_interp)
  return map_to_rz(data, projection, phi_tor=phi_tor, comp=comp,
      inplace=inplace, tag=tag, label=label)
# end


__all__ = [
    "Geometry", "RzProjection", "geometry_prefix", "gk_rz", "map_to_rz",
    "per_block_path", "projection_for", "resolve_geometry",
    "resolve_rz_projection", "rz_projections",
]
