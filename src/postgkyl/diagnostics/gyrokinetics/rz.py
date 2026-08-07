"""Gyrokinetic R-Z mapping: interpolate DG data and project it onto the
physical poloidal (R, Z) plane.

Ported from ``src_bak/postgkyl/commands/gk_rz.py`` (PR #214): 2-D data maps
onto (R, Z) directly from the simulation's geometry file; 3-D (field-aligned)
data is reconstructed on the poloidal plane at a chosen toroidal angle via an
FFT-based twist-and-shift binormal projection. Geometry resolution
(:func:`resolve_geometry`) and the projection grid it builds
(:func:`resolve_rz_projection`) are shared with
:mod:`~postgkyl.diagnostics.gyrokinetics.fluxsurf`, which extracts a
theta-phi flux surface from the same field-aligned data instead of a
poloidal cross section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from postgkyl import numerics
from postgkyl.gdata import GData

from . import nodes, utils


@dataclass(frozen=True)
class Geometry:
  """Physical (R, Z[, phi]) geometry on its own cell centers.

  ``corner`` closes the poloidal domain's theta = +/-pi ends for the 3-D
  R-Z projection (see :func:`resolve_rz_projection`); ``None`` when no
  ``'<prefix>-geo_corn_nodes.gkyl'`` file was found alongside the source
  geometry.
  """

  coords: list[np.ndarray]
  major_r: np.ndarray
  vert_z: np.ndarray
  phi: np.ndarray | None
  corner: tuple[list[np.ndarray], np.ndarray, np.ndarray] | None


@dataclass(frozen=True)
class RzProjection:
  """Precomputed R-Z mapping, shared across every dataset that shares one
  computational grid and geometry (see :func:`resolve_rz_projection`).

  ``zc``/``zf``/``box``/``wind``/``phi0_zf`` are populated only when
  ``num_dims == 3`` (the field-aligned binormal-projection case);
  :func:`map_to_rz` maps 2-D data onto ``(r, z)`` directly.
  """

  num_dims: int
  r: np.ndarray
  z: np.ndarray
  zc: np.ndarray | None = None
  zf: np.ndarray | None = None
  box: float | None = None
  wind: np.ndarray | None = None
  phi0_zf: np.ndarray | None = None


def _file_prefix(file_name: str | None) -> str | None:
  """Simulation name prefix a dataset's source file belongs to (the part
  before the last ``'-'``), used to default-locate its geometry files."""
  if not file_name:
    return None
  # end
  return os.path.splitext(file_name)[0].rsplit("-", 1)[0]
# end


def _gauss_nodes(edges: np.ndarray) -> np.ndarray:
  """Physical coordinates of the p1 nodal points (cell center +/-
  h/(2*sqrt(3))) of a 1-D edge grid -- where a nodal geometry file's values
  live."""
  c = 0.5 * (edges[:-1] + edges[1:])
  off = np.diff(edges) / (2.0 * np.sqrt(3.0))
  return np.ravel(np.column_stack([c - off, c + off]))
# end


def _read_mapc2p_geometry(path: str):
  """Interpolate a modal mapc2p (or geo R, Z, phi) file to physical R, Z,
  phi on its cell centers."""
  gdat = GData(path).interpolate()
  cells = gdat.values.shape[:-1]
  grid = numerics.nodal_to_cell_centered_grid(gdat.grid, cells)
  values = gdat.values
  if nodes.is_geo_mapc2p(gdat.ctx):
    x, y, z = values[..., 0], values[..., 1], values[..., 2]
    return grid, np.sqrt(x**2 + y**2), z, np.arctan2(y, x)
  # end
  r, z = values[..., 0], values[..., 1]
  phi = values[..., 2] if r.ndim == 3 else None
  return grid, r, z, phi
# end


def _read_nodes_geometry(path: str):
  """Read a pointwise nodal geometry file to physical R, Z, phi.

  The stored grid is 2x-refined (one edge per value plus one); the values
  themselves sit at the two p1 nodes of each cell, whose edges are every
  other stored grid point.
  """
  grid, values, gdat = utils.read_gfile(path)
  coords = []
  for dim, g in enumerate(grid):
    if g.shape[0] != values.shape[dim] + 1 or values.shape[dim] % 2:
      raise ValueError(f"Unrecognized nodal geometry layout in {path}")
    # end
    coords.append(_gauss_nodes(g[::2]))
  # end

  if nodes.is_geo_mapc2p(gdat.ctx):
    x, y, z = values[..., 0], values[..., 1], values[..., 2]
    return coords, np.sqrt(x**2 + y**2), z, np.arctan2(y, x)
  # end
  r, z = values[..., 0], values[..., 1]
  phi = values[..., 2] if r.ndim == 3 else None
  return coords, r, z, phi
# end


def _read_corner_rz(path: str):
  """R and Z from a pointwise corner geometry file
  (``'-geo_corn_nodes.gkyl'``).

  The corner file stores one value per cell corner, endpoints included, so
  unlike the interior nodes/mapc2p files it covers the domain boundary (in
  particular z = +/-pi, where the poloidal cross section closes).
  """
  grid, values, gdat = utils.read_gfile(path)
  coords = [np.linspace(g[0], g[-1], n) for g, n in zip(grid, values.shape[:-1])]
  if nodes.is_geo_mapc2p(gdat.ctx):
    x, y, z = values[..., 0], values[..., 1], values[..., 2]
    return coords, np.sqrt(x**2 + y**2), z
  # end
  return coords, values[..., 0], values[..., 1]
# end


def resolve_geometry(file_name: str, *, mapc2p: str | None = None,
    nodes_file: str | None = None) -> Geometry:
  """Resolve and load the R-Z(-phi) geometry for the simulation
  ``file_name`` belongs to.

  The pointwise ``'<prefix>-geo_int_nodes.gkyl'`` is preferred over the
  modal ``'<prefix>-geo_int_mapc2p.gkyl'`` (exact node values; the modal
  representation loses amplitude where the toroidal winding is
  under-resolved), unless overridden by ``nodes_file``/``mapc2p``. A
  ``'<prefix>-geo_corn_nodes.gkyl'`` corner file is picked up automatically
  when present (used by :func:`resolve_rz_projection`'s 3-D case).

  Args:
    file_name: Source file of a dataset in the simulation whose geometry is
      wanted; only its prefix (the part before the last ``'-'``) is used.
    mapc2p: Explicit modal mapc2p geometry file; ``''`` requests the
      default ``'<prefix>-geo_int_mapc2p.gkyl'`` lookup. Mutually exclusive
      with ``nodes_file``.
    nodes_file: Explicit pointwise nodes geometry file. Mutually exclusive
      with ``mapc2p``.

  Raises:
    ValueError: if both ``mapc2p`` and ``nodes_file`` are given, or no
      geometry file can be found.
  """
  if mapc2p is not None and nodes_file is not None:
    raise ValueError("Pass either mapc2p= or nodes_file=, not both.")
  # end

  prefix = _file_prefix(file_name)
  if nodes_file is not None:
    path, kind = nodes_file, "nodes"
  elif mapc2p is not None:
    path = mapc2p or (f"{prefix}-geo_int_mapc2p.gkyl" if prefix else None)
    kind = "mapc2p"
  elif prefix is not None:
    path, kind = f"{prefix}-geo_int_nodes.gkyl", "nodes"
    if not os.path.exists(path):
      path, kind = f"{prefix}-geo_int_mapc2p.gkyl", "mapc2p"
    # end
  else:
    path, kind = None, None
  # end

  if path is None or not os.path.exists(path):
    raise ValueError(
        "Could not find a geometry file; pass nodes_file= or mapc2p= explicitly.")
  # end

  coords, major_r, vert_z, phi = (
      _read_nodes_geometry(path) if kind == "nodes" else _read_mapc2p_geometry(path))

  corner = None
  if prefix is not None:
    corner_path = f"{prefix}-geo_corn_nodes.gkyl"
    if os.path.exists(corner_path):
      corner = _read_corner_rz(corner_path)
    # end
  # end

  return Geometry(coords=coords, major_r=major_r, vert_z=vert_z, phi=phi, corner=corner)
# end


def _fft_poloidal_project(values: np.ndarray, zc: np.ndarray, box: float,
    wind: np.ndarray, phi0_zf: np.ndarray, zf: np.ndarray,
    phi_tor: float) -> np.ndarray:
  """Project a 3-D field-aligned dataset onto the poloidal plane at
  ``phi = phi_tor`` via an FFT-based twist-and-shift binormal projection.
  """
  Nx, Ny, Nz = values.shape
  fk = np.fft.rfft(values, axis=1, norm="forward")  # (Nx, K, Nz)
  K = fk.shape[1]

  # Twist-and-shift reconnection: add a ghost z-cell at each domain edge
  # (+/-pi) whose value is the opposite end phase-shifted by
  # exp(i k n0 wind).
  dz = zc[1] - zc[0]
  z_ex = np.concatenate(([zc[0] - dz / 2], zc, [zc[-1] + dz / 2]))
  fk_ex = np.zeros((Nx, K, Nz + 2), dtype=complex)
  fk_ex[:, :, 1:-1] = fk
  psh = (2.0 * np.pi / box) * wind  # per-mode phase = n0 * wind(x)
  for k in range(K):
    ph = np.exp(-1j * k * psh)
    fk_ex[:, k, -1] = 0.5 * (fk[:, k, -1] + ph * fk[:, k, 0])
    fk_ex[:, k, 0] = 0.5 * (fk[:, k, 0] + np.conj(ph) * fk[:, k, -1])
  # end

  # Up-sample along z (interpolate real and imaginary parts separately).
  fk_zf = (PchipInterpolator(z_ex, fk_ex.real, axis=2)(zf)
      + 1j * PchipInterpolator(z_ex, fk_ex.imag, axis=2)(zf))

  # Phase-sum: reconstruct the field where the physical toroidal angle ==
  # phi_tor.
  frac = (phi_tor - phi0_zf) / box
  out = np.zeros((Nx, len(zf)))
  for k in range(K):
    # rfft: modes 0 < k < Nyquist represent both +/-k; do not double k=0
    # or Nyquist.
    weight = 1.0 if (k == 0 or (Ny % 2 == 0 and k == K - 1)) else 2.0
    out += weight * np.real(fk_zf[:, k, :] * np.exp(-1j * 2.0 * np.pi * k * frac))
  # end
  return out
# end


def resolve_rz_projection(first: GData, geo: Geometry, *,
    z_axis: float = 0.0, nz_interp: int = 8) -> RzProjection:
  """Precompute the R-Z mapping for every dataset that shares ``first``'s
  computational grid and ``geo``'s geometry.

  Args:
    first: A representative (not yet interpolated) dataset; only its
      computational grid is used.
    geo: The simulation's geometry (see :func:`resolve_geometry`).
    z_axis: Vertical position of the magnetic axis (m), added to the
      geometry Z. mapc2p files store Z relative to the axis; pass Z_axis
      from the simulation input file to project in machine coordinates.
    nz_interp: 3-D only: parallel (z) up-sampling factor for smooth
      projected surfaces.

  Raises:
    ValueError: if ``first`` is 3-D and ``geo`` has no toroidal-angle
      component.
  """
  edges, centers, _ = utils.interpolated_grid_values(first)
  vert_z = geo.vert_z + z_axis

  if first.num_dims == 2:
    # Evaluate R, Z at the field's cell corners (its edge grid) so
    # pcolormesh gets explicit cell edges, not non-monotonic curvilinear
    # cell centers.
    r = utils.resample_grid(geo.major_r, geo.coords, edges)
    z = utils.resample_grid(vert_z, geo.coords, edges)
    return RzProjection(num_dims=2, r=r, z=z)
  # end

  if geo.phi is None:
    raise ValueError(
        "The geometry file has no toroidal-angle component; cannot project "
        "3-D data.")
  # end

  xc, yc, zc = centers
  Nx, Ny, Nz = xc.size, yc.size, zc.size

  phi = np.unwrap(np.unwrap(np.unwrap(geo.phi, axis=2), axis=1), axis=0)
  phi_field = utils.resample_grid(phi, geo.coords, centers)
  # The binormal domain spans one 1/n0 toroidal sector (wedge).
  box = np.mean(np.diff(phi_field[Nx // 2, :, Nz // 2])) * Ny
  n0 = max(1, int(round(abs(2.0 * np.pi / box))))
  box = np.sign(box) * 2.0 * np.pi / n0
  wind = phi_field[:, 0, -1] - phi_field[:, 0, 0]

  xn, _, zn = edges
  zf_edges = np.linspace(zn[0], zn[-1], nz_interp * Nz + 1)
  zf = 0.5 * (zf_edges[:-1] + zf_edges[1:])
  # Toroidal angle at the binormal origin along (x, zf), used by the
  # phase-sum.
  phi0_zf = np.array([np.interp(zf, zc, phi_field[ix, 0, :]) for ix in range(Nx)])

  # R, Z (y-independent) on the (x, z) plane.
  gx, gy, gz = geo.coords
  R2d, Z2d, gz_rz = geo.major_r[:, 0, :], vert_z[:, 0, :], gz
  if geo.corner is not None:
    corn_coords, corn_r, corn_z = geo.corner
    cx, cz = corn_coords[0], corn_coords[2]
    corn_r, corn_z = corn_r[:, 0, :], corn_z[:, 0, :] + z_axis
    R2d = np.concatenate([
        np.interp(gx, cx, corn_r[:, 0])[:, None], R2d,
        np.interp(gx, cx, corn_r[:, -1])[:, None]], axis=1)
    Z2d = np.concatenate([
        np.interp(gx, cx, corn_z[:, 0])[:, None], Z2d,
        np.interp(gx, cx, corn_z[:, -1])[:, None]], axis=1)
    gz_rz = np.concatenate([[cz[0]], gz, [cz[-1]]])
  # end

  r = utils.resample_grid(R2d, [gx, gz_rz], [xn, zf_edges])
  z = utils.resample_grid(Z2d, [gx, gz_rz], [xn, zf_edges])

  return RzProjection(num_dims=3, r=r, z=z, zc=zc, zf=zf, box=box, wind=wind,
      phi0_zf=phi0_zf)
# end


def map_to_rz(data: GData, projection: RzProjection, *, phi_tor: float = 0.0,
    inplace: bool = False, tag: str | None = None, label: str | None = None) -> GData:
  """Interpolate ``data``'s DG coefficients and map them onto the R-Z plane
  using a precomputed ``projection`` (see :func:`resolve_rz_projection`).

  2-D data maps onto ``projection``'s (r, z) directly. 3-D (field-aligned)
  data is reconstructed on the poloidal plane at toroidal angle ``phi_tor``
  via the FFT-based binormal projection ``projection`` was built for.
  """
  _, _, values = utils.interpolated_grid_values(data)

  if projection.num_dims == 2:
    out = values[..., np.newaxis]
  else:
    out = _fft_poloidal_project(values, projection.zc, projection.box,
        projection.wind, projection.phi0_zf, projection.zf, phi_tor)[..., np.newaxis]
  # end

  return data._result([projection.r, projection.z], out, inplace=inplace, tag=tag,
      label=label, interpolated=True)
# end
