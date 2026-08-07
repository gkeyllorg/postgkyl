"""Gyrokinetic flux-surface extraction: a theta-phi surface at fixed radius.

Ported from ``src_bak/postgkyl/commands/gk_fluxsurf.py`` (PR #214): extracts
a 2-D (theta, phi) surface from 3-D field-aligned data at a given radial
(x) index, by scanning a set of toroidal angles and, for each, interpolating
the field along the binormal direction of the geometry it was computed
from. Geometry resolution is shared with
:mod:`~postgkyl.diagnostics.gyrokinetics.rz`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from postgkyl.gdata import GData

from . import utils
from .rz import Geometry, per_block_path, geometry_prefix, resolve_geometry


@dataclass(frozen=True)
class FluxSurfaceGrid:
  """Precomputed toroidal-angle sampling grid for extracting a flux surface
  at radial index ``x_idx``, shared across every dataset that shares
  ``first``'s computational grid and geometry (see
  :func:`resolve_flux_surface_grid`).
  """

  x_idx: int
  zc: np.ndarray
  zf: np.ndarray
  phi_tor_list: np.ndarray
  phi_2d: np.ndarray  # geometry's toroidal angle at (y, zf), shape (Ny, len(zf))


def resolve_flux_surface_grid(first: GData, geo: Geometry, *, x_idx: int = 0,
    nphi: int = 128, nz_interp: int = 8) -> FluxSurfaceGrid:
  """Precompute the toroidal-angle sampling grid for extracting the flux
  surface at radial index ``x_idx``, shared across every dataset that
  shares ``first``'s computational grid and ``geo``'s geometry.

  Args:
    first: A representative (not yet interpolated) 3-D dataset; only its
      computational grid is used.
    geo: The simulation's geometry (see
      :func:`postgkyl.diagnostics.gyrokinetics.rz.resolve_geometry`).
    x_idx: Radial (x) cell index of the flux surface.
    nphi: Number of toroidal-angle (phi) slices to scan.
    nz_interp: Parallel (z) up-sampling factor used to smooth the
      projected surface.

  Raises:
    ValueError: if ``first`` is not 3-D, ``geo`` has no toroidal-angle
      component, or ``x_idx`` is out of bounds.
  """
  if first.num_dims != 3:
    raise ValueError("gk_fluxsurf requires 3-D data to scan over toroidal angle (phi).")
  # end
  if geo.phi is None:
    raise ValueError(
        "The geometry file has no toroidal-angle component; cannot extract "
        "a flux surface.")
  # end

  edges, centers, _ = utils.interpolated_grid_values(first)
  xc, yc, zc = centers
  if not 0 <= x_idx < xc.size:
    raise ValueError(f"x_idx {x_idx} is out of bounds for data with Nx={xc.size}")
  # end
  Nz = zc.size

  # Up-sample z for a smooth parallel mapping.
  zf_edges = np.linspace(edges[2][0], edges[2][-1], nz_interp * Nz + 1)
  zf = 0.5 * (zf_edges[:-1] + zf_edges[1:])

  # Unwrap toroidal angle coordinates to track continuous winding.
  phi = np.unwrap(np.unwrap(np.unwrap(geo.phi, axis=2), axis=1), axis=0)
  phi_grid = utils.resample_grid(phi, geo.coords, [xc, yc, zf])
  phi_2d = phi_grid[x_idx, :, :]

  # Array of toroidal angles to scan over (standard 0 to 2*pi).
  phi_tor_list = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)

  return FluxSurfaceGrid(x_idx=x_idx, zc=zc, zf=zf, phi_tor_list=phi_tor_list,
      phi_2d=phi_2d)
# end


def flux_surface_grids(datasets, *, mapc2p: str | None = None,
    nodes_file: str | None = None, x_idx: int = 0, nphi: int = 128,
    nz_interp: int = 8) -> dict:
  """Resolve one flux-surface sampling grid **per block**, keyed by geometry
  prefix -- the :func:`postgkyl.diagnostics.gyrokinetics.rz.rz_projections`
  counterpart for this diagnostic.

  Each block of a multiblock run has its own geometry file, so it needs its
  own toroidal-angle sampling grid; datasets sharing a prefix (one block's
  frames) resolve once.

  Returns:
    ``{geometry_prefix: FluxSurfaceGrid}``. Pair with :func:`grid_for`.
  """
  grids: dict = {}
  for data in datasets:
    key = geometry_prefix(data.file_name)
    if key in grids:
      continue
    # end
    block = data.ctx.get("block")
    geo = resolve_geometry(data.file_name,
        mapc2p=per_block_path(mapc2p, block),
        nodes_file=per_block_path(nodes_file, block))
    grids[key] = resolve_flux_surface_grid(data, geo, x_idx=x_idx, nphi=nphi,
        nz_interp=nz_interp)
  # end
  return grids
# end


def grid_for(grids: dict, data: GData) -> FluxSurfaceGrid:
  """The entry of a :func:`flux_surface_grids` mapping belonging to ``data``.

  Raises:
    KeyError: if ``data``'s block was not among the datasets the mapping was
      built from.
  """
  return grids[geometry_prefix(data.file_name)]
# end


def extract_flux_surface(data: GData, fs_grid: FluxSurfaceGrid, *,
    inplace: bool = False, tag: str | None = None, label: str | None = None) -> GData:
  """Extract the theta-phi flux surface at radial index ``fs_grid.x_idx``
  from ``data``'s 3-D field, using a precomputed ``fs_grid`` (see
  :func:`resolve_flux_surface_grid`).

  The field is up-sampled along z to ``fs_grid.zf``, then, for each z,
  projected onto every requested toroidal angle by periodic interpolation
  along the binormal (y) direction.

  Raises:
    ValueError: if ``fs_grid.x_idx`` is out of bounds for ``data``.
  """
  _, _, values = utils.interpolated_grid_values(data)
  if not 0 <= fs_grid.x_idx < values.shape[0]:
    raise ValueError(
        f"x_idx {fs_grid.x_idx} is out of bounds for data with Nx={values.shape[0]}")
  # end

  # Up-sample the 3-D field along the z (parallel) direction, then extract
  # ONLY the target radial index to save massive amounts of compute.
  vals_zf = PchipInterpolator(fs_grid.zc, values, axis=-1, extrapolate=True)(fs_grid.zf)
  vals_2d = vals_zf[fs_grid.x_idx, :, :]  # shape (Ny, len(zf))
  phi_2d = fs_grid.phi_2d
  Ny = vals_2d.shape[0]

  # Vectorized projection: loop only over z, evaluate all phi_tor
  # simultaneously.
  flux_surf_data = np.empty((fs_grid.phi_tor_list.size, fs_grid.zf.size))
  for iz in range(fs_grid.zf.size):
    phi_y = phi_2d[:, iz]
    val_y = vals_2d[:, iz]

    # Toroidal angle subtended by one full (periodic) binormal box.
    box = np.mean(np.diff(phi_y)) * Ny

    # Extend domain for periodic interpolation.
    phi_ext = np.concatenate([phi_y - box, phi_y, phi_y + box])
    val_ext = np.concatenate([val_y, val_y, val_y])

    # Sort to prepare for numpy interpolation.
    order = np.argsort(phi_ext)
    phi_ext_sorted = phi_ext[order]
    val_ext_sorted = val_ext[order]

    # Fold ALL requested phi angles into the local domain at once.
    pt_array = phi_y[0] + np.mod(fs_grid.phi_tor_list - phi_y[0], box)

    # Interpolate and assign the entire column of toroidal angles in one
    # call.
    flux_surf_data[:, iz] = np.interp(pt_array, phi_ext_sorted, val_ext_sorted)
  # end

  return data._result([fs_grid.phi_tor_list, fs_grid.zf], flux_surf_data[..., np.newaxis],
      inplace=inplace, tag=tag, label=label, interpolated=True)
# end
