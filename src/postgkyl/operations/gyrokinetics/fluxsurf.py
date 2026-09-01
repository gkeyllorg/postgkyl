"""Extract theta-phi flux surfaces from gyrokinetic field-aligned data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

from .geometry import (
    Geometry,
    _interpolate_component,
    _interpolation_grid,
    _resample_grid,
    _same_grid,
    _validate_component,
    _validate_geometry,
    _validate_modal_data,
    _validate_positive_int,
    resolve_geometry,
)

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


@dataclass(frozen=True)
class FluxSurfaceGrid:
  """Precomputed toroidal sampling grid for one radial flux surface."""

  x_idx: int
  zc: np.ndarray
  zf: np.ndarray
  phi_tor_list: np.ndarray
  phi_2d: np.ndarray
  computational_grid: tuple[np.ndarray, ...] | None = None


def resolve_flux_surface_grid(first: "GDataState", geo: Geometry, *,
    x_idx: int = 0, nphi: int = 128, nz_interp: int = 8) -> FluxSurfaceGrid:
  """Precompute a flux-surface sampling grid for compatible 3-D fields."""
  _validate_modal_data(first, "gk_fluxsurf", (3,))
  nphi = _validate_positive_int(nphi, "nphi")
  nz_interp = _validate_positive_int(nz_interp, "nz_interp")
  _validate_geometry(geo, 3)
  if geo.phi is None:
    raise ValueError(
        "The geometry file has no toroidal-angle component; cannot extract a flux surface.")
  # end
  if isinstance(x_idx, bool) or not isinstance(x_idx, (int, np.integer)):
    raise ValueError("x_idx must be an integer radial index.")
  # end

  edges, centers = _interpolation_grid(first)
  xc, yc, zc = centers
  x_idx = int(x_idx)
  if not 0 <= x_idx < xc.size:
    raise ValueError(f"x_idx {x_idx} is out of bounds for data with Nx={xc.size}.")
  # end
  if zc.size < 2 or yc.size < 2:
    raise ValueError("gk_fluxsurf requires at least two interpolated y and z points.")
  # end

  zf_edges = np.linspace(edges[2][0], edges[2][-1], nz_interp * zc.size + 1)
  zf = 0.5 * (zf_edges[:-1] + zf_edges[1:])
  phi = np.unwrap(np.unwrap(np.unwrap(geo.phi, axis=2), axis=1), axis=0)
  phi_grid = _resample_grid(phi, geo.coords, [xc, yc, zf])
  phi_2d = phi_grid[x_idx, :, :]
  phi_tor_list = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
  return FluxSurfaceGrid(x_idx=x_idx, zc=zc, zf=zf,
      phi_tor_list=phi_tor_list, phi_2d=phi_2d,
      computational_grid=tuple(np.array(axis, copy=True) for axis in edges))
# end


def extract_flux_surface(data: "GDataState", fs_grid: FluxSurfaceGrid, *,
    comp: int = 0, inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Extract component ``comp`` using a reusable ``fs_grid``."""
  _validate_modal_data(data, "gk_fluxsurf", (3,))
  _validate_component(data, comp)
  edges, _ = _interpolation_grid(data)
  if fs_grid.computational_grid is not None \
      and not _same_grid(fs_grid.computational_grid, edges):
    raise ValueError(
        "Incompatible flux-surface grid: data computational grid does not "
        "match the grid used to build the projection.")
  # end
  nx, ny, nz = (axis.size - 1 for axis in edges)
  if not 0 <= fs_grid.x_idx < nx:
    raise ValueError(f"x_idx {fs_grid.x_idx} is out of bounds for data with Nx={nx}.")
  # end
  if (fs_grid.zc.shape != (nz,) or fs_grid.phi_2d.shape != (ny, fs_grid.zf.size)
      or fs_grid.phi_tor_list.ndim != 1):
    raise ValueError("Incompatible flux-surface grid: projection and data grid shapes differ.")
  # end

  _, _, values = _interpolate_component(data, comp)
  vals_zf = PchipInterpolator(
      fs_grid.zc, values, axis=-1, extrapolate=True)(fs_grid.zf)
  vals_2d = vals_zf[fs_grid.x_idx, :, :]

  flux_surf_data = np.empty((fs_grid.phi_tor_list.size, fs_grid.zf.size))
  for iz in range(fs_grid.zf.size):
    phi_y = fs_grid.phi_2d[:, iz]
    val_y = vals_2d[:, iz]
    box = np.mean(np.diff(phi_y)) * ny
    if not np.isfinite(box) or np.isclose(box, 0.0):
      raise ValueError("Toroidal geometry has a zero or non-finite binormal angular span.")
    # end
    phi_ext = np.concatenate([phi_y - box, phi_y, phi_y + box])
    val_ext = np.concatenate([val_y, val_y, val_y])
    order = np.argsort(phi_ext)
    folded = phi_y[0] + np.mod(fs_grid.phi_tor_list - phi_y[0], box)
    flux_surf_data[:, iz] = np.interp(
        folded, phi_ext[order], val_ext[order])
  # end

  return data._result([fs_grid.phi_tor_list, fs_grid.zf],
      flux_surf_data[..., np.newaxis], inplace=inplace, tag=tag,
      label=label, interpolated=True)
# end


def gk_fluxsurf(data: "GDataState", *, mapc2p: str | None = None,
    nodes_file: str | None = None, x_idx: int = 0, nphi: int = 128,
    nz_interp: int = 8, comp: int = 0, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Extract one field component on a toroidal flux surface.

  Args:
    data: Three-dimensional field-aligned modal dataset.
    mapc2p: Explicit modal geometry path.
    nodes_file: Explicit nodal geometry path.
    x_idx: Radial cell index identifying the surface.
    nphi: Number of toroidal-angle slices.
    nz_interp: Parallel-direction interpolation factor.
    comp: Physical field component to extract.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.
  """
  geometry = resolve_geometry(data.file_name, mapc2p=mapc2p,
      nodes_file=nodes_file)
  fs_grid = resolve_flux_surface_grid(data, geometry, x_idx=x_idx,
      nphi=nphi, nz_interp=nz_interp)
  return extract_flux_surface(data, fs_grid, comp=comp, inplace=inplace,
      tag=tag, label=label)
# end


__all__ = ["FluxSurfaceGrid", "extract_flux_surface", "gk_fluxsurf",
    "resolve_flux_surface_grid"]
