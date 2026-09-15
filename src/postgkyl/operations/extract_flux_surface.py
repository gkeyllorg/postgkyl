"""Sample modal fields on a toroidal surface, with reusable sampling grids."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.numerics.resample import resample_grid
from .geometry import Geometry, resolve_geometry, _validate_geometry
from ._mapping import (
    DEFAULT_NZ_INTERP,
    _interpolate_component,
    _interpolation_grid,
    validate_mapping_grid,
    _validate_component,
    _validate_modal_data,
    _validate_positive_int,
    mappings_by_geometry,
    mapping_for,
)
from postgkyl.numerics.rz import sample_flux_surface

_DEFAULT_NPHI = 128


@dataclass(frozen=True)
class FluxSurfaceGrid:
  """Precomputed toroidal sampling grid for one radial flux surface."""

  x_idx: int
  zc: np.ndarray
  zf: np.ndarray
  phi_tor_list: np.ndarray
  phi_2d: np.ndarray
  computational_grid: tuple[np.ndarray, ...]


def resolve_flux_surface_grid(
    first: "GDataState",
    geo: Geometry,
    *,
    x_idx: int = 0,
    nphi: int = _DEFAULT_NPHI,
    nz_interp: int = DEFAULT_NZ_INTERP) -> FluxSurfaceGrid:
  """Precompute a flux-surface sampling grid for compatible 3-D fields."""
  _validate_modal_data(first, "extract_flux_surface", (3, ))
  nphi = _validate_positive_int(nphi, "nphi")
  nz_interp = _validate_positive_int(nz_interp, "nz_interp")
  _validate_geometry(geo, 3)
  if geo.phi is None:
    raise ValueError(
        "The geometry has no toroidal-angle component; cannot extract a flux surface."
    )
  if isinstance(x_idx, bool) or not isinstance(x_idx, (int, np.integer)):
    raise ValueError("x_idx must be an integer radial index.")

  edges, centers = _interpolation_grid(first)
  xc, yc, zc = centers
  x_idx = int(x_idx)
  if not 0 <= x_idx < xc.size:
    raise ValueError(
        f"x_idx {x_idx} is out of bounds for data with Nx={xc.size}.")
  if zc.size < 2 or yc.size < 2:
    raise ValueError(
        "extract_flux_surface requires at least two interpolated y and z points."
    )

  zf_edges = np.linspace(edges[2][0], edges[2][-1], nz_interp * zc.size + 1)
  zf = 0.5 * (zf_edges[:-1] + zf_edges[1:])
  phi = np.unwrap(np.unwrap(np.unwrap(geo.phi, axis=2), axis=1), axis=0)
  phi_grid = resample_grid(phi, geo.coords, [xc, yc, zf])
  phi_2d = phi_grid[x_idx, :, :]
  phi_tor_list = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
  return FluxSurfaceGrid(x_idx=x_idx,
                         zc=zc,
                         zf=zf,
                         phi_tor_list=phi_tor_list,
                         phi_2d=phi_2d,
                         computational_grid=tuple(
                             np.array(axis, copy=True) for axis in edges))


def extract_flux_surface(data: "GDataState",
                         *,
                         fs_grid: FluxSurfaceGrid | None = None,
                         mapc2p: str | None = None,
                         nodes_file: str | None = None,
                         x_idx: int = 0,
                         nphi: int = _DEFAULT_NPHI,
                         nz_interp: int = DEFAULT_NZ_INTERP,
                         comp: int = 0,
                         inplace: bool = False,
                         tag: str | None = None,
                         label: str | None = None) -> "GDataState":
  """Sample a field component on a toroidal surface.

  Args:
    data: Three-dimensional field-aligned modal DG dataset.
    fs_grid: Reusable Python grid from resolve_flux_surface_grid; omit to
      load geometry. Geometry and sampling options belong on its builder.
    mapc2p: Explicit modal geometry path, or '' for the inferred modal file.
    nodes_file: Explicit interior-node geometry path; excludes mapc2p.
      Omit both paths to infer geometry from data.file_name.
    x_idx: Radial index on the interpolated computational grid.
    nphi: Number of toroidal-angle samples.
    nz_interp: Positive parallel-direction up-sampling factor.
    comp: Zero-based physical field component to extract.
    inplace: Mutate and return data instead of creating a dataset.
    tag: Optional result tag.
    label: Optional result label.
  """
  _validate_modal_data(data, "extract_flux_surface", (3, ))
  _validate_component(data, comp)
  if fs_grid is None:
    geometry = resolve_geometry(data.file_name,
                                mapc2p=mapc2p,
                                nodes_file=nodes_file)
    fs_grid = resolve_flux_surface_grid(data,
                                        geometry,
                                        x_idx=x_idx,
                                        nphi=nphi,
                                        nz_interp=nz_interp)
  elif (mapc2p is not None or nodes_file is not None or x_idx != 0
        or nphi != _DEFAULT_NPHI or nz_interp != DEFAULT_NZ_INTERP):
    raise ValueError(
        "fs_grid cannot be combined with geometry or sampling options; "
        "set them in resolve_flux_surface_grid instead.")
  edges = validate_mapping_grid(data, fs_grid.computational_grid)
  nx, ny, nz = (axis.size - 1 for axis in edges)
  if not 0 <= fs_grid.x_idx < nx:
    raise ValueError(
        f"x_idx {fs_grid.x_idx} is out of bounds for data with Nx={nx}.")
  if (fs_grid.zc.shape != (nz, )
      or fs_grid.phi_2d.shape != (ny, fs_grid.zf.size)
      or fs_grid.phi_tor_list.ndim != 1):
    raise ValueError(
        "Incompatible flux-surface grid: projection and data grid shapes differ."
    )

  values = _interpolate_component(data, comp)
  flux_surf_data = sample_flux_surface(values[fs_grid.x_idx], fs_grid.zc,
                                       fs_grid.zf, fs_grid.phi_2d,
                                       fs_grid.phi_tor_list)

  return data._result([fs_grid.phi_tor_list, fs_grid.zf],
                      flux_surf_data[..., np.newaxis],
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True)


def flux_surface_grids(
    datasets,
    *,
    mapc2p: str | None = None,
    nodes_file: str | None = None,
    x_idx: int = 0,
    nphi: int = _DEFAULT_NPHI,
    nz_interp: int = DEFAULT_NZ_INTERP) -> dict[str | None, FluxSurfaceGrid]:
  """Build one reusable toroidal sampling grid per block across fields/frames."""
  return mappings_by_geometry(datasets,
                              partial(resolve_flux_surface_grid,
                                      x_idx=x_idx,
                                      nphi=nphi,
                                      nz_interp=nz_interp),
                              mapc2p=mapc2p,
                              nodes_file=nodes_file)


grid_for = mapping_for
