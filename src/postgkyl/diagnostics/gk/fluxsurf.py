"""Gkeyll geometry discovery composed with explicit fluxsurf mapping."""
from __future__ import annotations

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.operations.map import resolve_flux_surface_grid, extract_flux_surface
from postgkyl.operations.geometry import FluxSurfaceGrid, validate_mapping_grid
from .geometry import geometry_prefix, per_block_path, resolve_geometry


def flux_surface_grids(datasets,
                       *,
                       mapc2p: str | None = None,
                       nodes_file: str | None = None,
                       x_idx: int = 0,
                       nphi: int = 128,
                       nz_interp: int = 8) -> dict[str | None, FluxSurfaceGrid]:
  """Build one reusable flux-surface grid per block geometry."""
  grids: dict[str | None, FluxSurfaceGrid] = {}
  for data in datasets:
    key = geometry_prefix(data.file_name)
    if key in grids:
      validate_mapping_grid(data, grids[key].computational_grid)
      continue
    block = data.ctx.get("block")
    geometry = resolve_geometry(data.file_name,
                                mapc2p=per_block_path(mapc2p, block),
                                nodes_file=per_block_path(nodes_file, block))
    grids[key] = resolve_flux_surface_grid(data,
                                           geometry,
                                           x_idx=x_idx,
                                           nphi=nphi,
                                           nz_interp=nz_interp)
  return grids


def grid_for(grids: dict[str | None, FluxSurfaceGrid],
             data: "GDataState") -> FluxSurfaceGrid:
  """Return the sampling grid belonging to ``data``'s block."""
  return grids[geometry_prefix(data.file_name)]


def fluxsurf(data: "GDataState",
             *,
             mapc2p: str | None = None,
             nodes_file: str | None = None,
             x_idx: int = 0,
             nphi: int = 128,
             nz_interp: int = 8,
             comp: int = 0,
             inplace: bool = False,
             tag: str | None = None,
             label: str | None = None) -> "GDataState":
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
  geometry = resolve_geometry(data.file_name,
                              mapc2p=mapc2p,
                              nodes_file=nodes_file)
  fs_grid = resolve_flux_surface_grid(data,
                                      geometry,
                                      x_idx=x_idx,
                                      nphi=nphi,
                                      nz_interp=nz_interp)
  return extract_flux_surface(data,
                              fs_grid=fs_grid,
                              comp=comp,
                              inplace=inplace,
                              tag=tag,
                              label=label)


__all__ = ["fluxsurf", "flux_surface_grids", "grid_for"]
