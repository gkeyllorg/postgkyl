"""Compatibility aliases for gyrokinetic flux-surface operations.

Canonical imports live in :mod:`postgkyl.operations.gyrokinetics`; this path
is scheduled for removal in the next major version.
"""

from postgkyl.operations.gyrokinetics.fluxsurf import (
    FluxSurfaceGrid,
    Geometry,
    extract_flux_surface,
    flux_surface_grids,
    grid_for,
    resolve_flux_surface_grid,
)

__all__ = [
    "Geometry", "FluxSurfaceGrid", "extract_flux_surface",
    "flux_surface_grids", "grid_for", "resolve_flux_surface_grid",
]
