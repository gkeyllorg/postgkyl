"""Gyrokinetic data transformations.

Placement answers two independent questions: ``operations`` says these
functions re-express data rather than derive physical conclusions, while
``gyrokinetics`` identifies the domain knowledge their geometry requires.
"""

from .geometry import GKYL_GEOMETRY_ID, Geometry, is_geo_mapc2p, resolve_geometry
from .rz import RzProjection, gk_rz, map_to_rz, resolve_rz_projection
from .fluxsurf import FluxSurfaceGrid, extract_flux_surface, resolve_flux_surface_grid

__all__ = [
    "GKYL_GEOMETRY_ID", "Geometry", "is_geo_mapc2p", "resolve_geometry",
    "RzProjection", "gk_rz", "map_to_rz", "resolve_rz_projection",
    "FluxSurfaceGrid", "extract_flux_surface", "resolve_flux_surface_grid",
]
