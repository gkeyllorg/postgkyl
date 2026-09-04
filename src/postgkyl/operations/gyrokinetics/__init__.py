"""Gyrokinetic data transformations.

Placement answers two independent questions: ``operations`` says these
functions re-express data rather than derive physical conclusions, while
``gyrokinetics`` identifies the domain knowledge their geometry requires.
"""

from .geometry import GKYL_GEOMETRY_ID, Geometry, is_geo_mapc2p, resolve_geometry
from .rz import RzProjection, gk_rz, map_to_rz, resolve_rz_projection
from .fluxsurf import (
    FluxSurfaceGrid,
    extract_flux_surface,
    gk_fluxsurf,
    resolve_flux_surface_grid,
)

from postgkyl.cli_spec import CommandSpec, Execution, Section, command, hidden
from postgkyl.gdatastate.gdatastate import GDataState

gk_rz.__globals__.setdefault("GDataState", GDataState)
gk_fluxsurf.__globals__.setdefault("GDataState", GDataState)
command(CommandSpec(Section.VERBS, Execution.MAP_REPLACE))(gk_rz)
command(CommandSpec(Section.VERBS, Execution.MAP_REPLACE))(gk_fluxsurf)
for _function in (
    is_geo_mapc2p,
    resolve_geometry,
    map_to_rz,
    resolve_rz_projection,
    extract_flux_surface,
    resolve_flux_surface_grid,
):
  hidden("lower-level geometry API requires Python geometry objects")(_function)

__all__ = [
    "GKYL_GEOMETRY_ID",
    "Geometry",
    "is_geo_mapc2p",
    "resolve_geometry",
    "RzProjection",
    "gk_rz",
    "map_to_rz",
    "resolve_rz_projection",
    "FluxSurfaceGrid",
    "extract_flux_surface",
    "gk_fluxsurf",
    "resolve_flux_surface_grid",
]
