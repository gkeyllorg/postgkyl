"""Gyrokinetic diagnostics: loading and equation-specific physics.

The whole gyrokinetic-quantity stack -- naming-convention file resolution
(``quantity.py``), the derived-quantity physics (``quantities.py``), the
registry (``registry.py``), and the "physics-ready data by name" entry point
(``load_quantity.py``) -- lives together in this subpackage (see the layer-12
instruction file's decision record): splitting resolution from physics would
give gyrokinetics two homes for one piece of equation knowledge. Only the
equation-blind stem/frame discovery is shared, via
``postgkyl.diagnostics.discovery``. Geometry-only transformations live below
this physics layer in ``postgkyl.operations.gyrokinetics``; the R-Z and
flux-surface names exported here are compatibility aliases.
"""

from __future__ import annotations

from .distf import load_distf, resolve_frames
from .load_quantity import available_quantities, load_quantity
from .quantities import (
    fetch_beta_from_bmag_press,
    fetch_diamag_vel,
    fetch_ExB_vel,
    fetch_gradB_vel,
    fetch_M1_from_H,
    fetch_press_from_BiMax,
    fetch_press_from_Max,
    fetch_press_p,
    fetch_Tpar_from_BiMax,
    fetch_Tpar_from_M0_M1_M2par,
    fetch_temp_from_Max,
    fetch_temp_from_Tpar_Tperp,
    fetch_Tperp_from_BiMax,
    fetch_Tperp_from_M0_M2perp,
)
from .registry import gk_quant_registry

# Layer 13: program-scale diagnostics ported from src_bak's apps/gk_*.py.
from .energy_balance import EnergyBalanceTraces, energy_balance_error, energy_balance
from .particle_balance import ParticleBalanceTraces, particle_balance, particle_balance_error
from .nodes import GKYL_GEOMETRY_ID, nodes, is_geo_mapc2p, multib_tag, nodes_to_RZ

# Compatibility exports: canonical transformation APIs now live under
# postgkyl.operations.gyrokinetics. These imports are exact aliases.
from .rz import Geometry, RzProjection, gk_rz, map_to_rz, resolve_geometry, resolve_rz_projection
from .fluxsurf import FluxSurfaceGrid, extract_flux_surface, resolve_flux_surface_grid

from typing import Annotated

from postgkyl.cli_spec import (
    CommandSpec, Execution, KeyValue, ResultPolicy, Section, command,
)

_LOAD_SPEC = CommandSpec(Section.DIAGNOSTICS, Execution.LOAD)
_REPORT_SPEC = CommandSpec(Section.DIAGNOSTICS, Execution.LOAD,
    result=ResultPolicy.VALUE)
command(_LOAD_SPEC)(load_distf)
command(_LOAD_SPEC)(load_quantity)
energy_balance.__annotations__["bflux_files"] = Annotated[
    dict[str, str] | None, KeyValue()]
particle_balance.__annotations__["bflux_files"] = Annotated[
    dict[str, str] | None, KeyValue()]
for _function in (energy_balance, particle_balance, nodes):
  command(_REPORT_SPEC)(_function)
# end

__all__ = [
    "load_distf", "resolve_frames",
    "available_quantities", "load_quantity", "gk_quant_registry",
    "fetch_beta_from_bmag_press", "fetch_diamag_vel", "fetch_ExB_vel",
    "fetch_gradB_vel", "fetch_M1_from_H", "fetch_press_from_BiMax",
    "fetch_press_from_Max", "fetch_press_p", "fetch_Tpar_from_BiMax",
    "fetch_Tpar_from_M0_M1_M2par", "fetch_temp_from_Max",
    "fetch_temp_from_Tpar_Tperp", "fetch_Tperp_from_BiMax",
    "fetch_Tperp_from_M0_M2perp",
    "EnergyBalanceTraces", "energy_balance_error", "energy_balance",
    "ParticleBalanceTraces", "particle_balance", "particle_balance_error",
    "GKYL_GEOMETRY_ID", "nodes", "is_geo_mapc2p", "multib_tag", "nodes_to_RZ",
    "Geometry", "RzProjection", "gk_rz", "map_to_rz", "resolve_geometry", "resolve_rz_projection",
    "FluxSurfaceGrid", "extract_flux_surface", "resolve_flux_surface_grid",
]
