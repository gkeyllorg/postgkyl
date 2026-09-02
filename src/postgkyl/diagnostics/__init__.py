"""Equation-specific physics grouped by Gkeyll model family.

Folds together the old ``models`` (array math) and ``operations`` physics-verb
(GData wrapping) layers into a single home per equation system: functions
here take loaded ``GData``/``GDataState`` (one or several) plus physical
scalars as keyword-only options, and return a ``GDataState`` (via
``_result``) or, in later layers, a ``Figure``. Equation-blind core verbs
stay in flat ``operations`` modules. Domain-specific transformations live in
operation subpackages (for example ``operations.gyrokinetics``); this layer
is reserved for code that knows what field components physically mean.

The four public packages mirror Gkeyll's model families: ``gyrokinetics``,
``vlasov``, ``pkpm``, and ``moments``. The equation-blind ``discovery`` module
stays shared at this package root. There is no separate ``loaders`` package;
each model family owns its loading and program-scale diagnostics.
"""

from . import discovery, gyrokinetics, moments, pkpm, vlasov

from typing import Annotated, Literal

from postgkyl.command_spec import (
    CommandSpec, DatasetRef, Execution, ResultPolicy, Section, command,
    command_spec, hidden, hidden_spec,
)
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.gdata.gdata import GData


_DIAG_MAP = CommandSpec(Section.DIAGNOSTICS, Execution.MAP_REPLACE)
_DIAG_COMBINE = CommandSpec(Section.DIAGNOSTICS, Execution.COMBINE,
    consumes_inputs=True)
_DIAG_LOAD = CommandSpec(Section.DIAGNOSTICS, Execution.LOAD)
_DIAG_REPORT = CommandSpec(Section.DIAGNOSTICS, Execution.LOAD,
    result=ResultPolicy.VALUE)


def _resolve(function) -> None:
  function.__globals__.setdefault("GDataState", GDataState)
  function.__globals__.setdefault("_GDataState", GDataState)
  function.__globals__.setdefault("GData", GData)
# end


def _map(function) -> None:
  _resolve(function)
  command(_DIAG_MAP)(function)
# end


def _combine(function, dataset_names: tuple[str, ...]) -> None:
  _resolve(function)
  for name in dataset_names:
    function.__annotations__[name] = Annotated[GDataState, DatasetRef()]
  # end
  command(_DIAG_COMBINE)(function)
# end


for _module, _names in (
    (moments.five_moment, ("density", "xvel", "yvel", "zvel", "vel", "pressure",
        "ke", "temp", "sound", "mach")),
    (moments.ten_moment, ("pressure", "ke", "temp", "sound", "mach", "pxx", "pxy",
        "pxz", "pyy", "pyz", "pzz", "pressure_tensor")),
    (moments.mhd, ("bx", "by", "bz", "bi", "mag_pressure", "pressure", "temp",
        "sound", "mach")),
    (moments.plasma, ("magB", "vt", "omegaC", "omegaP", "d", "lambdaD")),
):
  for _name in _names:
    _map(getattr(_module, _name))
  # end
# end

_resolve(moments.multispecies.accumulate_current)
command(CommandSpec(Section.DIAGNOSTICS, Execution.MAP_APPEND,
    consumes_inputs=True))(moments.multispecies.accumulate_current)

for _function, _datasets in (
    (moments.five_moment.velocity, ("density", "momentum")),
    (moments.ten_moment.p_par, ("ptensor", "bfield")),
    (moments.ten_moment.p_perp, ("ptensor", "bfield")),
    (moments.ten_moment.agyro, ("ptensor", "bfield")),
    (moments.ten_moment.mom_agyro, ("species", "field")),
    (moments.plasma.vA, ("species", "field")),
    (moments.plasma.rho, ("species", "field")),
    (moments.plasma.beta, ("species", "field")),
    (moments.multispecies.energetics, ("elc", "ion", "field")),
    (moments.rotations.parrotate, ("array", "rotator")),
    (moments.rotations.perprotate, ("array", "rotator")),
    (moments.rotations.bparrotate, ("array", "field")),
    (moments.rotations.bperprotate, ("array", "field")),
    (vlasov.kinetic.transform_frame, ("distribution", "bulk")),
    (pkpm.laguerre_compose, ("distribution", "variables")),
):
  _combine(_function, _datasets)
# end

moments.ten_moment.agyro.__annotations__["measure"] = Literal["swisdak", "frobenius"]
moments.ten_moment.mom_agyro.__annotations__["measure"] = Literal["swisdak", "frobenius"]

for _function in (pkpm.load_pkpm, discovery.find_output_stems,
    discovery.available_frames, moments.enstrophy.enstrophy,
    moments.ke_dke.ke_dke):
  _resolve(_function)
  command(_DIAG_LOAD if _function is pkpm.load_pkpm else _DIAG_REPORT)(_function)
# end
pkpm.load_pkpm.__annotations__["idx"] = str

_resolve(vlasov.trajectory.trajectory)
command(CommandSpec(Section.DIAGNOSTICS, Execution.TERMINAL_ALL,
    result=ResultPolicy.VALUE))(vlasov.trajectory.trajectory)

for _function in (
    gyrokinetics.load_gk_distf, gyrokinetics.load_gk_quantity,
    gyrokinetics.gk_energy_balance, gyrokinetics.gk_particle_balance,
    gyrokinetics.gk_nodes,
):
  _resolve(_function)
# end


# These exclusions are an explicit audit, not a catch-all over module
# contents.  Adding a public diagnostic callable without classifying it now
# fails discovery instead of being silently hidden.
for _name in (
    "resolve_frames", "available_quantities", "fetch_beta_from_bmag_press",
    "fetch_diamag_vel", "fetch_ExB_vel", "fetch_gradB_vel", "fetch_M1_from_H",
    "fetch_press_from_BiMax", "fetch_press_from_Max", "fetch_press_p",
    "fetch_Tpar_from_BiMax", "fetch_Tpar_from_M0_M1_M2par",
    "fetch_temp_from_Max", "fetch_temp_from_Tpar_Tperp",
    "fetch_Tperp_from_BiMax", "fetch_Tperp_from_M0_M2perp",
    "energy_balance_error", "particle_balance_error", "is_geo_mapc2p",
    "multib_tag", "nodes_to_RZ", "map_to_rz", "resolve_geometry",
    "resolve_rz_projection", "extract_flux_surface",
    "resolve_flux_surface_grid",
):
  _function = getattr(gyrokinetics, _name)
  if command_spec(_function) is None and hidden_spec(_function) is None:
    hidden("requires Python objects or is a registry/provider helper")(_function)
  # end
# end
# end

__all__ = [
    "gyrokinetics", "vlasov", "pkpm", "moments", "discovery",
]
