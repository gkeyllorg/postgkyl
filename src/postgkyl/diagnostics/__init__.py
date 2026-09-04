"""Equation-specific physics grouped by Gkeyll model family.

Folds together the old ``models`` (array math) and ``operations`` physics-verb
(GData wrapping) layers into a single home per equation system: functions
here take loaded ``GData``/``GDataState`` (one or several) plus physical
scalars as keyword-only options, and return a ``GDataState`` (via
``_result``) or, in later layers, a ``Figure``. Equation-blind core verbs
stay in flat ``operations`` modules. Domain-specific transformations live in
operation subpackages (for example ``operations.gyrokinetics``); this layer
is reserved for code that knows what field components physically mean.

The four public packages mirror Gkeyll's model families: ``gk``, ``vm``,
``pkpm``, and ``mom``. The equation-blind ``discovery`` module
stays shared at this package root. There is no separate ``loaders`` package;
each model family owns its loading and program-scale diagnostics.
"""

from . import discovery, gk, mom, pkpm, vm

from typing import Annotated, Literal

from postgkyl.cli_spec import (
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
    (mom.five_moment, ("density", "xvel", "yvel", "zvel", "vel", "pressure",
        "ke", "temp", "sound", "mach")),
    (mom.ten_moment, ("pressure", "ke", "temp", "sound", "mach", "pxx", "pxy",
        "pxz", "pyy", "pyz", "pzz", "pressure_tensor")),
    (mom.mhd, ("bx", "by", "bz", "bi", "mag_pressure", "pressure", "temp",
        "sound", "mach")),
    (mom.plasma, ("magB", "vt", "omegaC", "omegaP", "d", "lambdaD")),
):
  for _name in _names:
    _map(getattr(_module, _name))
  # end
# end

_resolve(mom.multispecies.accumulate_current)
command(CommandSpec(Section.DIAGNOSTICS, Execution.MAP_APPEND,
    consumes_inputs=True))(mom.multispecies.accumulate_current)

for _function, _datasets in (
    (mom.five_moment.velocity, ("density", "momentum")),
    (mom.ten_moment.p_par, ("ptensor", "bfield")),
    (mom.ten_moment.p_perp, ("ptensor", "bfield")),
    (mom.ten_moment.agyro, ("ptensor", "bfield")),
    (mom.ten_moment.mom_agyro, ("species", "field")),
    (mom.plasma.vA, ("species", "field")),
    (mom.plasma.rho, ("species", "field")),
    (mom.plasma.beta, ("species", "field")),
    (mom.multispecies.energetics, ("elc", "ion", "field")),
    (mom.rotations.parrotate, ("array", "rotator")),
    (mom.rotations.perprotate, ("array", "rotator")),
    (mom.rotations.bparrotate, ("array", "field")),
    (mom.rotations.bperprotate, ("array", "field")),
    (vm.kinetic.transform_frame, ("distribution", "bulk")),
    (pkpm.laguerre_compose, ("distribution", "variables")),
):
  _combine(_function, _datasets)
# end

mom.ten_moment.agyro.__annotations__["measure"] = Literal["swisdak", "frobenius"]
mom.ten_moment.mom_agyro.__annotations__["measure"] = Literal["swisdak", "frobenius"]

for _function in (pkpm.load_pkpm, discovery.find_output_stems,
    discovery.available_frames, mom.enstrophy.enstrophy,
    mom.ke_dke.ke_dke):
  _resolve(_function)
  command(_DIAG_LOAD if _function is pkpm.load_pkpm else _DIAG_REPORT)(_function)
# end
pkpm.load_pkpm.__annotations__["idx"] = str

_resolve(vm.trajectory.trajectory)
command(CommandSpec(Section.DIAGNOSTICS, Execution.TERMINAL_ALL,
    result=ResultPolicy.VALUE))(vm.trajectory.trajectory)

for _function in (
    gk.load_distf, gk.load_quantity,
    gk.energy_balance, gk.particle_balance,
    gk.nodes,
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
  _function = getattr(gk, _name)
  if command_spec(_function) is None and hidden_spec(_function) is None:
    hidden("requires Python objects or is a registry/provider helper")(_function)
  # end
# end
# end

__all__ = [
    "gk", "vm", "mom", "pkpm", "discovery",
]
