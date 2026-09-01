"""Equation-specific physics -- the COMPOSITION tier, one module per equation
model.

Folds together the old ``models`` (array math) and ``operations`` physics-verb
(GData wrapping) layers into a single home per equation system: functions
here take loaded ``GData``/``GDataState`` (one or several) plus physical
scalars as keyword-only options, and return a ``GDataState`` (via
``_result``) or, in later layers, a ``Figure``. Equation-blind core verbs
stay in flat ``operations`` modules. Domain-specific transformations live in
operation subpackages (for example ``operations.gyrokinetics``); this layer
is reserved for code that knows what field components physically mean.

Layer 12 added the equation-internal loaders: ``gyrokinetics/`` (distribution
functions + the derived-quantity registry), the shared ``discovery.py``
stem/frame discovery, and ``pkpm.load_pkpm``. Layer 13 extends this package
further with the program-scale diagnostics: three gyrokinetic programs
(``gyrokinetics.gk_energy_balance``/``gk_particle_balance``/``gk_nodes``,
ported from the old ``apps/gk_*.py``) plus ``trajectory``, ``enstrophy``, and
``ke_dke`` (ported from ``apps/trajectory.py`` and ``tools/calc_*.py``) --
there is no separate ``loaders/`` package anywhere.
"""

from . import (
    five_moment,
    ten_moment,
    mhd,
    plasma,
    multispecies,
    rotations,
    kinetic,
    pkpm,
    discovery,
    gyrokinetics,
    trajectory,
    enstrophy,
    ke_dke,
)

import inspect
import re
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
_DIAG_LOAD = CommandSpec(Section.DIAGNOSTICS, Execution.LOAD, selectable=False)
_DIAG_REPORT = CommandSpec(Section.DIAGNOSTICS, Execution.LOAD,
    selectable=False, result=ResultPolicy.VALUE)


def _complete_command_doc(function) -> None:
  """Give legacy short diagnostic docs one complete canonical Args block."""
  doc = inspect.getdoc(function) or function.__name__.replace("_", " ").capitalize() + "."
  parameters = tuple(inspect.signature(function).parameters.values())
  documented = set(re.findall(
      r"^\s{2,}(\*{0,2}[A-Za-z_]\w*)(?:\s*\([^)]*\))?:", doc,
      flags=re.MULTILINE))
  documented = {name.lstrip("*") for name in documented}
  if {parameter.name for parameter in parameters} <= documented:
    return
  # end
  pieces = doc.split("\n\n", 1)
  first = pieces[0]
  entries = []
  for parameter in parameters:
    display = parameter.name
    entries.append(f"  {display}: Value for ``{display}``.")
  # end
  function.__doc__ = first + "\n\nArgs:\n" + "\n".join(entries)
  if len(pieces) > 1:
    function.__doc__ += "\n\nNotes:\n" + "\n".join(
        "  " + line for line in pieces[1].splitlines())
  # end
# end


def _resolve(function) -> None:
  function.__globals__.setdefault("GDataState", GDataState)
  function.__globals__.setdefault("_GDataState", GDataState)
  function.__globals__.setdefault("GData", GData)
# end


def _map(function) -> None:
  _resolve(function)
  _complete_command_doc(function)
  command(_DIAG_MAP)(function)
# end


def _combine(function, dataset_names: tuple[str, ...]) -> None:
  _resolve(function)
  for name in dataset_names:
    function.__annotations__[name] = Annotated[GDataState, DatasetRef()]
  # end
  _complete_command_doc(function)
  command(_DIAG_COMBINE)(function)
# end


for _module, _names in (
    (five_moment, ("density", "xvel", "yvel", "zvel", "vel", "pressure",
        "ke", "temp", "sound", "mach")),
    (ten_moment, ("pressure", "ke", "temp", "sound", "mach", "pxx", "pxy",
        "pxz", "pyy", "pyz", "pzz", "pressure_tensor")),
    (mhd, ("bx", "by", "bz", "bi", "mag_pressure", "pressure", "temp",
        "sound", "mach")),
    (plasma, ("magB", "vt", "omegaC", "omegaP", "d", "lambdaD")),
):
  for _name in _names:
    _map(getattr(_module, _name))
  # end
# end

_resolve(multispecies.accumulate_current)
_complete_command_doc(multispecies.accumulate_current)
command(CommandSpec(Section.DIAGNOSTICS, Execution.MAP_APPEND,
    consumes_inputs=True))(multispecies.accumulate_current)

for _function, _datasets in (
    (five_moment.velocity, ("density", "momentum")),
    (ten_moment.p_par, ("ptensor", "bfield")),
    (ten_moment.p_perp, ("ptensor", "bfield")),
    (ten_moment.agyro, ("ptensor", "bfield")),
    (ten_moment.mom_agyro, ("species", "field")),
    (plasma.vA, ("species", "field")),
    (plasma.rho, ("species", "field")),
    (plasma.beta, ("species", "field")),
    (multispecies.energetics, ("elc", "ion", "field")),
    (rotations.parrotate, ("array", "rotator")),
    (rotations.perprotate, ("array", "rotator")),
    (rotations.bparrotate, ("array", "field")),
    (rotations.bperprotate, ("array", "field")),
    (kinetic.transform_frame, ("distribution", "bulk")),
    (pkpm.laguerre_compose, ("distribution", "variables")),
):
  _combine(_function, _datasets)
# end

ten_moment.agyro.__annotations__["measure"] = Literal["swisdak", "frobenius"]
ten_moment.mom_agyro.__annotations__["measure"] = Literal["swisdak", "frobenius"]

for _function in (pkpm.load_pkpm, discovery.find_output_stems,
    discovery.available_frames, enstrophy.enstrophy, ke_dke.ke_dke):
  _resolve(_function)
  _complete_command_doc(_function)
  command(_DIAG_LOAD if _function is pkpm.load_pkpm else _DIAG_REPORT)(_function)
# end
pkpm.load_pkpm.__annotations__["idx"] = str

_resolve(trajectory.trajectory)
_complete_command_doc(trajectory.trajectory)
command(CommandSpec(Section.DIAGNOSTICS, Execution.TERMINAL_ALL,
    result=ResultPolicy.VALUE))(trajectory.trajectory)

for _function in (
    gyrokinetics.load_gk_distf, gyrokinetics.load_gk_quantity,
    gyrokinetics.gk_energy_balance, gyrokinetics.gk_particle_balance,
    gyrokinetics.gk_nodes,
):
  _resolve(_function)
  _complete_command_doc(_function)
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
    "five_moment", "ten_moment", "mhd", "plasma", "multispecies",
    "rotations", "kinetic", "pkpm", "discovery", "gyrokinetics",
    "trajectory", "enstrophy", "ke_dke",
]
