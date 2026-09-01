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

__all__ = [
    "five_moment", "ten_moment", "mhd", "plasma", "multispecies",
    "rotations", "kinetic", "pkpm", "discovery", "gyrokinetics",
    "trajectory", "enstrophy", "ke_dke",
]
