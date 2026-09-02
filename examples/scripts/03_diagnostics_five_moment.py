"""The diagnostics layer: equation-specific physics on top of the same
``GData`` you get from ``pg.load``.

``diagnostics`` sits *above* the fluent API (``operations``/``api``) and is
equation-blind-free by design: ``postgkyl.diagnostics.moments.five_moment`` knows the
Euler fluid moment layout (``[rho, rho*vx, rho*vy, rho*vz, E]``) and turns raw
conserved moments into primitive variables (density, velocity, pressure,
Mach number, ...). Diagnostics are **free functions**, not ``GData`` methods
-- ``fm.density(d)``, never ``d.density()`` -- because a diagnostic knows
about one specific equation system and a ``GData`` doesn't.

This script builds a small 1D Sod shock-tube initial condition by hand (no
file needed -- any interpolated/NumPy-backed ``GData`` will do, whether it
came from ``pg.load(...).interpolate()`` or, as here, straight from
``.push()``) and reads off density, pressure, and Mach number.

Run directly:
    MPLBACKEND=Agg PYTHONPATH=src python examples/scripts/03_diagnostics_five_moment.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np

import postgkyl as pg
from postgkyl.diagnostics.moments import five_moment as fm

from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()

GAS_GAMMA = 5.0 / 3.0

# A Sod shock tube: high density/pressure on the left, low on the right, at
# rest everywhere. Conserved moments are laid out
# ``[rho, rho*vx, rho*vy, rho*vz, E]`` -- see the five_moment module
# docstring.
grid = [np.linspace(0.0, 1.0, 100)]
x = grid[0]
rho = np.where(x < 0.5, 1.0, 0.125)
p = np.where(x < 0.5, 1.0, 0.1)
vx = vy = vz = np.zeros_like(x)
energy = p / (GAS_GAMMA - 1) + 0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2)
moments = np.stack([rho, rho * vx, rho * vy, rho * vz, energy], axis=-1)

d = pg.GData()
d.push(grid, moments)               # diagnostics require field-domain data

# Diagnostics are free functions of a GData(State), returning a new one --
# the same ``inplace``/``tag``/``label`` contract as every ``operations`` verb.
density = fm.density(d)
pressure = fm.pressure(d, gas_gamma=GAS_GAMMA)
mach = fm.mach(d, gas_gamma=GAS_GAMMA)

print("density matches the input rho profile:", np.allclose(density.values.ravel(), rho))
print("pressure matches the input p profile: ", np.allclose(pressure.values.ravel(), p))
print("Mach number at rest:                  ", mach.values.ravel()[0])

# Raw modal DG coefficients have no "density"/"pressure" until interpolated
# -- the diagnostics layer refuses the same way ``operations`` verbs do.
modal = pg.load(TEST_DATA / "generated" / "1d_ms_p1.gkyl")
try:
  fm.density(modal)
except ValueError as exc:
  print("fm.density(modal) refuses:", exc)

fig = density.plot(title="density (Sod shock tube)", show=False)
fig.savefig(OUTPUT_DIR / "03_diagnostics_density.png")

fig = pressure.plot(title="pressure (Sod shock tube)", show=False)
fig.savefig(OUTPUT_DIR / "03_diagnostics_pressure.png")

print("03_diagnostics_five_moment: OK")
