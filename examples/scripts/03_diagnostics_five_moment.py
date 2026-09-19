"""The diagnostics layer: equation-specific physics on top of the same
``GData`` you get from ``pg.load``.

``diagnostics`` sits *above* the fluent API (``operations``/``api``) and is
equation-blind-free by design: ``postgkyl.diagnostics.mom.five_moment`` knows the
Euler fluid moment layout (``[rho, rho*vx, rho*vy, rho*vz, E]``) and turns raw
conserved moments into primitive variables (density, velocity, pressure,
Mach number, ...). Diagnostics are **free functions**, not ``GData`` methods
-- ``fm.density(d)``, never ``d.density()`` -- because a diagnostic knows
about one specific equation system and a ``GData`` doesn't.

This script loads a generated stationary shock-tube initial condition and
checks density, pressure, and Mach number against the prescribed state.
Run ``python tests/generate_test_data.py`` first.

Run directly:
    MPLBACKEND=Agg PYTHONPATH=src python examples/scripts/03_diagnostics_five_moment.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

import postgkyl as pg
from postgkyl.diagnostics.mom import five_moment as fm

from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()

GAS_GAMMA = 5.0 / 3.0

# Load the generated p0 modal state and evaluate its physical values.
d = pg.load(TEST_DATA / "generated/shock_tube_1d_p0.gkyl").interpolate()
x = 0.5 * (d.grid[0][1:] + d.grid[0][:-1])
rho = np.where(x < 0.5, 1.0, 0.125)
p = np.where(x < 0.5, 1.0, 0.1)

# Diagnostics are free functions of a GData(State), returning a new one --
# the same ``inplace``/``tag``/``label`` contract as every ``operations`` verb.
density = fm.density(d)
pressure = fm.pressure(d, gas_gamma=GAS_GAMMA)
mach = fm.mach(d, gas_gamma=GAS_GAMMA)

np.testing.assert_allclose(density.values.ravel(), rho)
np.testing.assert_allclose(pressure.values.ravel(), p)
np.testing.assert_allclose(mach.values, 0.0)

# Raw modal DG coefficients have no "density"/"pressure" until interpolated
# -- the diagnostics layer refuses the same way ``operations`` verbs do.
modal = pg.load(TEST_DATA / "generated" / "1d_ms_p1.gkyl")
try:
  fm.density(modal)
except ValueError as exc:
  print("fm.density(modal) refuses:", exc)

fig = density.plot(title="density (Sod shock tube)", no_show=True)
fig.savefig(OUTPUT_DIR / "03_diagnostics_density.png")

fig = pressure.plot(title="pressure (Sod shock tube)", no_show=True)
fig.savefig(OUTPUT_DIR / "03_diagnostics_pressure.png")

print("03_diagnostics_five_moment: OK")
