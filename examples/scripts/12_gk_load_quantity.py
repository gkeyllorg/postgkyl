"""Load thermal speed, normalized potential, and Larmor radius by name.

Run ``python tests/generate_test_data.py`` first, then:
    MPLBACKEND=Agg python examples/scripts/12_gk_load_quantity.py

The synthetic electron profiles use SI units, with temperature rising from
20 to 30 eV, potential varying sinusoidally, and magnetic field from 1 to
1.5 T. The MaxwellianMoments file stores [density, upar, T/m], plus electron
mass and charge metadata. All sources share the same 1D grid and p1 basis;
their slopes vanish within each cell, making the nonlinear checks exact.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy import constants

import postgkyl as pg

from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()
NAME = "gk_quantity_1d_p1"
PATH = str(TEST_DATA / "generated")

# Each call returns a list, even for one species and one frame. The loader
# finds the required source files by simulation name, species, and frame.
# Mass and charge come from the moments metadata; for files without them,
# pass mass=... and charge=... in the simulation's unit system.
vt, = pg.gk.load_quantity("vt", "elc", NAME, "0", path=PATH)

# The potential is species independent, but its normalization needs the
# electron temperature, so supply "elc" here too. e is the positive
# elementary charge: phi_norm = e*phi/T_e, even for negative-charge electrons.
phi_norm, = pg.gk.load_quantity("phi_norm",
                                "elc",
                                NAME,
                                "0",
                                path=PATH,
                                label=r"$e\phi/T_e$")

# The magnetic geometry file has no frame suffix and is loaded automatically.
larmor_radius, = pg.gk.load_quantity("larmor_radius",
                                     "elc",
                                     NAME,
                                     "0",
                                     path=PATH)

for quantity, data in (("vt", vt), ("phi_norm", phi_norm), ("larmor_radius",
                                                            larmor_radius)):
  # All arithmetic happens on native modal fields. Interpolate only after
  # computing the quantity, to obtain point values for checking and plotting.
  assert data.ctx["value_form"] == "modal"
  samples = data.interpolate(num_interp=1)
  x = 0.5 * (samples.grid[0][1:] + samples.grid[0][:-1])
  temperature = constants.elementary_charge * (20.0 + 10.0 * x)
  potential = 5.0 * np.sin(2 * np.pi * x)
  bmag = 1.0 + 0.5 * x
  expected = {
      "vt":
      np.sqrt(temperature / constants.electron_mass),
      "phi_norm":
      constants.elementary_charge * potential / temperature,
      "larmor_radius":
      np.sqrt(constants.electron_mass * temperature) /
      (constants.elementary_charge * bmag),
  }
  np.testing.assert_allclose(samples.values.ravel(),
                             expected[quantity],
                             rtol=1e-12,
                             atol=1e-14)
  fig = samples.plot(title=f"Electron {quantity}",
                     xlabel="x [m]",
                     ylabel=data.get_label(),
                     no_show=True)
  fig.savefig(OUTPUT_DIR / f"12_gk_load_quantity_{quantity}.png")

print("12_gk_load_quantity: vt, phi_norm, larmor_radius OK")
