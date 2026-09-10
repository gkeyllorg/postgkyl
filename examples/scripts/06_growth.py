"""Recover a known energy growth rate from a generated diagnostic history."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import postgkyl as pg

from _example_paths import TEST_DATA, prepare_output_dir

data = pg.load(TEST_DATA / "generated/exponential_energy.gkyl")
log_energy = pg.evaluate("f log", data)
fit = log_energy.fit("linear")
time = data.grid[0]
energy_rate = np.polyfit(time, fit.values.ravel(), 1)[0]
np.testing.assert_allclose(energy_rate, 0.4, rtol=1e-10)
np.testing.assert_allclose(fit.values, log_energy.values, atol=1e-12)

# Plot the reconstructed fitted energy with the public plotting function.
# The CLI uses evaluate "f log", fit linear, and evaluate "f1 exp" (fit appends its result).
pg.evaluate("f exp", fit).plot(title="Fitted energy growth",
                               xlabel="Time [normalized]",
                               ylabel="Energy [normalized]",
                               logy=True,
                               no_show=True,
                               saveas=prepare_output_dir() / "06_growth.png",
                               dpi=100)
