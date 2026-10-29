"""Compare two simulations' output with ``pg.diagnostics.compare.sims``.

The fluent-API equivalent of ``compare_sims`` in ``examples/cli_tutorial.md``
(section 11): finds the output quantities two simulation directories share,
ranks them by how much they differ, and plots the biggest differences.

Run ``python tests/generate_test_data.py`` first, then:
    MPLBACKEND=Agg python examples/scripts/13_compare_sims.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import postgkyl as pg

from _example_paths import TEST_DATA, prepare_output_dir

COMPARE_SIMS = TEST_DATA / "generated" / "compare_sims"
OUTPUT_DIR = prepare_output_dir()

# toy_sim's field_energy differs between baseline and candidate;
# elc_integrated_moms does not (see generate_test_data.py).
report = pg.diagnostics.compare.sims(
    str(COMPARE_SIMS / "baseline" / "toy_sim"),
    str(COMPARE_SIMS / "candidate" / "toy_sim"),
    labels="before,after",
    diff=True,
    file_per_fig=True,
    out_dir=str(OUTPUT_DIR),
)

# Without --batch, sims() returns one entry, keyed by path0's basename.
toy_sim = report["toy_sim"]
print("compared:", [d.quantity for d in toy_sim.compared])
print("plotted:", toy_sim.plotted)

# field_energy differs, elc_integrated_moms doesn't -- both are still
# plotted (only two quantities exist, well under the default num_plots),
# but field_energy ranks first by how much it differs.
assert toy_sim.compared[0].quantity == "field_energy"
assert set(toy_sim.plotted) == {"field_energy", "elc_integrated_moms"}
for quantity in toy_sim.plotted:
  assert (OUTPUT_DIR / f"{quantity}.jpg").is_file()

print("13_compare_sims: OK")
