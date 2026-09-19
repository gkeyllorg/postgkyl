"""Plot one field across three blocks with a shared color scale.

Run from the repository root (or the extracted example bundle):
    python tests/generate_test_data.py
    MPLBACKEND=Agg python examples/scripts/11_multiblock.py
"""

import numpy as np
import postgkyl as pg
from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()

blocks = pg.load(TEST_DATA / "generated/mb_sim_b*-elc_M0_0.gkyl").interpolate()

fig = pg.plot(*blocks,
              title="One field across three blocks",
              xlabel="x",
              ylabel="y",
              clabel="Synthetic field [arbitrary units]",
              fixaspect=True,
              no_show=True,
              saveas=OUTPUT_DIR / "11_multiblock.png",
              dpi=100)

# Verify that the whole domain appears in one panel with a shared scale.
assert len(fig.axes) == 2  # field panel and colorbar
meshes = fig.axes[0].collections
assert len(meshes) == len(blocks)
limits = (min(float(np.min(block.values)) for block in blocks),
          max(float(np.max(block.values)) for block in blocks))
for mesh in meshes:
  np.testing.assert_allclose(mesh.get_clim(), limits)
np.testing.assert_allclose(fig.axes[0].get_xlim(), [0.0, 3.0])
np.testing.assert_allclose(fig.axes[0].get_ylim(), [0.0, 1.0])

print("11_multiblock: OK")
