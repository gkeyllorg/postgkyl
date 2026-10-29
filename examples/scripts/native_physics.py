"""Use optional native collision and gyrokinetic kernels on loaded GData.

Build with POSTGKYL_GKEYLL_MODULE=gyrokinetic (or pkpm), then generate the
fixtures with python tests/generate_test_data.py before running this script.
A core-only installation prints the build instruction and exits normally.
"""

import numpy as np
import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.vm.collisions import spitzer_coll_freq
from postgkyl.diagnostics.gk import mom_gyrokinetic
from _example_paths import TEST_DATA


def main():
  modules = gpython.module_api_versions()
  if "gyrokinetic" not in modules:
    print(
        "This example needs POSTGKYL_GKEYLL_MODULE=gyrokinetic at installation."
    )
    return
  generated = TEST_DATA / "generated"
  moms = pg.load(generated / "collision_moms_p1.gkyl")
  bmag = pg.load(generated / "collision_bmag_p1.gkyl")
  nu = spitzer_coll_freq(moms,
                         moms,
                         bmag,
                         mass_self=1.,
                         mass_other=1.,
                         charge_self=-1.,
                         charge_other=1.,
                         epsilon_0=1.,
                         hbar=0.)
  assert np.all(nu.interpolate().values > 0)
  print("Modal collision frequency:", nu)

  distribution = pg.load(generated / "native_gk_1x2v_p1_f.gkyl")
  bmag = pg.load(generated / "native_gk_1x2v_p1_bmag.gkyl")
  density = mom_gyrokinetic(distribution, bmag, mass=2., moment="M0")
  # F=2+x+v, v in [-.5,2.5], mu in [.25,1.75]:
  # (2*pi/m) integral F dv dmu = 4.5*pi*(3+x).
  centers = (density.grid[0][1:] + density.grid[0][:-1]) / 2
  np.testing.assert_allclose(density.values[:, 0],
                             np.sqrt(2) * 4.5 * np.pi * (3 + centers),
                             rtol=3e-14,
                             atol=3e-14)
  print("Modal gyrokinetic density:", density)


if __name__ == "__main__":
  main()
