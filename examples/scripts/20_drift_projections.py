"""Load physical drift projections without changing their modal representation."""
import numpy as np
import postgkyl as pg
from _example_paths import TEST_DATA

for direction in ("0", "n0", "n1", "n2", "R", "Z", "phi"):
  drift, = pg.gk.load_quantity("ExB_vel",
                               "ion",
                               "drift_2x_cart",
                               "0",
                               path=str(TEST_DATA / "generated"),
                               direction=direction)
  assert drift.num_dims == 2
  assert drift.ctx["value_form"] == "modal"
  assert drift.ctx["drift_units"] == ("coordinate units/s"
                                      if direction == "0" else "m/s")
  if direction == "Z":
    # This fixture has b along Cartesian Z, so b x grad(phi) has no Z component.
    np.testing.assert_allclose(drift.values, 0, atol=2e-13)
  print(direction, drift.get_label())
