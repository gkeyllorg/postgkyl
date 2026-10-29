"""Project GK drifts and fields, or explicitly supplied vector components."""
import numpy as np
import postgkyl as pg
from _example_paths import TEST_DATA

for direction in ("0", "n0", "n1", "n2", "R", "Z", "phi"):
  drift, = pg.gk.load_quantity("ExB_vel_contra",
                               "ion",
                               "drift_2x_cart",
                               "0",
                               path=str(TEST_DATA / "generated"),
                               direction=direction)
  assert drift.num_dims == 2
  assert drift.ctx["value_form"] == "modal"
  assert drift.ctx["projection_units"] == ("coordinate units/s"
                                           if direction == "0" else "m/s")
  if direction == "Z":
    # This fixture has b along Cartesian Z, so b x grad(phi) has no Z component.
    np.testing.assert_allclose(drift.values, 0, atol=2e-13)
  print(direction, drift.get_label())

# Covariant and contravariant electric fields reconstruct the same vector.
radial_cov, = pg.gk.load_quantity("E_field_cov",
                                  None,
                                  "drift_2x_cart",
                                  "0",
                                  path=str(TEST_DATA / "generated"),
                                  direction="R")
radial_contra, = pg.gk.load_quantity("E_field_contra",
                                     None,
                                     "drift_2x_cart",
                                     "0",
                                     path=str(TEST_DATA / "generated"),
                                     direction="R")
np.testing.assert_allclose(radial_cov.values,
                           radial_contra.values,
                           rtol=2e-13,
                           atol=2e-13)
assert radial_cov.ctx["projection_units"] == "V/m"


# The operations-layer projector also works with any explicit vector.
def load_field(suffix):
  """Read one fixture field without GK quantity discovery."""
  return pg.load(str(TEST_DATA / "generated" / f"drift_2x_cart-{suffix}.gkyl"))


vector = load_field("vector_cartesian")
vertical = pg.operations.project_vector(
    [vector.select(comp=i) for i in range(3)], basis="cartesian", direction="Z")
np.testing.assert_array_equal(vertical.values, vector.select(comp=2).values)
