"""Project GK drifts and fields, or explicitly supplied vector components."""
import numpy as np
import postgkyl as pg
from _example_paths import TEST_DATA

for rep, direction in (("contra", 0), ("norm", 0), ("norm", 1), ("norm", 2),
                       ("cyl", 0), ("cyl", 1), ("cyl", 2), ("cart",
                                                            0), ("covar", 0)):
  drift, = pg.gk.load_quantity("ExB_vel",
                               "ion",
                               "drift_2x_cart",
                               "0",
                               path=str(TEST_DATA / "generated"),
                               direction=direction,
                               rep=rep)
  assert drift.num_dims == 2
  assert drift.ctx["value_form"] == "modal"
  if rep in ("cyl", "cart", "norm"):
    assert drift.ctx["projection_units"] == "m/s"
  if rep == "cyl" and direction == 2:
    # This fixture has b along Cartesian Z, so b x grad(phi) has no Z component.
    np.testing.assert_allclose(drift.values, 0, atol=2e-13)
  print(rep, direction, drift.get_label())

# The same electric-field quantity supports either coordinate representation.
covariant = [
    pg.gk.load_quantity("E_field",
                        None,
                        "drift_2x_cart",
                        "0",
                        path=str(TEST_DATA / "generated"),
                        rep="covar",
                        direction=i)[0] for i in range(3)
]
contravariant, = pg.gk.load_quantity("E_field",
                                     None,
                                     "drift_2x_cart",
                                     "0",
                                     path=str(TEST_DATA / "generated"),
                                     rep="contra",
                                     direction=0)
metric = pg.load(str(TEST_DATA / "generated" /
                     "drift_2x_cart-geo_int_gij.gkyl"))
raised = sum(metric.select(comp=i) * covariant[i] for i in range(3))
np.testing.assert_allclose(contravariant.values,
                           raised.values,
                           rtol=2e-13,
                           atol=2e-13)


# The operations-layer projector also works with any explicit vector.
def load_field(suffix):
  """Read one fixture field without GK quantity discovery."""
  return pg.load(str(TEST_DATA / "generated" / f"drift_2x_cart-{suffix}.gkyl"))


vector = load_field("vector_cartesian")
vertical = pg.operations.project_vector(
    [vector.select(comp=i) for i in range(3)],
    basis="cartesian",
    rep="cart",
    direction=2)
np.testing.assert_array_equal(vertical.values, vector.select(comp=2).values)

# In 2x, both fluctuation choices subtract a J-weighted parallel (z) mean.
for quantity in ("flux_particle_ExB", "flux_particle_dB"):
  fluxes = [
      pg.gk.load_quantity(quantity,
                          "ion",
                          "repr_2x_cart",
                          "0",
                          path=str(TEST_DATA / "generated"),
                          rep="cart",
                          direction=1,
                          fluct=fluct)[0] for fluct in ("y", "yz")
  ]
  np.testing.assert_array_equal(fluxes[0].values, fluxes[1].values)
  assert fluxes[0].num_dims == 2
  assert fluxes[0].ctx["projection_units"] == "1/(m^2 s)"
