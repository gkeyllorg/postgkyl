"""GK quantities must compute on DG fields before any interpolation.

For 1-D p1 the weak product and weak inverse are diagonal at the two
Gauss points: projection of a product has degree at most three in its
integrands. This gives an independent numerical oracle for the formulas,
including every higher mode, without calling the production arithmetic.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import constants

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.gk import quantities as ff
from postgkyl.diagnostics.gk.load_quantity import load_quantity

pytestmark = pytest.mark.skipif(not gpython.available(),
                                reason="requires Gkeyll")
DATA = Path(__file__).parent / "test_data" / "generated" / "gk_moments_p1.gkyl"


@pytest.fixture
def moments():
  return pg.load(str(DATA))


def _samples(data):
  coefficients = data.values.reshape(2, -1, 2)
  return (coefficients[..., 0, None] +
          coefficients[..., 1, None] * np.array([-1., 1.])) / np.sqrt(2.)


def _project(samples):
  return np.stack([samples.sum(axis=-1), samples[..., 1] - samples[..., 0]],
                  axis=-1).reshape(2, -1) / np.sqrt(2.)


# All scalar fetch formulas, with nonconstant, nonidentical sources. Each
# source index picks one complete physical field from the packed fixture.
@pytest.mark.parametrize("fetch,components,formula", [
    (ff.fetch_s0c0_add_s1c0, [0, 1], lambda a, b: a + b),
    (ff.fetch_s0c0_sub_s1c0, [0, 1], lambda a, b: a - b),
    (ff.fetch_s0c0_mul_s1c0, [0, 1], lambda a, b: a * b),
    (ff.fetch_s1c0_div_s0c0, [0, 1], lambda a, b: b / a),
    (ff.fetch_Tpar_from_M0_M1_M2par, [0, 1, 2], lambda n, m1, m2: 2 *
     (m2 - m1**2 / n) / n),
    (ff.fetch_Tperp_from_M0_M2perp, [0, 3], lambda n, m2: m2 / n),
    (ff.fetch_temp_from_Tpar_Tperp, [2, 3], lambda a, b: (a + 2 * b) / 3),
    (ff.fetch_press_p, [0, 2], lambda n, t: n * t),
    (ff.fetch_qpar, [2], lambda m3: m3),
    (ff.fetch_qperp, [3], lambda m3: m3),
    (ff.fetch_qpar_fluid, [0, 1, 2, 3],
     lambda n, m1, m2, m3: m3 - 3 * m1 * m2 / n + 2 * m1**3 / n**2),
    (ff.fetch_qperp_fluid, [0, 1, 2, 3
                            ], lambda n, m1, m2, m3: m3 - m1 * m2 / n),
    (ff.fetch_vt, [2], lambda t: np.sqrt(t / 2)),
    (ff.fetch_larmor_radius, [2, 3], lambda t, b: np.sqrt(2 * t) / (3 * b)),
    (ff.fetch_debye_length, [2, 0
                             ], lambda t, n: np.sqrt(constants.epsilon_0 * t /
                                                     (9 * n))),
    (ff.fetch_beta_from_bmag_press, [3, 2],
     lambda b, p: 2 * constants.mu_0 * p / b**2),
    (ff.fetch_qpar_norm, [3, 0, 2, 1], lambda q, n, t, cs: q / (n * t * cs)),
    (ff.fetch_qperp_norm, [3, 0, 2, 1], lambda q, n, t, cs: q / (n * t * cs)),
    (ff.fetch_rho_over_lambda, [2, 3], lambda rho, ld: rho / ld),
    (ff.fetch_phi_norm, [1, 2],
     lambda phi, t: constants.elementary_charge * phi / t),
])
def test_scalar_quantities_keep_modal_coefficients(moments, fetch, components,
                                                   formula):
  sources = [moments.select(comp=c) for c in components]
  before = [s.values.copy() for s in sources]
  expected = _project(formula(*[_samples(s) for s in sources]))
  out = fetch(sources)
  assert out.backend == "gkyl"
  assert out.ctx["value_form"] == "modal"
  assert not out.is_interpolated
  np.testing.assert_allclose(out.values, expected, rtol=2e-12, atol=1e-30)
  np.testing.assert_array_equal(out.grid[0], moments.grid[0])
  for source, original in zip(sources, before):
    np.testing.assert_array_equal(source.values, original)


@pytest.mark.parametrize("fetch,formula", [
    (ff.fetch_s0c0, lambda n, u, tp, tt: n),
    (ff.fetch_s0c1, lambda n, u, tp, tt: u),
    (ff.fetch_s0c2, lambda n, u, tp, tt: tp),
    (ff.fetch_s0c3, lambda n, u, tp, tt: tt),
    (ff.fetch_s0c2_add_s0c3, lambda n, u, tp, tt: tp + tt),
    (ff.fetch_s0c0_mul_s0c1, lambda n, u, tp, tt: n * u),
    (ff.fetch_M1_from_H, lambda n, u, tp, tt: n * u / 2),
    (ff.fetch_Tpar_from_BiMax, lambda n, u, tp, tt: 2 * tp),
    (ff.fetch_Tperp_from_BiMax, lambda n, u, tp, tt: 2 * tt),
    (ff.fetch_temp_from_Max, lambda n, u, tp, tt: 2 * tp),
    (ff.fetch_press_from_Max, lambda n, u, tp, tt: 2 * n * tp),
    (ff.fetch_press_from_BiMax, lambda n, u, tp, tt: 2 * n * (tp + 2 * tt) / 3),
])
def test_packed_quantities_select_physical_fields(moments, fetch, formula):
  samples = _samples(moments)
  expected = _project(formula(*[samples[:, c, :] for c in range(4)]))
  out = fetch([moments])
  assert out.backend == "gkyl"
  np.testing.assert_allclose(out.values, expected, rtol=2e-12)


def test_all_components_retained(moments):
  out = ff.fetch_s0cAll([moments])
  assert out.backend == "gkyl"
  np.testing.assert_array_equal(out.values, moments.values)
  assert out is not moments


@pytest.mark.parametrize("kind", ["thermo", "ion_acoustic"])
def test_sound_speed_multiple_ions(moments, kind):
  n, t = moments.select(comp=0), moments.select(comp=2)
  sources = [[n, t], [n * 0.6, t * 0.4], [n * 0.2, t * 0.7]]
  # Quasineutral ions with charge states 1 and 2; species overrides also
  # exercise resolution of mass/charge independent of source metadata.
  out = ff.fetch_c_s(sources,
                     kind=kind,
                     mass=[1., 4., 8.],
                     charge=[-constants.e, constants.e, 2 * constants.e],
                     gamma_e=1.5,
                     gamma_i=2.5)
  if kind == "thermo":
    cs_sq = _samples(t) * (1.5 + 2.5 *
                           (0.6 * 0.4 + 0.2 * 0.7)) / (0.6 * 4 + 0.2 * 8)
  else:
    cs_sq = _samples(t) * (0.6 / 4 + 0.2 * 4 / 8) / (0.6 + 0.2 * 2)
  assert out.backend == "gkyl"
  np.testing.assert_allclose(out.values, _project(np.sqrt(cs_sq)), rtol=2e-12)


def test_projected_square_root_floors_negative_values(moments):
  negative = -moments.select(comp=2)
  out = ff.fetch_vt([negative])
  expected = np.array([[np.sqrt(2.) * 1e-40, 0.]] * 2)
  np.testing.assert_allclose(out.values, expected, rtol=2e-12, atol=1e-55)


def test_pressure_differs_from_multiplication_after_interpolation(moments):
  n, t = moments.select(comp=0), moments.select(comp=2)
  weak = ff.fetch_press_p([n, t]).interpolate()
  pointwise = n.interpolate() * t.interpolate()
  assert not np.allclose(weak.values, pointwise.values)


def test_recursive_loader_returns_modal_temperature(tmp_path):
  # Only primitive moments exist: Tpar must recursively load all three and
  # compute the temperature, keeping DG coefficients at every stage.
  from generate_test_data import write_gkyl_field
  source = pg.load(str(DATA))
  for comp, name in enumerate(["M0", "M1", "M2par"]):
    write_gkyl_field(tmp_path / f"sim-ion_{name}_0.gkyl", [2], [0.], [2.],
                     source.values[:, 2 * comp:2 * comp + 2],
                     1,
                     "serendipity",
                     metadata={"mass": 2.})
  out, = load_quantity("Tpar", "ion", "sim", "0", path=str(tmp_path))
  n, m1, m2 = [_samples(source)[:, c, :] for c in range(3)]
  assert out.backend == "gkyl"
  np.testing.assert_allclose(out.values, _project(2 * (m2 - m1**2 / n) / n))


@pytest.mark.parametrize(
    "fetch", [ff.fetch_ExB_vel, ff.fetch_gradB_vel, ff.fetch_diamag_vel])
@pytest.mark.parametrize("direction", [0, 1, 2])
def test_drifts_use_cell_local_derivatives(moments, fetch, direction):
  jac, bmag, scalar = [moments.select(comp=c) for c in [0, 3, 2]]
  b_i = moments.select(comp="0:3")
  sources = [jac, bmag, b_i, scalar]
  if fetch is ff.fetch_diamag_vel:
    sources.append(scalar)
  differentiated = bmag if fetch is ff.fetch_gradB_vel else scalar
  # Unit-width cells: df/dx = 2*b for f=a+b*xi, independently of jumps.
  deriv = np.sqrt(6.) * differentiated.values[:, 1, None]
  if direction == 0:
    cross = _samples(b_i)[:, 1, :] * deriv
  elif direction == 1:
    cross = -_samples(b_i)[:, 0, :] * deriv
  else:
    cross = np.zeros((2, 2))
  expected = cross * _samples(jac)[:, 0, :]
  if fetch is ff.fetch_gradB_vel:
    expected *= _samples(scalar)[:, 0, :] / _samples(bmag)[:, 0, :] / 3
  elif fetch is ff.fetch_diamag_vel:
    expected /= _samples(scalar)[:, 0, :] * 3
  out = fetch(sources, dir=direction)
  assert out.backend == "gkyl"
  np.testing.assert_allclose(out.values,
                             _project(expected),
                             rtol=2e-12,
                             atol=1e-14)


def test_quantities_reject_mixed_representations(moments):
  n = moments.select(comp=0)
  t = moments.select(comp=2)
  with pytest.raises(ValueError, match="value_forms"):
    ff.fetch_press_p([n, t.represent(to="nodal")])
  with pytest.raises(ValueError, match="different backends"):
    ff.fetch_press_p([n, t.interpolate()])


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("direction", [0, 1, 2])
def test_cross_product_in_reduced_and_full_configuration_space(ndim, direction):
  data = pg.load(str(DATA.parent / f"gk_drift_{ndim}d_p1.gkyl"))
  phi, jac, bi = data.select(comp=0), data.select(comp=1), data.select(
      comp="2:5")
  gradient = np.zeros(3)
  gradient[2] = 2 * ndim
  if ndim > 1:
    gradient[0] = 2
  if ndim == 3:
    gradient[1] = 4
  expected = np.zeros_like(phi.values)
  expected[..., 0] = 2 * np.cross([3., 4., 5.], gradient)[direction] * np.sqrt(
      2**ndim)
  out = ff.fetch_ExB_vel([jac, None, bi, phi], dir=direction)
  assert out.backend == "gkyl"
  np.testing.assert_allclose(out.values, expected, atol=1e-13)
