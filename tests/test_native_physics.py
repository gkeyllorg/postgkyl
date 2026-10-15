"""Independent analytic checks of optional Gkeyll physics bindings."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.gk import mom_gyrokinetic
from postgkyl.diagnostics.vm.collisions import coulomb_log, spitzer_coll_freq

GEN = Path(__file__).parent / "test_data" / "generated"
MODULES = gpython.module_api_versions() if gpython.available() else {}
needs_vlasov = pytest.mark.skipif("vlasov" not in MODULES,
                                  reason="requires optional Vlasov module")
needs_gk = pytest.mark.skipif("gyrokinetic" not in MODULES,
                              reason="requires optional gyrokinetic module")
CONSTANTS = dict(mass_self=1.,
                 mass_other=1.,
                 charge_self=-1.,
                 charge_other=1.,
                 epsilon_0=1.,
                 hbar=0.)


def collision_fields(order=1):
  return [
      pg.load(str(GEN / f"collision_{name}_p{order}.gkyl"))
      for name in ("density", "temperature", "bmag", "moms")
  ]


@needs_vlasov
@pytest.mark.parametrize("hbar", [0., 100.])
def test_coulomb_impact_parameters(hbar):
  # Equal species: Debye/magnetic cutoff squared is 2/(n+B^2),
  # classical closest approach is 1/(12*pi), quantum cutoff hbar/sqrt(6e).
  n = np.array([[1.], [2.], [4.]])
  magnetic = np.array([0., 1., 3., 5.])
  closest = max(1 / (12 * np.pi), hbar / np.sqrt(6 * np.e))
  expected = .5 * np.log1p(2 / ((n + magnetic**2) * closest**2))
  result = gpython.vlasov.coulomb_log(n, n, 1., 1., magnetic,
                                      **(CONSTANTS | {
                                          "hbar": hbar
                                      }))
  assert_allclose(result, expected, rtol=2e-15, atol=2e-15)


@needs_vlasov
@pytest.mark.parametrize("order", [1, 2])
def test_coulomb_gdata_projection_and_points(order):
  density, temperature, bmag, _ = collision_fields(order)
  before = density.values.copy()
  result = coulomb_log(density,
                       density,
                       temperature,
                       temperature,
                       bmag,
                       **CONSTANTS,
                       num_quad=8,
                       tag="logLambda")
  # Independently project logLambda=log(1+288*pi^2/(2+x))/2 with
  # the explicitly requested 8-point rule and orthonormal Legendre basis.
  nodes, weights = np.polynomial.legendre.leggauss(8)
  centers = (density.grid[0][1:] + density.grid[0][:-1]) / 2
  values = .5 * np.log1p(288 * np.pi**2 / (2 + centers[:, None] + .25 * nodes))
  legendre = np.polynomial.legendre.legvander(nodes, order)
  expected = (values * weights) @ (legendre * np.sqrt(
      (2 * np.arange(order + 1) + 1) / 2))
  assert_allclose(result.values, expected, rtol=3e-14, atol=3e-14)
  assert_allclose(density.values, before, rtol=0, atol=0)
  assert result is not density and result.ctx["value_form"] == "modal"
  for backend in ("numpy", "gkyl"):
    n, t, b = [field.interpolate() for field in (density, temperature, bmag)]
    if backend == "gkyl":
      n, t, b = [
          d._result(d.grid,
                    gpython.GkylArray.from_numpy(d.values.reshape(-1, 1)))
          for d in (n, t, b)
      ]
    out = coulomb_log(n, n, t, t, b, **CONSTANTS)
    assert out.backend == backend
    assert_allclose(out.values,
                    .5 * np.log1p(288 * np.pi**2 / n.values),
                    rtol=3e-14,
                    atol=3e-14)


@needs_vlasov
@pytest.mark.parametrize("order", [1, 2])
def test_spitzer_exact_affine_density(order):
  density, _, bmag, moms = collision_fields(order)
  result = spitzer_coll_freq(moms, moms, bmag, **CONSTANTS, nufrac=.7)
  # Gkeyll uses the cell-average density only in logLambda. With unit
  # thermal speeds, the remaining multiplier is n/(12*pi^(3/2)).
  mean_n = density.values[:, 0] / np.sqrt(2)
  log_lambda = .5 * np.log1p(288 * np.pi**2 / mean_n)
  expected = density.values * (.7 * log_lambda / (12 * np.pi**1.5))[:, None]
  assert_allclose(result.values, expected, rtol=4e-14, atol=4e-14)
  assert result.values.shape == density.values.shape
  assert result.ctx["num_comps"] == order + 1
  low = gpython.vlasov.spitzer_coll_freq("serendipity",
                                         1,
                                         order,
                                         moms.native,
                                         moms.native,
                                         bmag.native,
                                         **CONSTANTS,
                                         nufrac=.7)
  assert_allclose(low.view(), expected, rtol=4e-14, atol=4e-14)


GK_CASES = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 2, 1), (2, 2, 2),
            (3, 2, 1)]
MOMENTS = [
    "M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp", "ThreeMoments",
    "FourMoments", "HamiltonianMoments"
]


@needs_gk
@pytest.mark.parametrize("cdim,vdim,order", GK_CASES)
@pytest.mark.parametrize("moment", MOMENTS)
def test_gyrokinetic_analytic_moments(cdim, vdim, order, moment):
  stem = f"native_gk_{cdim}x{vdim}v_p{order}"
  f, bmag, phi = [
      pg.load(str(GEN / f"{stem}_{name}.gkyl")) for name in ("f", "bmag", "phi")
  ]
  if vdim == 1 and moment in ("M2perp", "M3perp"):
    with pytest.raises(ValueError):
      mom_gyrokinetic(f, bmag, mass=2., moment=moment)
    return
  original = f.values.copy()
  result = mom_gyrokinetic(f, bmag, mass=2., charge=-.5, phi=phi, moment=moment)
  # F=2+x+v; integrate v^k F using its polynomial antiderivative.
  # In 2v the measure is pi dv dmu (m=2); 2*mu*B/m=3*mu.
  centers = np.meshgrid(*[(e[1:] + e[:-1]) / 2 for e in bmag.grid],
                        indexing="ij")
  x = centers[0]
  nb = bmag.values.shape[-1]

  def integral(k, perpendicular=False):
    a, b = -.5, 2.5
    vk = (b**(k + 1) - a**(k + 1)) / (k + 1)
    vk1 = (b**(k + 2) - a**(k + 2)) / (k + 2)
    measure = 1. if vdim == 1 else np.pi * (1.75 - .25)
    if perpendicular:
      measure *= 3 * (1.75 + .25) / 2
    expected = np.zeros((*x.shape, nb))
    expected[..., 0] = 2**(cdim / 2) * measure * ((2 + x) * vk + vk1)
    expected[..., 1] = 2**(cdim / 2) / np.sqrt(3) * .25 * measure * vk
    return expected

  n, momentum, parallel = [integral(k) for k in range(3)]
  perpendicular = integral(0, True) if vdim == 2 else np.zeros_like(n)
  energy = parallel + perpendicular
  # Native 1v FourMoments is the same three-field layout as ThreeMoments.
  choices = dict(M0=n,
                 M1=momentum,
                 M2=energy,
                 M2par=parallel,
                 M2perp=perpendicular,
                 M3par=integral(3),
                 M3perp=integral(1, True),
                 ThreeMoments=np.concatenate((n, momentum, energy), axis=-1),
                 FourMoments=np.concatenate(
                     (n, momentum, parallel, perpendicular) if vdim == 2 else
                     (n, momentum, parallel),
                     axis=-1),
                 HamiltonianMoments=np.concatenate(
                     (n, 2 * momentum, -n + energy), axis=-1))
  assert_allclose(result.values, choices[moment], rtol=3e-14, atol=3e-13)
  assert result.num_dims == cdim and result.ctx["basis_type"] == "serendipity"
  assert result.ctx["value_form"] == "modal"
  assert_allclose(f.values, original, rtol=0, atol=0)
  for got, want in zip(result.grid, bmag.grid):
    assert_allclose(got, want, rtol=0, atol=0)


@needs_vlasov
@pytest.mark.parametrize("changes", [{
    "mass_self": 0
}, {
    "hbar": -1
}, {
    "charge_other": 0
}])
def test_coulomb_invalid_constants(changes):
  with pytest.raises(ValueError):
    gpython.vlasov.coulomb_log(1, 1, 1, 1, 0, **(CONSTANTS | changes))


@needs_gk
def test_gyrokinetic_rejects_incompatible_inputs():
  stem = "native_gk_1x2v_p1"
  f, b = [pg.load(str(GEN / f"{stem}_{name}.gkyl")) for name in ("f", "bmag")]
  for kwargs in ({
      "moment": "unknown"
  }, {
      "moment": "HamiltonianMoments"
  }, {
      "mass": -1
  }):
    with pytest.raises(ValueError):
      mom_gyrokinetic(f, b, **({"mass": 1.} | kwargs))
  with pytest.raises(ValueError, match="point values"):
    mom_gyrokinetic(f.interpolate(), b, mass=1.)
  f.ctx["mapped_axes"] = (1, )
  with pytest.raises(ValueError, match="mapped"):
    mom_gyrokinetic(f, b, mass=1.)


@needs_gk
@pytest.mark.parametrize("moment", ["M0", "HamiltonianMoments"])
def test_gyrokinetic_generated_cli(moment):
  from postgkyl.cli import PostgkylSession
  session = PostgkylSession()
  stem = "native_gk_1x2v_p1"
  session.load(GEN / f"{stem}_f.gkyl", tag="f")
  session.load(GEN / f"{stem}_bmag.gkyl", tag="B")
  kwargs = {}
  if moment == "HamiltonianMoments":
    session.load(GEN / f"{stem}_phi.gkyl", tag="phi")
    kwargs["phi"] = "phi"
  session.gk_mom_gyrokinetic(distribution="f",
                             bmag="B",
                             mass=2.,
                             charge=-.5,
                             moment=moment,
                             **kwargs)
  result, = session.datasets
  assert result.num_dims == 1
  assert result.values.shape == (3, 6 if kwargs else 2)
  expected_m0 = np.sqrt(2) * 4.5 * np.pi * (3 + np.array([0., .5, 1.]))
  assert_allclose(result.values[:, 0], expected_m0, rtol=3e-14, atol=3e-14)


@needs_vlasov
@pytest.mark.parametrize("command", ["spitzer_coll_freq", "coulomb_log"])
def test_collision_generated_cli(command):
  from postgkyl.cli import PostgkylSession
  session = PostgkylSession()
  session.load(GEN / "collision_bmag_p1.gkyl", tag="B")
  if command == "spitzer_coll_freq":
    session.load(GEN / "collision_moms_p1.gkyl", tag="m")
    session.vm_collisions_spitzer_coll_freq(moms_self="m",
                                            moms_other="m",
                                            bmag="B",
                                            **CONSTANTS)
  else:
    session.load(GEN / "collision_density_p1.gkyl", tag="n")
    session.load(GEN / "collision_temperature_p1.gkyl", tag="T")
    session.vm_collisions_coulomb_log(density_self="n",
                                      density_other="n",
                                      temperature_self="T",
                                      temperature_other="T",
                                      bmag="B",
                                      **CONSTANTS)
  result, = session.datasets
  assert result.values.shape == (3, 2)
  assert np.all(result.interpolate().values > 0)


@pytest.mark.parametrize("module, function, args, kwargs", [
    ("vlasov", "coulomb_log", (1, 1, 1, 1, 0), CONSTANTS),
    ("vlasov", "spitzer_coll_freq",
     ("serendipity", 1, 1, None, None, None), CONSTANTS),
    ("gyrokinetic", "mom_gyrokinetic",
     (1, 1, 1, None, None, [0, 0], [1, 1], [1, 1]), {
         "mass": 1
     }),
])
def test_missing_optional_module_is_actionable(monkeypatch, module, function,
                                               args, kwargs):
  from postgkyl.gpython import _lib
  from types import SimpleNamespace
  lib = SimpleNamespace(module_api_versions=lambda: {"core": 9})
  monkeypatch.setattr(_lib, "require", lambda: lib)
  with pytest.raises(RuntimeError) as error:
    getattr(getattr(gpython, module), function)(*args, **kwargs)
  message = str(error.value)
  assert f"Cannot call {function}" in message
  assert f"requires Gkeyll's {module} module" in message
  assert "Gkeyll was built without that module" in message
  assert f"rebuild Postgkyl with the {module} layer" in message
  assert f"POSTGKYL_GKEYLL_MODULE={module} python -m pip install -e ." in message


@needs_vlasov
def test_coulomb_species_order_is_preserved():
  # Unit masses/densities/charge magnitudes with T_s=1, T_r=4:
  # r_min=1/(30*pi); r_max^2=28/11 in this order, 208/29 in reverse.
  forward = gpython.vlasov.coulomb_log(1, 1, 1, 4, 0, **CONSTANTS)
  reverse = gpython.vlasov.coulomb_log(1, 1, 4, 1, 0, **CONSTANTS)
  assert_allclose(forward,
                  .5 * np.log1p(900 * np.pi**2 * 28 / 11),
                  rtol=2e-15,
                  atol=2e-15)
  assert_allclose(reverse,
                  .5 * np.log1p(900 * np.pi**2 * 208 / 29),
                  rtol=2e-15,
                  atol=2e-15)
