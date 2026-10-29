"""Independent tensor-Gauss checks of drift rates and physical projections."""
from pathlib import Path
import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl import gpython
from postgkyl.cli.app import cli
from postgkyl.diagnostics.gk import quantities as ff
from postgkyl.diagnostics.gk.load_quantity import load_quantity
from postgkyl.diagnostics.gk.drift_projection import drift_direction
from generate_test_data import drift_projection_reference

pytestmark = pytest.mark.skipif(not gpython.available(),
                                reason="requires Gkeyll")
DATA = Path(__file__).parent / "test_data" / "generated"


def _expected(quantity, selector, ndim=3):
  """Project the analytic physical drift, independently of production helpers."""
  cells, _, _, _, basis, tangents, normals, positions, _, cross, speeds = drift_projection_reference(
      ndim)
  velocity = speeds[quantity][..., None] * cross
  if str(selector) in ("0", "1", "2"):
    target = np.linalg.inv(tangents)[int(selector)]
  elif selector.startswith("n"):
    target = normals[int(selector[1])]
  elif selector == "Z":
    target = np.array([0., 0., 1.])
  else:
    Rmajor = np.hypot(positions[..., 0], positions[..., 1])
    target = np.zeros_like(positions)
    if selector == "R":
      target[...,
             0], target[...,
                        1] = positions[..., 0] / Rmajor, positions[...,
                                                                   1] / Rmajor
    else:
      target[...,
             0], target[...,
                        1] = -positions[..., 1] / Rmajor, positions[...,
                                                                    0] / Rmajor
  samples = np.sum(velocity * target, axis=-1)
  return np.einsum("...q,qm->...m", samples, basis).reshape(*cells, 2**ndim)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("kind", ["cart", "cyl"])
@pytest.mark.parametrize("selector",
                         [0, "0", 1, 2, "n0", "n1", "n2", "R", "Z", "phi"])
@pytest.mark.parametrize("quantity", ["ExB_vel", "gradB_vel", "diamag_vel"])
def test_drift_projections(quantity, selector, kind, ndim):
  """Every coefficient agrees with an independent analytic Gauss projection."""
  prefix = f"drift_{kind}" if ndim == 3 else f"drift_2x_{kind}"

  # Direct fetch gives grad-B an independent constant Tperp and diamagnetic
  # drift its prescribed pressure, bypassing moment-source preference.
  def read(suffix):
    return pg.load(str(DATA / f"{prefix}-{suffix}.gkyl"))

  geometry = {
      "tangents": read("geo_int_dxdz"),
      "normals": read("geo_int_normals"),
      "positions": read("geo_int_mapc2p")
  }
  jac, bmag, bi = [
      read("geo_int_" + s) for s in ("jacobtot_inv", "bmag", "b_i")
  ]
  scalar = read("field_0")
  if quantity == "ExB_vel":
    sources = [jac, bmag, bi, scalar]
  elif quantity == "gradB_vel":
    sources = [jac, bmag, bi, scalar * 0 + 4]
  else:
    sources = [jac, bmag, bi, scalar * 0 + 2, read("ion_M2perp_0")]
  out = getattr(ff, "fetch_" + quantity)(sources,
                                         dir=selector,
                                         drift_geometry=geometry,
                                         charge=1.)
  np.testing.assert_allclose(out.values,
                             _expected(quantity, selector, ndim),
                             rtol=2e-11,
                             atol=2e-13)
  assert out.ctx["value_form"] == "modal"
  if str(selector) not in ("0", "1", "2"):
    assert out.ctx["drift_units"] == "m/s"


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("selector", ["n0", "n1", "n2", "R", "Z", "phi", "0"])
def test_loader_and_cli(selector, ndim):
  """The loader and generated CLI share the physical-selector contract."""
  prefix = "drift_cart" if ndim == 3 else "drift_2x_cart"
  out, = load_quantity("ExB_vel",
                       None,
                       prefix,
                       "0",
                       path=str(DATA),
                       direction=selector)
  np.testing.assert_allclose(out.values,
                             _expected("ExB_vel", selector, ndim),
                             rtol=2e-11,
                             atol=2e-13)
  result = CliRunner().invoke(cli, [
      "gk_load_q", "-q", "ExB_vel", "-n", prefix, "-s", "ion", "-p",
      str(DATA), "-f", "0", "-d", selector, "info"
  ])
  assert result.exit_code == 0, result.output
  if selector != "0":
    assert "m/s" in out.get_label()
  else:
    assert "coordinate units/s" in out.get_label()


def test_invalid_selector_and_missing_geometry(tmp_path):
  """Do not silently fall back to a different projection or units."""
  from postgkyl.diagnostics.gk.drift_projection import load_drift_geometry, project_drift
  for value in (None, -1, 3, "n3", "r", "theta", 1.5):
    with pytest.raises(ValueError, match="direction="):
      drift_direction(value)
  assert load_drift_geometry(str(tmp_path), "absent", "0") == {}
  with pytest.raises(FileNotFoundError, match="geo_int_dxdz"):
    load_drift_geometry(str(tmp_path), "absent", "n0")
  data = pg.load(str(DATA / "gk_moments_p1.gkyl"))
  with pytest.raises(ValueError, match="2x or 3x"):
    project_drift(lambda i: data, "n0", data)


def test_cylindrical_axis_rejected_and_speed_reconstructed():
  """Cylindrical projections preserve speed at quadrature and reject R=0."""
  from postgkyl.diagnostics.gk.drift_projection import _cylindrical_unit
  prefix = DATA / "drift_cart"
  positions = pg.load(str(prefix) + "-geo_int_mapc2p.gkyl")
  with pytest.raises(ValueError, match="R > 0"):
    _cylindrical_unit(positions * 0, "R")
  _, _, _, _, basis, _, _, _, _, cross, speeds = drift_projection_reference()
  fields = [
      load_quantity("ExB_vel",
                    "ion",
                    "drift_cart",
                    "0",
                    path=str(DATA),
                    direction=d)[0] for d in ("R", "Z", "phi")
  ]
  samples = [np.einsum("...m,qm->...q", f.values, basis) for f in fields]
  np.testing.assert_allclose(sum(v * v for v in samples),
                             speeds["ExB_vel"]**2 * np.dot(cross, cross),
                             rtol=2e-11,
                             atol=2e-13)


def test_read_options_selector_and_tokamak_units():
  """Legacy dir= input and TOKAMAK coordinate-rate labels remain usable."""
  result = CliRunner().invoke(cli, [
      "gk_load_q", "-q", "ExB_vel", "-n", "drift_cart", "-s", "ion", "-p",
      str(DATA), "-f", "0", "-r", "dir=n0", "info"
  ])
  assert result.exit_code == 0, result.output
  for selector, unit in (("0", "Wb/(rad s)"), ("1", "rad/s"), ("2", "rad/s")):
    out, = load_quantity("ExB_vel",
                         "ion",
                         "drift_cyl",
                         "0",
                         path=str(DATA),
                         direction=selector)
    assert out.ctx["drift_units"] == unit


@pytest.mark.parametrize("selector", ["n0", "n1", "n2", "R", "Z", "phi"])
def test_2x_cylindrical_loader(selector):
  """Load full cylindrical position vectors on the reduced 2x grid."""
  out, = load_quantity("ExB_vel",
                       "ion",
                       "drift_2x_cyl",
                       "0",
                       path=str(DATA),
                       direction=selector)
  np.testing.assert_allclose(out.values,
                             _expected("ExB_vel", selector, 2),
                             rtol=2e-11,
                             atol=2e-13)
  assert out.num_dims == 2
  assert out.ctx["drift_units"] == "m/s"
