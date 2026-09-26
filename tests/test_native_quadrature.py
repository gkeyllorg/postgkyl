"""Public quadrature representation and exact native integration contracts."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl import gpython, io
from postgkyl.cli.app import cli

pytestmark = pytest.mark.skipif(not gpython.available(),
                                reason="compiled Gkeyll unavailable")
GENERATED = Path(__file__).parent / "test_data" / "generated"
HYBRID = GENERATED / "fsimple_hyb.gkyl"


@pytest.mark.parametrize("representation", ["modal", "quad", "gauss"])
@pytest.mark.parametrize("stem", [
    "fsimple", "1d_ms_p1", "1d_ms_p2", "2d_ms_p1", "2d_ms_p2", "3d_ms_p1",
    "gk_moments_p1", "gk_drift_3d_p1"
])
def test_integrate_generated_field_is_exact(stem, representation):
  data = pg.load(GENERATED / f"{stem}.gkyl")
  # Only the constant orthonormal mode has a nonzero cell integral.
  cell_volume = np.prod(
      (data.ctx["upper"] - data.ctx["lower"]) / data.num_cells)
  nb = gpython.basis.num_basis(data.ctx["basis_type"], data.num_dims,
                               data.ctx["poly_order"])
  means = data.values.reshape(-1, data.num_comps // nb, nb)[..., 0]
  expected = means.sum(axis=0) * cell_volume / 2**(data.num_dims / 2)
  if representation != "modal":
    data = data.represent(to="quad",
                          num_quad=3 if representation == "gauss" else None)
  before = data.values.copy()
  np.testing.assert_allclose(data.integrate(), expected, rtol=2e-12, atol=2e-12)
  np.testing.assert_array_equal(data.values, before)


@pytest.mark.parametrize("stem", ["1d_ms_p2", "2d_ms_p2", "3d_ms_p1"])
@pytest.mark.parametrize("representation", ["modal", "quad", "gauss"])
def test_integral_of_square_matches_modal_norm(stem, representation):
  data = pg.load(GENERATED / f"{stem}.gkyl")
  # Orthonormality gives int(f^2) = sum(c_k^2) times the cell Jacobian.
  jacobian = np.prod(
      (data.ctx["upper"] - data.ctx["lower"]) / (2 * data.num_cells))
  expected = np.sum(data.values**2) * jacobian
  if representation == "modal":
    result = data.integrate(op="sq")
  else:
    quad = data.represent(to="quad",
                          num_quad=3 if representation == "gauss" else None)
    result = (quad**2).integrate()
  np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("num_quad", [None, 3])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("axis", [0, 1, (0, 2)])
def test_partial_quad_integration_requires_explicit_projection(
    num_quad, inplace, axis):
  modal = pg.load(GENERATED / "3d_ms_p1.gkyl")
  # Squaring gives a retained quadratic dependence outside the p1 basis.
  # A silent projection would change the partial integral's point values.
  data = modal.represent(to="quad", num_quad=num_quad)
  data = data**2
  before = data.values.copy()
  message = "no direct partial quadrature integration kernel"
  with pytest.warns(RuntimeWarning, match=message):
    with pytest.raises(NotImplementedError, match=message):
      data.integrate(axis, inplace=inplace)
  np.testing.assert_array_equal(data.values, before)
  assert data.ctx["value_form"] == "quad"
  assert data.num_dims == 3
  reduced = data.represent(to="modal").integrate(axis)
  np.testing.assert_allclose(reduced.integrate(),
                             data.integrate(),
                             rtol=2e-12,
                             atol=2e-12)


@pytest.mark.parametrize("stem", ["2d_mt_p2", "fsimple_hyb"])
@pytest.mark.parametrize("value_form", ["modal", "quad"])
def test_integrate_warns_and_rejects_unsupported_native_basis(stem, value_form):
  data = pg.load(GENERATED / f"{stem}.gkyl").represent(to=value_form)
  message = "gkyl_array_integrate kernels.*serendipity p1-p2"
  with pytest.warns(RuntimeWarning, match=message):
    with pytest.raises(NotImplementedError, match=message):
      data.integrate()


def test_quad_integration_uses_the_same_rule_after_a_python_reader_load(
    tmp_path, monkeypatch):
  quad = pg.load(GENERATED / "fsimple.gkyl").represent(to="quad")
  path = quad.save(str(tmp_path / "quad.gkyl"))
  monkeypatch.setattr(io, "_READERS", {"test": io.GkylReader})
  data = pg.load(path)
  assert data.backend == "numpy"
  assert data.ctx["value_form"] == "quad"
  # f=4 on [-1,1]. Storage is independent of the retained quadrature rule.
  np.testing.assert_allclose(data.integrate(), 8.0, rtol=0, atol=2e-14)


@pytest.mark.parametrize("representation", [[], ["represent", "-t", "quad"]],
                         ids=["modal", "quad"])
def test_cli_integrates_constant_field_exactly(representation):
  result = CliRunner().invoke(
      cli, [str(GENERATED / "fsimple.gkyl"), *representation, "integrate"])
  assert result.exit_code == 0, result.output
  # f=4 on [-1, 1], so both command chains must print 8.
  np.testing.assert_allclose(float(result.output.strip()),
                             8.0,
                             rtol=2e-12,
                             atol=2e-12)


def test_highest_hybrid_mode_survives_default_quad_roundtrip():
  data = pg.load(HYBRID)
  quad = data.represent(to="quad")
  assert quad.num_comps == 6
  assert quad.ctx["num_quad"] == 6
  assert quad.ctx["quad_rule"] == "gkeyll"
  expected = [
      -1 / np.sqrt(5),
      np.sqrt(5) / 4, -1 / np.sqrt(5), 1 / np.sqrt(5), -np.sqrt(5) / 4,
      1 / np.sqrt(5)
  ]
  np.testing.assert_allclose(quad.values.ravel(), expected, atol=1e-14)
  np.testing.assert_allclose(quad.represent(to="modal").values,
                             data.values,
                             atol=1e-14)
  np.testing.assert_allclose(
      data.represent(to="nodal").represent(to="quad").values,
      quad.values,
      atol=1e-14)
  np.testing.assert_allclose(data.apply(lambda x: x).values,
                             data.values,
                             atol=1e-14)


@pytest.mark.parametrize("reader", [io.GkylReader, io.GkylCReader])
def test_basis_quadrature_saved_rule_roundtrips(tmp_path, monkeypatch, reader):
  data = pg.load(HYBRID)
  quad = data.represent(to="quad")
  path = quad.save(str(tmp_path / "quad.gkyl"))
  monkeypatch.setattr(io, "_READERS", {"test": reader})
  restored = pg.load(path)
  assert restored.ctx["quad_rule"] == "gkeyll"
  assert restored.ctx["num_quad"] == 6
  np.testing.assert_array_equal(restored.values, quad.values)
  if restored.backend == "gkyl":
    np.testing.assert_allclose(restored.represent(to="modal").values,
                               data.values,
                               atol=1e-14)


@pytest.mark.parametrize("order", [None, 3])
def test_repeated_represent_quad_preserves_rule_and_values(order):
  quad = pg.load(HYBRID).represent(to="quad", num_quad=order)
  again = quad.represent(to="quad")
  assert again.ctx["quad_rule"] == quad.ctx["quad_rule"]
  assert again.ctx["num_quad"] == quad.ctx["num_quad"]
  np.testing.assert_array_equal(again.values, quad.values)


def test_explicit_quad_order_resamples_existing_basis_quadrature():
  data = pg.load(HYBRID)
  resampled = data.represent(to="quad").represent(to="quad", num_quad=3)
  assert resampled.num_comps == 9
  assert resampled.ctx["quad_rule"] == "gauss"
  np.testing.assert_allclose(resampled.represent(to="modal").values,
                             data.values,
                             atol=1e-14)


def test_legacy_saved_quad_metadata_still_selects_uniform_gauss_rule():
  quad = pg.load(HYBRID).represent(to="quad", num_quad=3)
  del quad.ctx["quad_rule"]
  np.testing.assert_allclose(quad.represent(to="modal").values,
                             pg.load(HYBRID).values,
                             atol=1e-14)


@pytest.mark.parametrize("operation", [lambda a, b: a + b, np.multiply])
def test_pointwise_arithmetic_rejects_different_quadrature_orderings(operation):
  data = pg.load(GENERATED / "2d_ms_p1.gkyl")
  with pytest.raises(ValueError, match="different quadrature rules"):
    operation(data.represent(to="quad"), data.represent(to="quad", num_quad=2))


def test_cli_hybrid_quad_modal_info_and_print():
  result = CliRunner().invoke(cli, [
      str(HYBRID), "info", "represent", "-t", "quad", "info", "represent", "-t",
      "modal", "info", "pr"
  ])
  assert result.exit_code == 0, result.output
  assert result.output.count("Number of components: 6") == 3
  assert "quad, num_quad=6, Gkeyll nodes" in result.output
  assert result.output.count("DG: hybrid p1 (modal)") == 2
  printed = result.output[result.output.rfind("["):].strip("[] \n")
  np.testing.assert_allclose(np.fromstring(printed, sep=" "),
                             [0, 0, 0, 0, 0, 1],
                             atol=1e-14)
