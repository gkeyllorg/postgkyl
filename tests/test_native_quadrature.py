"""Public basis quadrature: regression for the highest 1x1v hybrid mode."""

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
