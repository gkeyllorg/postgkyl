"""Independent numerical checks for value-unit conversion."""

from pathlib import Path

from click.testing import CliRunner
import numpy as np
import pytest
from scipy.constants import elementary_charge

import postgkyl as pg
from postgkyl.cli.app import cli
from postgkyl.operations.units import available_units


@pytest.mark.parametrize("source,dest,factor", [
    ("m^-3", "cm^-3", 1e-6),
    ("J", "eV", 1 / elementary_charge),
    ("J", "keV", 1 / (1e3 * elementary_charge)),
    ("eV", "keV", 1e-3),
    ("m/s", "km/s", 1e-3),
    ("V", "kV", 1e-3),
    ("V/m", "kV/m", 1e-3),
    ("T", "G", 1e4),
])
def test_point_value_pairs_and_reverse(source, dest, factor):
  data = pg.GData()
  data.push([np.array([0., 1., 2.])], np.array([[2., 3.], [5., 7.]]))
  data.set_label(f"quantity ({source})")
  converted = data.units(source=source, dest=dest)
  np.testing.assert_allclose(converted.values,
                             data.values * factor,
                             rtol=1e-14,
                             atol=0)
  np.testing.assert_array_equal(converted.grid[0], data.grid[0])
  assert converted.ctx["value_units"] == dest
  assert converted.get_label() == f"quantity ({dest})"
  restored = pg.units(converted, source=dest, dest=source)
  np.testing.assert_allclose(restored.values, data.values, rtol=1e-14, atol=0)
  np.testing.assert_array_equal(data.values, [[2., 3.], [5., 7.]])


def test_density_latex_label_and_explicit_label():
  data = pg.GData()
  data.push([np.array([0., 1.])], np.array([[1e6]]))
  data.set_label(r"$M_0$ (m$^{-3}$)")
  out = data.units(source="m^-3", dest="cm^-3")
  assert out.get_label() == r"$M_0$ (cm$^{-3}$)"
  assert out.units(source="cm^-3", dest="m^-3").get_label() == data.get_label()
  assert data.units(source="m^-3", dest="cm^-3",
                    label="my density").get_label() == "my density"


def test_invalid_units_do_not_change_inplace_data():
  data = pg.GData()
  data.push([np.array([0., 1.])], np.array([[2.]]))
  for source, dest in (("J", "V"), ("bad", "J"), ("J", "bad")):
    with pytest.raises(ValueError):
      data.units(source=source, dest=dest, inplace=True)
  np.testing.assert_array_equal(data.values, [[2.]])


def test_inplace_preserves_unmatched_label_and_sets_units():
  data = pg.GData()
  data.push([np.array([0., 1.])], np.array([[3.]]))
  data.set_label("electric field")
  result = data.units(source="V/m", dest="kV/m", inplace=True)
  assert result is data
  np.testing.assert_array_equal(data.values, [[0.003]])
  assert data.get_label() == "electric field"
  assert data.ctx["value_units"] == "kV/m"


def test_same_unit_preserves_label_and_values_with_new_tag():
  data = pg.GData()
  data.push([np.array([0., 1.])], np.array([[4.]]))
  data.set_label("potential (V)")
  result = data.units(source="V", dest="V", tag="converted")
  np.testing.assert_array_equal(result.values, [[4.]])
  assert result.get_label() == "potential (V)"
  assert result.tag == "converted"
  assert data.ctx.get("value_units") is None


def test_avail_catalog_comes_from_supported_families():
  rows = available_units()
  assert [name for name, _ in rows] == [
      "m^-3 / cm^-3", "J / eV / keV", "m/s / km/s", "V / kV", "V/m / kV/m",
      "T / G"
  ]
  command = cli.get_command(None, "units")
  option = next(p for p in command.params if p.name == "avail")
  assert option.is_eager
  assert option.is_flag


@pytest.mark.parametrize("flag", ["--avail", "-a"])
def test_avail_cli_needs_no_dataset(flag, monkeypatch):
  from postgkyl.cli import compiler

  def no_dispatch(*args, **kwargs):
    pytest.fail("avail must exit before pipeline execution")

  monkeypatch.setattr(compiler, "execute_model", no_dispatch)
  result = CliRunner().invoke(cli, ["units", flag])
  assert result.exit_code == 0, result.output
  assert "Supported unit translations:" in result.output
  for family, _ in available_units():
    assert family in result.output
  assert "any pair, either direction" not in result.output


def test_units_help_points_to_catalog_without_listing_families():
  result = CliRunner().invoke(cli, ["units", "--help"])
  assert result.exit_code == 0, result.output
  assert "--avail" in result.output
  assert "m^-3 / cm^-3" not in result.output


@pytest.mark.skipif(not pg.gpython.available(), reason="requires native Gkeyll")
def test_native_modal_scaling_and_cli():
  path = Path(__file__).parent / "test_data/generated/fsimple.gkyl"
  data = pg.load(path)
  scaled = data.units(source="T", dest="G")
  assert scaled.backend == data.backend == "gkyl"
  assert scaled.ctx["value_form"] == data.ctx["value_form"] == "modal"
  np.testing.assert_allclose(scaled.values,
                             data.values * 1e4,
                             rtol=1e-14,
                             atol=0)
  result = CliRunner().invoke(
      cli, [str(path), "units", "--source", "T", "--dest", "G", "print"])
  assert result.exit_code == 0, result.output
  expected = np.array2string(scaled.values.squeeze(), precision=16)
  assert result.output.strip() == expected
