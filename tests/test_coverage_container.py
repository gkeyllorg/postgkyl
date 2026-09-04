"""Coverage-completing tests for gdata/gdata, gdatastate/state, gdatastate/collection, cli/*.

These target branches the golden-path tests in test_postgkyl.py don't reach:
state readers on empty/bare containers, the modal .mul()/.div() aliases, the
CLI's abbreviation/ambiguity/fail paths, and the write/plot command edges.

Run:  PYTHONPATH=src pytest tests/test_coverage_container.py -v
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

import matplotlib

matplotlib.use("Agg")

import postgkyl as pg  # noqa: E402
from postgkyl import gpython  # noqa: E402
from postgkyl.gdatastate.gdatastate import GDataState  # noqa: E402
from postgkyl.gdatastate.collection import flatten_datasets  # noqa: E402

DATA = os.path.join(ROOT, "tests", "test_data")
F1 = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


# --------------------------------------------------------------------- gdata
@needs_gkeyll
def test_mul_div_explicit_aliases_match_operators():
  a, b = pg.load(F1), pg.load(F1)
  np.testing.assert_allclose(a.mul(b).values, (a * b).values)
  a2, b2 = pg.load(F1), pg.load(F1)
  np.testing.assert_allclose(a2.div(b2).values, (a2 / b2).values)


# --------------------------------------------------------------------- tags
def test_tag_setter_ignores_falsy_value():
  d = GDataState()
  d.tag = "custom"
  assert d.tag == "custom"
  d.tag = ""  # falsy: must not clobber the existing tag
  assert d.tag == "custom"


def test_label_getter_setter():
  d = GDataState()
  assert d.label == ""
  d.label = "raw-label"
  assert d.label == "raw-label"


# ---------------------------------------------------------------- shape info
def test_num_cells_falls_back_to_values_shape_then_empty():
  d = GDataState()
  assert d.num_cells.size == 0  # no ctx, no values

  d.push([np.linspace(0.0, 1.0, 4)], np.zeros((3, 2)))
  del d.ctx["cells"]
  assert np.array_equal(d.num_cells, [3])


def test_num_comps_falls_back_to_values_shape_then_zero():
  d = GDataState()
  assert d.num_comps == 0  # no ctx, no values

  d.push([np.linspace(0.0, 1.0, 4)], np.zeros((3, 2)))
  del d.ctx["num_comps"]
  assert d.num_comps == 2


@needs_gkeyll
def test_num_comps_falls_back_for_gkyl_backed_values():
  d = pg.load(F1)
  del d.ctx["num_comps"]
  assert d.num_comps == d.native.ncomp


def test_num_dims_falls_back_to_values_ndim_then_zero():
  d = GDataState()
  assert d.num_dims == 0  # no ctx, no values

  d.push([np.linspace(0.0, 1.0, 4)], np.zeros((3, 2)))
  del d.ctx["cells"]
  assert d.num_dims == 1


def test_bounds_falls_back_to_grid_then_none():
  d = GDataState()
  assert d.bounds == (None, None)

  d.push([np.linspace(0.0, 2.0, 4)], np.zeros((3, 2)))
  del d.ctx["lower"]
  del d.ctx["upper"]
  lo, up = d.bounds
  np.testing.assert_allclose(lo, [0.0])
  np.testing.assert_allclose(up, [2.0])


# ------------------------------------------------ getitem / setitem / copy
def test_getitem_raises_when_empty():
  d = GDataState()
  with pytest.raises(ValueError):
    d[0]


def test_getitem_uses_numpy_axis_order_when_loaded():
  d = GDataState()
  d.push([np.linspace(0.0, 1.0, 4)], np.arange(6, dtype=float).reshape(3, 2))
  np.testing.assert_allclose(d[:, 1], [1.0, 3.0, 5.0])
  np.testing.assert_allclose(d[1], [2.0, 3.0])


def test_setitem_uses_numpy_axis_order_when_loaded():
  d = GDataState()
  d.push([np.linspace(0.0, 1.0, 4)], np.arange(12, dtype=float).reshape(3, 4))
  d[:, 2:4] *= 5
  np.testing.assert_allclose(d.values, [
      [0.0, 1.0, 10.0, 15.0],
      [4.0, 5.0, 30.0, 35.0],
      [8.0, 9.0, 50.0, 55.0],
  ])


def test_setitem_raises_when_empty():
  d = GDataState()
  with pytest.raises(ValueError, match="cannot assign"):
    d[0] = 1.0


@needs_gkeyll
def test_setitem_rejects_native_storage():
  d = pg.load(F1)
  with pytest.raises(ValueError, match="native Gkeyll storage"):
    d[0] = 1.0


def test_copy_with_data_deep_copies_numpy_backend():
  d = GDataState()
  d.push([np.linspace(0.0, 1.0, 4)], np.ones((3, 2)))
  c = d.clone()
  c.values[0, 0] = 99.0
  assert d.values[0, 0] == 1.0
  assert c.grid[0] is not d.grid[0]


@needs_gkeyll
def test_copy_with_data_deep_copies_gkyl_backend():
  d = pg.load(F1)
  c = d.clone()
  assert c.native is not d.native
  np.testing.assert_allclose(c.values, d.values)


@needs_gkeyll
def test_result_applies_explicit_tag_and_label():
  d = pg.load(F1).interpolate(tag="custom-tag", label="custom-label")
  assert d.tag == "custom-tag"
  assert d.label == "custom-label"


def test_require_operable_raises_on_empty_dataset():
  d = GDataState()
  with pytest.raises(ValueError):
    d._require_operable()


# -------------------------------------------------------------------- info
@needs_gkeyll
def test_info_reports_nodal_and_quad_representation():
  a = pg.load(F1)
  assert "nodal" in a.to_nodal().info()
  assert "quad" in a.to_quad().info()


def test_output_identity_handles_an_unrecognized_name(monkeypatch):
  monkeypatch.setattr("postgkyl.gdatastate.gdatastate.io.parse_output_name",
                      lambda _path: None)
  d = GDataState()
  d._file_name = "unrecognized"
  d._stamp_output_name()
  assert d.output_name is None
  assert "sim" not in d.ctx


def test_info_reports_all_optional_metadata(capsys):
  d = GDataState(
      ctx={
          "time": 1.5,
          "frame": 3,
          "block": 2,
          "sim": "demo",
          "basis_type": "serendipity",
          "poly_order": None,
          "value_form": "quad",
          "num_quad": 3,
          "changeset": "abc123",
          "builddate": "today",
          "geometry_type": "tokamak",
          "geqdsk_sign_convention": -1,
          "mass": 2.0,
          "charge": -1.0,
          "gas_gamma": 5.0 / 3.0,
          "vdim": 2,
          "custom_metadata": "kept",
      })
  d.push([np.linspace(0.0, 1.0, 3)], np.arange(4, dtype=float).reshape(2, 2))

  out = d.info(no_header=True)

  assert "GEQDSK sign convention: -1" in out
  assert "Adiabatic index" in out
  assert "custom_metadata: kept" in out
  assert "default#0" not in out
  assert capsys.readouterr().out == out + "\n"


@pytest.mark.parametrize("ctx", [{
    "builddate": "today"
}, {
    "changeset": "abc123"
}, {
    "geqdsk_sign_convention": 1
}, {
    "mass": 1.0
}, {
    "charge": -1.0
}])
def test_info_reports_independent_optional_metadata(ctx):
  d = GDataState(ctx=ctx)
  d.push([np.linspace(0.0, 1.0, 3)], np.ones((2, 1)))
  assert d.info()


def test_info_handles_values_without_a_grid_and_a_grid_without_values():
  values_only = GDataState()
  values_only.values = np.ones((2, 1))
  assert "Maximum" in values_only.info(no_header=True)
  assert "Grid:" not in values_only.info(no_header=True)

  grid_only = GDataState()
  grid_only.grid = [np.linspace(0.0, 1.0, 3)]
  assert "Grid:" in grid_only.info(no_header=True)
  assert "Maximum" not in grid_only.info(no_header=True)


# ---------------------------------------------------------- repr/str/summary
def test_repr_and_str_on_empty_dataset():
  d = GDataState()
  assert "empty" in repr(d)
  assert repr(d) == str(d)


def test_repr_and_str_on_loaded_modal_dataset():
  d = pg.load(F1)
  r = repr(d)
  assert "comp" in r and "tag" in r
  s = str(d)
  assert s.startswith(r)
  assert "modal" in r or "gkyl-native" in r


def test_repr_on_interpolated_dataset():
  d = pg.load(F1).interpolate()
  r = repr(d)
  assert "interpolate" in r


def test_repr_handles_values_without_grid_or_basis_metadata():
  d = GDataState()
  d.values = np.ones((2, 1))
  assert "[" not in repr(d)


def test_repr_handles_basis_without_poly_order():
  d = GDataState(ctx={"basis_type": "serendipity"})
  d.values = np.ones((2, 1))
  assert "serendipity" in repr(d)
  assert " p" not in repr(d)


@needs_gkeyll
def test_repr_on_nodal_and_quad_datasets():
  a = pg.load(F1)
  assert "nodal" in repr(a.to_nodal())
  assert "quad" in repr(a.to_quad())


# ------------------------------------------------------------- collections
def test_flatten_datasets_passes_through_non_dataset_items():
  out = flatten_datasets([1, [2, 3], "x"])
  assert out == [1, 2, 3, "x"]


# ------------------------------------------------------------------ cli app
def test_cli_hidden_alias_pl_resolves_to_plot(tmp_path):
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  out = tmp_path / "alias.png"
  result = CliRunner().invoke(cli, [
      F1, "interp", "sel", "--comp", "0", "pl", "--no_show", "--saveas",
      str(out)
  ])
  assert result.exit_code == 0, result.output
  assert out.exists()


def test_cli_ambiguous_abbreviation_fails():
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  result = CliRunner().invoke(
      cli, [F1, "in"])  # "in" prefixes both info and interpolate
  assert result.exit_code != 0
  assert "Ambiguous command" in result.output


def test_cli_unknown_token_is_neither_command_nor_file():
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  result = CliRunner().invoke(cli, ["not-a-command-or-file-xyz"])
  assert result.exit_code != 0
  assert "No such command 'not-a-command-or-file-xyz'" in result.output


def test_cli_plot_without_datasets_raises_usage_error():
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  result = CliRunner().invoke(cli, ["plot"])
  assert result.exit_code != 0
  assert "no datasets selected" in result.output


def test_cli_has_no_manual_batch_mode(tmp_path, monkeypatch):
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  monkeypatch.chdir(tmp_path)
  runner = CliRunner()
  result = runner.invoke(cli, ["--batch_mode", F1, "info"])
  assert result.exit_code != 0
  assert "No such option '--batch_mode'" in result.output


def test_cli_module_entry_point_runs_as_script(monkeypatch, capsys):
  """Exercise ``if __name__ == "__main__": cli()`` in-process (so it's
  visible to coverage), rather than via subprocess."""
  import runpy
  monkeypatch.setattr(sys, "argv", ["pgkyl", "--help"])
  app_path = os.path.join(SRC, "postgkyl", "cli", "app.py")
  with pytest.raises(SystemExit) as exc:
    runpy.run_path(app_path, run_name="__main__")
  assert exc.value.code == 0
  assert "Postprocessing and plotting tool" in capsys.readouterr().out


def test_dataspace_is_iterable():
  from postgkyl.cli.state import DataSpace
  ds = DataSpace(datasets=[1, 2, 3])
  assert list(ds) == [1, 2, 3]


def test_cli_save_command(tmp_path):
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  out = tmp_path / "written.txt"
  result = CliRunner().invoke(cli, [
      F1, "interp", "sel", "--comp", "0", "save", "--out_name",
      str(out), "--extension", "txt"
  ])
  assert result.exit_code == 0, result.output
  assert out.exists()
  assert str(out) in result.output
