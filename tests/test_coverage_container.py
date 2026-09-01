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
F1 = os.path.join(DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
    reason="no compiled Gkeyll (libg0core.so) found")


# --------------------------------------------------------------------- gdata
@needs_gkeyll
def test_mul_div_explicit_aliases_match_operators():
  a, b = pg.load(F1), pg.load(F1)
  np.testing.assert_allclose(a.mul(b).values, (a * b).values)
  a2, b2 = pg.load(F1), pg.load(F1)
  np.testing.assert_allclose(a2.div(b2).values, (a2 / b2).values)
# end


# --------------------------------------------------------------------- tags
def test_tag_setter_ignores_falsy_value():
  d = GDataState()
  d.tag = "custom"
  assert d.tag == "custom"
  d.tag = ""  # falsy: must not clobber the existing tag
  assert d.tag == "custom"
# end


def test_label_getter_setter():
  d = GDataState()
  assert d.label == ""
  d.label = "raw-label"
  assert d.label == "raw-label"
# end


# ---------------------------------------------------------------- shape info
def test_num_cells_falls_back_to_values_shape_then_empty():
  d = GDataState()
  assert d.num_cells.size == 0  # no ctx, no values

  d.push([np.linspace(0.0, 1.0, 4)], np.zeros((3, 2)))
  del d.ctx["cells"]
  assert np.array_equal(d.num_cells, [3])
# end


def test_num_comps_falls_back_to_values_shape_then_zero():
  d = GDataState()
  assert d.num_comps == 0  # no ctx, no values

  d.push([np.linspace(0.0, 1.0, 4)], np.zeros((3, 2)))
  del d.ctx["num_comps"]
  assert d.num_comps == 2
# end


@needs_gkeyll
def test_num_comps_falls_back_for_gkyl_backed_values():
  d = pg.load(F1)
  del d.ctx["num_comps"]
  assert d.num_comps == d.native.ncomp
# end


def test_num_dims_falls_back_to_values_ndim_then_zero():
  d = GDataState()
  assert d.num_dims == 0  # no ctx, no values

  d.push([np.linspace(0.0, 1.0, 4)], np.zeros((3, 2)))
  del d.ctx["cells"]
  assert d.num_dims == 1
# end


def test_bounds_falls_back_to_grid_then_none():
  d = GDataState()
  assert d.bounds == (None, None)

  d.push([np.linspace(0.0, 2.0, 4)], np.zeros((3, 2)))
  del d.ctx["lower"]
  del d.ctx["upper"]
  lo, up = d.bounds
  np.testing.assert_allclose(lo, [0.0])
  np.testing.assert_allclose(up, [2.0])
# end


# ------------------------------------------------ getitem / setitem / copy
def test_getitem_raises_when_empty():
  d = GDataState()
  with pytest.raises(ValueError):
    d[0]
  # end
# end


def test_getitem_uses_numpy_axis_order_when_loaded():
  d = GDataState()
  d.push([np.linspace(0.0, 1.0, 4)], np.arange(6, dtype=float).reshape(3, 2))
  np.testing.assert_allclose(d[:, 1], [1.0, 3.0, 5.0])
  np.testing.assert_allclose(d[1], [2.0, 3.0])
# end


def test_setitem_uses_numpy_axis_order_when_loaded():
  d = GDataState()
  d.push([np.linspace(0.0, 1.0, 4)], np.arange(12, dtype=float).reshape(3, 4))
  d[:, 2:4] *= 5
  np.testing.assert_allclose(d.values, [
      [0.0, 1.0, 10.0, 15.0],
      [4.0, 5.0, 30.0, 35.0],
      [8.0, 9.0, 50.0, 55.0],
  ])
# end


def test_setitem_raises_when_empty():
  d = GDataState()
  with pytest.raises(ValueError, match="cannot assign"):
    d[0] = 1.0
  # end
# end


def test_copy_with_data_deep_copies_numpy_backend():
  d = GDataState()
  d.push([np.linspace(0.0, 1.0, 4)], np.ones((3, 2)))
  c = d.clone(data=True)
  c.values[0, 0] = 99.0
  assert d.values[0, 0] == 1.0
  assert c.grid[0] is not d.grid[0]
# end


@needs_gkeyll
def test_copy_with_data_deep_copies_gkyl_backend():
  d = pg.load(F1)
  c = d.clone(data=True)
  assert c.native is not d.native
  np.testing.assert_allclose(c.values, d.values)
# end


@needs_gkeyll
def test_result_applies_explicit_tag_and_label():
  d = pg.load(F1).interpolate(tag="custom-tag", label="custom-label")
  assert d.tag == "custom-tag"
  assert d.label == "custom-label"
# end


def test_require_operable_raises_on_empty_dataset():
  d = GDataState()
  with pytest.raises(ValueError):
    d._require_operable()
  # end
# end


# -------------------------------------------------------------------- info
@needs_gkeyll
def test_info_reports_nodal_and_quad_representation():
  a = pg.load(F1)
  assert "nodal" in a.to_nodal().info()
  assert "quad" in a.to_quad().info()
# end


# ---------------------------------------------------------- repr/str/summary
def test_repr_and_str_on_empty_dataset():
  d = GDataState()
  assert "empty" in repr(d)
  assert repr(d) == str(d)
# end


def test_repr_and_str_on_loaded_modal_dataset():
  d = pg.load(F1)
  r = repr(d)
  assert "comp" in r and "tag" in r
  s = str(d)
  assert s.startswith(r)
  assert "modal" in r or "gkyl-native" in r
# end


def test_repr_on_interpolated_dataset():
  d = pg.load(F1).interpolate()
  r = repr(d)
  assert "interpolate" in r
# end


@needs_gkeyll
def test_repr_on_nodal_and_quad_datasets():
  a = pg.load(F1)
  assert "nodal" in repr(a.to_nodal())
  assert "quad" in repr(a.to_quad())
# end


# ------------------------------------------------------------- collections
def test_flatten_datasets_passes_through_non_dataset_items():
  out = flatten_datasets([1, [2, 3], "x"])
  assert out == [1, 2, 3, "x"]
# end


# ------------------------------------------------------------------ cli app
def test_cli_hidden_alias_pl_resolves_to_plot(tmp_path):
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  out = tmp_path / "alias.png"
  result = CliRunner().invoke(cli, [
      "--batch-mode", F1, "interp", "sel", "--comp", "0", "pl", "--saveas", str(out)])
  assert result.exit_code == 0, result.output
  assert out.exists()
# end


def test_cli_ambiguous_abbreviation_fails():
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  result = CliRunner().invoke(cli, [F1, "in"])  # "in" prefixes both info and interpolate
  assert result.exit_code != 0
  assert "Ambiguous command" in result.output
# end


def test_cli_unknown_token_is_neither_command_nor_file():
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  result = CliRunner().invoke(cli, ["not-a-command-or-file-xyz"])
  assert result.exit_code != 0
  assert "is not a command name nor a data file" in result.output
# end


def test_cli_plot_without_datasets_raises_usage_error():
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  result = CliRunner().invoke(cli, ["plot"])
  assert result.exit_code != 0
  assert "no datasets to plot" in result.output
# end


def test_cli_plot_batch_mode_default_save_path():
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  runner = CliRunner()
  with runner.isolated_filesystem():
    result = runner.invoke(cli, [
        "--batch-mode", "--saveframes-prefix", "myrun",
        F1, "interp", "sel", "--comp", "0", "plot"])
    assert result.exit_code == 0, result.output
    # main's batch-mode file name is "<prefix>_<dataset index>.png".
    assert os.path.exists("myrun_0.png")
  # end
# end


def test_cli_module_entry_point_runs_as_script(monkeypatch, capsys):
  """Exercise ``if __name__ == "__main__": cli()`` in-process (so it's
  visible to coverage), rather than via subprocess."""
  import runpy
  monkeypatch.setattr(sys, "argv", ["pgkyl", "--help"])
  app_path = os.path.join(SRC, "postgkyl", "cli", "app.py")
  with pytest.raises(SystemExit) as exc:
    runpy.run_path(app_path, run_name="__main__")
  # end
  assert exc.value.code == 0
  assert "Postprocessing and plotting tool" in capsys.readouterr().out
# end


def test_dataspace_is_iterable():
  from postgkyl.cli.state import DataSpace
  ds = DataSpace(datasets=[1, 2, 3])
  assert list(ds) == [1, 2, 3]
# end


def test_cli_save_command(tmp_path):
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  out = tmp_path / "written.txt"
  result = CliRunner().invoke(cli, [
      F1, "interp", "sel", "--comp", "0", "save", "-o", str(out), "-f", "txt"])
  assert result.exit_code == 0, result.output
  assert out.exists()
  assert "wrote" in result.output
# end
