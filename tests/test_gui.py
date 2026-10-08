"""Tests for the notebook GUI: the processing chain in
``postgkyl.gui.pipeline``, the launcher, and (with marimo installed) a
headless run of the notebook.

The chain is checked against the public API it stands for: running a chain
must give the same datasets as calling the verbs directly, and the Python
script it renders must reproduce them.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import shlex

from click.testing import CliRunner
import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.cli import gui_launch as launch
from postgkyl.cli.app import cli
from postgkyl.gui import pipeline as gp

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
                                  reason="requires Gkeyll")

DATA = Path(__file__).parent / "test_data"
M0_3X = DATA / "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl"
GENERATED = DATA / "generated"


def _touch(directory, *names):
  for name in names:
    (directory / name).touch()


# ---------------------------------------------------------------- discovery
class TestDiscovery:

  def test_frame_series_static_files_and_simulations(self, tmp_path):
    _touch(tmp_path, "sim-ion_M0_0.gkyl", "sim-ion_M0_2.gkyl",
           "sim-ion_M0_10.gkyl", "sim-geo_int_bmag.gkyl",
           "sim-ion_M0_3_restart.gkyl", "other_b1-field_4.gkyl")
    outputs = gp.scan_outputs(str(tmp_path))
    assert set(outputs) == {"ion_M0", "geo_int_bmag (static)", "field"}
    assert outputs["ion_M0"].frames == [0, 2, 10]
    assert outputs["ion_M0"].file_name(10).endswith("sim-ion_M0_10.gkyl")
    assert outputs["geo_int_bmag (static)"].frames == []
    assert outputs["field"].sim == "other_b1"
    assert gp.list_simulations(str(tmp_path)) == ["other_b1", "sim"]

  def test_clashing_quantities_keep_their_simulation(self, tmp_path):
    _touch(tmp_path, "a-elc_M0_0.gkyl", "b-elc_M0_0.gkyl")
    assert set(gp.scan_outputs(str(tmp_path))) == {"a-elc_M0", "b-elc_M0"}

  def test_missing_directory_has_no_outputs(self, tmp_path):
    assert gp.scan_outputs(str(tmp_path / "nope")) == {}

  def test_weight_file(self, tmp_path):
    assert gp.weight_file(str(tmp_path), "sim") is None
    _touch(tmp_path, "sim-geo_int_jacobgeo.gkyl")
    assert gp.weight_file(str(tmp_path),
                          "sim") == str(tmp_path / "sim-geo_int_jacobgeo.gkyl")

  def test_quantity_frames(self):
    assert gp.quantity_frames(str(GENERATED), "gk_quantity_1d_p1", "vt",
                              "elc") == [0]
    assert gp.quantity_frames(str(GENERATED), "gk_quantity_1d_p1", "apar",
                              None) == []


@pytest.mark.parametrize("text, expected", [
    ("", None),
    (":", [0, 2, 4, 6]),
    ("::2", [0, 4]),
    ("-2:", [4, 6]),
    ("-1", [6]),
    ("1", [2]),
])
def test_pick_frames(text, expected):
  assert gp.pick_frames([0, 2, 4, 6], text) == expected


@pytest.mark.parametrize("text", ["9", "a:b", "1:2:3:4"])
def test_pick_frames_rejects_bad_ranges(text):
  with pytest.raises(ValueError, match="frame range"):
    gp.pick_frames([0, 2, 4, 6], text)


def test_parse_options():
  assert gp.parse_options("ti_over_te=2 den_ref=1e19,2e19, mode=fast") == {
      "ti_over_te": 2,
      "den_ref": [1e19, 2e19],
      "mode": "fast"
  }
  assert gp.parse_options("  ") == {}
  with pytest.raises(ValueError, match="key=value"):
    gp.parse_options("=3")


# ----------------------------------------------------------- building chains
# The real frame 5 and a second, never-loaded frame for multi-frame chains.
_OUTPUT = gp.Output(label="elc_M0",
                    files=((5, str(M0_3X)), (6, "frame_6.gkyl")),
                    sim="rt_gk_tcv_nt_iwl_3x2v_p1")


def _file_settings(**overrides):
  settings = dict(directory=str(DATA),
                  mode="file",
                  frames=(5, ),
                  output=_OUTPUT,
                  sim=_OUTPUT.sim,
                  source_ndim=3,
                  ndim=3)
  return gp.Settings(**{**settings, **overrides})


def _verbs(steps):
  return [step.verb for step in steps]


class TestBuildChain:

  def test_order_is_fluctuation_average_transform_select_collect(self):
    steps = gp.build_chain(
        _file_settings(frames=(5, 6),
                       fluct="yz",
                       weight="w.gkyl",
                       average=(1, ),
                       select=((2, 0.5), ),
                       comp=0,
                       collect=True))
    assert _verbs(steps) == [
        "load", "load", "load", "activate", "fluctuation", "average",
        "interpolate", "select", "collect"
    ]
    weight, first, second, activate, fluct, average, _, select, _ = steps
    # The weight is loaded, then set aside so only the data is transformed.
    assert weight.kwargs == {"file_name": "w.gkyl", "tag": "weight"}
    assert [first.kwargs, second.kwargs] == [{
        "file_name": str(M0_3X)
    }, {
        "file_name": "frame_6.gkyl"
    }]
    assert activate.kwargs == {"tags": ["default"]}
    assert fluct.kwargs == {"dims": "1,2", "weight": "weight"}
    assert average.kwargs == {"dims": "1", "weight": "weight"}
    # Averaging dimension 1 moves dimension 2 to index 1.
    assert select.kwargs == {"z1": 0.5, "comp": 0}

  def test_an_unused_weight_is_not_loaded(self):
    assert _verbs(gp.build_chain(
        _file_settings(weight="w.gkyl"))) == ["load", "interpolate"]

  def test_fluctuations_need_3x_data(self):
    """'y' is the second of three axes only in a 3x run; on 2x data it
    would be z."""
    with pytest.raises(ValueError, match="need 3x"):
      gp.build_chain(_file_settings(fluct="y", source_ndim=2, ndim=2))
    with pytest.raises(ValueError, match="need 3x"):
      gp.build_chain(_file_settings(fluct="yz", source_ndim=2), probe_only=True)

  def test_averaging_everything_keeps_each_mean_as_a_dataset(self):
    steps = gp.build_chain(_file_settings(average=(0, 1, 2)))
    assert _verbs(steps) == ["load", "average"]
    assert steps[1].kwargs == {"dims": "0,1,2", "as_dataset": True}

  def test_probe_chain_stops_after_the_transform_on_one_frame(self):
    steps = gp.build_chain(_file_settings(frames=(5, 6),
                                          average=(1, ),
                                          select=((2, 0.5), ),
                                          transform="local_poly",
                                          num_interp=3),
                           probe_only=True)
    assert steps == (gp.Step.of("load", file_name=str(M0_3X)),
                     gp.Step.of("local_poly", npoints=3))

  def test_geometry_transforms_refuse_averages(self):
    with pytest.raises(ValueError, match="full configuration space"):
      gp.build_chain(_file_settings(average=(1, ), transform="map_to_rz"))

  def test_quantity_source(self):
    steps = gp.build_chain(
        gp.Settings(directory="d",
                    mode="quantity",
                    frames=(3, 4),
                    sim="s",
                    quantity="D",
                    species="ion",
                    options=(("conv", 2.5), ),
                    transform="none"))
    assert steps == (gp.Step.of("gk_load_quantity",
                                quantity="D",
                                species="ion",
                                name="s",
                                frame="3,4",
                                path="d",
                                conv=2.5), )


def test_script_makes_the_session_calls():
  steps = (gp.Step.of("load", file_name="f.gkyl"),
           gp.Step.of("fluctuation", dims="1",
                      weight="weight"), gp.Step.of("plot", title="t"))
  assert gp.python_script(steps).splitlines() == [
      "from postgkyl.cli import PostgkylSession", "", "s = PostgkylSession()",
      "s.load(file_name='f.gkyl')", "s.fluctuation(dims='1', weight='weight')",
      "s.plot(title='t')"
  ]


# --------------------------------------------------------------- execution
def _pgkyl_prints(command_line):
  """What running ``command_line`` (``pgkyl ...``) and then ``print`` prints."""
  program, *args = shlex.split(command_line)
  assert program == "pgkyl"
  result = CliRunner().invoke(cli, args + ["print"])
  assert result.exit_code == 0, result.output
  return result.output


def _printed(data):
  return np.array2string(data.values.squeeze(), precision=16) + "\n"


@needs_gkeyll
class TestRun:

  def test_steps_match_the_python_api_their_script_and_their_command(self):
    steps = gp.build_chain(
        _file_settings(average=(1, ), select=((2, 0.0), ), num_interp=3))
    session = gp.run(steps)
    (data, ) = session.datasets
    direct = pg.load(str(M0_3X)).average(
        [1]).interpolate(num_interp=3).select(z1=0.0)
    np.testing.assert_array_equal(data.values, direct.values)

    namespace = {}
    exec(gp.python_script(steps), namespace)  # noqa: S102 - our own script.
    assert namespace["s"].command() == session.command()
    np.testing.assert_array_equal(namespace["s"].datasets[0].values,
                                  direct.values)
    assert _pgkyl_prints(session.command()) == _printed(direct)

  def test_a_weighted_fluctuation_never_transforms_the_weight(self):
    # The density, as its own single-field weight on the same grid.
    steps = gp.build_chain(
        _file_settings(fluct="yz", weight=str(M0_3X), select=((2, 0.0), )))
    session = gp.run(steps)
    (data, ) = session.datasets
    direct = pg.load(str(M0_3X)).fluctuation("1,2", weight=pg.load(
        str(M0_3X))).interpolate().select(z2=0.0)
    np.testing.assert_array_equal(data.values, direct.values)
    assert " activate --tags default fluctuation " in session.command()
    assert _pgkyl_prints(session.command()) == _printed(direct)

  def test_full_averages_collect_into_a_time_trace(self):
    """Every direction averaged: one mean per frame, stamped with the
    frame's time, which collect stacks into a 1-D trace."""
    output = gp.Output(label="elc_M0",
                       files=((5, str(M0_3X)), (6, str(M0_3X))),
                       sim=_OUTPUT.sim)
    steps = gp.build_chain(
        _file_settings(frames=(5, 6),
                       output=output,
                       average=(0, 1, 2),
                       collect=True))
    (trace, ) = gp.run(steps).datasets
    mean = pg.load(str(M0_3X)).average([0, 1, 2])
    assert len(trace.grid[0]) == 2
    np.testing.assert_allclose(trace.values.ravel(), [mean, mean], rtol=1e-14)

  def test_probe_counts_fields_and_curvilinear_axes(self):
    info = gp.probe(gp.build_chain(_file_settings(), probe_only=True))
    assert info.cells == (96, 64, 32) and info.num_fields == 1
    assert info.curvilinear == (False, False, False)
    mapc2p = str(DATA / "rt_gk_tcv_nt_iwl_3x2v_p1-geo_int_mapc2p.gkyl")
    info = gp.probe(
        gp.build_chain(_file_settings(transform="map_to_rz", mapc2p=mapc2p),
                       probe_only=True))
    # The R-Z axes select on minor radius and poloidal angle, not by index.
    assert info.curvilinear == (False, False)
    np.testing.assert_allclose(info.lower, (0.0, -np.pi))
    np.testing.assert_allclose(info.upper, (0.12, np.pi))

  def test_several_frames_are_overlaid_on_one_figure(self):
    output = gp.Output(label="elc_M0",
                       files=((5, str(M0_3X)), (6, str(M0_3X))),
                       sim=_OUTPUT.sim)
    session = gp.run(
        gp.build_chain(
            _file_settings(frames=(5, 6),
                           output=output,
                           select=((1, 0.0), (2, 0.0)))))
    plot = gp.plot_step({"title": "t"}, session.datasets)
    assert plot == gp.Step.of("plot", title="t", figure=0)
    gp.apply(session, plot)
    figure = session.result
    try:
      assert len(figure.axes[0].lines) == 2
    finally:
      gp.figure_png(figure)
    assert gp.plot_step({}, session.datasets[:1]) == gp.Step.of("plot")

  def test_figure_and_movie(self, tmp_path):
    steps = gp.build_chain(_file_settings(select=((2, 0.0), )))
    session = gp.run(steps)
    gp.apply(session, gp.Step.of("plot", title="t", cmap="RdBu_r"))
    assert session.command().endswith(" plot --title t --cmap RdBu_r")
    png = gp.figure_png(session.result)
    assert png.startswith(b"\x89PNG")
    data = gp.processed(steps)
    movie = gp.make_movie([data, data],
                          str(tmp_path / "m.gif"),
                          fps=5,
                          plot_options={
                              "cmap": "RdBu_r",
                              "surface": True
                          },
                          fixed_range=True)
    assert Path(movie).stat().st_size > 0


# ------------------------------------------------------------------- launch
@needs_gkeyll
def test_movie_frames_carry_the_frame_after_a_typed_title(
    monkeypatch, tmp_path):
  titles = []
  animate_module = importlib.import_module("postgkyl.render.animate")
  plot = animate_module.backend.plot

  def recording_plot(*frame, **kwargs):
    titles.append(kwargs.get("title"))
    return plot(*frame, **kwargs)

  monkeypatch.setattr(animate_module.backend, "plot", recording_plot)
  data = gp.processed(gp.build_chain(_file_settings(select=((2, 0.0), ))))
  gp.make_movie([data],
                str(tmp_path / "m.gif"),
                fps=5,
                plot_options={"title": "n_e"},
                fixed_range=True)
  assert titles and all(t.startswith("n_e   frame: 5 time: ") for t in titles)


def test_launch_command_serves_the_packaged_notebook(tmp_path):
  cmd = launch.command(str(tmp_path))
  # -y (a global marimo flag, before the subcommand): Ctrl+C quits at once.
  assert cmd[1:5] == ["-m", "marimo", "-y", "run"]
  assert Path(cmd[5]).name == "notebook.py" and Path(cmd[5]).is_file()
  assert cmd[-2:] == ["--path", str(tmp_path)]


def test_launch_without_marimo_explains_how_to_install(monkeypatch, capsys):
  monkeypatch.setattr(launch.importlib.util, "find_spec", lambda name: None)
  assert launch.main(["--path", "."]) == 1
  assert "postgkyl[gui]" in capsys.readouterr().err


def test_launch_defaults_to_the_sample_data(monkeypatch, tmp_path):
  """Without --path the checkout's tests/test_data opens, from any working
  directory; an installation without the tests opens the current one."""
  calls = []
  monkeypatch.setattr(launch.importlib.util, "find_spec", lambda name: True)
  monkeypatch.setattr(launch, "serve", lambda cmd: calls.append(cmd) or 0)
  monkeypatch.chdir(tmp_path)
  assert launch.main([]) == 0
  assert calls[-1][-2:] == ["--path", str(DATA)]

  monkeypatch.setattr(launch, "SAMPLE_DATA", str(tmp_path / "missing"))
  assert launch.main([]) == 0
  assert calls[-1][-2:] == ["--path", str(tmp_path)]


def test_pgkyl_help_lists_the_gui_program():
  from click.testing import CliRunner

  from postgkyl.cli.app import cli

  result = CliRunner().invoke(cli, ["-h"])
  assert result.exit_code == 0, result.output
  programs = result.output.split("Programs:")[1]
  assert launch.PROGRAM in programs and launch.DESCRIPTION in programs


def test_serve_ignores_ctrl_c_only_while_marimo_runs(monkeypatch):
  """The launcher must not die with a traceback when Ctrl+C reaches it with
  marimo; marimo starts with the default handling, and the launcher's own
  handler is restored once marimo has exited."""
  import signal

  seen = {}

  class FakeProcess:

    def __init__(self, cmd):
      seen["at_start"] = signal.getsignal(signal.SIGINT)

    def wait(self):
      seen["while_waiting"] = signal.getsignal(signal.SIGINT)
      return 3

  before = signal.getsignal(signal.SIGINT)
  monkeypatch.setattr(launch.subprocess, "Popen", FakeProcess)
  assert launch.serve(["marimo"]) == 3
  assert seen["at_start"] is before
  assert seen["while_waiting"] is signal.SIG_IGN
  assert signal.getsignal(signal.SIGINT) is before


def test_launch_runs_marimo(monkeypatch, tmp_path):
  calls = []
  monkeypatch.setattr(launch.importlib.util, "find_spec", lambda name: True)
  monkeypatch.setattr(launch, "serve", lambda cmd: calls.append(cmd) or 0)
  assert launch.main(["-p", str(tmp_path)]) == 0
  assert calls == [launch.command(str(tmp_path))]


# ----------------------------------------------------------------- notebook
@needs_gkeyll
def test_notebook_runs_headless_and_draws_a_figure(monkeypatch):
  pytest.importorskip("marimo")
  from postgkyl.gui.notebook import app

  monkeypatch.chdir(DATA)
  _, defs = app.run()
  assert defs["field_dropdown"].value == "elc_M0"
  assert defs["grid_msg"] is None
  assert defs["figure_bytes"].startswith(b"\x89PNG")
  # Blank axes unless labels are typed, then on every subplot.
  # Text fields must live in a ui.dictionary: marimo ignores widgets kept in
  # a plain dict, so typed limits and labels would never reach the plot.
  import marimo
  assert isinstance(defs["texts"], marimo.ui.dictionary)
  options = defs["plot_options"]()
  assert options["xlabel"] == options["ylabel"] == ""
  # The colorbar label is always given, so a z scale never rewrites it.
  assert options["clabel"] == ""
  # Grid off by default; contour levels only with a contour plot.
  assert options["no_showgrid"] is True
  assert "cnlevels" not in options
  assert "subplot_xlabels" not in options
  typed = defs["plot_options"](subplot_xlabels="x (m)")
  assert typed["subplot_xlabels"] == "x (m)"
  # Below the figure: the session Python, then the command line it ran.
  view = defs["plot_view"].text
  python, command = (view.index("Equivalent Python"),
                     view.index("Equivalent command line"))
  assert python < command
  assert "s = PostgkylSession()" in view
  assert "pgkyl " in view[command:]
