"""Tests for the notebook GUI: the processing chain in
``postgkyl.gui.pipeline``, the launcher, and (with marimo installed) a
headless run of the notebook.

The chain is checked against the public API it stands for: running a chain
must give the same datasets as calling the verbs directly, and the Python
script it renders must reproduce them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.gui import launch
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
                  ndim=3)
  return gp.Settings(**{**settings, **overrides})


def _verbs(chain):
  return [step.verb for step in chain.steps]


class TestBuildChain:

  def test_order_is_fluctuation_average_transform_select_collect(self):
    chain = gp.build_chain(
        _file_settings(frames=(5, 6),
                       fluct="yz",
                       weight="w.gkyl",
                       average=(1, ),
                       select=((2, 0.5), ),
                       comp=0,
                       collect=True))
    assert _verbs(chain) == [
        "fluctuation", "average", "interpolate", "select", "collect"
    ]
    fluct, average, _, select, _ = chain.steps
    assert fluct.kwargs == {"dims": [1, 2], "weight": gp.WEIGHT}
    assert average.kwargs == {"dims": [1], "weight": gp.WEIGHT}
    # Averaging dimension 1 moves dimension 2 to index 1.
    assert select.kwargs == {"z1": 0.5, "comp": 0}
    assert chain.weight == "w.gkyl"

  def test_averaging_everything_is_a_full_average_without_transform(self):
    chain = gp.build_chain(_file_settings(average=(0, 1, 2)))
    assert _verbs(chain) == ["full_average"]

  def test_probe_chain_stops_after_the_transform_on_one_frame(self):
    chain = gp.build_chain(_file_settings(frames=(5, 6),
                                          average=(1, ),
                                          select=((2, 0.5), ),
                                          transform="local_poly",
                                          num_interp=3),
                           probe_only=True)
    assert chain.source.kwargs["file_name"] == (str(M0_3X), )
    assert [(s.verb, s.kwargs) for s in chain.steps] == [("local_poly", {
        "npoints": 3
    })]

  def test_geometry_transforms_refuse_averages(self):
    with pytest.raises(ValueError, match="full configuration space"):
      gp.build_chain(_file_settings(average=(1, ), transform="map_to_rz"))

  def test_quantity_source(self):
    chain = gp.build_chain(
        gp.Settings(directory="d",
                    mode="quantity",
                    frames=(3, 4),
                    sim="s",
                    quantity="D",
                    species="ion",
                    options=(("conv", 2.5), )))
    assert chain.source.verb == "gk.load_quantity"
    assert chain.source.kwargs == {
        "quantity": "D",
        "species": "ion",
        "name": "s",
        "frame": "3,4",
        "path": "d",
        "conv": 2.5
    }

  def test_transport_source(self):
    settings = gp.Settings(directory="d",
                           mode="transport",
                           frames=(3, 4),
                           sim="s",
                           species="ion",
                           transform="none",
                           fluct="y",
                           collect=True)
    chain = gp.build_chain(settings)
    assert chain.source.kwargs["per_frame"] is True
    assert chain.source.kwargs["fluct"] == "y"
    assert _verbs(chain) == ["collect"]
    with pytest.raises(ValueError, match="already flux-surface"):
      gp.build_chain(
          gp.Settings(directory="d",
                      mode="transport",
                      frames=(3, ),
                      sim="s",
                      species="ion"))


# --------------------------------------------------------------- execution
@needs_gkeyll
class TestRun:

  def test_chain_matches_the_direct_api_and_its_script(self):
    chain = gp.build_chain(
        _file_settings(average=(1, ), select=((2, 0.0), ), num_interp=3))
    (data, ) = gp.run(chain)
    direct = pg.load(str(M0_3X)).average(
        [1]).interpolate(num_interp=3).select(z1=0.0)
    np.testing.assert_array_equal(data.values, direct.values)

    namespace = {}
    exec(gp.python_script(chain), namespace)  # noqa: S102 - our own script.
    np.testing.assert_array_equal(namespace["data"][0].values, direct.values)

  def test_script_states_the_weight_and_the_plot(self):
    chain = gp.Chain(gp.Step.of("load", file_name=("f.gkyl", )),
                     weight="w.gkyl",
                     steps=(gp.Step.of("fluctuation",
                                       dims=[1],
                                       weight=gp.WEIGHT), ))
    script = gp.python_script(chain, {"title": "t"})
    assert script.splitlines() == [
        "import postgkyl as pg", "", "weight = pg.load('w.gkyl')",
        "data = pg.GDataGroup([pg.load(f) for f in ['f.gkyl']])",
        "data = data.fluctuation(dims=[1], weight=weight)",
        "pg.plot(data, title='t')"
    ]

  def test_full_averages_collect_into_a_time_trace(self):
    """Every direction averaged: one mean per frame, stamped with the
    frame's time, which collect stacks into a 1-D trace."""
    data = pg.GDataGroup([pg.load(str(M0_3X)), pg.load(str(M0_3X))])
    data[1].ctx["time"] = 2.0 * data[0].ctx["time"]
    traces = gp.full_average(data, [0, 1, 2])
    mean = data[0].average([0, 1, 2])
    np.testing.assert_allclose([t.values[0, 0] for t in traces], [mean, mean],
                               rtol=1e-14)
    trace = traces.collect()
    np.testing.assert_allclose(trace.grid[0],
                               [data[0].ctx["time"], data[1].ctx["time"]])

  def test_probe_counts_fields_and_curvilinear_axes(self):
    info = gp.probe(gp.build_chain(_file_settings(), probe_only=True))
    assert info.cells == (96, 64, 32) and info.num_fields == 1
    assert info.curvilinear == (False, False, False)
    mapc2p = str(DATA / "rt_gk_tcv_nt_iwl_3x2v_p1-geo_int_mapc2p.gkyl")
    info = gp.probe(
        gp.build_chain(_file_settings(transform="map_to_rz", mapc2p=mapc2p),
                       probe_only=True))
    assert info.curvilinear == (True, True)
    assert info.upper == (info.cells[0] - 1, info.cells[1] - 1)

  def test_figure_and_movie(self, tmp_path):
    data = gp.run(gp.build_chain(_file_settings(select=((2, 0.0), ))))
    png = gp.figure_png(data, {"title": "t", "cmap": "RdBu_r"})
    assert png.startswith(b"\x89PNG")
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
def test_launch_command_serves_the_packaged_notebook(tmp_path):
  cmd = launch.command(str(tmp_path))
  assert cmd[1:4] == ["-m", "marimo", "run"]
  assert Path(cmd[4]).name == "notebook.py" and Path(cmd[4]).is_file()
  assert cmd[-2:] == ["--path", str(tmp_path)]


def test_launch_without_marimo_explains_how_to_install(monkeypatch, capsys):
  monkeypatch.setattr(launch.importlib.util, "find_spec", lambda name: None)
  assert launch.main(["--path", "."]) == 1
  assert "postgkyl[gui]" in capsys.readouterr().err


def test_launch_runs_marimo(monkeypatch, tmp_path):
  calls = []
  monkeypatch.setattr(launch.importlib.util, "find_spec", lambda name: True)
  monkeypatch.setattr(launch.subprocess, "call",
                      lambda cmd: calls.append(cmd) or 0)
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
