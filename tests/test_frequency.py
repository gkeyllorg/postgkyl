"""Frequency measurements from time traces and spatial Fourier modes."""

import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl.cli.app import MODELS, cli
from postgkyl.cli.session import PostgkylSession


def _trace(time, values):
  return pg.GData(ctx={
      "interpolated": True,
      "tau": 2.0
  }).push([time],
          np.asarray(values)[:, None])


def test_frequency_option_contract():
  trace = _trace(np.arange(12.) * 0.1, np.sin(np.arange(12.) * 0.1))
  with pytest.raises(ValueError, match="only to plain"):
    pg.frequency(trace, spectrum=True, dir=0, inplace=True)
  with pytest.raises(ValueError, match="require --spectrum"):
    pg.frequency(trace, interpolated=True)
  with pytest.raises(ValueError, match="do not apply to --fft"):
    pg.frequency(trace, fft=True, guess="1,1,0,0")


def test_frequency_two_traces_and_short_fit(capsys):
  t = np.linspace(0, 8, 161)
  traces = [_trace(t, np.sin(omega * t)) for omega in (1.5, 2.5)]
  result = pg.frequency(*traces, time_window=(0, 8))
  assert len(result) == 2
  for data, omega in zip(result, (1.5, 2.5)):
    np.testing.assert_allclose(abs(data.ctx["fit_params"][0][1]),
                               omega,
                               atol=1e-8)
  assert capsys.readouterr().out.count("frequency component 0:") == 2
  with pytest.raises(ValueError, match="at least six time samples"):
    pg.frequency(traces[0], time_window=(0, 0.2))
  with pytest.raises(ValueError, match="min_n >= 6"):
    pg.frequency(traces[0], min_n=5)


@pytest.mark.parametrize("time,values,message", [
    (np.arange(4.), np.ones((2, 2, 1)), "one-dimensional time trace"),
    (np.arange(5.), np.ones((3, 1)), "coordinates do not match"),
    (np.array([0., 1., 1., 2.]), np.ones((4, 1)), "finite and increasing"),
])
def test_frequency_rejects_invalid_trace(time, values, message):
  grid = [time] if values.ndim == 2 else [time, np.arange(3.)]
  trace = pg.GData(ctx={"interpolated": True}).push(grid, values)
  with pytest.raises(ValueError, match=message):
    pg.frequency(trace)


def test_frequency_window_and_fft_contract():
  t = np.arange(12.) * 0.1
  trace = _trace(t, np.sin(2 * np.pi * t))
  with pytest.raises(ValueError, match="increasing finite bounds"):
    pg.frequency(trace, time_window=(1., 0.))
  with pytest.raises(ValueError, match="fewer than three samples"):
    pg.frequency(trace, time_window=(0., 0.1))
  with pytest.raises(ValueError, match="uniform time samples"):
    pg.frequency(_trace(np.array([0., 0.1, 0.21, 0.3]), np.ones(4)), fft=True)


def test_frequency_spectrum_with_metadata_expressions():
  edges = np.linspace(-1, 3, 17)
  centers = (edges[:-1] + edges[1:]) / 2
  frames = []
  for t in np.linspace(0, 4, 9):
    values = (np.cos(2 * np.pi * centers / 4 - 0.7 * t) +
              0.4 * np.cos(4 * np.pi * centers / 4 + 0.4 * t))
    frames.append(
        pg.GData(
            ctx={
                "time": t,
                "R_axis": 4.0,
                "c_s": 2.0,
                "rho_s": 0.6,
                "Cy": 0.2,
                "interpolated": True
            }).push([edges], values[:, None]))
  result = pg.frequency(*frames,
                        spectrum=True,
                        dir=0,
                        time_norm="(R_axis + 2*c_s)/4",
                        space_norm="rho_s/Cy",
                        time_window=(0, 2))
  np.testing.assert_allclose(result.values[:2, 0], [1.4, -0.8], atol=1e-12)
  np.testing.assert_allclose(result.grid[0][:2], [1.5 * np.pi, 3 * np.pi],
                             atol=1e-12)
  redundant_fft = pg.frequency(*frames,
                               spectrum=True,
                               fft=True,
                               dir=0,
                               time_norm="2",
                               time_window=(0, 2))
  np.testing.assert_allclose(redundant_fft.values[:2, 0],
                             result.values[:2, 0],
                             atol=1e-12)


def test_plain_frequency_fit_prints_and_preserves_time_grid(capsys):
  t = np.linspace(0, 10, 101)
  values = 1.5 * np.sin(3 * t + 0.2) + 0.1
  out = pg.frequency(_trace(t, values),
                     tag="omega",
                     time_norm="(tau + 2)/2",
                     time_window=(0, 10))
  assert out.tag == "omega"
  np.testing.assert_allclose(out.grid[0], t)
  np.testing.assert_allclose(out.values[:, 0], values, atol=1e-8)
  np.testing.assert_allclose(abs(out.ctx["fit_params"][0][1]), 6, atol=1e-8)
  assert "frequency component 0: 6" in capsys.readouterr().out


def test_frequency_fft_returns_temporal_spectrum(capsys):
  n = 128
  t = np.arange(n) * 0.1
  omega = 2 * np.pi * 5 / (n * 0.1)
  out = pg.frequency(_trace(t, np.cos(omega * t)), fft=True, tag="fft")
  assert out.tag == "fft"
  np.testing.assert_allclose(out.grid[0][4], omega)
  np.testing.assert_allclose(out.values[4, 0], 0.5, atol=1e-12)
  assert f"frequency component 0: {omega:.12g}" in capsys.readouterr().out
  with np.testing.assert_raises_regex(ValueError, "uniform time"):
    pg.frequency(_trace(t + 0.001 * t**2, np.cos(omega * t)), fft=True)


def test_frequency_cli_tag_and_no_independent_growth_spectrum(tmp_path):
  t = np.linspace(0, 10, 101)
  edges = np.linspace(-0.05, 10.05, 102)
  data = pg.GData(ctx={
      "interpolated": True
  }).push([edges],
          np.sin(3 * t)[:, None])
  path = pg.save(data, str(tmp_path / "trace.gkyl"))
  result = CliRunner().invoke(
      cli,
      [path, "frequency", "-t", "omega", "activate", "--tags", "omega", "info"])
  assert result.exit_code == 0, result.output
  assert "frequency component 0:" in result.output
  assert "omega" in result.output
  assert "growth_spectrum" not in {model.name for model in MODELS}


def test_frequency_activates_result_and_retains_tagged_source(tmp_path):
  t = np.linspace(0, 10, 101)
  edges = np.linspace(-0.05, 10.05, 102)
  frequency_file = pg.save(
      pg.GData(ctx={
          "interpolated": True
      }).push([edges],
              np.sin(3 * t)[:, None]), str(tmp_path / "frequency.gkyl"))
  session = PostgkylSession().load(frequency_file, tag="raw")
  session.frequency()
  assert [dataset.tag for dataset in session.datasets] == ["frequency"]
  session.activate(tags=["raw"])
  assert [dataset.tag for dataset in session.datasets] == ["raw"]
  session.activate(tags=["frequency"])
  assert [dataset.tag for dataset in session.datasets] == ["frequency"]
