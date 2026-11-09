"""Growth rates from time traces and spatial Fourier modes."""

import numpy as np
from click.testing import CliRunner

import postgkyl as pg
from postgkyl.cli.app import cli
from postgkyl.cli.session import PostgkylSession


def _frames():
  x = np.linspace(0, 2, 9)
  y = np.linspace(-1, 3, 17)
  xc = (x[1:] + x[:-1]) / 2
  yc = (y[1:] + y[:-1]) / 2
  frames = []
  for i, t in enumerate(np.linspace(0, 4, 9)):
    field = (
        np.exp(0.3 * t) * np.cos(2 * np.pi * yc[None, :] / 4 - 0.7 * t) +
        0.4 * np.exp(-0.2 * t) * np.cos(4 * np.pi * yc[None, :] / 4 + 0.4 * t))
    values = np.broadcast_to(field, (len(xc), len(yc)))[..., None].copy()
    frames.append(
        pg.GData(ctx={
            "time": t,
            "tau": 2.0,
            "ell": 3.0,
            "interpolated": True
        }).push([x, y], values))
  return frames


def test_plain_growth_prints_rate_and_returns_curve(capsys):
  t = np.linspace(0, 10, 101)
  values = 3 * np.exp(0.4 * t)
  data = pg.GData(ctx={"interpolated": True}).push([t], values[:, None])
  out = pg.growth(data, tag="gamma")
  assert out.tag == "gamma"
  np.testing.assert_allclose(out.values[:, 0], values, rtol=1e-8)
  np.testing.assert_allclose(out.ctx["fit_params"][0][1], 0.2, rtol=1e-8)
  assert "growth component 0: 0.2" in capsys.readouterr().out


def test_plain_growth_normalized_explicit_window(capsys):
  t = np.linspace(0, 10, 101)
  values = 3 * np.exp(0.4 * t)
  data = pg.GData(ctx={
      "interpolated": True,
      "tau": 2.0
  }).push([t], values[:, None])
  out = pg.growth(data, time_norm="tau/1", time_window=(2, 4))
  np.testing.assert_allclose(out.values[:, 0], values, rtol=1e-8)
  np.testing.assert_allclose(out.ctx["fit_params"][0][1], 0.4, rtol=1e-8)
  assert "growth component 0: 0.4" in capsys.readouterr().out


def test_growth_activates_result_and_retains_tagged_source(tmp_path):
  t = np.linspace(0, 10, 101)
  edges = np.linspace(-0.05, 10.05, 102)
  data = pg.GData(ctx={
      "interpolated": True
  }).push([edges], (3 * np.exp(0.4 * t))[:, None])
  path = pg.save(data, str(tmp_path / "growth.gkyl"))
  session = PostgkylSession().load(path, tag="raw")
  session.growth()
  assert [dataset.tag for dataset in session.datasets] == ["growth"]
  session.activate(tags=["raw"])
  assert [dataset.tag for dataset in session.datasets] == ["raw"]
  session.activate(tags=["growth"])
  assert [dataset.tag for dataset in session.datasets] == ["growth"]


def test_analytic_modes_and_normalization():
  frames = _frames()
  result = pg.growth(*frames,
                     spectrum=True,
                     dir=1,
                     time_norm="tau",
                     space_norm="ell",
                     time_window=(0, 8))
  np.testing.assert_allclose(result.grid[0][:2],
                             [2 * np.pi / 4 * 3, 4 * np.pi / 4 * 3])
  np.testing.assert_allclose(result.values[:2, 0], [0.6, -0.4],
                             rtol=1e-12,
                             atol=1e-12)
  assert result.ctx["growth_window"] == (0, 2)


def test_metadata_expressions_and_numeric_strings():
  frames = _frames()
  for frame in frames:
    frame.ctx.update(R_axis=4.0, c_s=2.0, rho_s=0.6, Cy=0.2)
  growth = pg.growth(*frames,
                     spectrum=True,
                     dir=1,
                     time_norm="R_axis/c_s",
                     space_norm="rho_s/Cy",
                     time_window=(0, 2))
  np.testing.assert_allclose(growth.values[:2, 0], [0.6, -0.4], atol=1e-12)
  np.testing.assert_allclose(growth.grid[0][:2], [1.5 * np.pi, 3 * np.pi],
                             atol=1e-12)


def test_normalization_expression_errors():
  frames = _frames()
  for frame in frames:
    frame.ctx.update(a=2.0, b=1.0)
  frames[0].ctx["b"] = 0.0
  for expression, message in [("a/missing", "missing numeric"),
                              ("a/b", "divides by zero"),
                              ("a**2", "supports only"), ("a+", "invalid"),
                              ("-a", "finite and positive")]:
    with np.testing.assert_raises_regex(ValueError, message):
      pg.growth(frames[0],
                *frames[1:],
                spectrum=True,
                dir=1,
                time_norm=expression)
  frames[1].ctx.update(a=3.0, b=1.0)
  for frame in frames[2:]:
    frame.ctx.update(a=2.0, b=1.0)
  frames[0].ctx["b"] = 1.0
  with np.testing.assert_raises_regex(ValueError, "differs across frames"):
    pg.growth(*frames, spectrum=True, dir=1, time_norm="a/b")


def test_automatic_window_and_bad_inputs():
  frames = _frames()
  result = pg.growth(*frames, spectrum=True, dir=1)
  np.testing.assert_allclose(result.values[:2, 0], [0.3, -0.2], atol=1e-12)
  with np.testing.assert_raises_regex(ValueError, "time_window"):
    pg.growth(*frames, spectrum=True, dir=1, time_window=(10, 12))
  with np.testing.assert_raises_regex(ValueError, "missing numeric"):
    pg.growth(*frames, spectrum=True, dir=1, time_norm="absent")


def test_modal_cell_average_matches_point_field():
  frames = _frames()
  modal = [
      pg.GData(
          ctx={
              **frame.ctx, "value_form": "modal",
              "basis_type": "serendipity",
              "poly_order": 0
          }).push(frame.grid, frame.values * 2.0) for frame in frames
  ]
  for frame in modal:
    frame.ctx.pop("interpolated")
  expected = pg.growth(*frames, spectrum=True, dir=1, time_window=(0, 4))
  actual = pg.growth(*modal, spectrum=True, dir=1, time_window=(0, 4))
  np.testing.assert_allclose(actual.values,
                             expected.values,
                             rtol=1e-12,
                             atol=1e-12)


def test_each_spatial_direction_has_its_own_wavenumber():
  edges = [np.linspace(-1, 1, 9), np.linspace(0, 3, 13), np.linspace(2, 6, 17)]
  for direction in range(3):
    coordinates = (edges[direction][1:] + edges[direction][:-1]) / 2
    shape = [1, 1, 1]
    shape[direction] = len(coordinates)
    frames = []
    for t in range(5):
      wave = np.exp(0.25 * t) * np.cos(2 * np.pi *
                                       (coordinates - edges[direction][0]) /
                                       np.ptp(edges[direction]) - 0.4 * t)
      values = np.broadcast_to(wave.reshape(shape), (8, 12, 16))
      frames.append(
          pg.GData(ctx={
              "time": t,
              "interpolated": True
          }).push(edges, values[..., None].copy()))
    result = pg.growth(*frames,
                       spectrum=True,
                       dir=direction,
                       time_window=(0, 4),
                       interpolated=True)
    np.testing.assert_allclose(result.grid[0][0],
                               2 * np.pi / np.ptp(edges[direction]))
    np.testing.assert_allclose(result.values[0, 0], 0.25, atol=1e-12)


def test_cli_growth_spectrum_spelling(tmp_path):
  paths = []
  for i, frame in enumerate(_frames()):
    frame.ctx.update(R_axis=4.0, c_s=2.0, rho_s=0.6, Cy=0.2)
    paths.append(pg.save(frame, str(tmp_path / f"sample-field_{i}.gkyl")))
  result = CliRunner().invoke(cli, [
      str(tmp_path / "sample-field_*.gkyl"), "growth", "--spectrum", "-d", "1",
      "-t", "gamma", "--time_norm", "R_axis/c_s", "--time_window", "0,2",
      "--space_norm", "rho_s/Cy", "activate", "--tags", "gamma", "info"
  ])
  assert result.exit_code == 0, result.output
  assert "Number of components: 1" in result.output
  assert "gamma" in result.output
  numeric = CliRunner().invoke(cli, [
      str(tmp_path / "sample-field_*.gkyl"), "growth", "--spectrum", "-d", "1",
      "--time_norm", "0.5", "--space_norm", "0.5", "--time_window", "0,8",
      "info"
  ])
  assert numeric.exit_code == 0, numeric.output
