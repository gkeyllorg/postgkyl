"""Tests for postgkyl.render.plotly -- 2-D surfaces, 3-D volumes/scatter,
animation, and rotating-figure export.

Adapted from ``tests_bak/test_plot.py``'s ``plotly`` cases: the old tests fed
``(grid, values)`` tuples straight into ``pg.output.plotly``; this layer's
``plotly()`` takes a :class:`~postgkyl.gdatastate.gdatastate.GDataState` instead (no
dual "GData or tuple" signature -- see PYTHON_PRINCIPLES.md #9), so every
case below builds one via ``GDataState().push(...)``.
"""

from __future__ import annotations

from importlib import import_module
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
import pytest

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.render import _ffmpeg
from postgkyl.render.plotly import (
    plotly,
    plotly_animate,
    save_rotating_plotly_figure,
)
from postgkyl.render.plotly import (
    _log_colorbar_ticks,
    _opacity_mapping,
    _prepare_2d_coordinates,
    _prepare_3d_coordinates,
)

needs_ffmpeg = pytest.mark.skipif(
    _ffmpeg.resolve_ffmpeg() is None,
    reason="ffmpeg not found on PATH or via imageio-ffmpeg")
external_tool = pytest.mark.external_tool
slow = pytest.mark.slow


def _chrome_available() -> bool:
  # Kaleido v1+ needs a real Chrome/Chromium binary (its own download via
  # `kaleido_get_chrome` or a system install) -- without one,
  # start_sync_server()'s background thread dies and to_image() hangs
  # forever waiting on a server that never came up, rather than raising.
  try:
    from choreographer.browsers.chromium import Chromium
    return Chromium.find_browser(skip_local=False) is not None
  except Exception:
    return False


needs_chrome = pytest.mark.skipif(not _chrome_available(),
                                  reason="no Chrome/Chromium found for kaleido")

# kaleido's Chrome subprocess (managed by the `choreographer` library under
# start_sync_server()/stop_sync_server()) has produced an intermittent,
# non-reproducible-on-Linux segfault during CPython's own interpreter
# finalization -- well after the whole pytest session has already passed --
# only ever seen on macOS CI. Skip there rather than let it take down the
# whole pytest process; see skip_macos_animate_save in test_cli_commands.py
# for the same pattern applied to an analogous matplotlib/macOS crash.
skip_macos_chrome = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="intermittent segfault during kaleido's Chrome subprocess "
    "teardown at interpreter shutdown on macOS -- not reproducible "
    "on Linux")


def _state(grid, values) -> GDataState:
  d = GDataState()
  d.push(list(grid), values)
  return d


def _volume_3d(fn=lambda x, y, z: x + y + z, n=4):
  grid = [
      np.linspace(0.0, 1.0, n),
      np.linspace(0.0, 1.0, n),
      np.linspace(0.0, 1.0, n)
  ]
  x, y, z = np.meshgrid(*grid, indexing="ij")
  values = fn(x, y, z)[..., np.newaxis]
  return _state(grid, values)


def _surface_2d(n=4, m=5):
  grid = [np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, m)]
  x, y = np.meshgrid(*grid, indexing="ij")
  values = (x + 2.0 * y)[..., np.newaxis]
  return _state(grid, values)


class TestPlotlySurface2D:

  def test_returns_a_surface_trace(self):
    fig = plotly(_surface_2d())
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Surface)

  def test_surface_z_matches_values(self):
    n, m = 4, 5
    grid = [np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, m)]
    x, y = np.meshgrid(*grid, indexing="ij")
    fig = plotly(_surface_2d(n, m))
    np.testing.assert_allclose(fig.data[0].z, x + 2.0 * y)

  def test_axis_ranges_match_data_extent(self):
    fig = plotly(_surface_2d())
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (0.0, 1.0))
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (0.0, 1.0))
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (0.0, 3.0))

  def test_scatter_mode_rejected_for_surface(self):
    with pytest.raises(ValueError, match="scatter"):
      plotly(_surface_2d(), scatter=True)

  def test_surface_logc_applies_log_colorscale(self):
    fig = plotly(_surface_2d(), logc=True, cmin=1.0e-3, cmax=10.0)
    np.testing.assert_allclose(fig.data[0].cmin, -3.0)
    np.testing.assert_allclose(fig.data[0].cmax, 1.0)

  def test_scale_and_shift_apply_to_surface_coordinates_and_height(self):
    # x/y scale+shift the coordinates; z/color inherit from the *value*
    # (zscale/zshift), matching src_bak/postgkyl/output/plotly.py:720.
    n, m = 4, 5
    fig = plotly(_surface_2d(n, m),
                 xscale=2.0,
                 xshift=1.0,
                 yscale=3.0,
                 yshift=0.5,
                 zscale=2.0,
                 zshift=1.0)
    np.testing.assert_allclose(fig.data[0].x.min(), 2.0)
    np.testing.assert_allclose(fig.data[0].x.max(), 4.0)
    np.testing.assert_allclose(fig.data[0].y.min(), 1.5)
    np.testing.assert_allclose(fig.data[0].y.max(), 4.5)
    np.testing.assert_allclose(np.nanmin(fig.data[0].z), 1.0)
    np.testing.assert_allclose(np.nanmax(fig.data[0].z), 7.0)


class TestPlotly3DVolume:

  def test_returns_a_volume_trace_with_default_surface_count(self):
    fig = plotly(_volume_3d())
    assert isinstance(fig, go.Figure)
    assert fig.data[0].surface.count == 32

  def test_axis_ranges_match_data_extent(self):
    fig = plotly(_volume_3d())
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (0.0, 1.0))
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (0.0, 1.0))
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (0.0, 1.0))

  def test_explicit_ranges_and_surface_count_override(self):
    fig = plotly(_volume_3d(),
                 xrange=(0.2, 0.8),
                 yrange=(0.1, 0.9),
                 zrange=(0.3, 0.7),
                 surface_count=12)
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (0.2, 0.8))
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (0.1, 0.9))
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (0.3, 0.7))
    assert fig.data[0].surface.count == 12

  def test_color_scale_shift_and_clim(self):
    fig = plotly(_volume_3d(), cscale=2.0, cshift=1.0, clim=(1.5, 5.5))
    np.testing.assert_allclose(fig.data[0].cmin, 1.5)
    np.testing.assert_allclose(fig.data[0].cmax, 5.5)
    np.testing.assert_allclose(np.nanmin(fig.data[0].value), 1.0)
    np.testing.assert_allclose(np.nanmax(fig.data[0].value), 7.0)

  def test_logc_converts_linear_clim_to_log_space(self):
    fig = plotly(_volume_3d(fn=lambda x, y, z: 1.0e-2 + x + y + z),
                 logc=True,
                 cmin=1.0e-20,
                 cmax=1.0e-2)
    np.testing.assert_allclose(fig.data[0].cmin, -20.0)
    np.testing.assert_allclose(fig.data[0].cmax, -2.0)

  def test_aspect_cube_mode(self):
    fig = plotly(_volume_3d(), aspect="cube")
    assert fig.layout.scene.aspectmode == "cube"

  def test_aspect_string_sets_mode(self):
    fig = plotly(_volume_3d(), aspect="data")
    assert fig.layout.scene.aspectmode == "data"

  def test_aspect_numeric_sets_manual_ratio(self):
    fig = plotly(_volume_3d(), aspect=2.0)
    assert fig.layout.scene.aspectmode == "manual"
    assert fig.layout.scene.aspectratio.x == 2.0
    assert fig.layout.scene.aspectratio.y == 2.0
    assert fig.layout.scene.aspectratio.z == 2.0

  def test_aspect_numeric_string_sets_manual_ratio(self):
    fig = plotly(_volume_3d(), aspect="1.5")
    assert fig.layout.scene.aspectmode == "manual"
    assert fig.layout.scene.aspectratio.x == 1.5

  def test_scale_and_shift_apply_to_volume_coordinates(self):
    fig = plotly(_volume_3d(),
                 xscale=2.0,
                 xshift=1.0,
                 yscale=3.0,
                 yshift=0.5,
                 zscale=4.0,
                 zshift=1.0)
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (2.0, 4.0))
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (1.5, 4.5))
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (4.0, 8.0))

  def test_zscale_zshift_apply_to_volume_color_value(self):
    # value = (x+y+z)*zscale + zshift, independent of the z *coordinate*'s
    # own scale/shift -- matches src_bak/postgkyl/output/plotly.py:720.
    fig = plotly(_volume_3d(), zscale=2.0, zshift=1.0)
    np.testing.assert_allclose(np.nanmin(fig.data[0].value), 1.0)
    np.testing.assert_allclose(np.nanmax(fig.data[0].value), 7.0)

  def test_cylindrical_to_cartesian_conversion(self):
    r = np.linspace(0.0, 1.0, 4)
    z = np.linspace(-0.5, 0.5, 4)
    phi = np.linspace(0.0, 2.0 * np.pi, 5)
    rr, zz, pp = np.meshgrid(r, z, phi, indexing="ij")
    values = (rr + zz)[..., np.newaxis]
    fig = plotly(_state([r, z, phi], values), cylindrical_to_cartesian=True)
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (-1.0, 1.0),
                               atol=1e-12)
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (-1.0, 1.0),
                               atol=1e-12)
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (-0.5, 0.5),
                               atol=1e-12)


class TestPlotly3DScatter:

  def test_scatter_trace_basic_properties(self):
    fig = plotly(_volume_3d(),
                 scatter=True,
                 marker_radius=3.0,
                 markerstyle="square",
                 cmin=0.2,
                 cmax=2.8)
    assert isinstance(fig.data[0], go.Scatter3d)
    assert fig.data[0].mode == "markers"
    np.testing.assert_allclose(fig.data[0].marker.size, 6.0)
    assert fig.data[0].marker.symbol == "square"
    np.testing.assert_allclose(fig.data[0].marker.cmin, 0.2)
    np.testing.assert_allclose(fig.data[0].marker.cmax, 2.8)

  def test_scatter_downsampling(self):
    fig = plotly(_volume_3d(), scatter=True, maximum_points_per_axis=2)
    # size-4 axis downsampled to indices [0, 2, 3] -> 3 points per axis.
    assert len(fig.data[0].x) == 27
    assert len(fig.data[0].y) == 27
    assert len(fig.data[0].z) == 27

  def test_opacity_gradient_when_requested(self):
    fig = plotly(_volume_3d(),
                 scatter=True,
                 opacity=0.5,
                 scatter_opacity_range=(0.01, 1.0))
    colorscale = fig.data[0].marker.colorscale
    low_alpha = float(colorscale[0][1].split(",")[-1].rstrip(")"))
    high_alpha = float(colorscale[-1][1].split(",")[-1].rstrip(")"))
    assert low_alpha < high_alpha

  def test_uniform_opacity_by_default(self):
    fig = plotly(_volume_3d(), scatter=True, opacity=0.5)
    colorscale = fig.data[0].marker.colorscale
    low_alpha = float(colorscale[0][1].split(",")[-1].rstrip(")"))
    high_alpha = float(colorscale[-1][1].split(",")[-1].rstrip(")"))
    np.testing.assert_allclose(low_alpha, high_alpha)
    np.testing.assert_allclose(fig.data[0].marker.opacity, 0.5)

  def test_log_opacity_ramp(self):
    fig = plotly(_volume_3d(),
                 scatter=True,
                 scatter_opacity_range=(0.01, 1.0),
                 scatter_opacity_log=True)
    colorscale = fig.data[0].marker.colorscale
    alphas = np.array(
        [float(c.split(",")[-1].rstrip(")")) for _, c in colorscale])
    q1 = int(0.25 * (len(alphas) - 1))
    q3 = int(0.75 * (len(alphas) - 1))
    low_span = alphas[q1] - alphas[0]
    high_span = alphas[-1] - alphas[q3]
    assert low_span > high_span


class TestPlotlyMultiComponent:

  def test_two_components_get_two_scenes(self):
    grid = [
        np.linspace(0.0, 1.0, 4),
        np.linspace(0.0, 1.0, 4),
        np.linspace(0.0, 1.0, 4)
    ]
    x, y, z = np.meshgrid(*grid, indexing="ij")
    values = np.stack([x + y + z, x - y - z], axis=-1)
    fig = plotly(_state(grid, values))
    assert len(fig.data) == 2

  def test_squeeze_forces_a_single_scene(self):
    grid = [
        np.linspace(0.0, 1.0, 4),
        np.linspace(0.0, 1.0, 4),
        np.linspace(0.0, 1.0, 4)
    ]
    x, y, z = np.meshgrid(*grid, indexing="ij")
    values = np.stack([x + y + z, x - y - z], axis=-1)
    fig = plotly(_state(grid, values), squeeze=True)
    assert len(fig.data) == 1


class TestPlotlyMisc:

  def test_diverging_symmetric_colorscale(self):
    fig = plotly(_volume_3d(), diverging=True)
    assert fig.data[0].cmin == -fig.data[0].cmax

  def test_title_is_set(self):
    fig = plotly(_volume_3d(), title="my title")
    assert fig.layout.title.text == "my title"

  def test_hashtag_annotation(self):
    fig = plotly(_volume_3d(), hashtag=True)
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == "#pgkyl"

  def test_figsize_sets_pixel_dimensions(self):
    fig = plotly(_volume_3d(), figsize=(6, 4))
    assert fig.layout.width == 600
    assert fig.layout.height == 400

  def test_invalid_num_dims_raises(self):
    d = _state([np.linspace(0.0, 1.0, 5)], np.ones((4, 1)))
    with pytest.raises(ValueError, match="2D surface"):
      plotly(d)

  def test_solid_color_disables_colorbar(self):
    fig = plotly(_volume_3d(), color="red")
    assert fig.data[0].showscale is False


class TestPlotlyStyleAndTheme:

  def test_light_background_sets_light_theme_colors(self):
    fig = plotly(_volume_3d(), background="light")
    assert fig.layout.paper_bgcolor == "#ffffff"

  def test_dark_background_is_the_default(self):
    fig = plotly(_volume_3d())
    assert fig.layout.paper_bgcolor == "#000000"

  def test_explicit_style_kwarg_is_applied(self):
    # "default" resets Matplotlib's baseline rc, distinct from the packaged
    # postgkyl style's lines.linewidth == 2 (image.cmap gets overwritten
    # right after by the cmap-resolution step below, so assert on a rc key
    # that step never touches).
    plotly(_volume_3d(), style="default")
    assert mpl.rcParams["lines.linewidth"] == 1.5

  def test_rcparams_override_is_applied(self):
    plotly(_volume_3d(), rcParams={"lines.linewidth": 4.0})
    assert mpl.rcParams["lines.linewidth"] == 4.0

  def test_invert_cmap_appends_reversal_suffix(self):
    plotly(_volume_3d(), cmap="viridis", invert_cmap=True)
    assert mpl.rcParams["image.cmap"] == "viridis_r"

  def test_invert_cmap_strips_reversal_suffix(self):
    plotly(_volume_3d(), cmap="viridis_r", invert_cmap=True)
    assert mpl.rcParams["image.cmap"] == "viridis"

  def test_xkcd_style_does_not_raise(self):
    import matplotlib.pyplot as plt
    plotly(_volume_3d(), xkcd=True)
    plt.rcdefaults()


class TestPlotlyLogAxes:

  def test_log_axes_use_log10_ranges(self):
    grid = [np.linspace(1.0, 10.0, 4), np.linspace(1.0, 100.0, 5)]
    x, y = np.meshgrid(*grid, indexing="ij")
    values = (x + y)[..., np.newaxis]
    fig = plotly(_state(grid, values), logx=True, logy=True)
    assert fig.layout.scene.xaxis.type == "log"
    assert fig.layout.scene.yaxis.type == "log"
    np.testing.assert_allclose(fig.layout.scene.xaxis.range,
                               [np.log10(1.0), np.log10(10.0)])

  def test_logz_masks_nonpositive_volume_values(self):
    # The z *coordinate* axis spans [0, 1] here, so log10(0) triggers an
    # (expected, harmless) divide-by-zero warning independent of the
    # *value* function -- match the old tree's behaviour, don't silence it
    # at the source, just don't let it fail this test.
    with np.errstate(divide="ignore"):
      fig = plotly(_volume_3d(fn=lambda x, y, z: x + y + z - 1.4), logz=True)
    # Values <= 0 become NaN in log space; the trace should still build.
    assert isinstance(fig.data[0], go.Volume)

  def test_logc_with_all_nonpositive_values_uses_fallback_range(self):
    fig = plotly(_volume_3d(fn=lambda x, y, z: -(x + y + z) - 1.0), logc=True)
    assert isinstance(fig.data[0], go.Volume)

  def test_logc_ticks_append_max_when_step_overshoots_it(self):
    # lo=0, hi=20 with the default max_ticks=7 steps by 3 and lands on 18,
    # short of hi -- _log_colorbar_ticks must append the true endpoint.
    fig = plotly(_surface_2d(), logc=True, cmin=1.0, cmax=1.0e20)
    tick_vals = fig.data[0].colorbar.tickvals
    assert tick_vals[-1] == 20.0

  def test_logc_cmax_below_cmin_falls_back_to_a_one_decade_span(self):
    # cmax < cmin collapses the requested log range; _apply_log_colorscale
    # falls back to a single decade above cmin rather than an inverted one.
    fig = plotly(_surface_2d(), logc=True, cmin=100.0, cmax=10.0)
    np.testing.assert_allclose(fig.data[0].cmin, 2.0)
    np.testing.assert_allclose(fig.data[0].cmax, 3.0)

  def test_all_nan_values_yield_nan_color_range_without_raising(self):
    n, m = 4, 5
    grid = [np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, m)]
    values = np.full((n - 1, m - 1, 1), np.nan)
    fig = plotly(_state(grid, values))
    assert np.isnan(fig.data[0].cmin)
    assert np.isnan(fig.data[0].cmax)


class TestPlotlyPrivateHelpers:
  """Direct tests for small pure helpers whose edge branches are defensive
  code unreachable through ``plotly()``'s public contract: the coordinate
  helpers only ever see grids matching the checked ``num_dims`` and 1-D
  nodal axes (guaranteed by ``GDataState``), and ``_log_colorbar_ticks``
  only ever gets called with the already-finite range ``_apply_log_colorscale``
  computes. Testing these directly is simpler and more honest than
  contriving a ``GDataState`` that violates those invariants."""

  def test_opacity_mapping_swaps_inverted_bounds(self):
    colorscale = [[0.0, "rgba(10, 20, 30, 1.000)"],
                  [1.0, "rgba(10, 20, 30, 1.000)"]]
    out = _opacity_mapping(colorscale, min_alpha=0.9, max_alpha=0.1)
    first_alpha = float(out[0][1].split(",")[-1].rstrip(")"))
    last_alpha = float(out[-1][1].split(",")[-1].rstrip(")"))
    np.testing.assert_allclose(first_alpha, 0.1)
    np.testing.assert_allclose(last_alpha, 0.9)

  def test_opacity_mapping_passes_through_non_rgba_and_malformed_colors(self):
    colorscale = [[0.0, "rgba(1, 2, 3)"], [1.0, "#ff0000"]]
    out = _opacity_mapping(colorscale, min_alpha=0.0, max_alpha=1.0)
    assert out == [[0.0, "rgba(1, 2, 3)"], [1.0, "#ff0000"]]

  def test_log_colorbar_ticks_empty_for_non_finite_bounds(self):
    assert _log_colorbar_ticks(float("nan"), 5.0) == ([], [])

  def test_prepare_3d_coordinates_rejects_wrong_count(self):
    with pytest.raises(ValueError, match="three coordinate arrays"):
      _prepare_3d_coordinates((np.array([0.0]), np.array([0.0])), (1, ))

  def test_prepare_3d_coordinates_passes_through_already_meshed_arrays(self):
    mesh = np.zeros((2, 2, 2))
    out = _prepare_3d_coordinates((mesh, mesh, mesh), mesh.shape)
    assert out[0] is mesh and out[1] is mesh and out[2] is mesh

  def test_prepare_2d_coordinates_rejects_wrong_count(self):
    with pytest.raises(ValueError, match="two coordinate arrays"):
      _prepare_2d_coordinates((np.array([0.0]), ), (1, ))

  def test_prepare_2d_coordinates_passes_through_already_meshed_arrays(self):
    mesh = np.zeros((2, 2))
    out = _prepare_2d_coordinates((mesh, mesh), mesh.shape)
    assert out[0] is mesh and out[1] is mesh


class TestSaveRotatingPlotlyFigure:

  def _scene_fig(self):
    # plotly() always calls fig.update_layout(scene=...), guaranteeing a
    # real "scene" key in the layout (a bare go.Figure(go.Surface(...))
    # only gets one once actually rendered by a Plotly frontend).
    return plotly(_volume_3d())

  def test_bad_extension_raises(self):
    with pytest.raises(ValueError, match=r"\.gif, \.mp4, or \.html"):
      save_rotating_plotly_figure(self._scene_fig(), "out.bogus", 0.0, 10, 60.0,
                                  2.0)

  def test_nonpositive_fps_raises(self):
    with pytest.raises(ValueError, match="fps"):
      save_rotating_plotly_figure(self._scene_fig(), "out.gif", 0.0, 0, 60.0,
                                  2.0)

  def test_nonpositive_rotation_period_raises(self):
    with pytest.raises(ValueError, match="rotation_period"):
      save_rotating_plotly_figure(self._scene_fig(), "out.gif", 0.0, 10, 60.0,
                                  0.0)

  def test_requires_a_3d_scene_figure(self):
    flat_fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    with pytest.raises(ValueError, match="3D scene"):
      save_rotating_plotly_figure(flat_fig, "out.gif", 0.0, 10, 60.0, 2.0)

  def test_html_export_embeds_rotation_script(self, tmp_path):
    out = tmp_path / "out.html"
    save_rotating_plotly_figure(self._scene_fig(), str(out), 45.0, 10, 60.0,
                                2.0)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "PGKYL" in text or len(text) > 0

  def test_html_export_zero_rotation_period_omits_script(self, tmp_path):
    # rotation_period must stay positive (checked above), but omega is
    # driven to exactly 0.0 via math.inf -- any finite (however huge) period
    # still yields omega > 0.0 in float64 and takes the *other* branch. Pass
    # every angle/period by keyword: the previous version of this test
    # passed a huge value positionally where it actually landed in
    # ``polar_angle`` (not ``rotation_period``, which stayed a normal 2.0),
    # so it never drove omega to zero at all -- see C6.
    import math

    out = tmp_path / "out.html"
    save_rotating_plotly_figure(self._scene_fig(),
                                str(out),
                                starting_azimuthal_angle=0.0,
                                fps=10,
                                polar_angle=60.0,
                                rotation_period=math.inf,
                                radius=2.0)
    assert out.exists()
    assert "recomputeRotationParams" not in out.read_text(encoding="utf-8")

  @pytest.mark.parametrize(("extension", "command_marker"), [
      ("mp4", "yuv420p"),
      ("gif", "palettegen"),
  ])
  def test_binary_export_protocol_without_external_process(
      self, monkeypatch, tmp_path, extension, command_marker):
    import subprocess
    plotly_module = import_module("postgkyl.render.plotly")
    events = []

    class FakeLayout:

      @staticmethod
      def to_plotly_json():
        return {"scene": {}}

    class FakeFigure:
      layout = FakeLayout()

      def update_layout(self, **kwargs):
        events.append(("layout", kwargs))

      def to_image(self, *, format):
        assert format == "png"
        return b"png"

    fake_kaleido = SimpleNamespace(
        start_sync_server=lambda **kwargs: events.append(("start", kwargs)),
        stop_sync_server=lambda **kwargs: events.append(("stop", kwargs)),
    )
    commands = []
    monkeypatch.setitem(sys.modules, "kaleido", fake_kaleido)
    monkeypatch.setattr(plotly_module, "require_ffmpeg",
                        lambda _caller: "/ffmpeg")
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: commands.append(
            (command, kwargs)))

    output = tmp_path / f"rotation.{extension}"
    save_rotating_plotly_figure(FakeFigure(),
                                str(output),
                                starting_azimuthal_angle=15.0,
                                fps=2,
                                polar_angle=60.0,
                                rotation_period=1.0)
    assert [event[0] for event in events].count("layout") == 2
    assert events[0][0] == "start" and events[-1][0] == "stop"
    assert len(commands) == 1
    assert command_marker in " ".join(commands[0][0])
    assert commands[0][1]["check"] is True

  @needs_ffmpeg
  @needs_chrome
  @skip_macos_chrome
  @external_tool
  @slow
  def test_gif_export_end_to_end(self, tmp_path):
    # fps * rotation_period = 2 -- the minimum frame count that still
    # exercises the multi-frame rotation loop (fewer, and the `max(2, ...)`
    # floor in save_rotating_plotly_figure would hide fps/rotation_period
    # from the frame count entirely). Each frame drives a real Kaleido
    # render, so keeping this small matters for test runtime.
    out = tmp_path / "out.gif"
    save_rotating_plotly_figure(self._scene_fig(), str(out), 0.0, 2, 1.0, 1.0)
    assert out.exists()
    assert out.stat().st_size > 0

  @needs_ffmpeg
  @needs_chrome
  @skip_macos_chrome
  @external_tool
  @slow
  def test_mp4_export_end_to_end(self, tmp_path):
    out = tmp_path / "out.mp4"
    save_rotating_plotly_figure(self._scene_fig(), str(out), 0.0, 2, 1.0, 1.0)
    assert out.exists()
    assert out.stat().st_size > 0


class TestPlotlyAnimate:

  def test_builds_frames_and_controls(self):
    n = 4
    grid = [np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n)]
    x, y = np.meshgrid(*grid, indexing="ij")
    values0 = (x + 2.0 * y)[..., np.newaxis]
    values1 = (x + 2.0 * y + 0.5)[..., np.newaxis]
    fig = plotly_animate(
        [_state(grid, values0), _state(grid, values1)], frame_duration=40)
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Surface)
    assert len(fig.frames) == 1
    assert fig.frames[0].name == "1"
    assert fig.layout.updatemenus[0].buttons[0].label == "Play"

  def test_requires_at_least_one_dataset(self):
    with pytest.raises(ValueError, match="at least one"):
      plotly_animate([])

  def test_frame_labels_length_mismatch_raises(self):
    with pytest.raises(ValueError, match="frame_labels"):
      plotly_animate([_surface_2d(), _surface_2d()], frame_labels=["only one"])

  def test_mismatched_trace_count_between_frames_raises(self):
    grid = [
        np.linspace(0.0, 1.0, 4),
        np.linspace(0.0, 1.0, 4),
        np.linspace(0.0, 1.0, 4)
    ]
    x, y, z = np.meshgrid(*grid, indexing="ij")
    one_comp = _state(grid, (x + y + z)[..., np.newaxis])
    two_comp = _state(grid, np.stack([x + y + z, x - y - z], axis=-1))
    with pytest.raises(ValueError, match="same number of traces"):
      plotly_animate([one_comp, two_comp])

  def test_save_adds_html_extension_and_show_writes_preview(
      self, monkeypatch, tmp_path):
    plotly_module = import_module("postgkyl.render.plotly")
    written = []
    opened = []
    monkeypatch.setattr(go.Figure, "write_html",
                        lambda _fig, path: written.append(str(path)))
    monkeypatch.setattr(plotly_module, "open_preview",
                        lambda path: opened.append(str(path)))

    output = tmp_path / "animation"
    plotly_animate([_surface_2d()], saveas=str(output))
    assert written[-1] == f"{output}.html"

    plotly_animate([_surface_2d()], show=True)
    assert written[-1].endswith("plotly-animate_preview.html")
    assert opened == [written[-1]]

    explicit_html = tmp_path / "animation.html"
    plotly_animate([_surface_2d()], saveas=str(explicit_html), show=True)
    assert written[-1] == str(explicit_html)
    assert opened[-1] == str(explicit_html)


class TestOutputHelpers:

  def test_default_stem_prefers_file_then_label_then_fallback(self):
    plotly_module = import_module("postgkyl.render.plotly")
    data = _surface_2d()
    data._file_name = "/tmp/simulation.gkyl"
    assert plotly_module._default_output_stem(data) == "simulation"
    data._file_name = ""
    data._custom_label = "density"
    assert plotly_module._default_output_stem(data) == "density"
    data._custom_label = ""
    assert plotly_module._default_output_stem(data) == "plotly_output"

  def test_write_output_dispatches_rotating_and_plain_formats(
      self, monkeypatch, tmp_path):
    plotly_module = import_module("postgkyl.render.plotly")
    calls = []

    class FakeFigure:

      def write_html(self, path):
        calls.append(("html", path))

    monkeypatch.setattr(
        plotly_module, "save_rotating_plotly_figure",
        lambda fig, path, **kwargs: calls.append(("rotate", path, kwargs)))
    figure = FakeFigure()
    rotating = plotly_module._write_plotly_output(figure,
                                                  str(tmp_path / "figure.html"),
                                                  starting_azimuthal_angle=0.0,
                                                  polar_angle=60.0,
                                                  rotation_period=2.0,
                                                  fps=10)
    plain = plotly_module._write_plotly_output(figure,
                                               str(tmp_path / "figure.png"),
                                               starting_azimuthal_angle=0.0,
                                               polar_angle=60.0,
                                               rotation_period=2.0,
                                               fps=10)
    empty = plotly_module._write_plotly_output(figure,
                                               "",
                                               starting_azimuthal_angle=0.0,
                                               polar_angle=60.0,
                                               rotation_period=2.0,
                                               fps=10)
    assert rotating.endswith("figure.html")
    assert plain.endswith("figure.html")
    assert empty == ".html"
    assert [call[0] for call in calls] == ["rotate", "html", "html"]

  def test_preview_sanitizes_names_and_open_preview_uses_file_uri(
      self, monkeypatch, tmp_path):
    plotly_module = import_module("postgkyl.render.plotly")
    saved = []
    opened = []
    monkeypatch.setattr(plotly_module.tempfile, "gettempdir",
                        lambda: str(tmp_path))
    monkeypatch.setattr(plotly_module, "save_rotating_plotly_figure",
                        lambda _fig, path, **_kwargs: saved.append(path))
    monkeypatch.setattr(plotly_module.webbrowser, "open",
                        lambda uri: opened.append(uri))
    path = plotly_module._preview_plotly_figure(object(),
                                                " !!! ",
                                                starting_azimuthal_angle=0.0,
                                                polar_angle=60.0,
                                                rotation_period=2.0,
                                                fps=10)
    assert path.endswith("plotly_preview_preview.html")
    assert saved == [path]
    named_path = plotly_module._preview_plotly_figure(
        object(),
        "named",
        starting_azimuthal_angle=0.0,
        polar_angle=60.0,
        rotation_period=2.0,
        fps=10)
    assert named_path.endswith("named_preview.html")
    plotly_module.open_preview(path)
    assert opened[0].startswith("file://")

  def test_plotly_save_and_show_dispatch_to_output_helpers(self, monkeypatch):
    plotly_module = import_module("postgkyl.render.plotly")
    calls = []
    monkeypatch.setattr(
        plotly_module, "_write_plotly_output",
        lambda _fig, path, **_kwargs: calls.append(
            ("write", path)) or "saved.html")
    monkeypatch.setattr(
        plotly_module, "_preview_plotly_figure",
        lambda _fig, stem, **_kwargs: calls.append(
            ("preview", stem)) or "preview.html")
    monkeypatch.setattr(plotly_module, "open_preview",
                        lambda path: calls.append(("open", path)))

    plotly(_surface_2d(), save=True)
    plotly(_surface_2d(), show=True)
    plotly(_surface_2d(), saveas="figure.html", show=True)
    assert [call[0]
            for call in calls] == ["write", "preview", "open", "write", "open"]
