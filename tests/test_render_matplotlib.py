"""Tests for postgkyl.render.matplotlib -- multi-panel figures, the pgkyl
colorbar, log axes, vmin/vmax, aspect, and mapped (curvilinear) grids.

``render.plot``'s basic single/multi-dataset 1-D and 2-D behaviour is already
covered by ``tests/test_coverage_leaf.py`` and ``tests/test_postgkyl.py``;
this file focuses on the features layer 09 adds on top.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, operations
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.render import matplotlib as backend

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
    reason="no compiled Gkeyll (libg0core.so) found")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
GEN = os.path.join(DATA, "generated")


def _line(n=8, offset=0.0) -> GDataState:
  d = GDataState()
  d.push([np.linspace(0.0, 1.0, n + 1)], (np.arange(n, dtype=float) + offset)[:, None])
  return d
# end


def _field_2d(n=8, ncomp=1) -> GDataState:
  d = GDataState()
  grid = [np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, n + 1)]
  values = np.stack([np.arange(n * n, dtype=float).reshape(n, n) + 10.0 * c
      for c in range(ncomp)], axis=-1)
  d.push(grid, values)
  return d
# end


@pytest.fixture(autouse=True)
def _close_figs():
  plt.close("all")
  yield
  plt.close("all")
# end


# --------------------------------------------------------------------------
# Multi-panel (multi-component) layout
# --------------------------------------------------------------------------

class TestMultiPanel:
  def test_two_components_get_two_axes(self):
    fig = backend.plot(_field_2d(ncomp=2), show=False)
    assert len(fig.axes) >= 2
  # end

  def test_four_components_use_a_square_grid(self):
    fig = backend.plot(_field_2d(ncomp=4), show=False)
    # 4 components -> 2x2 grid -> 4 panel axes, each with its own colorbar axes.
    assert len(fig.axes) == 8
  # end

  def test_five_components_hides_the_leftover_axis(self):
    fig = backend.plot(_field_2d(ncomp=5), show=False)
    off_axes = [ax for ax in fig.axes if not ax.axison]
    assert len(off_axes) == 1
  # end

  def test_single_component_has_no_per_panel_title(self):
    fig = backend.plot(_field_2d(ncomp=1), show=False)
    assert fig.axes[0].get_title() == ""
  # end

  def test_legend_can_be_limited_to_one_subplot_and_relocated(self):
    a = _line()
    b = _line(offset=3.0)
    a.values = np.column_stack((a.values[:, 0], a.values[:, 0] + 1.0))
    b.values = np.column_stack((b.values[:, 0], b.values[:, 0] + 1.0))

    fig = backend.plot(a, b, multiblock=True, show=False,
        legend_labels=["first", "second"],
        legend_subplot=1, legend_loc="lower left")

    assert fig.axes[0].get_legend() is None
    legend = fig.axes[1].get_legend()
    assert legend is not None
    assert legend._loc == 3  # Matplotlib's code for "lower left".
    assert [text.get_text() for text in legend.get_texts()] == ["first", "second"]
  # end

  def test_legend_subplot_rejects_an_out_of_range_index(self):
    with pytest.raises(ValueError, match="between 0 and 0"):
      backend.plot(_line(), show=False, legend_subplot=1)
    # end
  # end
# end


# --------------------------------------------------------------------------
# The pgkyl colorbar
# --------------------------------------------------------------------------

class TestColorbar:
  def test_colorbar_true_adds_an_axes(self):
    fig = backend.plot(_field_2d(), show=False, colorbar=True)
    assert len(fig.axes) == 2  # the panel + the appended colorbar axes
  # end

  def test_colorbar_false_omits_it(self):
    fig = backend.plot(_field_2d(), show=False, colorbar=False)
    assert len(fig.axes) == 1
  # end

  def test_clabel_reaches_the_colorbar(self):
    fig = backend.plot(_field_2d(), show=False, colorbar=True, clabel="density")
    cbar_ax = fig.axes[1]
    assert cbar_ax.get_ylabel() == "density"
  # end
# end


# --------------------------------------------------------------------------
# Log axes
# --------------------------------------------------------------------------

class TestLogAxes:
  def test_logx_1d(self):
    fig = backend.plot(_line(), show=False, logx=True)
    assert fig.axes[0].get_xscale() == "log"
  # end

  def test_logy_1d(self):
    fig = backend.plot(_line(), show=False, logy=True)
    assert fig.axes[0].get_yscale() == "log"
  # end

  def test_logz_uses_lognorm_on_2d_colormap(self):
    d = _field_2d()
    d.values[...] = d.values + 1.0  # keep strictly positive for LogNorm
    fig = backend.plot(d, show=False, logz=True)
    im = fig.axes[0].collections[0]
    from matplotlib.colors import LogNorm
    assert isinstance(im.norm, LogNorm)
  # end
# end


# --------------------------------------------------------------------------
# Grid indices
# --------------------------------------------------------------------------

class TestGridIndices:
  def test_1d_uses_zero_based_indices_instead_of_grid_values(self):
    data = GDataState()
    time = np.array([0.0, 0.1, 0.4, 1.2])
    data.push([time], np.arange(time.size, dtype=float)[:, None])

    fig = backend.plot(data, show=False, grid_indices=True)

    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(),
        np.arange(time.size))
    assert fig.get_supxlabel() == r"$i_0$"
    np.testing.assert_array_equal(data.grid[0], time)
  # end

  def test_2d_puts_cell_centers_at_integer_indices(self):
    data = _field_2d(n=4)

    fig = backend.plot(data, show=False, grid_indices=True, colorbar=False)

    coordinates = fig.axes[0].collections[0].get_coordinates()
    assert coordinates[..., 0].min() == pytest.approx(-0.5)
    assert coordinates[..., 0].max() == pytest.approx(3.5)
    assert coordinates[..., 1].min() == pytest.approx(-0.5)
    assert coordinates[..., 1].max() == pytest.approx(3.5)
    assert fig.get_supxlabel() == r"$i_0$"
    assert fig.get_supylabel() == r"$i_1$"
  # end
# end


# --------------------------------------------------------------------------
# Joined linear/log split panels
# --------------------------------------------------------------------------

class TestSplitLinearLog:
  @staticmethod
  def _split_line(ncomp=1):
    d = GDataState()
    # Cell centers are exactly [-2, -1, 0, 1, 2], pinning the split-point
    # ownership rule (the point itself belongs to the right panel).
    grid = [np.linspace(-2.5, 2.5, 6)]
    base = np.array([1.0, 2.0, 3.0, 10.0, 100.0])
    values = np.stack([base * (comp + 1) for comp in range(ncomp)], axis=-1)
    d.push(grid, values)
    return d
  # end

  def test_each_component_becomes_a_linear_left_log_right_pair(self):
    fig = backend.plot(self._split_line(ncomp=2), show=False,
        split_linear_log=True)

    assert len(fig.axes) == 4
    assert [axis.get_yscale() for axis in fig.axes] == [
        "linear", "log", "linear", "log"]
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [-2.0, -1.0])
    np.testing.assert_allclose(fig.axes[1].lines[0].get_xdata(), [0.0, 1.0, 2.0])
    assert fig.axes[0].get_xlim()[1] == pytest.approx(0.0)
    assert fig.axes[1].get_xlim()[0] == pytest.approx(0.0)
  # end

  def test_split_point_and_log_side_are_configurable(self):
    fig = backend.plot(self._split_line(), show=False, split_linear_log=True,
        split_point=1.0, split_log_side="left", split_log_base=2)

    left, right = fig.axes
    assert left.get_yscale() == "log"
    assert right.get_yscale() == "linear"
    assert left.yaxis._scale.base == 2
    np.testing.assert_allclose(left.lines[0].get_xdata(), [-2.0, -1.0, 0.0])
    np.testing.assert_allclose(right.lines[0].get_xdata(), [1.0, 2.0])
  # end

  def test_per_component_linear_and_log_limits(self):
    fig = backend.plot(self._split_line(ncomp=2), show=False,
        split_linear_log=True,
        split_linear_ylim=[(0.0, 5.0), (-2.0, 8.0)],
        split_log_ylim={0: (1.0, 200.0), 1: (2.0, 400.0)})

    assert fig.axes[0].get_ylim() == (0.0, 5.0)
    assert fig.axes[1].get_ylim() == (1.0, 200.0)
    assert fig.axes[2].get_ylim() == (-2.0, 8.0)
    assert fig.axes[3].get_ylim() == (2.0, 400.0)
  # end

  def test_shared_limit_pair_applies_to_every_component(self):
    fig = backend.plot(self._split_line(ncomp=2), show=False,
        split_linear_log=True, split_linear_ylim=(0.0, 10.0))
    assert fig.axes[0].get_ylim() == (0.0, 10.0)
    assert fig.axes[2].get_ylim() == (0.0, 10.0)
  # end

  def test_legend_subplot_uses_log_half_of_logical_subplot(self):
    a = self._split_line(ncomp=2)
    b = self._split_line(ncomp=2)
    fig = backend.plot(a, b, multiblock=True, show=False, split_linear_log=True,
        legend_labels=["a", "b"], legend_subplot=1, split_legend_side="log")

    assert fig.axes[0].get_legend() is None
    assert fig.axes[1].get_legend() is None
    assert fig.axes[2].get_legend() is None
    legend = fig.axes[3].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["a", "b"]
  # end

  def test_pair_geometry_and_logical_labels(self):
    fig = backend.plot(self._split_line(), show=False, split_linear_log=True,
        split_width_ratios=(2.0, 1.0), split_gap=0.05,
        subplot_titles="density", subplot_ylabels="n", subplot_xlabels="z")
    left, right = fig.axes
    assert left.get_position().width / right.get_position().width == pytest.approx(2.0)
    assert left.get_title() == "density"
    assert left.get_ylabel() == "n"
    assert left.get_xlabel() == "z"
    assert right.yaxis.get_ticks_position() == "right"
  # end

  def test_left_owns_seam_tick_label_by_default(self):
    fig = backend.plot(self._split_line(), show=False, split_linear_log=True)
    left_ticks = fig.axes[0].get_xticks()
    right_ticks = fig.axes[1].get_xticks()
    assert left_ticks[-1] == pytest.approx(0.0)
    assert right_ticks[0] > 0.0
  # end

  @pytest.mark.parametrize("kwargs, message", [
      ({"split_log_side": "middle"}, "split_log_side"),
      ({"split_legend_side": "middle"}, "split_legend_side"),
      ({"split_width_ratios": (1.0, 0.0)}, "split_width_ratios"),
      ({"split_gap": -0.1}, "split_gap"),
      ({"split_log_nonpositive": "drop"}, "split_log_nonpositive"),
      ({"split_seam_ticklabels": "middle"}, "split_seam_ticklabels"),
      ({"split_log_base": 1.0}, "split_log_base"),
  ])
  def test_invalid_split_options_raise(self, kwargs, message):
    with pytest.raises(ValueError, match=message):
      backend.plot(self._split_line(), show=False, split_linear_log=True, **kwargs)
    # end
  # end

  def test_split_rejects_2d_transpose_and_logy(self):
    with pytest.raises(ValueError, match="only supported for 1D"):
      backend.plot(_field_2d(), show=False, split_linear_log=True)
    # end
    with pytest.raises(ValueError, match="transpose"):
      backend.plot(self._split_line(), show=False, split_linear_log=True,
          transpose=True)
    # end
    with pytest.raises(ValueError, match="logy"):
      backend.plot(self._split_line(), show=False, split_linear_log=True,
          logy=True)
    # end
  # end
# end


# --------------------------------------------------------------------------
# value ranges: ymin/ymax (1-D), zmin/zmax (2-D color range)
# --------------------------------------------------------------------------

class TestValueRange:
  def test_ymin_ymax_set_1d_ylim(self):
    fig = backend.plot(_line(), show=False, ymin=-5.0, ymax=50.0)
    assert fig.axes[0].get_ylim() == (-5.0, 50.0)
  # end

  def test_zmin_zmax_set_2d_colormap_range(self):
    fig = backend.plot(_field_2d(), show=False, zmin=0.0, zmax=1.0)
    im = fig.axes[0].collections[0]
    assert im.get_clim() == (0.0, 1.0)
  # end
# end


# --------------------------------------------------------------------------
# Aspect
# --------------------------------------------------------------------------

class TestAspect:
  def test_aspect_applies_to_2d_axes(self):
    # aspect only takes effect with fixaspect=True -- --aspect on the CLI
    # implies --fix-aspect (see cli/commands/plot.py), but the render engine
    # itself keeps the two independent, exactly as main's output.plot did.
    fig = backend.plot(_field_2d(), show=False, fixaspect=True, aspect=1.0)
    assert fig.axes[0].get_aspect() == 1.0
  # end

  def test_aspect_none_leaves_default(self):
    fig = backend.plot(_field_2d(), show=False)
    assert fig.axes[0].get_aspect() == "auto"
  # end
# end


# --------------------------------------------------------------------------
# cmap / diverging
# --------------------------------------------------------------------------

class TestColormap:
  def test_explicit_cmap_is_used(self):
    fig = backend.plot(_field_2d(), show=False, cmap="plasma")
    im = fig.axes[0].collections[0]
    assert im.get_cmap().name == "plasma"
  # end

  def test_diverging_uses_rdbu(self):
    fig = backend.plot(_field_2d(), show=False, diverging=True)
    im = fig.axes[0].collections[0]
    assert im.get_cmap().name == "RdBu_r"
  # end
# end


# --------------------------------------------------------------------------
# style / rcParams
# --------------------------------------------------------------------------

class TestStyleAndRcParams:
  def test_style_kwarg_applies_named_style(self):
    backend.plot(_line(), show=False, style="default")
    import matplotlib as mpl
    assert mpl.rcParams["image.cmap"] == "viridis"
  # end

  def test_rcparams_dict_overrides(self):
    backend.plot(_line(), show=False, rcParams={"lines.linewidth": 5.0})
    import matplotlib as mpl
    assert mpl.rcParams["lines.linewidth"] == 5.0
  # end
# end


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

class TestSaving:
  @pytest.mark.parametrize("extension", [".png", ".pdf"])
  def test_saveas_writes_supported_formats(self, tmp_path, extension):
    output = tmp_path / f"figure{extension}"
    fig = backend.plot(_line(), show=False, saveas=output)

    assert output.exists()
    assert fig is plt.gcf()
  # end

  def test_extensionless_saveas_defaults_to_png(self, tmp_path):
    output = tmp_path / "figure"
    backend.plot(_line(), show=False, saveas=output)

    assert (tmp_path / "figure.png").exists()
  # end

  def test_empty_saveas_is_inert(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend.plot(_line(), show=False, saveas="")
    assert list(tmp_path.iterdir()) == []
  # end

  def test_save_true_derives_a_png_name(self, tmp_path, monkeypatch):
    data = _line()
    data._file_name = "/input/run.gkyl"
    monkeypatch.chdir(tmp_path)

    backend.plot(data, show=False, save=True)

    assert (tmp_path / "run.png").exists()
  # end

  def test_saveas_sequence_writes_each_requested_format(self, tmp_path):
    outputs = [tmp_path / "figure.png", tmp_path / "figure.pdf"]
    backend.plot(_line(), show=False, saveas=outputs)
    assert all(output.exists() for output in outputs)
  # end

  def test_unsupported_save_extension_raises(self, tmp_path):
    with pytest.raises(ValueError, match="Supported formats are: .png, .pdf"):
      backend.plot(_line(), show=False, saveas=tmp_path / "figure.svg")
    # end
  # end
# end


# --------------------------------------------------------------------------
# fig reuse (the hook render.animate needs)
# --------------------------------------------------------------------------

class TestFigureReuse:
  def test_reusing_a_figure_clears_previous_axes(self):
    fig = plt.figure()
    backend.plot(_line(), show=False, figure=fig, clear=True)
    first_axes_id = id(fig.axes[0])
    backend.plot(_line(offset=5.0), show=False, figure=fig, clear=True)
    assert len(fig.axes) == 1
    assert id(fig.axes[0]) != first_axes_id
  # end
# end


# --------------------------------------------------------------------------
# transpose -- swap the horizontal and vertical axes (upstream PR #225)
# --------------------------------------------------------------------------

def _field_2d_rect(n0=4, n1=8) -> GDataState:
  d = GDataState()
  grid = [np.linspace(0.0, 1.0, n0 + 1), np.linspace(0.0, 2.0, n1 + 1)]
  values = np.arange(n0 * n1, dtype=float).reshape(n0, n1)[..., None]
  d.push(grid, values)
  return d
# end


class TestTranspose:
  def test_1d_puts_the_coordinate_on_the_vertical_axis(self):
    fig = backend.plot(_line(), show=False, transpose=True)
    line = fig.axes[0].lines[0]
    edges = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(line.get_ydata(), 0.5 * (edges[:-1] + edges[1:]))
    np.testing.assert_allclose(line.get_xdata(), np.arange(8, dtype=float))
  # end

  def test_1d_default_label_follows_the_coordinate(self):
    fig = backend.plot(_line(), show=False, transpose=True)
    assert fig.get_supylabel() == r"$z_0$"
    assert fig.get_supxlabel() == ""
  # end

  def test_2d_swaps_the_mesh_axes(self):
    n0, n1 = 4, 8
    d = _field_2d_rect(n0, n1)
    fig = backend.plot(d, show=False, transpose=True)
    im = fig.axes[0].collections[0]
    # The horizontal axis now carries dimension 1 (extent 0..2), the
    # vertical dimension 0 (extent 0..1); the quads' value layout follows.
    assert im.get_coordinates().shape == (n0 + 1, n1 + 1, 2)
    np.testing.assert_allclose(fig.axes[0].get_xlim(), (0.0, 2.0))
    np.testing.assert_allclose(fig.axes[0].get_ylim(), (0.0, 1.0))
    np.testing.assert_allclose(
        np.asarray(im.get_array()).reshape(n0, n1), d.values[..., 0])
  # end

  def test_2d_swaps_the_default_labels(self):
    fig = backend.plot(_field_2d_rect(), show=False, transpose=True)
    assert fig.get_supxlabel() == r"$z_1$"
    assert fig.get_supylabel() == r"$z_0$"
  # end

  def test_2d_does_not_mutate_the_dataset(self):
    d = _field_2d_rect()
    cells_before = d.num_cells.copy()
    values_before = d.values.copy()
    backend.plot(d, show=False, transpose=True)
    np.testing.assert_array_equal(d.num_cells, cells_before)
    np.testing.assert_array_equal(d.values, values_before)
  # end
# end


# --------------------------------------------------------------------------
# Mapped (curvilinear) grids -- MAPPING.md's BACKEND row
# --------------------------------------------------------------------------

@needs_gkeyll
class TestMappedGrids:
  def test_2d_curvilinear_grid_plots_via_pcolormesh(self):
    data = pg.load(os.path.join(GEN, "2d_ms_p1.gkyl")).interpolate()
    mapped = operations.map(data, os.path.join(GEN, "2d_c2p_stretch_ms_p1.gkyl"),
        space="conf")
    assert mapped.grid[0].ndim == 2  # genuinely curvilinear
    fig = mapped.plot(show=False)
    assert fig is not None
    im = fig.axes[0].collections[0]
    assert im.get_array().size > 0
  # end

  def test_1d_non_uniform_mapped_axis_uses_true_centers(self):
    """A 1-D vel map produces non-uniform edges; _centers must handle them
    generically (it already does -- this pins the behaviour)."""
    edges = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # non-uniform, monotone
    d = GDataState()
    d.push([edges], np.arange(4, dtype=float)[:, None])
    fig = backend.plot(d, show=False)
    line = fig.axes[0].lines[0]
    x_plotted = line.get_xdata()
    np.testing.assert_allclose(x_plotted, 0.5 * (edges[:-1] + edges[1:]))
  # end
# end


# --------------------------------------------------------------------------
# surface plots
# --------------------------------------------------------------------------

class TestSurface:
  def test_surface_uses_3d_axes(self):
    fig = backend.plot(_field_2d(), show=False, surface=True)
    assert fig.axes[0].name == "3d"
  # end

  def test_surface_without_comparison_gets_a_colorbar(self):
    fig = backend.plot(_field_2d(), show=False, surface=True)
    assert len(fig.axes) == 2  # the 3D panel + its colorbar
  # end

  def test_surface_alpha_is_applied(self):
    fig = backend.plot(_field_2d(), show=False, surface=True, alpha=0.3)
    poly3d = fig.axes[0].collections[0]
    assert poly3d.get_alpha() == pytest.approx(0.3)
  # end
# end


# --------------------------------------------------------------------------
# multi-dataset 2D overlay comparison (surface/contour)
# --------------------------------------------------------------------------

class TestComparisonOverlay:
  def test_contour_comparison_gives_each_dataset_its_own_color_and_legend(self):
    fig = backend.plot(_field_2d(), _field_2d(), multiblock=True, show=False,
        contour=True, comparison=True, legend_labels=["a", "b"])
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    handles = ax.get_legend().legend_handles
    assert len(handles) == 2
    assert handles[0].get_facecolor() != handles[1].get_facecolor()
  # end

  def test_surface_comparison_gives_each_dataset_its_own_color_and_legend(self):
    fig = backend.plot(_field_2d(), _field_2d(), multiblock=True, show=False,
        surface=True, comparison=True, legend_labels=["a", "b"])
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    assert len(ax.get_legend().legend_handles) == 2
  # end
# end


# --------------------------------------------------------------------------
# cval-based colormap coloring for 1D lines
# --------------------------------------------------------------------------

class TestCvalColoring:
  def test_line_colored_by_cval(self):
    fig = backend.plot(_line(), show=False, cmap="viridis",
        cval=0.0, cval_min=0.0, cval_max=1.0)
    assert fig.axes[0].lines[0].get_color() == plt.get_cmap("viridis")(0.0)
  # end

  def test_second_call_into_the_same_figure_uses_its_own_cval(self):
    fig = backend.plot(_line(), show=False, cmap="viridis",
        cval=0.0, cval_min=0.0, cval_max=1.0)
    backend.plot(_line(offset=1), figure=fig, show=False, cmap="viridis",
        cval=1.0, cval_min=0.0, cval_max=1.0)
    colors = [line.get_color() for line in fig.axes[0].lines]
    assert colors[0] == plt.get_cmap("viridis")(0.0)
    assert colors[1] == plt.get_cmap("viridis")(1.0)
  # end

  def test_cval_without_cmap_is_ignored(self):
    fig = backend.plot(_line(), show=False, cval=0.5, color="red")
    assert fig.axes[0].lines[0].get_color() == "red"
  # end
# end


# --------------------------------------------------------------------------
# Explicit colors for 1D lines
# --------------------------------------------------------------------------

class TestLineColors:
  def test_color_sequence_assigns_one_color_to_each_dataset(self):
    fig = backend.plot(_line(), _line(offset=1), _line(offset=2),
        multiblock=True, show=False,
        color=["tab:red", "tab:green", "tab:blue"])

    assert [line.get_color() for line in fig.axes[0].lines] == [
        "tab:red", "tab:green", "tab:blue"]
  # end

  def test_scalar_color_still_applies_to_every_line(self):
    fig = backend.plot(_line(), _line(offset=1), multiblock=True,
        show=False, color="purple")
    assert [line.get_color() for line in fig.axes[0].lines] == ["purple", "purple"]
  # end

  def test_dataset_colors_repeat_across_component_panels(self):
    a = _line()
    b = _line(offset=2)
    a.values = np.column_stack((a.values[:, 0], a.values[:, 0] + 1))
    b.values = np.column_stack((b.values[:, 0], b.values[:, 0] + 1))

    fig = backend.plot(a, b, multiblock=True, show=False,
        color=["red", "blue"])

    assert [line.get_color() for line in fig.axes[0].lines] == ["red", "blue"]
    assert [line.get_color() for line in fig.axes[1].lines] == ["red", "blue"]
  # end

  def test_color_sequence_follows_dataset_then_component_order(self):
    a = _line()
    b = _line(offset=2)
    a.values = np.column_stack((a.values[:, 0], a.values[:, 0] + 1))
    b.values = np.column_stack((b.values[:, 0], b.values[:, 0] + 1))

    fig = backend.plot(a, b, multiblock=True, show=False,
        color=["red", "orange", "blue", "cyan"])

    assert [line.get_color() for line in fig.axes[0].lines] == ["red", "blue"]
    assert [line.get_color() for line in fig.axes[1].lines] == ["orange", "cyan"]
  # end

  def test_rgb_tuple_remains_a_single_color(self):
    rgb = (0.1, 0.2, 0.3)
    fig = backend.plot(_line(), _line(offset=1), multiblock=True,
        show=False, color=rgb)
    assert [line.get_color() for line in fig.axes[0].lines] == [rgb, rgb]
  # end

  def test_color_sequence_length_must_match_line_count(self):
    with pytest.raises(ValueError, match="2 entries.*expected either 3.*or 3"):
      backend.plot(_line(), _line(offset=1), _line(offset=2),
          multiblock=True, show=False,
          color=["red", "blue"])
    # end
  # end
# end


# --------------------------------------------------------------------------
# Per-dataset linestyles for 1D lines
# --------------------------------------------------------------------------

class TestLineStyles:
  def test_linestyle_sequence_assigns_one_style_to_each_dataset(self):
    fig = backend.plot(_line(), _line(offset=1), multiblock=True, show=False,
        linestyle=["-", "--"])

    assert [line.get_linestyle() for line in fig.axes[0].lines] == ["-", "--"]
  # end

  def test_scalar_linestyle_applies_to_every_dataset(self):
    fig = backend.plot(_line(), _line(offset=1), multiblock=True,
        show=False, linestyle=":")
    assert [line.get_linestyle() for line in fig.axes[0].lines] == [":", ":"]
  # end

  def test_single_entry_sequence_applies_to_every_dataset(self):
    fig = backend.plot(_line(), _line(offset=1), multiblock=True,
        show=False, linestyle=["-."])
    assert [line.get_linestyle() for line in fig.axes[0].lines] == ["-.", "-."]
  # end

  def test_omitted_linestyle_does_not_override_plot_format(self):
    fig = backend.plot(_line(), show=False, args=["--"])
    assert fig.axes[0].lines[0].get_linestyle() == "--"
  # end

  def test_dataset_linestyles_repeat_across_component_panels(self):
    a = _line()
    b = _line(offset=2)
    a.values = np.column_stack((a.values[:, 0], a.values[:, 0] + 1))
    b.values = np.column_stack((b.values[:, 0], b.values[:, 0] + 1))

    fig = backend.plot(a, b, multiblock=True, show=False,
        linestyle=["-", "--"])

    assert [line.get_linestyle() for line in fig.axes[0].lines] == ["-", "--"]
    assert [line.get_linestyle() for line in fig.axes[1].lines] == ["-", "--"]
  # end

  def test_dataset_linestyles_apply_to_both_split_axes(self):
    fig = backend.plot(_line(), _line(offset=1), multiblock=True, show=False,
        linestyle=["-", "--"], split_linear_log=True, split_point=0.5)

    for axis in fig.axes:
      assert [line.get_linestyle() for line in axis.lines] == ["-", "--"]
    # end
  # end

  def test_linestyle_sequence_length_must_match_dataset_count(self):
    with pytest.raises(ValueError, match="2 entries.*expected either 1.*or 3"):
      backend.plot(_line(), _line(offset=1), _line(offset=2),
          multiblock=True, show=False,
          linestyle=["-", "--"])
    # end
  # end
# end
