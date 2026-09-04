"""Tests for postgkyl.render._prep -- the dataset -> plottable-array prep
shared by every render backend (formerly axis_and_grid_prep + load_plot_data)."""

from __future__ import annotations

import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.render._prep import (
    default_axis_labels,
    format_axis_label,
    prep_plot_data,
    resolve_axis_labels,
    squeeze_collapsed_axes,
    subplot_grid,
)

# --------------------------------------------------------------------------
# default_axis_labels / format_axis_label
# --------------------------------------------------------------------------


class TestDefaultAxisLabels:

  def test_returns_one_label_per_dim(self):
    labels = default_axis_labels(3)
    assert labels == [r"$z_0$", r"$z_1$", r"$z_2$"]

  def test_zero_dims_is_empty(self):
    assert default_axis_labels(0) == []


class TestFormatAxisLabel:

  def test_no_shift_no_scale_passthrough(self):
    assert format_axis_label("x", 0.0, 1.0) == "x"

  def test_shift_only(self):
    result = format_axis_label("x", 1.0, 1.0)
    assert result == r"x + 1.00e+00"

  def test_scale_only(self):
    result = format_axis_label("x", 0.0, 2.0)
    assert result == r"x $\times$ 2.00e+00"

  def test_shift_and_scale(self):
    result = format_axis_label("x", 1.0, 2.0)
    assert result == r"(x + 1.00e+00) $\times$ 2.00e+00"


# --------------------------------------------------------------------------
# resolve_axis_labels
# --------------------------------------------------------------------------


class TestResolveAxisLabels:

  def test_defaults_for_2d(self):
    xl, yl, zl, cl = resolve_axis_labels(xlabel=None,
                                         ylabel=None,
                                         zlabel=None,
                                         clabel="",
                                         num_dims=2)
    assert xl == r"$z_0$"
    assert yl == r"$z_1$"

  def test_1d_has_no_default_ylabel(self):
    xl, yl, zl, cl = resolve_axis_labels(xlabel=None,
                                         ylabel=None,
                                         zlabel=None,
                                         clabel="",
                                         num_dims=1)
    assert xl == r"$z_0$"
    assert yl == ""

  def test_custom_labels_pass_through(self):
    xl, yl, zl, cl = resolve_axis_labels(xlabel="myX",
                                         ylabel="myY",
                                         zlabel="myZ",
                                         clabel="myC",
                                         num_dims=2,
                                         zscale=2.0)
    assert xl == "myX"
    assert yl == "myY"
    assert "2.00" in cl

  def test_3d_zlabel_defaults_to_third_axis(self):
    xl, yl, zl, cl = resolve_axis_labels(xlabel=None,
                                         ylabel=None,
                                         zlabel=None,
                                         clabel="",
                                         num_dims=3)
    assert zl == r"$z_2$"

  def test_clabel_annotated_with_zscale(self):
    _, _, _, cl = resolve_axis_labels(xlabel=None,
                                      ylabel=None,
                                      zlabel=None,
                                      clabel="density",
                                      num_dims=2,
                                      zscale=3.0)
    assert cl == r"density $\times$ 3.000e+00"

  def test_clabel_zscale_with_no_base_label(self):
    _, _, _, cl = resolve_axis_labels(xlabel=None,
                                      ylabel=None,
                                      zlabel=None,
                                      clabel="",
                                      num_dims=2,
                                      zscale=3.0)
    assert cl == r"$\times$ 3.000e+00"


# --------------------------------------------------------------------------
# squeeze_collapsed_axes
# --------------------------------------------------------------------------


class TestSqueezeCollapsedAxes:

  def test_no_collapsed_axes_is_a_passthrough(self):
    grid = [np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 4)]
    values = np.ones((4, 3, 2))
    out_grid, out_values = squeeze_collapsed_axes(grid, values)
    assert len(out_grid) == 2
    assert out_values.shape == (4, 3, 2)

  def test_drops_a_singleton_axis(self):
    x = np.linspace(0.0, 1.0, 4)
    y = np.array([0.5, 0.6])  # 1-cell axis (select()-ed)
    z = np.linspace(-1.0, 1.0, 5)
    values = np.zeros((3, 1, 4, 2))
    grid, out_values = squeeze_collapsed_axes([x, y, z], values)
    assert len(grid) == 2
    assert out_values.shape == (3, 4, 2)

  def test_drops_multiple_singleton_axes(self):
    x = np.linspace(0.0, 1.0, 4)
    y = np.array([0.0])
    z = np.array([0.0])
    values = np.zeros((3, 1, 1, 2))
    grid, out_values = squeeze_collapsed_axes([x, y, z], values)
    assert len(grid) == 1
    assert out_values.shape == (3, 2)

  def test_curvilinear_axis_is_averaged_not_indexed(self):
    # A 2-D (curvilinear) coordinate array spanning both dims; dropping dim 1
    # (a singleton) should mean-reduce dim 1 out of the coordinate array too.
    x2d = np.arange(12.0).reshape(4, 3)  # (dim0=4 edges, dim1=3 edges)
    y2d = np.arange(12.0).reshape(4, 3) * 2.0
    values = np.zeros((3, 1, 2))  # 3 cells in dim0, 1 cell in dim1
    grid, out_values = squeeze_collapsed_axes([x2d, y2d], values)
    assert len(grid) == 1
    assert grid[0].shape == (4, )
    np.testing.assert_allclose(grid[0], np.mean(x2d, axis=1))
    assert out_values.shape == (3, 2)


# --------------------------------------------------------------------------
# subplot_grid
# --------------------------------------------------------------------------


class TestSubplotGrid:

  def test_perfect_square(self):
    assert subplot_grid(4) == (2, 2)

  def test_single_panel(self):
    assert subplot_grid(1) == (1, 1)

  def test_non_square_uses_near_square_layout(self):
    rows, cols = subplot_grid(3)
    assert rows * cols >= 3

  def test_explicit_num_rows(self):
    assert subplot_grid(6, num_rows=2) == (2, 3)

  def test_explicit_num_cols(self):
    assert subplot_grid(6, num_cols=3) == (2, 3)

  def test_five_panels_layout(self):
    rows, cols = subplot_grid(5)
    assert rows * cols >= 5
    assert rows * cols <= 6


# --------------------------------------------------------------------------
# prep_plot_data
# --------------------------------------------------------------------------


def _make_state(grid, values) -> GDataState:
  d = GDataState()
  d.push(grid, values)
  return d


class TestPrepPlotData:

  def test_1d_basic(self):
    grid = [np.linspace(0.0, 1.0, 9)]
    values = np.ones((8, 1))
    panel = prep_plot_data(_make_state(grid, values))
    assert panel.num_dims == 1
    assert panel.num_comps == 1
    assert panel.xlabel == r"$z_0$"
    assert panel.ylabel == ""

  def test_2d_basic(self):
    grid = [np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 6)]
    values = np.ones((4, 5, 2))
    panel = prep_plot_data(_make_state(grid, values))
    assert panel.num_dims == 2
    assert panel.num_comps == 2
    assert panel.xlabel == r"$z_0$"
    assert panel.ylabel == r"$z_1$"

  def test_squeezes_a_selected_axis(self):
    x = np.linspace(0.0, 1.0, 4)
    y = np.array([0.4, 0.6])
    values = np.zeros((3, 1, 2))
    panel = prep_plot_data(_make_state([x, y], values))
    assert panel.num_dims == 1
    assert panel.values.shape == (3, 2)

  def test_custom_xlabel_overrides_default(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    values = np.ones((4, 1))
    panel = prep_plot_data(_make_state(grid, values), xlabel="time")
    assert panel.xlabel == "time"

  def test_clabel_gets_zscale_annotation(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    values = np.ones((4, 1))
    panel = prep_plot_data(_make_state(grid, values), clabel="n_e", zscale=2.0)
    assert "2.00" in panel.clabel
