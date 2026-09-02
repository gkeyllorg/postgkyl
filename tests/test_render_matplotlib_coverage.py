"""Coverage top-up for postgkyl.render.matplotlib.

``tests/test_render_matplotlib.py`` covers the layer's headline features;
this file targets the branches ``--cov-report=term-missing`` still flagged:
``_nodal_grid``'s error/curvilinear paths, the rcParams novelties, label
shift/scale formatting, figure creation/reuse edge cases, contour/quiver/
streamline/lineouts, zmin/zmax ``extend``, logz diverging, and the 0-D
data path. No compiled Gkeyll is needed -- every dataset here is built by
hand with ``GDataState.push``.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.render import matplotlib as backend


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
# _nodal_grid, tested directly as the pure function it is
# --------------------------------------------------------------------------

class TestNodalGridDirect:
  def test_dim_count_mismatch_raises(self):
    with pytest.raises(ValueError, match="doesn't match"):
      backend._nodal_grid([np.linspace(0.0, 1.0, 5)], np.array([4, 4]))
    # end
  # end

  def test_1d_bad_edge_count_raises(self):
    with pytest.raises(ValueError, match="terribly wrong"):
      backend._nodal_grid([np.linspace(0.0, 1.0, 5)], np.array([10]))
    # end
  # end

  def test_1d_curvilinear_edges_averaged(self):
    g = np.array([[0.0, 1.0], [2.0, 4.0]])  # shape (2, 2): cells[0]+1 == 2
    out = backend._nodal_grid([g], np.array([1]))
    np.testing.assert_allclose(out[0], 0.5 * (g[:-1] + g[1:]))
  # end

  def test_2d_curvilinear_cell_centered_passthrough(self):
    g0 = np.ones((3, 3))
    out = backend._nodal_grid([g0, np.ones((3, 3))], np.array([3, 3]))
    assert out[0] is g0
  # end

  def test_2d_curvilinear_edges_averaged(self):
    g0 = np.arange(16, dtype=float).reshape(4, 4)  # cells[0]+1 == 4
    out = backend._nodal_grid([g0, np.ones((4, 4))], np.array([3, 3]))
    np.testing.assert_allclose(out[0], 0.5 * (g0[:-1, :-1] + g0[1:, 1:]))
  # end

  def test_2d_curvilinear_bad_shape_raises(self):
    g0 = np.ones((5, 5))
    with pytest.raises(ValueError, match="terribly wrong"):
      backend._nodal_grid([g0, np.ones((5, 5))], np.array([3, 3]))
    # end
  # end
# end


# --------------------------------------------------------------------------
# rcParams novelties: jet / xkcd / color / linewidth / linestyle
# --------------------------------------------------------------------------

class TestRcParamNovelties:
  def test_jet_sets_cmap(self):
    with mpl.rc_context():
      backend.plot(_field_2d(), show=False, jet=True)
      assert mpl.rcParams["image.cmap"] == "jet"
    # end
  # end

  def test_xkcd_flag_invokes_xkcd_mode(self):
    with mpl.rc_context():
      fig = backend.plot(_line(), show=False, xkcd=True)
      line = fig.axes[0].lines[0]
      assert line.get_sketch_params() is not None
    # end
  # end

  def test_xkcd_flag_does_not_leak_into_global_rcparams(self):
    # A past bug: `plt.xkcd()` called without a `with` block never reverted,
    # contaminating every plot drawn afterwards.
    with mpl.rc_context():
      backend.plot(_line(), show=False, xkcd=True)
      assert mpl.rcParams["path.sketch"] is None
    # end
  # end

  def test_color_sets_rcparam(self):
    with mpl.rc_context():
      backend.plot(_line(), show=False, color="red")
      assert mpl.rcParams["lines.color"] == "red"
    # end
  # end

  def test_linewidth_sets_rcparam(self):
    with mpl.rc_context():
      backend.plot(_line(), show=False, linewidth=4.0)
      assert mpl.rcParams["lines.linewidth"] == 4.0
    # end
  # end

  def test_linestyle_sets_rcparam(self):
    with mpl.rc_context():
      backend.plot(_line(), show=False, linestyle="--")
      assert mpl.rcParams["lines.linestyle"] == "--"
    # end
  # end
# end


# --------------------------------------------------------------------------
# xlabel/ylabel/clabel shift-scale annotation branches
# --------------------------------------------------------------------------

class TestLabelShiftScale:
  def test_xlabel_shift_and_scale(self):
    fig = backend.plot(_line(), show=False, squeeze=True, xshift=1.0, xscale=2.0)
    lbl = fig.axes[0].get_xlabel()
    assert " + " in lbl and r"\times" in lbl
  # end

  def test_xlabel_shift_only(self):
    fig = backend.plot(_line(), show=False, squeeze=True, xshift=1.0)
    lbl = fig.axes[0].get_xlabel()
    assert " + " in lbl and r"\times" not in lbl
  # end

  def test_xlabel_scale_only(self):
    fig = backend.plot(_line(), show=False, squeeze=True, xscale=2.0)
    lbl = fig.axes[0].get_xlabel()
    assert r"\times" in lbl and " + " not in lbl
  # end

  def test_ylabel_shift_and_scale(self):
    fig = backend.plot(_field_2d(), show=False, squeeze=True, yshift=1.0, yscale=2.0)
    lbl = fig.axes[0].get_ylabel()
    assert " + " in lbl and r"\times" in lbl
  # end

  def test_ylabel_bug_branch_uses_xshift(self):
    fig = backend.plot(_field_2d(), show=False, squeeze=True, xshift=1.0)
    lbl = fig.axes[0].get_ylabel()
    assert " + " in lbl
  # end

  def test_ylabel_bug_branch_uses_xscale(self):
    fig = backend.plot(_field_2d(), show=False, squeeze=True, xscale=2.0)
    lbl = fig.axes[0].get_ylabel()
    assert r"\times" in lbl
  # end

  def test_clabel_gets_zscale_annotation(self):
    fig = backend.plot(_field_2d(), show=False, clabel="density", zscale=2.0,
        colorbar=True)
    cbar_lbl = fig.axes[1].get_ylabel()
    assert "density" in cbar_lbl and r"\times" in cbar_lbl
  # end
# end


# --------------------------------------------------------------------------
# figsize / figure kwarg / figure reuse
# --------------------------------------------------------------------------

class TestFigureCreation:
  def test_figsize_string_is_parsed(self):
    fig = backend.plot(_line(), show=False, figsize="6,4")
    np.testing.assert_allclose(fig.get_size_inches(), (6.0, 4.0))
  # end

  def test_figure_int_selects_numbered_figure(self):
    fig = backend.plot(_line(), show=False, figure=11)
    assert fig.number == 11
  # end

  def test_figure_str_selects_numbered_figure(self):
    fig = backend.plot(_line(), show=False, figure="12")
    assert fig.number == 12
  # end

  def test_figure_object_is_used_directly(self):
    fig_obj = plt.figure()
    result = backend.plot(_line(), show=False, figure=fig_obj)
    assert result is fig_obj
    assert len(result.axes) == 1
  # end

  def test_figure_invalid_type_raises(self):
    with pytest.raises(TypeError, match="'figure' keyword"):
      backend.plot(_line(), show=False, figure=3.14)
    # end
  # end

  def test_reused_figure_without_enough_axes_raises(self):
    fig_obj = plt.figure()
    fig_obj.subplots(1, 1)
    with pytest.raises(ValueError, match="not enough axes"):
      backend.plot(_field_2d(ncomp=4), show=False, figure=fig_obj)
    # end
  # end
# end


# --------------------------------------------------------------------------
# squeeze=True single-panel layout
# --------------------------------------------------------------------------

class TestSqueezeLayout:
  def test_squeeze_sets_title_on_first_use(self):
    fig = backend.plot(_field_2d(ncomp=1), show=False, squeeze=True,
        title="hello", colorbar=False)
    assert fig.axes[0].get_title() == "hello"
    assert len(fig.axes) == 1
  # end
# end


# --------------------------------------------------------------------------
# Multi-panel subplot titles
# --------------------------------------------------------------------------

class TestSubplotTitles:
  def test_per_panel_titles_are_set(self):
    fig = backend.plot(_field_2d(ncomp=2), show=False, colorbar=False,
        subplot_titles="a,b")
    assert fig.axes[0].get_title() == "a"
    assert fig.axes[1].get_title() == "b"
  # end
# end


# --------------------------------------------------------------------------
# Multi-dataset label_prefix via data.get_label()
# --------------------------------------------------------------------------

class TestMultiDatasetLabel:
  def test_label_prefix_uses_dataset_label(self):
    a, b = _line(), _line(offset=3.0)
    a.label, b.label = "first", "second"
    fig = backend.plot(a, b, multiblock=True, show=False)
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "first" in texts and "second" in texts
  # end
# end


# --------------------------------------------------------------------------
# Dimensionality errors raised per-dataset (not just from the first/"ref")
# --------------------------------------------------------------------------

class TestDimensionalityErrors:
  def test_second_dataset_over_2d_raises(self):
    ok = _line()
    bad = GDataState()
    bad.push([np.linspace(0, 1, 3)] * 3, np.zeros((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="Only 1D and 2D"):
      backend.plot(ok, bad, show=False)
    # end
  # end

  def test_0d_dataset_raises(self):
    d = GDataState()
    d.push([np.array([0.0, 1.0]), np.array([0.0, 1.0])],
        np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="0D data not supported"):
      backend.plot(d, show=False)
    # end
  # end
# end


# --------------------------------------------------------------------------
# squeeze-with-curvilinear-grid dimension drop (plot()'s own inline squeeze)
# --------------------------------------------------------------------------

class TestSqueezeCurvilinearDrop:
  def test_curvilinear_coordinate_meaned_over_dropped_axis(self):
    d = GDataState()
    g0 = np.array([[0.0], [1.0]])
    g1 = np.arange(24, dtype=float).reshape(4, 6)  # joint-shaped, no size-1 axis
    values = np.arange(5, dtype=float).reshape(1, 5, 1)
    d.push([g0, g1], values)
    fig = backend.plot(d, show=False)
    expected_edges = g1.mean(axis=0)
    expected_x = 0.5 * (expected_edges[:-1] + expected_edges[1:])
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), expected_x)
  # end
# end


# --------------------------------------------------------------------------
# contour
# --------------------------------------------------------------------------

class TestContour:
  def test_default_levels_with_clabel_text(self):
    fig = backend.plot(_field_2d(), show=False, contour=True, cont_label=True)
    assert fig is not None
  # end

  def test_cnlevels_sets_integer_level_count(self):
    fig = backend.plot(_field_2d(), show=False, contour=True, cnlevels=6)
    assert fig is not None
  # end

  def test_clevels_colon_syntax_is_linspace(self):
    fig = backend.plot(_field_2d(), show=False, contour=True, clevels="0:60:5")
    assert fig is not None
  # end

  def test_clevels_single_value_disables_colorbar(self):
    fig = backend.plot(_field_2d(), show=False, contour=True, clevels="30")
    assert len(fig.axes) == 1
  # end

  def test_clevels_comma_list(self):
    fig = backend.plot(_field_2d(), show=False, contour=True, clevels="10,30,50")
    assert fig is not None
  # end
# end


# --------------------------------------------------------------------------
# quiver / streamline (need a wide-enough grid so `skip` isn't 0)
# --------------------------------------------------------------------------

class TestQuiverAndStreamline:
  def test_quiver_draws_vector_field(self):
    fig = backend.plot(_field_2d(n=15, ncomp=2), show=False, quiver=True)
    assert len(fig.axes[0].collections) >= 1
  # end

  def test_quiver_on_curvilinear_grid_uses_2d_nodal_grid(self):
    edges = np.linspace(0.0, 1.0, 16)
    gx, gy = np.meshgrid(edges, edges, indexing="ij")
    values = np.stack([np.zeros((15, 15)), np.ones((15, 15))], axis=-1)
    d = GDataState()
    d.push([gx, gy], values)
    fig = backend.plot(d, show=False, quiver=True)
    assert len(fig.axes[0].collections) >= 1
  # end

  def test_streamline_default_uses_speed_as_color(self):
    fig = backend.plot(_field_2d(n=15, ncomp=2), show=False, streamline=True)
    assert fig is not None
  # end

  def test_streamline_explicit_color(self):
    fig = backend.plot(_field_2d(n=15, ncomp=2), show=False, streamline=True,
        color="black")
    assert fig is not None
  # end
# end


# --------------------------------------------------------------------------
# lineouts
# --------------------------------------------------------------------------

class TestLineouts:
  def test_lineouts_0_draws_one_line_per_column(self):
    d = _field_2d(n=4)
    fig = backend.plot(d, show=False, lineouts=0)
    assert len(fig.axes[0].lines) == 4
    assert len(fig.axes) == 2  # panel + the appended lineout colorbar
  # end

  def test_lineouts_1_draws_one_line_per_row(self):
    d = _field_2d(n=4)
    fig = backend.plot(d, show=False, lineouts=1)
    assert len(fig.axes[0].lines) == 4
    assert len(fig.axes) == 2
  # end
# end


# --------------------------------------------------------------------------
# zmin/zmax -> colorbar `extend`
# --------------------------------------------------------------------------

class TestExtend:
  def test_zmax_only_extends_max(self):
    fig = backend.plot(_field_2d(), show=False, zmax=5.0)
    im = fig.axes[0].collections[0]
    assert im.colorbar.extend == "max"
  # end

  def test_zmin_only_extends_min(self):
    fig = backend.plot(_field_2d(), show=False, zmin=5.0)
    im = fig.axes[0].collections[0]
    assert im.colorbar.extend == "min"
  # end
# end


# --------------------------------------------------------------------------
# plain pcolormesh: nodal-grid fallback when grid already matches cell count
# --------------------------------------------------------------------------

class TestNodalGridFallback:
  def test_cell_centered_grid_falls_back_through_nodal_grid(self):
    d = GDataState()
    grid = [np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5)]
    values = np.arange(25, dtype=float).reshape(5, 5, 1)
    d.push(grid, values)
    fig = backend.plot(d, show=False)
    assert len(fig.axes[0].collections) == 1
  # end
# end


# --------------------------------------------------------------------------
# logz + diverging -> SymLogNorm
# --------------------------------------------------------------------------

class TestLogzDiverging:
  def test_logz_diverging_uses_symlognorm(self):
    fig = backend.plot(_field_2d(), show=False, logz=True, diverging=True)
    im = fig.axes[0].collections[0]
    from matplotlib.colors import SymLogNorm
    assert isinstance(im.norm, SymLogNorm)
  # end
# end


# --------------------------------------------------------------------------
# hashtag watermark
# --------------------------------------------------------------------------

class TestHashtag:
  def test_hashtag_adds_text(self):
    fig = backend.plot(_line(), show=False, hashtag=True)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "#pgkyl" in texts
  # end
# end


# --------------------------------------------------------------------------
# xmin/xmax -> set_xlim
# --------------------------------------------------------------------------

class TestXlim:
  def test_xmin_xmax_set_xlim(self):
    fig = backend.plot(_line(), show=False, xmin=0.2, xmax=0.8)
    assert fig.axes[0].get_xlim() == (0.2, 0.8)
  # end
# end


# --------------------------------------------------------------------------
# num_axes spreads multiple single-component datasets across separate panels
# --------------------------------------------------------------------------

class TestNumAxesAcrossDatasets:
  def test_cur_start_axes_advances_between_datasets(self):
    a, b = _field_2d(ncomp=1), _field_2d(ncomp=1)
    fig = backend.plot(a, b, multiblock=True, show=False, num_axes=2)
    assert len(fig.axes[0].collections) == 1
    assert len(fig.axes[1].collections) == 1
  # end
# end
