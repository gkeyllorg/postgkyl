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
  d.push([np.linspace(0.0, 1.0, n + 1)],
         (np.arange(n, dtype=float) + offset)[:, None])
  return d


def _field_2d(n=8, ncomp=1) -> GDataState:
  d = GDataState()
  grid = [np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, n + 1)]
  values = np.stack([
      np.arange(n * n, dtype=float).reshape(n, n) + 10.0 * c
      for c in range(ncomp)
  ],
                    axis=-1)
  d.push(grid, values)
  return d


class _UnlabelledState:

  _file_name = ""


@pytest.fixture(autouse=True)
def _close_figs():
  plt.close("all")
  yield
  plt.close("all")


# --------------------------------------------------------------------------
# Pure output/style normalization helpers
# --------------------------------------------------------------------------


class TestOutputNormalization:

  def test_indexed_saveas_handles_one_path_and_a_sequence(self):
    assert backend._indexed_saveas("plot.pdf", 3, True) == "plot_3.pdf"
    assert backend._indexed_saveas(("a", "b.png"), 2,
                                   True) == ("a_2", "b_2.png")

  def test_default_output_stem_uses_labels_and_fallbacks(self):
    labelled = _line()
    labelled.label = "ion density"
    assert backend._default_output_stem([labelled, _UnlabelledState()
                                         ]) == "ion_density_dataset_1"
    assert backend._default_output_stem([]) == "matplotlib_output"

  def test_output_paths_rejects_a_non_iterable(self):
    with pytest.raises(TypeError, match="path or an iterable"):
      backend._output_paths(False, 3, [])

  def test_output_paths_rejects_a_non_path_entry(self):
    with pytest.raises(TypeError, match="entry must be path-like"):
      backend._output_paths(False, ["ok.png", object()], [])


class TestLineStyleNormalization:

  def test_invalid_color_string_is_left_to_matplotlib(self):
    assert backend._normalize_line_colors("not-a-color") is None

  def test_non_iterable_color_is_treated_as_a_scalar(self):
    assert backend._normalize_line_colors(object()) is None

  def test_empty_color_sequence_raises(self):
    with pytest.raises(ValueError, match="must not be an empty sequence"):
      backend._normalize_line_colors([])

  def test_invalid_color_sequence_entry_raises(self):
    with pytest.raises(ValueError, match="every entry"):
      backend._normalize_line_colors(["red", "not-a-color"])

  def test_custom_dash_pattern_is_one_linestyle(self):
    assert backend._normalize_linestyles((0, (5, 2)), 2) is None

  def test_non_iterable_linestyle_is_treated_as_a_scalar(self):
    assert backend._normalize_linestyles(object(), 2) is None

  def test_empty_linestyle_sequence_raises(self):
    with pytest.raises(ValueError, match="must not be an empty sequence"):
      backend._normalize_linestyles([], 2)


class TestSmallPlotHelpers:

  def test_xkcd_without_a_suitable_font_warns_and_uses_sans_serif(
      self, monkeypatch):
    monkeypatch.setattr(backend.fm.fontManager, "ttflist", [])
    with pytest.warns(UserWarning, match="No xkcd-style font"):
      context, rc = backend.get_xkcd_safely()
    assert callable(context)
    assert rc == {"font.family": "sans-serif"}

  def test_shared_range_skips_missing_components_and_nonfinite_values(self):
    one_comp = _field_2d(n=2)
    two_comps = _field_2d(n=2, ncomp=2)
    two_comps._values[..., 1] = np.nan
    assert backend._shared_component_range([one_comp, two_comps], 0.0,
                                           1.0) == [(0.0, 3.0), (None, None)]

  @pytest.mark.parametrize(("limits", "comp", "expected"),
                           [({}, 0, None), ([(0.0, 1.0)], 2, None),
                            ([None], 0, None)])
  def test_split_ylim_missing_component_is_automatic(self, limits, comp,
                                                     expected):
    assert backend._split_ylim_for_component(limits, comp) is expected

  def test_split_ylim_rejects_a_non_container(self):
    with pytest.raises(TypeError, match="split y-limits"):
      backend._split_ylim_for_component(1.0, 0)

  @pytest.mark.parametrize("limits", [[(0.0, 1.0, 2.0)], {0: 1.0}])
  def test_split_ylim_rejects_a_malformed_pair(self, limits):
    with pytest.raises(ValueError, match="must be a .* pair"):
      backend._split_ylim_for_component(limits, 0)


# --------------------------------------------------------------------------
# _nodal_grid, tested directly as the pure function it is
# --------------------------------------------------------------------------


class TestNodalGridDirect:

  def test_dim_count_mismatch_raises(self):
    with pytest.raises(ValueError, match="doesn't match"):
      backend._nodal_grid([np.linspace(0.0, 1.0, 5)], np.array([4, 4]))

  def test_1d_bad_edge_count_raises(self):
    with pytest.raises(ValueError, match="terribly wrong"):
      backend._nodal_grid([np.linspace(0.0, 1.0, 5)], np.array([10]))

  def test_1d_curvilinear_edges_averaged(self):
    g = np.array([[0.0, 1.0], [2.0, 4.0]])  # shape (2, 2): cells[0]+1 == 2
    out = backend._nodal_grid([g], np.array([1]))
    np.testing.assert_allclose(out[0], 0.5 * (g[:-1] + g[1:]))

  def test_2d_curvilinear_cell_centered_passthrough(self):
    g0 = np.ones((3, 3))
    out = backend._nodal_grid([g0, np.ones((3, 3))], np.array([3, 3]))
    assert out[0] is g0

  def test_2d_curvilinear_edges_averaged(self):
    g0 = np.arange(16, dtype=float).reshape(4, 4)  # cells[0]+1 == 4
    out = backend._nodal_grid([g0, np.ones((4, 4))], np.array([3, 3]))
    np.testing.assert_allclose(out[0], 0.5 * (g0[:-1, :-1] + g0[1:, 1:]))

  def test_2d_curvilinear_bad_shape_raises(self):
    g0 = np.ones((5, 5))
    with pytest.raises(ValueError, match="terribly wrong"):
      backend._nodal_grid([g0, np.ones((5, 5))], np.array([3, 3]))


# --------------------------------------------------------------------------
# rcParams novelties: jet / xkcd / color / linewidth / linestyle
# --------------------------------------------------------------------------


class TestRcParamNovelties:

  def test_jet_sets_cmap(self):
    with mpl.rc_context():
      backend.plot(_field_2d(), no_show=True, jet=True)
      assert mpl.rcParams["image.cmap"] == "jet"

  @pytest.mark.filterwarnings("ignore:No xkcd-style font found:UserWarning")
  def test_xkcd_flag_invokes_xkcd_mode(self):
    with mpl.rc_context():
      fig = backend.plot(_line(), no_show=True, xkcd=True)
      line = fig.axes[0].lines[0]
      assert line.get_sketch_params() is not None

  @pytest.mark.filterwarnings("ignore:No xkcd-style font found:UserWarning")
  def test_xkcd_flag_does_not_leak_into_global_rcparams(self):
    # A past bug: `plt.xkcd()` called without a `with` block never reverted,
    # contaminating every plot drawn afterwards.
    with mpl.rc_context():
      backend.plot(_line(), no_show=True, xkcd=True)
      assert mpl.rcParams["path.sketch"] is None

  def test_color_sets_rcparam(self):
    with mpl.rc_context():
      backend.plot(_line(), no_show=True, color="red")
      assert mpl.rcParams["lines.color"] == "red"

  def test_linewidth_sets_rcparam(self):
    with mpl.rc_context():
      backend.plot(_line(), no_show=True, linewidth=4.0)
      assert mpl.rcParams["lines.linewidth"] == 4.0

  def test_linestyle_sets_rcparam(self):
    with mpl.rc_context():
      backend.plot(_line(), no_show=True, linestyle="--")
      assert mpl.rcParams["lines.linestyle"] == "--"


# --------------------------------------------------------------------------
# xlabel/ylabel/clabel shift-scale annotation branches
# --------------------------------------------------------------------------


class TestLabelShiftScale:

  def test_xlabel_shift_and_scale(self):
    fig = backend.plot(_line(),
                       no_show=True,
                       squeeze=True,
                       xshift=1.0,
                       xscale=2.0)
    lbl = fig.axes[0].get_xlabel()
    assert " + " in lbl and r"\times" in lbl

  def test_xlabel_shift_only(self):
    fig = backend.plot(_line(), no_show=True, squeeze=True, xshift=1.0)
    lbl = fig.axes[0].get_xlabel()
    assert " + " in lbl and r"\times" not in lbl

  def test_xlabel_scale_only(self):
    fig = backend.plot(_line(), no_show=True, squeeze=True, xscale=2.0)
    lbl = fig.axes[0].get_xlabel()
    assert r"\times" in lbl and " + " not in lbl

  def test_ylabel_shift_and_scale(self):
    fig = backend.plot(_field_2d(),
                       no_show=True,
                       squeeze=True,
                       yshift=1.0,
                       yscale=2.0)
    lbl = fig.axes[0].get_ylabel()
    assert " + " in lbl and r"\times" in lbl

  def test_ylabel_bug_branch_uses_xshift(self):
    fig = backend.plot(_field_2d(), no_show=True, squeeze=True, xshift=1.0)
    lbl = fig.axes[0].get_ylabel()
    assert " + " in lbl

  def test_ylabel_bug_branch_uses_xscale(self):
    fig = backend.plot(_field_2d(), no_show=True, squeeze=True, xscale=2.0)
    lbl = fig.axes[0].get_ylabel()
    assert r"\times" in lbl

  def test_clabel_gets_zscale_annotation(self):
    fig = backend.plot(_field_2d(),
                       no_show=True,
                       clabel="density",
                       zscale=2.0,
                       no_colorbar=False)
    cbar_lbl = fig.axes[1].get_ylabel()
    assert "density" in cbar_lbl and r"\times" in cbar_lbl


# --------------------------------------------------------------------------
# figsize / figure kwarg / figure reuse
# --------------------------------------------------------------------------


class TestFigureCreation:

  def test_figsize_string_is_parsed(self):
    fig = backend.plot(_line(), no_show=True, figsize="6,4")
    np.testing.assert_allclose(fig.get_size_inches(), (6.0, 4.0))

  def test_figure_int_selects_numbered_figure(self):
    fig = backend.plot(_line(), no_show=True, figure=11)
    assert fig.number == 11

  def test_figure_str_selects_numbered_figure(self):
    fig = backend.plot(_line(), no_show=True, figure="12")
    assert fig.number == 12

  def test_figure_object_is_used_directly(self):
    fig_obj = plt.figure()
    result = backend.plot(_line(), no_show=True, figure=fig_obj)
    assert result is fig_obj
    assert len(result.axes) == 1

  def test_figure_invalid_type_raises(self):
    with pytest.raises(TypeError, match="'figure' keyword"):
      backend.plot(_line(), no_show=True, figure=3.14)

  def test_reused_figure_without_enough_axes_raises(self):
    fig_obj = plt.figure()
    fig_obj.subplots(1, 1)
    with pytest.raises(ValueError, match="not enough axes"):
      backend.plot(_field_2d(ncomp=4), no_show=True, figure=fig_obj)


# --------------------------------------------------------------------------
# squeeze=True single-panel layout
# --------------------------------------------------------------------------


class TestSqueezeLayout:

  def test_squeeze_sets_title_on_first_use(self):
    fig = backend.plot(_field_2d(ncomp=1),
                       no_show=True,
                       squeeze=True,
                       title="hello",
                       no_colorbar=True)
    assert fig.axes[0].get_title() == "hello"
    assert len(fig.axes) == 1


# --------------------------------------------------------------------------
# Multi-panel subplot titles
# --------------------------------------------------------------------------


class TestSubplotTitles:

  def test_per_panel_titles_are_set(self):
    fig = backend.plot(_field_2d(ncomp=2),
                       no_show=True,
                       no_colorbar=True,
                       subplot_titles="a,b")
    assert fig.axes[0].get_title() == "a"
    assert fig.axes[1].get_title() == "b"


# --------------------------------------------------------------------------
# Multi-dataset label_prefix via data.get_label()
# --------------------------------------------------------------------------


class TestMultiDatasetLabel:

  def test_label_prefix_uses_dataset_label(self):
    a, b = _line(), _line(offset=3.0)
    a.label, b.label = "first", "second"
    fig = backend.plot(a, b, multiblock=True, no_show=True)
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "first" in texts and "second" in texts


# --------------------------------------------------------------------------
# Dimensionality errors raised per-dataset (not just from the first/"ref")
# --------------------------------------------------------------------------


class TestDimensionalityErrors:

  def test_second_dataset_over_2d_raises(self):
    ok = _line()
    bad = GDataState()
    bad.push([np.linspace(0, 1, 3)] * 3, np.zeros((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="Only 1D and 2D"):
      backend.plot(ok, bad, no_show=True)

  def test_0d_dataset_raises(self):
    d = GDataState()
    d.push([np.array([0.0, 1.0]), np.array([0.0, 1.0])], np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="0D data not supported"):
      backend.plot(d, no_show=True)


# --------------------------------------------------------------------------
# squeeze-with-curvilinear-grid dimension drop (plot()'s own inline squeeze)
# --------------------------------------------------------------------------


class TestSqueezeCurvilinearDrop:

  def test_curvilinear_coordinate_meaned_over_dropped_axis(self):
    d = GDataState()
    g0 = np.array([[0.0], [1.0]])
    g1 = np.arange(24, dtype=float).reshape(4,
                                            6)  # joint-shaped, no size-1 axis
    values = np.arange(5, dtype=float).reshape(1, 5, 1)
    d.push([g0, g1], values)
    fig = backend.plot(d, no_show=True)
    expected_edges = g1.mean(axis=0)
    expected_x = 0.5 * (expected_edges[:-1] + expected_edges[1:])
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), expected_x)


# --------------------------------------------------------------------------
# contour
# --------------------------------------------------------------------------


class TestContour:

  def test_default_levels_with_clabel_text(self):
    fig = backend.plot(_field_2d(), no_show=True, contour=True, cont_label=True)
    assert fig is not None

  def test_cnlevels_sets_integer_level_count(self):
    fig = backend.plot(_field_2d(), no_show=True, contour=True, cnlevels=6)
    assert fig is not None

  def test_clevels_colon_syntax_is_linspace(self):
    fig = backend.plot(_field_2d(),
                       no_show=True,
                       contour=True,
                       clevels="0:60:5")
    assert fig is not None

  def test_clevels_single_value_disables_colorbar(self):
    fig = backend.plot(_field_2d(), no_show=True, contour=True, clevels="30")
    assert len(fig.axes) == 1

  def test_clevels_comma_list(self):
    fig = backend.plot(_field_2d(),
                       no_show=True,
                       contour=True,
                       clevels="10,30,50")
    assert fig is not None


# --------------------------------------------------------------------------
# quiver / streamline (need a wide-enough grid so `skip` isn't 0)
# --------------------------------------------------------------------------


class TestQuiverAndStreamline:

  def test_quiver_draws_vector_field(self):
    fig = backend.plot(_field_2d(n=15, ncomp=2), no_show=True, quiver=True)
    assert len(fig.axes[0].collections) >= 1

  def test_quiver_on_curvilinear_grid_uses_2d_nodal_grid(self):
    edges = np.linspace(0.0, 1.0, 16)
    gx, gy = np.meshgrid(edges, edges, indexing="ij")
    values = np.stack([np.zeros((15, 15)), np.ones((15, 15))], axis=-1)
    d = GDataState()
    d.push([gx, gy], values)
    fig = backend.plot(d, no_show=True, quiver=True)
    assert len(fig.axes[0].collections) >= 1

  def test_streamline_default_uses_speed_as_color(self):
    fig = backend.plot(_field_2d(n=15, ncomp=2), no_show=True, streamline=True)
    assert fig is not None

  def test_streamline_explicit_color(self):
    fig = backend.plot(_field_2d(n=15, ncomp=2),
                       no_show=True,
                       streamline=True,
                       color="black")
    assert fig is not None


# --------------------------------------------------------------------------
# lineouts
# --------------------------------------------------------------------------


class TestLineouts:

  def test_lineouts_0_draws_one_line_per_column(self):
    d = _field_2d(n=4)
    fig = backend.plot(d, no_show=True, lineouts=0)
    assert len(fig.axes[0].lines) == 4
    assert len(fig.axes) == 2  # panel + the appended lineout colorbar

  def test_lineouts_1_draws_one_line_per_row(self):
    d = _field_2d(n=4)
    fig = backend.plot(d, no_show=True, lineouts=1)
    assert len(fig.axes[0].lines) == 4
    assert len(fig.axes) == 2


# --------------------------------------------------------------------------
# zmin/zmax -> colorbar `extend`
# --------------------------------------------------------------------------


class TestExtend:

  def test_zmax_only_extends_max(self):
    fig = backend.plot(_field_2d(), no_show=True, zmax=5.0)
    im = fig.axes[0].collections[0]
    assert im.colorbar.extend == "max"

  def test_zmin_only_extends_min(self):
    fig = backend.plot(_field_2d(), no_show=True, zmin=5.0)
    im = fig.axes[0].collections[0]
    assert im.colorbar.extend == "min"


# --------------------------------------------------------------------------
# plain pcolormesh: nodal-grid fallback when grid already matches cell count
# --------------------------------------------------------------------------


class TestNodalGridFallback:

  def test_cell_centered_grid_falls_back_through_nodal_grid(self):
    d = GDataState()
    grid = [np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5)]
    values = np.arange(25, dtype=float).reshape(5, 5, 1)
    d.push(grid, values)
    fig = backend.plot(d, no_show=True)
    assert len(fig.axes[0].collections) == 1


# --------------------------------------------------------------------------
# logz + diverging -> SymLogNorm
# --------------------------------------------------------------------------


class TestLogzDiverging:

  def test_logz_diverging_uses_symlognorm(self):
    fig = backend.plot(_field_2d(), no_show=True, logz=True, diverging=True)
    im = fig.axes[0].collections[0]
    from matplotlib.colors import SymLogNorm
    assert isinstance(im.norm, SymLogNorm)


# --------------------------------------------------------------------------
# hashtag watermark
# --------------------------------------------------------------------------


class TestHashtag:

  def test_hashtag_adds_text(self):
    fig = backend.plot(_line(), no_show=True, hashtag=True)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "#pgkyl" in texts


# --------------------------------------------------------------------------
# xmin/xmax -> set_xlim
# --------------------------------------------------------------------------


class TestXlim:

  def test_xmin_xmax_set_xlim(self):
    fig = backend.plot(_line(), no_show=True, xmin=0.2, xmax=0.8)
    assert fig.axes[0].get_xlim() == (0.2, 0.8)


# --------------------------------------------------------------------------
# num_axes spreads multiple single-component datasets across separate panels
# --------------------------------------------------------------------------


class TestNumAxesAcrossDatasets:

  def test_cur_start_axes_advances_between_datasets(self):
    a, b = _field_2d(ncomp=1), _field_2d(ncomp=1)
    fig = backend.plot(a, b, multiblock=True, no_show=True, num_axes=2)
    assert len(fig.axes[0].collections) == 1
    assert len(fig.axes[1].collections) == 1


# --------------------------------------------------------------------------
# Remaining validation and specialized drawing branches
# --------------------------------------------------------------------------


class TestRemainingValidationBranches:

  def test_color_sequence_rejects_a_2d_plot(self):
    with pytest.raises(ValueError, match="only supported for 1D"):
      backend.plot(_field_2d(), no_show=True, color=["red", "blue"])

  @pytest.mark.parametrize(("kwargs", "error", "message"),
                           [({
                               "split_point": object()
                           }, TypeError, "split_point"),
                            ({
                                "split_point": np.inf
                            }, ValueError, "split_point"),
                            ({
                                "split_log_base": object()
                            }, TypeError, "split_log_base"),
                            ({
                                "split_gap": object()
                            }, TypeError, "split_gap")])
  def test_split_numeric_options_reject_invalid_values(self, kwargs, error,
                                                       message):
    with pytest.raises(error, match=message):
      backend.plot(_line(), no_show=True, split_linear_log=True, **kwargs)

  def test_legend_subplot_rejects_a_non_integer(self):
    with pytest.raises(TypeError, match="must be an integer"):
      backend.plot(_line(), no_show=True, legend_subplot="0")

  def test_second_dataset_over_2d_raises_inside_one_family(self):
    bad = GDataState()
    bad.push([np.linspace(0, 1, 3)] * 3, np.zeros((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="Only 1D and 2D"):
      backend.plot(_line(), bad, multiblock=True, no_show=True)

  def test_second_dataset_must_match_split_dimensionality(self):
    with pytest.raises(ValueError, match="every dataset must be 1D"):
      backend.plot(_line(),
                   _field_2d(),
                   multiblock=True,
                   no_show=True,
                   split_linear_log=True)


class TestRemainingSplitBranches:

  @pytest.mark.parametrize(("side", "legend_axis"), [("left", 0), ("right", 1),
                                                     ("linear", 0)])
  def test_explicit_split_legend_side(self, side, legend_axis):
    data = _line()
    data.label = "curve"
    fig = backend.plot(data,
                       no_show=True,
                       forcelegend=True,
                       split_linear_log=True,
                       split_legend_side=side)
    assert fig.axes[legend_axis].get_legend() is not None

  def test_split_layout_accepts_y_label_only_and_hides_right_ticks(self):
    fig = backend.plot(_line(),
                       no_show=True,
                       split_linear_log=True,
                       xlabel="",
                       ylabel="amplitude",
                       no_split_right_ticks=True)
    assert fig.get_supxlabel() == ""
    assert fig.get_supylabel() == "amplitude"
    assert fig.axes[1].yaxis.get_ticks_position() != "right"

  def test_split_plot_allows_logarithmic_x_axis(self):
    fig = backend.plot(_line(),
                       no_show=True,
                       split_linear_log=True,
                       split_point=0.5,
                       logx=True)
    assert all(axis.get_xscale() == "log" for axis in fig.axes)


class TestRemainingSurfaceBranches:

  @staticmethod
  def _mapped_field() -> GDataState:
    coordinates = np.linspace(0.0, 1.0, 4)
    gx, gy = np.meshgrid(coordinates, coordinates, indexing="ij")
    data = GDataState()
    data.push([gx, gy], np.arange(16, dtype=float).reshape(4, 4, 1))
    return data

  def test_transpose_transposes_joint_coordinate_arrays(self):
    fig = backend.plot(self._mapped_field(),
                       no_show=True,
                       transpose=True,
                       no_colorbar=True)
    assert len(fig.axes[0].collections) == 1

  def test_surface_transposes_joint_coordinates_without_a_colorbar(self):
    fig = backend.plot(self._mapped_field(),
                       no_show=True,
                       surface=True,
                       no_colorbar=True)
    assert fig.axes[0].name == "3d"
    assert len(fig.axes) == 1

  def test_surface_applies_color_label_and_z_limits(self):
    fig = backend.plot(_field_2d(),
                       no_show=True,
                       surface=True,
                       clabel="density",
                       zmin=1.0,
                       zmax=9.0)
    assert fig.axes[0].get_zlabel() == "density"
    assert fig.axes[0].get_zlim() == (1.0, 9.0)

  def test_unlabelled_surface_comparison_needs_no_legend_handle(self):
    fig = backend.plot(_field_2d(), no_show=True, surface=True, comparison=True)
    assert fig.axes[0].get_legend() is None

  def test_unlabelled_contour_comparison_needs_no_legend_handle(self):
    fig = backend.plot(_field_2d(), no_show=True, contour=True, comparison=True)
    assert fig.axes[0].get_legend() is None

  def test_cval_without_bounds_uses_colormap_midpoint(self):
    fig = backend.plot(_line(), no_show=True, cmap="viridis", cval=3.0)
    assert fig.axes[0].lines[0].get_color() == plt.get_cmap("viridis")(0.5)
