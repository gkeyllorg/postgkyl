"""Tests for the fluent surface (layer 11 -- api): every ``operations`` verb from
layers 07-09 as a ``GData`` method (or, for the multi-dataset verbs with no
single ``self``, a module-level function in ``api.verbs``), the fluent
``api.group.GDataGroup`` that broadcasts verbs over its members, and the
facade re-exports.

Diagnostics (layer 10: five_moment/ten_moment/mhd/plasma/multispecies/
rotations/kinetic/pkpm/gyrokinetics) are equation-specific and deliberately
NOT fluent methods -- this file only exercises the equation-blind core verbs.
"""

from __future__ import annotations

import base64
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, operations
from postgkyl.gdata.gdatagroup import GDataGroup as ApiGDataGroup
from postgkyl.gdata import verbs as api_verbs
from postgkyl.gdatastate.gdatastategroup import GDataStateGroup as CoreGDataStateGroup
from postgkyl.gdatastate.gdatastate import GDataState

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
    reason="no compiled Gkeyll (libg0core.so) found")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
GEN = os.path.join(DATA, "generated")
F1D = os.path.join(GEN, "1d_ms_p1.gkyl")
F2D_VEC = os.path.join(GEN, "2d_c2p_rot45_ms_p1.gkyl")  # 2 comps after interpolate


@pytest.fixture(autouse=True)
def _close_figs():
  plt.close("all")
  yield
  plt.close("all")
# end


class MyData(pg.GData):
  """A ``GData`` subclass, used to verify subclass propagation through every
  fluent method (the ``_result``/``type(self)`` contract)."""
# end


def _make(cls, grid, values, **ctx):
  d = cls(ctx=ctx or None)
  d.push(list(grid), values)
  return d
# end


def _line(cls=MyData, tag: str = "default", value: float = 1.0, n: int = 5):
  grid = [np.linspace(0.0, 1.0, n + 1)]
  return _make(cls, grid, np.full((n, 1), value), tag=tag)
# end


# ============================================================ method roster
# The full equation-blind verb inventory from operations/__init__.py, keyed by its
# fluent spelling: either a GData instance method, or a module-level function
# in api.verbs for the verbs that combine several datasets (see the group
# contract and api/gdata.py's ``grid`` note for the two exceptions).
INSTANCE_VERBS = ["interpolate", "local_poly", "select", "plot", "save",
    "mul", "div", "integrate", "integrate_axis", "to_modal", "to_nodal",
    "to_quad", "apply",
    "fft", "magsq", "mask", "val2coord", "extract_input", "fit",
    "differentiate", "map"]
MODULE_VERBS = ["collect", "evaluate", "relchange", "animate", "sort"]


class TestMethodInventory:
  def test_every_instance_verb_exists_and_is_callable(self):
    for name in INSTANCE_VERBS:
      assert hasattr(pg.GData, name), f"GData has no {name!r} method"
      assert callable(getattr(pg.GData, name))
    # end
  # end

  def test_every_module_verb_exists_in_api_verbs(self):
    for name in MODULE_VERBS:
      assert hasattr(api_verbs, name), f"api.verbs has no {name!r} function"
      assert callable(getattr(api_verbs, name))
    # end
  # end

  def test_grid_is_deliberately_not_a_fluent_method(self):
    """``operations.grid`` has no fluent spelling: ``GData.grid`` must stay the
    inherited axis-edge-array *property* (see api/gdata.py's note), not a
    verb method -- otherwise every other verb reading ``data.grid`` would
    silently break."""
    d = _line()
    assert isinstance(d.grid, list)
    assert not callable(d.grid)
    assert hasattr(operations, "grid") and callable(operations.grid)
  # end
# end


# ==================================================== subclass propagation
class TestSubclassPropagation:
  def test_fft(self):
    d = _line(value=1.0, n=16)
    out = d.fft()
    assert isinstance(out, MyData)
    out_psd = d.fft(psd=True)
    assert isinstance(out_psd, MyData)
  # end

  def test_magsq(self):
    d = _make(MyData, [np.linspace(0.0, 1.0, 5)], np.tile([1.0, 2.0, 3.0], (4, 1)))
    out = d.magsq()
    assert isinstance(out, MyData)
  # end

  def test_mask(self):
    d = _make(MyData, [np.linspace(0.0, 1.0, 6)], np.arange(5.0)[:, np.newaxis])
    out = d.mask(lower=2.0)
    assert isinstance(out, MyData)
  # end

  def test_relchange(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    ref = _make(MyData, grid, np.full((4, 1), 2.0))
    cur = _make(MyData, grid, np.full((4, 1), 3.0))
    out = api_verbs.relchange(ref, cur)
    assert isinstance(out, MyData)
  # end

  def test_val2coord_returns_fluent_group_of_the_subclass(self):
    d = _make(MyData, [np.arange(5.0)], np.arange(15.0).reshape(5, 3))
    group = d.val2coord(x="0", y="1,2")
    assert isinstance(group, ApiGDataGroup)
    assert len(group) == 2
    for member in group:
      assert isinstance(member, MyData)
    # end
  # end

  def test_extract_input_returns_a_plain_string(self):
    d = _line()
    assert d.extract_input() == ""
    text = "title = my sim\n"
    encoded = base64.encodebytes(text.encode("utf-8")).decode("utf-8")
    d2 = _make(MyData, [np.linspace(0.0, 1.0, 3)], np.ones((2, 1)),
        input_file=encoded)
    assert d2.extract_input() == text
  # end

  def test_fit(self):
    edges = np.linspace(0.0, 1.0, 21)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = 2.0 * centers + 1.0
    d = _make(MyData, [edges], y[:, np.newaxis])
    out = d.fit("linear")
    assert isinstance(out, MyData)
    np.testing.assert_allclose(out.ctx["fit_params"][0], [2.0, 1.0], atol=1e-8)
  # end

  def test_fit_window_growth_rate(self):
    edges = np.linspace(0.0, 1.0, 61)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = 1.0 * np.exp(2 * 0.5 * centers)
    d = _make(MyData, [edges], y[:, np.newaxis])
    out = d.fit("exp2", window=True)
    assert isinstance(out, MyData)
    assert out.ctx["fit_params"][0][1] == pytest.approx(0.5, abs=1e-2)
  # end

  def test_differentiate(self):
    edges = np.linspace(0.0, 1.0, 17)
    centers = 0.5 * (edges[:-1] + edges[1:])
    d = _make(MyData, [edges], (centers**2)[:, np.newaxis])
    out = d.differentiate()
    assert isinstance(out, MyData)
  # end

  def test_collect(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    a = _make(MyData, grid, np.full((4, 1), 2.0), time=0.0)
    b = _make(MyData, grid, np.full((4, 1), 3.0), time=1.0)
    out = api_verbs.collect(a, b)
    assert isinstance(out, MyData)
  # end

  def test_sort(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    a = _make(MyData, grid, np.full((4, 1), 2.0))
    a._file_name = "field_10.gkyl"
    b = _make(MyData, grid, np.full((4, 1), 3.0))
    b._file_name = "field_2.gkyl"
    out = api_verbs.sort(a, b)
    assert [d.file_name for d in out] == ["field_2.gkyl", "field_10.gkyl"]
  # end

  def test_evaluate(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    a = _make(MyData, grid, np.full((4, 1), 2.0))
    b = _make(MyData, grid, np.full((4, 1), 3.0))
    out = api_verbs.evaluate("f0 f1 +", a, b)
    assert isinstance(out, MyData)
    np.testing.assert_allclose(out.get_values(), 5.0)
  # end

  @needs_gkeyll
  def test_map(self):
    from postgkyl.gpython import basis as gpython_basis

    lower, upper, cells = 0.0, 4.0, 4
    node_eta = gpython_basis.node_coords("serendipity", 1, 1)[:, 0]
    n2m = gpython_basis.nodal_to_modal_matrix("serendipity", 1, 1)
    dz = (upper - lower) / cells
    centers = lower + (np.arange(cells) + 0.5) * dz
    nodal_z = centers[:, None] + 0.5 * dz * node_eta[None, :]
    modal = nodal_z @ n2m.T  # exact per-cell modal coeffs of the identity map

    mapping = GDataState()
    mapping.ctx.update(basis_type="serendipity", poly_order=1, value_form="modal",
        cells=np.array([cells], dtype=np.int64))
    mgrid = [np.linspace(lower, upper, cells + 1)]
    mapping.push(mgrid, gpython.GkylArray.from_numpy(modal))

    target = _make(MyData, [np.linspace(lower, upper, 17)], np.zeros((16, 1)))
    out = target.map(mapping, space="conf")
    assert isinstance(out, MyData)
    np.testing.assert_allclose(out.grid[0], target.grid[0], atol=1e-12)
  # end

  @needs_gkeyll
  def test_mul_div_interpolate_to_modal_nodal_quad_apply_integrate(self):
    F1 = os.path.join(DATA,
        "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
    a, b = MyData(F1), MyData(F1)
    assert isinstance(a.mul(b), MyData)
    a2, b2 = MyData(F1), MyData(F1)
    assert isinstance(a2.div(b2), MyData)
    assert isinstance(MyData(F1).interpolate(), MyData)
    assert isinstance(MyData(F1).to_nodal(), MyData)
    assert isinstance(MyData(F1).to_nodal().to_modal(), MyData)
    assert isinstance(MyData(F1).to_quad(), MyData)
    assert isinstance(MyData(F1).apply(np.abs), MyData)
    result = MyData(F1).integrate()
    assert result is not None
    assert isinstance(MyData(F1).interpolate().integrate_axis(0), MyData)
  # end
# end


# ============================================================ keyword pass-through
class TestKeywordPassthrough:
  def test_fft_psd_kwarg_reaches_the_verb(self):
    d = _line(value=1.0, n=16)
    full = d.fft(psd=False)
    half = d.fft(psd=True)
    assert half.values.shape[0] == full.values.shape[0] // 2
  # end

  def test_magsq_coords_kwarg_reaches_the_verb(self):
    d = _make(MyData, [np.linspace(0.0, 1.0, 5)], np.tile([1.0, 2.0, 3.0], (4, 1)))
    default = d.magsq()               # "0:3" -> 1+4+9
    partial = d.magsq(coords="0:2")   # 1+4
    np.testing.assert_allclose(default.get_values().flat[0], 14.0)
    np.testing.assert_allclose(partial.get_values().flat[0], 5.0)
  # end

  def test_mask_lower_vs_upper_kwarg_reaches_the_verb(self):
    d = _make(MyData, [np.linspace(0.0, 1.0, 6)], np.arange(5.0)[:, np.newaxis])
    lower = d.mask(lower=2.0)
    upper = d.mask(upper=2.0)
    assert lower.get_values().mask[0, 0] and not lower.get_values().mask[-1, 0]
    assert upper.get_values().mask[-1, 0] and not upper.get_values().mask[0, 0]
  # end
# end


# ======================================================== end-to-end chains
@needs_gkeyll
class TestEndToEndChains:
  def test_interpolate_magsq_plot(self):
    fig = pg.load(F2D_VEC).interpolate().magsq().plot(show=False)
    assert fig is not None
  # end

  def test_interpolate_select_fft(self):
    # fft's output grid is a frequency axis (one entry per value, not a
    # nodal N+1 edge array), so it is not directly re-plottable through the
    # same render path as the other chains -- exercised on values instead.
    out = pg.load(F1D).interpolate().select(comp=0).fft(psd=True)
    assert isinstance(out, pg.GData)
    assert out.values.shape[0] == pg.load(F1D).interpolate().select(comp=0).num_cells[0] // 2
  # end

  def test_interpolate_select_mask_fit(self):
    out = pg.load(F1D).interpolate().select(comp=0).mask(lower=-1e30).fit("linear")
    assert isinstance(out, pg.GData)
    assert "fit_params" in out.ctx
  # end
# end


# ================================================================== group
class TestGDataGroup:
  def _frames(self, cls=MyData):
    grid = [np.linspace(0.0, 1.0, 5)]
    return [_make(cls, grid, np.full((4, 1), v), time=t)
        for t, v in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))]
  # end

  def test_broadcast_non_terminal_verb_returns_a_group_of_the_same_class(self):
    g = ApiGDataGroup(self._frames())
    out = g.select(comp=0)
    assert isinstance(out, ApiGDataGroup)
    assert len(out) == 3
    for member in out:
      assert isinstance(member, MyData)
    # end
  # end

  def test_broadcast_chains(self):
    g = ApiGDataGroup(self._frames())
    out = g.select(comp=0).mask(lower=-1e30)
    assert isinstance(out, ApiGDataGroup)
    assert len(out) == 3
  # end

  def test_broadcast_terminal_verb_returns_a_plain_list(self):
    g = ApiGDataGroup(self._frames())
    inputs = g.extract_input()
    assert isinstance(inputs, list)
    assert len(inputs) == 3
  # end

  def test_plot_is_not_broadcast_but_one_shared_figure(self):
    # A group is a set of datasets that belong together -- above all a
    # multiblock family -- so .plot() is ONE figure with every member drawn
    # onto it, not one figure per member.
    import matplotlib.figure

    g = ApiGDataGroup(self._frames())
    fig = g.plot(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
  # end

  def test_broadcast_write_returns_a_list_of_paths(self, tmp_path):
    g = ApiGDataGroup(self._frames())
    paths = g.save(out_name=str(tmp_path / "frame"))
    assert isinstance(paths, list)
    assert len(paths) == 3
    for p in paths:
      assert os.path.isfile(p)
    # end
  # end

  def test_broadcast_non_callable_property_returns_a_plain_list(self):
    g = ApiGDataGroup(self._frames())
    dims = g.num_dims
    assert isinstance(dims, list)
    assert len(dims) == 3
    assert all(d == g[0].num_dims for d in dims)
  # end

  def test_info_is_explicit_not_broadcast_and_enumerates_members(self):
    g = ApiGDataGroup(self._frames())
    summaries = g.info()
    assert isinstance(summaries, list)
    assert len(summaries) == 3
    assert "#0" in summaries[0] and "#1" in summaries[1] and "#2" in summaries[2]
  # end

  def test_collect_combines_members_into_one_dataset(self):
    g = ApiGDataGroup(self._frames())
    out = g.collect()
    assert isinstance(out, MyData)
    np.testing.assert_allclose(out.get_grid()[0], [0.0, 1.0, 2.0])
  # end

  def test_evaluate_combines_named_members(self):
    g = ApiGDataGroup(self._frames()[:2])
    out = g.evaluate("f0 f1 +")
    assert isinstance(out, MyData)
    np.testing.assert_allclose(out.get_values(), 3.0)  # 1.0 + 2.0
  # end

  @needs_gkeyll
  def test_animate_is_explicit_not_broadcast(self):
    from matplotlib.animation import FuncAnimation
    frames = [pg.load(F1D).interpolate().select(comp=0) for _ in range(3)]
    g = ApiGDataGroup(frames)
    anim = g.animate(show=False)
    assert isinstance(anim, FuncAnimation)
  # end

  def test_with_and_and_preserve_the_concrete_class(self):
    a, b, c = self._frames()
    g = ApiGDataGroup([a, b])
    g2 = g.with_(c)
    assert isinstance(g2, ApiGDataGroup)
    assert len(g2) == 3
    g3 = g & c
    assert isinstance(g3, ApiGDataGroup)
  # end

  def test_slicing_preserves_the_concrete_class(self):
    g = ApiGDataGroup(self._frames())
    sub = g[0:2]
    assert isinstance(sub, ApiGDataGroup)
    assert len(sub) == 2
    assert isinstance(g[0], MyData)
  # end

  def test_private_and_unknown_attributes_are_not_broadcast(self):
    g = ApiGDataGroup(self._frames())
    with pytest.raises(AttributeError):
      g._not_a_real_attribute
    # end
    with pytest.raises(AttributeError):
      g.this_verb_does_not_exist()
    # end
  # end

  def test_is_a_core_dataset_group_too(self):
    """The fluent group is a genuine subclass of the verb-less container
    (mirrors GData/GDataState); every state-reading behavior still holds."""
    g = ApiGDataGroup(self._frames())
    assert isinstance(g, CoreGDataStateGroup)
    assert repr(g) == "<GDataGroup [3 datasets]>"
  # end
# end


# ================================================================== facade
class TestFacade:
  def test_documented_names_resolve(self):
    for name in ["GData", "load", "GDataGroup", "plot", "info", "integrate",
        "interpolate", "select", "represent", "apply",
        "save", "collect", "evaluate", "relchange", "animate", "__version__"]:
      assert hasattr(pg, name), f"postgkyl has no {name!r}"
    # end
  # end

  def test_all_is_consistent(self):
    assert hasattr(pg, "__all__")
    for name in pg.__all__:
      assert hasattr(pg, name), f"pg.__all__ names {name!r} but it is missing"
    # end
  # end

  def test_dataset_group_is_the_fluent_one(self):
    assert pg.GDataGroup is ApiGDataGroup
  # end

  def test_module_verbs_are_the_api_ones(self):
    assert pg.collect is api_verbs.collect
    assert pg.evaluate is api_verbs.evaluate
    assert pg.relchange is api_verbs.relchange
    assert pg.animate is api_verbs.animate
    assert pg.sort is api_verbs.sort
  # end
# end
