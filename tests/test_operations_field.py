"""Tests for the small field-domain operations verbs: fft, magsq, relchange, mask,
grid, val2coord, extract_input.
"""

from __future__ import annotations

import base64
import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, operations
from postgkyl.gdatastate.gdatastategroup import GDataStateGroup
from postgkyl.gdatastate.gdatastate import GDataState

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
F1 = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")


def _make(grid, values, **ctx):
  d = GDataState(ctx=ctx or None)
  d.push(list(grid), values)
  return d


# ============================================================== operations.fft
class TestFft:

  def test_analytic_sine_peak(self):
    N = 32
    edges = np.linspace(0.0, 1.0, N + 1)
    x_cc = 0.5 * (edges[:-1] + edges[1:])
    f0 = 4
    values = np.sin(2 * np.pi * f0 * x_cc)[:, np.newaxis]
    d = _make([edges], values)
    out = operations.fft(d)
    assert isinstance(out, GDataState)
    freq = out.get_grid()[0]
    ft = out.get_values()
    peak = freq[np.argmax(np.abs(ft[:, 0]))]
    assert abs(abs(peak) - f0) < 1e-9

  def test_psd_returns_positive_frequencies_only(self):
    N = 16
    d = _make([np.linspace(0.0, 1.0, N + 1)], np.ones((N, 1)))
    out = operations.fft(d, psd=True)
    assert out.get_values().shape[0] == N // 2

  def test_inplace_mutates(self):
    d = _make([np.linspace(0.0, 1.0, 17)], np.ones((16, 1)))
    out = operations.fft(d, inplace=True)
    assert out is d

  def test_tag_and_label(self):
    d = _make([np.linspace(0.0, 1.0, 17)], np.ones((16, 1)))
    out = operations.fft(d, tag="spec", label="lbl")
    assert out.get_tag() == "spec"
    assert out.get_label() == "lbl"

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      operations.fft(d)


# ============================================================ operations.magsq
class TestMagsq:

  def _vec3(self):
    return _make([np.linspace(0.0, 1.0, 5)], np.tile([1.0, 2.0, 3.0], (4, 1)))

  def test_value_and_num_comps(self):
    out = operations.magsq(self._vec3())
    np.testing.assert_allclose(out.get_values().flat[0], 14.0)  # 1+4+9
    assert out.get_num_comps() == 1

  def test_custom_coords(self):
    out = operations.magsq(self._vec3(), coords="1:3")
    np.testing.assert_allclose(out.get_values().flat[0], 13.0)  # 4+9

  def test_inplace(self):
    d = self._vec3()
    assert operations.magsq(d, inplace=True) is d

  def test_tag(self):
    out = operations.magsq(self._vec3(), tag="m")
    assert out.get_tag() == "m"

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      operations.magsq(d)


# ========================================================= operations.relchange
class TestRelchange:

  def test_value_componentwise(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    ref = _make(grid, np.full((4, 1), 2.0))
    cur = _make(grid, np.full((4, 1), 3.0))
    out = operations.relchange(ref, cur)
    np.testing.assert_allclose(out.get_values(), 0.5)  # (3-2)/2

  def test_value_with_explicit_comp(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    ref = _make(grid, np.tile([2.0, 10.0], (4, 1)))
    cur = _make(grid, np.tile([4.0, 4.0], (4, 1)))
    out = operations.relchange(ref, cur,
                               comp=0)  # normalize both by ref comp 0 (=2)
    np.testing.assert_allclose(out.get_values()[..., 0], 1.0)  # (4-2)/2
    np.testing.assert_allclose(out.get_values()[..., 1], -3.0)  # (4-10)/2

  def test_result_built_from_data_not_reference(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    ref = _make(grid, np.full((4, 1), 2.0), tag="ref")
    cur = _make(grid, np.full((4, 1), 3.0), tag="cur")
    out = operations.relchange(ref, cur, tag="rc")
    assert out.get_tag() == "rc"

  def test_inplace_mutates_data(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    ref = _make(grid, np.full((4, 1), 2.0))
    cur = _make(grid, np.full((4, 1), 4.0))
    out = operations.relchange(ref, cur, inplace=True)
    assert out is cur

  @needs_gkeyll
  def test_rejects_modal_data(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    numpy_side = _make(grid, np.full((4, 1), 2.0))
    modal = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      operations.relchange(modal, numpy_side)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      operations.relchange(numpy_side, modal)


# ============================================================== operations.mask
class TestMask:

  def _data(self):
    return _make([np.linspace(0.0, 1.0, 6)], np.arange(5.0)[:, np.newaxis])

  def test_mask_lower(self):
    out = operations.mask(self._data(), lower=2.0)
    assert np.ma.is_masked(out.get_values())
    assert out.get_values().mask[0, 0]
    assert not out.get_values().mask[-1, 0]

  def test_mask_upper(self):
    out = operations.mask(self._data(), upper=2.0)
    assert out.get_values().mask[-1, 0]

  def test_mask_outside(self):
    out = operations.mask(self._data(), lower=1.0, upper=3.0)
    assert np.ma.is_masked(out.get_values())

  def test_mask_from_dataset(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    d = _make(grid, np.ones((5, 2)))
    mask_field = _make(grid, np.array([[1.0], [-1.0], [1.0], [-1.0], [1.0]]))
    out = operations.mask(d, mask_field)
    values = out.get_values()
    assert np.ma.is_masked(values)
    assert values.mask[1, 0] and values.mask[1, 1]
    assert not values.mask[0, 0]

  def test_mask_no_args_raises(self):
    with pytest.raises(ValueError):
      operations.mask(self._data())

  def test_mask_from_dataset_multi_component_raises(self):
    """mask_data must have exactly one component (see mask.py's docstring);
    a multi-component mask does not "evenly divide" -- np.repeat produces
    k*num_comps entries, which np.ma.masked_where rejects outright."""
    grid = [np.linspace(0.0, 1.0, 6)]
    d = _make(grid, np.ones((5, 2)))
    mask_field = _make(
        grid,
        np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0],
                  [1.0, 1.0]]))
    with pytest.raises(IndexError):
      operations.mask(d, mask_field)

  def test_inplace(self):
    d = self._data()
    out = operations.mask(d, lower=2.0, inplace=True)
    assert out is d

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      operations.mask(d, lower=0.0)


# ============================================================== operations.grid
class TestGrid:

  def test_1d_values_equal_grid(self):
    edges = np.linspace(0.0, 1.0, 5)
    d = _make([edges], np.ones((4, 1)))
    out = operations.grid(d)
    np.testing.assert_allclose(out.get_values()[..., 0], edges)

  def test_2d_meshgrid_shape(self):
    edges = [np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 4)]
    d = _make(edges, np.ones((4, 3, 1)))
    out = operations.grid(d)
    assert out.get_num_comps() == 2
    assert out.get_values().shape == (5, 4, 2)

  def test_inplace(self):
    edges = np.linspace(0.0, 1.0, 5)
    d = _make([edges], np.ones((4, 1)))
    out = operations.grid(d, inplace=True)
    assert out is d

  def test_curvilinear_grid_passthrough(self):
    # A curvilinear (post-'map') grid: every per-axis array already has
    # the full nodal shape, not just a 1-D axis.
    nx, ny = 3, 2
    gx, gy = np.meshgrid(np.linspace(0.0, 1.0, nx + 1),
                         np.linspace(0.0, 1.0, ny + 1),
                         indexing="ij")
    d = _make([gx, gy], np.ones((nx, ny, 1)))
    out = operations.grid(d)
    assert out.get_values().shape == (nx + 1, ny + 1, 2)
    np.testing.assert_allclose(out.get_values()[..., 0], gx)
    np.testing.assert_allclose(out.get_values()[..., 1], gy)

  def test_dimension_mismatch_raises(self):
    d = _make([np.linspace(0.0, 1.0, 5)], np.ones((4, 1)))
    d.ctx["cells"] = np.array([4, 4])  # claims 2 dims; grid has 1 axis
    with pytest.raises(ValueError, match="dimension"):
      operations.grid(d)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      operations.grid(d)


# ========================================================= operations.val2coord
class TestVal2coord:

  def _table(self):
    # 5 samples, 3 columns: [x, y0, y1]
    return _make([np.arange(5.0)], np.arange(15.0).reshape(5, 3))

  def test_single_x_multiple_y(self):
    group = operations.val2coord(self._table(), x="0", y="1,2")
    assert isinstance(group, GDataStateGroup)
    assert len(group) == 2
    np.testing.assert_allclose(group[0].get_grid()[0], np.arange(5.0) * 3.0)
    np.testing.assert_allclose(group[0].get_values().flatten(),
                               np.arange(5.0) * 3.0 + 1.0)

  def test_periodic_appends_first_sample(self):
    group = operations.val2coord(self._table(), x="0", y="1", periodic=True)
    d = group[0]
    assert d.get_values().shape[0] == 6
    np.testing.assert_allclose(d.get_values().flatten()[-1],
                               d.get_values().flatten()[0])

  def test_mismatched_x_y_counts_raises(self):
    with pytest.raises(ValueError):
      operations.val2coord(self._table(), x="0,1", y="2")

  def test_colon_range_selector_with_negative_indices_and_step(self):
    # 4 columns; "-3:-1:1" exercises the negative-lo, negative-hi, and
    # explicit-step branches of the 'lo:hi[:step]' grammar in one shot.
    d = _make([np.arange(6.0)], np.arange(24.0).reshape(6, 4))
    group = operations.val2coord(d, x="0", y="-3:-1:1")
    assert len(group) == 2  # columns 1, 2

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      operations.val2coord(d, x="0", y="1")


# ===================================================== operations.extract_input
class TestExtractInput:

  def test_missing_returns_empty_string(self):
    d = _make([np.linspace(0.0, 1.0, 3)], np.ones((2, 1)))
    assert operations.extract_input(d) == ""

  def test_decodes_base64_ctx_field(self):
    text = "title = my sim\nnFrames = 10\n"
    encoded = base64.encodebytes(text.encode("utf-8")).decode("utf-8")
    d = _make([np.linspace(0.0, 1.0, 3)], np.ones((2, 1)), input_file=encoded)
    assert operations.extract_input(d) == text

  def test_returns_a_plain_string_not_a_dataset(self):
    d = _make([np.linspace(0.0, 1.0, 3)], np.ones((2, 1)))
    assert isinstance(operations.extract_input(d), str)
