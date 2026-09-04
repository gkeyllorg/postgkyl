"""Coverage-completing tests for the leaf/engine/backend layers: numerics,
dg (interpolate/modal/rep), the remaining gpython corners (array/kernels), and the
matplotlib render backend.

Run:  PYTHONPATH=src pytest tests/test_coverage_leaf.py -v
"""

import importlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

import matplotlib

matplotlib.use("Agg")

import postgkyl as pg  # noqa: E402
from postgkyl import gpython, dg  # noqa: E402
# NB: `postgkyl.numerics.idx_parser` (the submodule) is shadowed by the
# `idx_parser` FUNCTION that numerics/__init__.py re-exports under the same
# attribute name -- both plain `from ... import idx_parser` and
# `import a.b.idx_parser as x` (itself sugar for `x = a.b.idx_parser`, an
# *attribute* lookup) resolve to the function. `importlib` sidesteps the
# package's __init__ entirely and returns the actual submodule object.
ip = importlib.import_module("postgkyl.numerics.idx_parser")
from postgkyl.numerics import elementwise  # noqa: E402
from postgkyl.gdatastate.gdatastate import GDataState  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

DATA = os.path.join(ROOT, "tests", "test_data")
F1 = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")


# ============================================================ numerics/idx_parser
def test_find_nearest_index_raises_without_a_coordinate_array():
  with pytest.raises(TypeError, match="no coordinate array"):
    ip._find_nearest_index(None, 1.0)


def test_find_nearest_index_edge_cases():
  arr = np.array([0.0, 1.0, 2.0, 3.0])
  assert ip._find_nearest_index(arr, 10.0) == 2  # beyond the end -> idx-2
  assert ip._find_nearest_index(arr, -10.0) == 0  # before the start -> idx==0


def test_find_cell_index_raises_without_a_coordinate_array():
  with pytest.raises(TypeError, match="no coordinate array"):
    ip._find_cell_index(None, 1.0)


def test_string_to_index_rejects_non_strings():
  with pytest.raises(TypeError, match="not a string"):
    ip._string_to_index(1.5, np.array([0.0, 1.0]))


def test_string_to_index_parses_a_float_string():
  arr = np.array([0.0, 1.0, 2.0, 3.0])
  assert ip._string_to_index("1.4", arr) == 1
  assert ip._string_to_index("1.4", arr, nodal=True) == 2


def test_idx_parser_slice_with_empty_start_and_stop():
  arr = np.array([0.0, 1.0, 2.0, 3.0])
  s = ip.idx_parser("2:", arr)  # empty stop -> len(array)
  assert s == slice(2, 4)
  s2 = ip.idx_parser(":2", arr)  # empty start -> 0
  assert s2 == slice(0, 2)


def test_idx_parser_slice_negative_stop():
  arr = np.array([0.0, 1.0, 2.0, 3.0])
  assert ip.idx_parser("0:-1", arr) == slice(0, 4)


def test_idx_parser_slice_with_non_integer_stop_falls_back_to_float_lookup():
  """``hi`` failing int() parsing (a float-valued stop) is swallowed by the
  ``except ValueError: pass`` guard, then resolved via the float-coordinate
  path instead of the integer-count adjustment."""
  arr = np.array([0.0, 1.0, 2.0, 3.0])
  s = ip.idx_parser("0:1.4", arr)
  assert s == slice(0, 1)


def test_idx_parser_rejects_unsupported_types():
  with pytest.raises(TypeError, match="Unsupported selector type"):
    ip.idx_parser(3.0 + 4.0j)


# ============================================================ numerics/elementwise
def test_grids_compatible_rejects_different_ndims():
  a = [np.linspace(0.0, 1.0, 4)]
  b = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
  assert elementwise.grids_compatible(a, b) is False


def test_grid_is_prefix_rejects_out_of_range_lengths():
  same_len = [np.linspace(0.0, 1.0, 4)]
  assert elementwise.grid_is_prefix(same_len,
                                    same_len) is False  # not strictly smaller
  assert elementwise.grid_is_prefix([], same_len) is False  # empty


# ===================================================================== dg/interpolate
@needs_gkeyll
def test_interpolate_degenerates_1d_hybrid_to_serendipity():
  nb = dg.num_basis(1, 1, "serendipity")
  values = np.zeros((5, nb))
  grid = [np.linspace(0.0, 1.0, 6)]
  grid_out, out = dg.interpolate(values,
                                 grid,
                                 poly_order=1,
                                 basis_type="hybrid")
  assert out.shape[-1] == 1


@needs_gkeyll
def test_interpolate_converts_nodal_basis_data_through_nodal_to_modal():
  """A NumPy-backed dataset tagged ``value_form="nodal"`` at load time
  forces the nodal-basis-file convention -- exercising the nodal-to-modal
  conversion machinery inside ``dg.interpolate`` (the values themselves are
  meaningless here, only the code path and output shape matter). Native
  (gkyl-backed) data always enforces the modal value_form, so this must be
  plain NumPy from the start."""
  nb = dg.num_basis(1, 1, "serendipity")
  d = pg.GData(ctx={
      "poly_order": 1,
      "basis_type": "serendipity",
      "value_form": "nodal"
  })
  d.push([np.linspace(0.0, 1.0, 5)], np.zeros((4, nb)))
  out = d.interpolate()
  assert out.is_interpolated
  assert out.values.ndim == 2


@needs_gkeyll
def test_local_poly_degenerates_1d_hybrid_to_serendipity():
  nb = dg.num_basis(1, 1, "serendipity")
  values = np.zeros((5, nb))
  grid = [np.linspace(0.0, 1.0, 6)]
  grid_out, out = dg.local_poly(values, grid, poly_order=1, basis_type="hybrid")
  assert out.shape[-1] == 1


@needs_gkeyll
def test_local_poly_converts_nodal_basis_data_through_nodal_to_modal():
  nb = dg.num_basis(1, 1, "serendipity")
  d = pg.GData(ctx={
      "poly_order": 1,
      "basis_type": "serendipity",
      "value_form": "nodal"
  })
  d.push([np.linspace(0.0, 1.0, 5)], np.zeros((4, nb)))
  out = d.local_poly()
  assert out.is_interpolated
  assert out.values.ndim == 2


# ======================================================================= dg/modal
@needs_gkeyll
def test_modal_power_non_integer_exponent_uses_powsqrt():
  """A fractional exponent used to raise; it now routes through
  ``dg.modal.powsqrt`` (``gkyl_proj_powsqrt_on_basis``) instead, field by
  field, and must match the defining identity ``f ** 1.5 == f * (f ** 0.5)``
  (the latter computed via the same powsqrt kernel at a different exponent,
  so this pins the exponent-doubling translation rather than just "some
  value came out")."""
  a = pg.load(F1)
  cubed_half = a**1.5
  half = a**0.5
  expect = dg.modal.weak_mul(a.ctx["basis_type"], a.num_dims,
                             a.ctx["poly_order"], a.native, half.native)
  np.testing.assert_allclose(cubed_half.native.view(), expect.view(), atol=1e-8)


@needs_gkeyll
def test_modal_power_without_cells_raises_for_non_integer_exponent():
  """``cells`` builds the powsqrt kernel's index range; calling ``dg.modal.
  power`` directly (bypassing ``operations.arithmetic``, which always
  supplies it from ``ctx["cells"]``) with a non-integer exponent and no
  ``cells`` must fail clearly rather than crash inside the kernel."""
  a = _const_gkyl_array("serendipity", 1, 1, [4], 3.0)
  with pytest.raises(ValueError, match="needs cells="):
    dg.modal.power("serendipity", 1, 1, a, 0.5)


@needs_gkeyll
def test_modal_powsqrt_rejects_multi_field_ncomp_mismatch():
  """``a.ncomp`` must be a multiple of the basis's ``num_basis``."""
  a = gpython.GkylArray.alloc(3, 4)  # 3 is not a multiple of num_basis (2)
  with pytest.raises(ValueError, match="not a multiple"):
    dg.modal.powsqrt("serendipity", 1, 1, [4], a, 1.0)


def _const_gkyl_array(basis_type, ndim, p, cells, value):
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  b0 = 2.0**(-ndim / 2.0)
  coeffs = np.zeros((int(np.prod(cells)), nb))
  coeffs[:, 0] = value / b0
  return gpython.GkylArray.from_numpy(coeffs)


@needs_gkeyll
def test_modal_average_full_reduction_corrects_the_raw_kernel_value():
  """Unlike the raw kernel (test_array_average_full_reduction_unweighted_
  writes_a_raw_value in test_gpython_kernels.py), ``dg.modal.average``
  rescales the degenerate (every dim averaged), unweighted case back into a
  properly b0-normalized coefficient -- so it agrees with the weighted
  case (which the underlying weak division already normalizes) and with
  every other modal dataset's "value = coeff0 * b0" convention."""
  basis_type, p, cells = "serendipity", 1, [4]
  a = _const_gkyl_array(basis_type, 1, p, cells, 3.0)
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([2.0]),
      "cells": np.array(cells)
  }
  keep_dirs, cells_avg, out = dg.modal.average(grid, basis_type, 1, p, a, [0])
  assert keep_dirs == []
  assert cells_avg == [1]
  b0 = 2.0**(-1 / 2.0)
  np.testing.assert_allclose(out.view()[0, 0] * b0, 3.0, atol=1e-10)


@needs_gkeyll
def test_modal_average_multi_field_loops_and_reassembles_per_field():
  """``gkyl_array_average`` has no field-index argument, so a multi-field
  array (ncomp = nfields * num_basis) must be split, averaged one field at
  a time, and reassembled -- verify each field's result matches averaging
  it alone."""
  basis_type, p, cells = "serendipity", 1, [4]
  nb = gpython.basis.num_basis(basis_type, 1, p)
  values = [3.0, -1.5]
  coeffs = np.concatenate(
      [_const_gkyl_array(basis_type, 1, p, cells, v).view() for v in values],
      axis=-1)
  a = gpython.GkylArray.from_numpy(coeffs)
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([2.0]),
      "cells": np.array(cells)
  }
  keep_dirs, cells_avg, out = dg.modal.average(grid, basis_type, 1, p, a, [0])
  assert out.ncomp == 2 * nb
  b0 = 2.0**(-1 / 2.0)
  np.testing.assert_allclose(out.view()[0, 0] * b0, values[0], atol=1e-10)
  np.testing.assert_allclose(out.view()[0, nb] * b0, values[1], atol=1e-10)


@needs_gkeyll
def test_modal_average_rejects_dirs_out_of_range():
  basis_type, p, cells = "serendipity", 1, [4]
  a = _const_gkyl_array(basis_type, 1, p, cells, 1.0)
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([1.0]),
      "cells": np.array(cells)
  }
  with pytest.raises(ValueError, match="out of range"):
    dg.modal.average(grid, basis_type, 1, p, a, [1])


@needs_gkeyll
def test_modal_average_rejects_ncomp_not_a_multiple_of_num_basis():
  a = gpython.GkylArray.alloc(3, 4)  # num_basis for ser p1 1D is 2
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([1.0]),
      "cells": np.array([4])
  }
  with pytest.raises(ValueError, match="not a multiple"):
    dg.modal.average(grid, "serendipity", 1, 1, a, [0])


@needs_gkeyll
def test_modal_average_rejects_weight_ncomp_mismatch():
  basis_type, p, cells = "serendipity", 1, [4]
  a = _const_gkyl_array(basis_type, 1, p, cells, 1.0)
  w = gpython.GkylArray.alloc(3, 4)  # wrong ncomp for this basis
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([1.0]),
      "cells": np.array(cells)
  }
  with pytest.raises(ValueError, match="weight ncomp"):
    dg.modal.average(grid, basis_type, 1, p, a, [0], weight=w)


# ==================================================================== gpython/array
@needs_gkeyll
def test_gkylarray_from_numpy_rejects_scalar_input(monkeypatch):
  """``np.ascontiguousarray`` itself always promotes a 0-d input to 1-D, so
  this guard can't be reached through any real ndarray -- it defends against
  a hypothetical future NumPy behavior change. Drive it directly by faking
  ascontiguousarray's return value."""
  from postgkyl.gpython import array as array_mod
  monkeypatch.setattr(array_mod.np,
                      "ascontiguousarray",
                      lambda values, dtype=None: np.array(5.0, dtype=dtype))
  with pytest.raises(ValueError, match="at least a 1-D"):
    gpython.GkylArray.from_numpy(np.array(5.0))


# ==================================================================== gpython/kernels
@needs_gkeyll
def test_weak_mul_conf_phase_rejects_unsupported_phase_basis():
  from postgkyl.gpython import kernels as k
  cop = gpython.GkylArray.alloc(2, 3)
  pop = gpython.GkylArray.alloc(2, 12)
  with pytest.raises(NotImplementedError, match="cross-mul supports"):
    k.weak_mul_conf_phase("serendipity", 1, "bogus-basis", 2, 1, [3], [3, 4],
                          cop, pop)


@needs_gkeyll
def test_weak_mul_conf_phase_rejects_pop_ncomp_mismatch():
  from postgkyl.gpython import kernels as k
  cbasis = gpython.basis.get_basis("serendipity", 1, 1)
  pbasis = gpython.basis.get_basis("serendipity", 2, 1)
  cop = gpython.GkylArray.alloc(cbasis.num_basis, 3)
  pop = gpython.GkylArray.alloc(pbasis.num_basis + 1, 12)  # wrong ncomp
  with pytest.raises(ValueError, match="pop.ncomp"):
    k.weak_mul_conf_phase("serendipity", 1, "serendipity", 2, 1, [3], [3, 4],
                          cop, pop)


# ======================================================================= dg/rep
@needs_gkeyll
def test_apply_per_field_rejects_ncomp_not_a_multiple():
  arr = gpython.GkylArray.alloc(3, 4)  # ncomp=3, not a multiple of num_basis=2
  with pytest.raises(ValueError, match="not a multiple"):
    dg.rep.modal_to_nodal("serendipity", 1, 1, arr)


@needs_gkeyll
def test_materialize_rejects_ncomp_not_a_multiple_of_points_per_cell():
  a = pg.load(F1)
  arr = gpython.GkylArray.alloc(a.native.ncomp + 1, a.native.size)  # off by one
  with pytest.raises(ValueError, match="points/cell"):
    dg.rep.materialize("serendipity", 1, 1, arr, a.grid, "nodal")


@needs_gkeyll
def test_tensor_point_layout_rejects_a_non_tensor_lin_index_collision(
    monkeypatch):
  """A hand-crafted node set whose per-dimension unique counts multiply to
  ``num_basis`` (passing the coarse check) yet still contains a duplicate
  cell -> point mapping (failing the fine-grained tensor-linearization
  check): both are real defensive checks in ``_tensor_point_layout``, but
  Gkeyll's actual basis node sets never exhibit either failure mode, so we
  drive them directly by faking ``node_coords``."""
  from postgkyl.dg import rep

  duplicate_coords = np.array([[0., 0.], [0., 1.], [1., 0.], [0., 0.]])
  monkeypatch.setattr(rep.gpython_basis, "node_coords",
                      lambda *a, **k: duplicate_coords)
  with pytest.raises(ValueError, match="not a tensor product"):
    rep._tensor_point_layout("serendipity", 2, 1, "nodal", None)


@needs_gkeyll
def test_tensor_point_layout_rejects_misaligned_node_coordinates(monkeypatch):
  from postgkyl.dg import rep

  nan_coords = np.array([[0.0], [np.nan]])
  monkeypatch.setattr(rep.gpython_basis, "node_coords",
                      lambda *a, **k: nan_coords)
  with pytest.raises(ValueError, match="do not align on a tensor grid"):
    rep._tensor_point_layout("serendipity", 1, 1, "nodal", None)


# =================================================================== render
@needs_gkeyll
def test_plot_rejects_empty_and_valueless_datasets():
  from postgkyl import render
  with pytest.raises(ValueError, match="nothing to plot"):
    render.plot()

  empty = GDataState()
  with pytest.raises(ValueError, match="no values to plot"):
    render.plot(empty)


@needs_gkeyll
def test_plot_multi_dataset_1d_with_labels_shows_legend_and_title():
  from postgkyl import render
  a = pg.load(F1).interpolate().select(comp=0)
  b = pg.load(F1).interpolate().select(comp=0)
  fig = render.plot(a,
                    b,
                    multiblock=True,
                    legend_labels=["first", "second"],
                    title="my title",
                    show=False)
  assert fig is not None
  assert fig._suptitle is not None
  assert fig._suptitle.get_text() == "my title"


@needs_gkeyll
def test_plot_rejects_more_than_two_dimensions():
  from postgkyl import render
  d = GDataState()
  d.push([np.linspace(0, 1, 3),
          np.linspace(0, 1, 3),
          np.linspace(0, 1, 3)], np.zeros((2, 2, 2, 1)))
  with pytest.raises(ValueError,
                     match="Only 1D and 2D plots are currently supported"):
    render.plot(d)


@needs_gkeyll
def test_plot_show_true_does_not_error_with_agg_backend(monkeypatch):
  # matplotlib's own FigureCanvasBase.show() silently no-ops (no warning) on
  # a genuinely headless Linux host (no DISPLAY) -- a deliberate
  # headless-friendliness special case, not something this test is about.
  # Force a DISPLAY so the assertion below exercises the actual behavior
  # under test (Agg backend + show=True -> warn, don't raise) regardless of
  # whether this host has a real display.
  monkeypatch.setenv("DISPLAY", ":0")
  a = pg.load(F1).interpolate().select(comp=0)
  with pytest.warns(UserWarning, match="non-interactive"):
    fig = a.plot(show=True)
  assert fig is not None
