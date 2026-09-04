"""Coverage-completing tests for the ``operations`` verb layer.

The golden-path tests exercise ``comp=`` selection, the happy arithmetic
paths, and the default basis/poly_order. This file targets the error edges
and the less obvious dispatch branches: coordinate (``z0``) selection,
mixed-value_form/mixed-basis rejections, modal-scalar operator
combinations, ufunc edge cases, and every verb's metadata-missing guard.

Run:  PYTHONPATH=src pytest tests/test_coverage_ops.py -v
"""

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
from postgkyl import gpython, operations  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

DATA = os.path.join(ROOT, "tests", "test_data")
F1 = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
F3 = os.path.join(DATA, "generated", "3d_ms_p1.gkyl")


def _dynvec_dataset(tmp_path, time, values):
  from postgkyl.gpython import rio
  path = str(tmp_path / "series.gkyl")
  rio.write_dynvec(path, np.asarray(time), np.asarray(values))
  return pg.load(path)


# ============================================================== operations.select
@needs_gkeyll
def test_select_by_coordinate_on_a_nodal_grid(tmp_path):
  """A dynvector's grid length equals its value count exactly, so
  ``select``'s ``is_matching`` branch is True (unlike interpolated field
  data, whose grid is always one edge longer than its values)."""
  d = _dynvec_dataset(tmp_path, [0.0, 0.5, 1.0, 1.5],
                      [[1.0], [2.0], [3.0], [4.0]])

  by_int = d.select(z0=1)
  np.testing.assert_allclose(by_int.values, [[2.0]])
  np.testing.assert_allclose(by_int.grid[0], [0.5])

  by_float = d.select(z0=0.6)
  np.testing.assert_allclose(by_float.values, [[3.0]])

  by_slice = d.select(z0="1:3")
  np.testing.assert_allclose(by_slice.values, [[2.0], [3.0]])
  np.testing.assert_allclose(by_slice.grid[0], [0.5, 1.0])

  by_negative_int = d.select(z0=-1)
  np.testing.assert_allclose(by_negative_int.values, [[4.0]])

  with pytest.raises(TypeError, match="single index or a slice"):
    d.select(
        z0="1,2")  # comma selector is comp-only syntax, not valid for z-axes


@needs_gkeyll
def test_select_by_coordinate_on_a_non_matching_edge_grid():
  """Interpolated field data: the grid has one more point than the values
  along every axis (edges vs. cell values) -- the ``is_matching`` False
  path."""
  g = pg.load(F1).interpolate()
  assert g.grid[0].shape[0] == g.values.shape[0] + 1

  by_float = g.select(z0=0.0)
  assert by_float.values.shape[0] == 1
  by_slice = g.select(z0="2:5")
  assert by_slice.values.shape[0] == 3
  assert by_slice.grid[0].shape[0] == 4


@needs_gkeyll
def test_select_keeps_native_point_values_in_the_native_backend():
  nodal = pg.load(F1).to_nodal()
  selected = nodal.select(comp=0)
  assert selected.backend == "gkyl"
  assert selected.ctx["value_form"] == "nodal"
  np.testing.assert_array_equal(selected.ctx["cells"], [24])
  assert selected.native.ncomp == 1


# ========================================================== operations.arithmetic
@needs_gkeyll
def test_numpy_domain_rejects_incompatible_grids_and_shapes():
  a = pg.load(F1).interpolate()
  b = pg.load(F1).interpolate()
  b_sub = b.select(comp=0)  # different shape than the full 'a'
  with pytest.raises(ValueError, match="incompatible shapes"):
    a + b_sub

  c = pg.load(F1).interpolate()
  c.grid[0] = c.grid[0] + 1.0  # displace the grid -> no longer "compatible"
  with pytest.raises(ValueError, match="different grids"):
    a + c


@needs_gkeyll
def test_basis_of_raises_when_metadata_missing():
  a, b = pg.load(F1), pg.load(F1)
  del a.ctx["poly_order"]
  with pytest.raises(ValueError, match="basis_type/poly_order"):
    a * b


@needs_gkeyll
def test_modal_binary_rejects_mixing_with_a_plain_array():
  a = pg.load(F1)
  with pytest.raises(ValueError, match="cannot mix native modal data"):
    a * np.zeros((24, 6))


@needs_gkeyll
def test_modal_dataset_pair_rejects_grid_and_basis_mismatch():
  a = pg.load(F1)
  b = pg.load(F1)
  b.grid[0] = b.grid[0] + 100.0
  with pytest.raises(ValueError, match="different grids"):
    a * b

  c = pg.load(F1)
  c.ctx["basis_type"] = "tensor"
  with pytest.raises(ValueError, match="different DG bases"):
    a * c


@needs_gkeyll
def test_modal_dataset_pair_rejects_unsupported_op():
  a, b = pg.load(F1), pg.load(F1)
  with pytest.raises(ValueError, match="not defined between two"):
    a**b


@needs_gkeyll
def test_conf_phase_mul_requires_both_operands_modal():
  """Mixed value_form on a conf*phase multiply (different num_dims)
  must refuse just like the same-dims path, not silently coerce."""
  conf_edges = [np.linspace(0.0, 1.0, 4)]
  phase_edges = [np.linspace(0.0, 1.0, 4), np.linspace(-1.0, 1.0, 5)]
  cbasis = gpython.basis.get_basis("serendipity", 1, 1)
  pbasis = gpython.basis.get_basis("hybrid", 2, 1)

  conf = pg.GData()
  conf.ctx.update(basis_type="serendipity",
                  poly_order=1,
                  value_form="modal",
                  cells=np.array([3]))
  conf.push(conf_edges,
            gpython.array.GkylArray.from_numpy(np.zeros((3, cbasis.num_basis))))

  phase = pg.GData()
  phase.ctx.update(basis_type="hybrid",
                   poly_order=1,
                   value_form="modal",
                   cells=np.array([3, 4]))
  phase.push(
      phase_edges,
      gpython.array.GkylArray.from_numpy(np.zeros((12, pbasis.num_basis))))

  phase_nodal = phase.to_nodal()
  with pytest.raises(ValueError, match="modal DG coefficients only"):
    conf * phase_nodal


@needs_gkeyll
@pytest.mark.parametrize(
    "expr",
    [
        lambda a: a / 2.0,  # modal / scalar (linear divide)
        lambda a: 5.0 - a,  # scalar - modal
        lambda a: a - 5.0,  # modal - scalar
        lambda a: 5.0 / a,  # scalar / modal (weak reciprocal)
    ])
def test_modal_scalar_operator_combinations(expr):
  a = pg.load(F1)
  out = expr(a)
  assert isinstance(out, pg.GData)
  assert out.backend == "gkyl"


@needs_gkeyll
def test_modal_scalar_rejects_reflected_power():
  a = pg.load(F1)
  with pytest.raises(ValueError, match="not defined for modal"):
    2.0**a


@needs_gkeyll
def test_apply_ufunc_non_reduction_method_and_out_kwarg_are_rejected():
  a = pg.load(F1).interpolate()
  assert a.__array_ufunc__(np.add, "accumulate", a) is NotImplemented
  assert a.__array_ufunc__(np.sqrt, "__call__", a,
                           out=(np.zeros(1), )) is NotImplemented


def test_apply_ufunc_reductions_return_numpy_results():
  values = np.array([[3.0, 2.0], [-4.0, 5.0], [1.0, -2.0]])
  a = pg.GData()
  a.push([np.linspace(0.0, 1.0, 4)], values)

  assert np.max(a) == np.max(values)
  assert np.min(a) == np.min(values)
  assert np.sum(a) == np.sum(values)
  assert np.prod(a) == np.prod(values)
  np.testing.assert_allclose(np.max(a, axis=0), np.max(values, axis=0))
  np.testing.assert_allclose(
      np.sum(a, axis=1, keepdims=True, dtype=np.float64),
      np.sum(values, axis=1, keepdims=True, dtype=np.float64))

  bool_values = values > 0
  b = pg.GData()
  b.push([np.linspace(0.0, 1.0, 4)], bool_values)
  assert np.all(b) == np.all(bool_values)
  assert np.any(b) == np.any(bool_values)


def test_remaining_reflected_and_unary_operators_preserve_data():
  a = pg.GData()
  a.push([np.linspace(0.0, 1.0, 4)], np.array([[-2.0], [0.0], [3.0]]))

  np.testing.assert_allclose((1.0 + a).values, 1.0 + a.values)
  np.testing.assert_allclose(abs(a).values, np.abs(a.values))
  positive = +a
  np.testing.assert_allclose(positive.values, a.values)
  assert positive is not a


def test_apply_ufunc_reduction_rejects_non_dataset_inputs():
  assert operations.arithmetic.apply_ufunc(np.add, "reduce", 1.0,
                                           2.0) is NotImplemented


@needs_gkeyll
def test_apply_ufunc_rejects_shape_mismatch():
  a = pg.load(F1).interpolate()
  b = pg.load(F1).interpolate().select(comp=0)
  with pytest.raises(ValueError, match="incompatible shapes"):
    np.add(a, b)


@needs_gkeyll
def test_apply_ufunc_accepts_scalars_and_rejects_unhandled_types():
  a = pg.load(F1).interpolate()
  out = np.add(a, 2.0)
  np.testing.assert_allclose(out.values, a.values + 2.0)
  assert a.__array_ufunc__(np.add, "__call__", a,
                           "not-a-number") is NotImplemented


# ========================================================== operations.interpolate
def test_interpolate_requires_basis_type_when_none_given():
  d = pg.GData()
  d.push([np.linspace(0.0, 1.0, 4)], np.zeros((3, 2)))
  with pytest.raises(ValueError, match="no 'basis_type' metadata"):
    d.interpolate()


def test_interpolate_requires_poly_order_when_none_given():
  d = pg.GData()
  d.ctx["basis_type"] = "serendipity"
  d.push([np.linspace(0.0, 1.0, 4)], np.zeros((3, 2)))
  with pytest.raises(ValueError, match="no 'poly_order' metadata"):
    d.interpolate()


# =========================================================== operations.represent
@needs_gkeyll
def test_represent_rejects_numpy_backed_and_missing_metadata():
  interpolated = pg.load(F1).interpolate()
  with pytest.raises(ValueError, match="NumPy-backed"):
    interpolated.to_modal()

  a = pg.load(F1)
  del a.ctx["poly_order"]
  with pytest.raises(ValueError, match="no basis_type/poly_order"):
    a.to_nodal()


@needs_gkeyll
def test_represent_rejects_unknown_target():
  a = pg.load(F1)
  with pytest.raises(ValueError, match="unknown value_form"):
    operations.represent(a, to="bogus")


@needs_gkeyll
def test_represent_rejects_quad_dataset_missing_num_quad():
  q = pg.load(F1).to_quad()
  del q.ctx["num_quad"]
  with pytest.raises(ValueError, match="lost its 'num_quad'"):
    q.to_modal()


@needs_gkeyll
def test_represent_same_representation_clones():
  a = pg.load(F1)
  same = a.to_modal()  # already modal -> the "cur == to" clone branch
  np.testing.assert_allclose(same.values, a.values)
  assert same.native is not a.native


@needs_gkeyll
def test_apply_rejects_non_modal_data():
  a = pg.load(F1).to_nodal()
  with pytest.raises(ValueError, match="expects modal data"):
    a.apply(np.sqrt)


# =============================================================== operations.info
@needs_gkeyll
def test_info_verb_handles_multiple_datasets():
  a, b = pg.load(F1), pg.load(F1)
  summaries = pg.info(a, b)
  assert len(summaries) == 2
  assert all("Number of components" in s for s in summaries)


# ============================================================ operations.integrate
@needs_gkeyll
def test_integrate_requires_basis_metadata():
  a = pg.load(F1)
  del a.ctx["basis_type"]
  with pytest.raises(ValueError, match="basis_type/poly_order"):
    a.integrate()


# ============================================================== operations.average
@needs_gkeyll
def test_average_full_reduction_matches_integrate_over_volume():
  a = pg.load(F1)
  avg = a.average([0])
  assert avg.num_dims == 1
  assert avg.ctx["cells"].tolist() == [1]
  lo, up = a.bounds
  volume = float(up[0] - lo[0])
  integral = a.integrate()
  b0 = 2.0**(-avg.num_dims / 2.0)
  np.testing.assert_allclose(np.asarray(avg.native.view())[0, ::2] * b0,
                             np.asarray(integral) / volume,
                             rtol=1e-8)


@needs_gkeyll
def test_average_partial_reduction_of_a_constant_field():
  basis_type, p = "serendipity", 1
  cells = [4, 3]
  nb = gpython.basis.num_basis(basis_type, 2, p)
  b0 = 2.0**(-2 / 2.0)
  coeffs = np.zeros((int(np.prod(cells)), nb))
  coeffs[:, 0] = 3.0 / b0

  d = pg.GData()
  d.ctx.update(basis_type=basis_type,
               poly_order=p,
               cells=np.array(cells),
               value_form="modal")
  grid = [
      np.linspace(0.0, 2.0, cells[0] + 1),
      np.linspace(0.0, 1.0, cells[1] + 1)
  ]
  d.push(grid, gpython.array.GkylArray.from_numpy(coeffs))

  out = d.average([1])
  assert out.num_dims == 1
  assert out.ctx["cells"].tolist() == [cells[0]]
  assert out.grid[0].shape[0] == cells[0] + 1
  b0_avg = 2.0**(-1 / 2.0)
  np.testing.assert_allclose(np.asarray(out.native.view())[:, 0] * b0_avg,
                             3.0,
                             atol=1e-10)


@needs_gkeyll
def test_average_accepts_a_compatible_weight():
  basis_type, poly_order, cells = "serendipity", 1, [4]
  num_basis = gpython.basis.num_basis(basis_type, 1, poly_order)
  basis_constant = 2.0**0.5

  def constant_state(value):
    coefficients = np.zeros((cells[0], num_basis))
    coefficients[:, 0] = value * basis_constant
    data = pg.GData(
        ctx={
            "basis_type": basis_type,
            "poly_order": poly_order,
            "value_form": "modal",
            "cells": np.array(cells),
        })
    data.push([np.linspace(0.0, 1.0, cells[0] + 1)],
              gpython.GkylArray.from_numpy(coefficients))
    return data

  out = constant_state(3.0).average([0], weight=constant_state(2.0))
  np.testing.assert_allclose(out.native.view()[0, 0] / basis_constant,
                             3.0,
                             atol=1e-10)


@needs_gkeyll
def test_average_rejects_numpy_backed_and_non_modal():
  interpolated = pg.load(F1).interpolate()
  with pytest.raises(ValueError, match="native modal data"):
    interpolated.average([0])

  nodal = pg.load(F1).to_nodal()
  with pytest.raises(ValueError, match="modal value_form"):
    nodal.average([0])


@needs_gkeyll
def test_average_rejects_missing_basis_metadata():
  a = pg.load(F1)
  del a.ctx["poly_order"]
  with pytest.raises(ValueError, match="basis_type/poly_order"):
    a.average([0])


@needs_gkeyll
def test_average_rejects_weight_mismatch():
  a = pg.load(F1)
  weight_wrong_dims = pg.GData()
  weight_wrong_dims.ctx.update(basis_type="serendipity",
                               poly_order=1,
                               cells=np.array([4, 3]),
                               value_form="modal")
  weight_wrong_dims.push([np.linspace(0.0, 1.0, 5),
                          np.linspace(0.0, 1.0, 4)],
                         gpython.array.GkylArray.from_numpy(np.zeros((12, 4))))
  with pytest.raises(ValueError, match="dims but the field has"):
    a.average([0], weight=weight_wrong_dims)

  weight_wrong_basis = pg.load(F1)
  weight_wrong_basis.ctx["basis_type"] = "tensor"
  with pytest.raises(ValueError, match="basis_type"):
    a.average([0], weight=weight_wrong_basis)

  weight_wrong_p = pg.load(F1)
  weight_wrong_p.ctx["poly_order"] = 2
  with pytest.raises(ValueError, match="poly_order"):
    a.average([0], weight=weight_wrong_p)


@needs_gkeyll
def test_average_tag_and_label_and_inplace():
  a = pg.load(F1)
  out = a.average([0], tag="reduced", label="my label")
  assert out.tag == "reduced"
  assert out.label == "my label"
  assert a.num_dims == 1 and a.ctx["cells"].tolist() != [1
                                                         ]  # original untouched

  b = pg.load(F1)
  mutated = b.average([0], inplace=True)
  assert mutated is b
  assert b.ctx["cells"].tolist() == [1]


# ============================================================= operations.integrate
@needs_gkeyll
def test_integrate_partial_modal_stays_native_and_exact():
  a = pg.load(F3)
  reduced = a.integrate(2)
  assert reduced.backend == "gkyl"
  assert reduced.ctx["value_form"] == "modal"
  assert reduced.num_dims == 2
  np.testing.assert_allclose(reduced.integrate(), a.integrate(), rtol=1e-12)


def test_integrate_partial_point_data_removes_the_axis():
  a = pg.load(F3).interpolate()
  r = a.integrate(2)
  assert r.num_dims == 2
  assert r.num_cells.tolist() == list(a.num_cells[:2])
  assert r.ctx.get("interpolated") is True


def test_integrate_partial_point_data_matches_manual_sum():
  a = pg.load(F3).interpolate()
  r = a.integrate(2)
  dz = np.diff(a.grid[2])
  expected = np.tensordot(np.asarray(a.values), dz, axes=([2], [0]))
  np.testing.assert_allclose(np.asarray(r.values), expected)


def test_integrate_point_default_is_a_full_terminal_integral():
  a = pg.load(F1).interpolate()
  result = a.integrate()
  assert isinstance(result, np.ndarray)
  assert result.shape == (a.num_comps, )


@needs_gkeyll
def test_integrate_partial_on_native_nodal_representation():
  # A gkyl-native nodal/quad dataset materializes to its true point grid
  # before integrating -- same bridge ``plot`` uses (Doctrine V: one home).
  nodal = pg.load(F3).to_nodal()
  r = nodal.integrate(2)
  assert r.backend == "numpy"
  assert r.num_dims == 2
  assert r.ctx.get("value_form") is None  # stale tag cleared, not "nodal"


def test_integrate_partial_tag_and_label():
  a = pg.load(F3).interpolate()
  r = a.integrate(2, tag="reduced", label="my label")
  assert r.tag == "reduced"
  assert r.label == "my label"
  assert a.num_dims == 3  # original left untouched (inplace=False default)


def test_integrate_partial_inplace_mutates_the_dataset():
  a = pg.load(F3).interpolate()
  out = a.integrate(2, inplace=True)
  assert out is a
  assert a.num_dims == 2


def test_integrate_full_rejects_partial_result_options():
  a = pg.load(F1).interpolate()
  with pytest.raises(ValueError, match="partial integration"):
    a.integrate(tag="not-a-dataset")


@pytest.mark.parametrize(("axis", "message"), [
    ((), "at least one axis"),
    ((0, 0), "must be distinct"),
    ((1, ), "out of range"),
])
def test_integrate_rejects_invalid_axis_sets(axis, message):
  a = pg.GData()
  a.push([np.linspace(0.0, 1.0, 5)], np.ones((4, 1)))
  with pytest.raises(ValueError, match=message):
    a.integrate(axis)


def test_native_integration_guard_reports_backend_before_basis():
  from importlib import import_module
  integrate_module = import_module("postgkyl.operations.integrate")
  a = pg.GData()
  a.push([np.linspace(0.0, 1.0, 5)], np.ones((4, 1)))
  with pytest.raises(ValueError, match="needs native modal data"):
    integrate_module._native_basis(a)


@needs_gkeyll
def test_native_integration_guard_rejects_point_value_forms():
  from importlib import import_module
  integrate_module = import_module("postgkyl.operations.integrate")
  nodal = pg.load(F1).to_nodal()
  with pytest.raises(ValueError, match="expects the modal value_form"):
    integrate_module._native_basis(nodal)


def test_integrate_rejects_native_only_op_for_point_data():
  one_dim = pg.GData()
  one_dim.push([np.linspace(0.0, 1.0, 5)], np.ones((4, 1)))
  with pytest.raises(ValueError, match="full native-DG"):
    one_dim.integrate(op="abs")

  two_dim = pg.GData()
  two_dim.push([np.linspace(0.0, 1.0, 5),
                np.linspace(0.0, 1.0, 4)], np.ones((4, 3, 1)))
  with pytest.raises(ValueError, match="full native-DG"):
    two_dim.integrate(1, op="abs")


def test_remaining_mapped_axes_reindexes_surviving_groups():
  from importlib import import_module
  integrate_module = import_module("postgkyl.operations.integrate")
  data = pg.GData(ctx={"mapped_axes": {0: 0, 1: 0, 2: 2}})
  assert integrate_module._remaining_mapped_axes(data, [0, 1]) == {0: 0, 1: 0}


def test_curvilinear_lookup_ignores_flat_axes_and_can_miss_an_axis():
  from postgkyl.operations._curvilinear import block_for_axis, curvilinear_blocks

  blocks = curvilinear_blocks([np.arange(3), np.ones((3, 4))], {
      0: 0,
      1: 1,
  })
  assert blocks == {1: [1]}
  assert block_for_axis(blocks, 0) is None


def test_fft_preserves_an_already_point_aligned_grid():
  data = pg.GData()
  data.push([np.linspace(0.0, 1.0, 8, endpoint=False)],
            np.arange(8, dtype=float)[:, None])
  out = data.fft()
  assert out.values.shape == data.values.shape


def test_val2coord_range_accepts_negative_slice_endpoints():
  from postgkyl.operations.val2coord import _get_range

  np.testing.assert_array_equal(_get_range("-4:-1", 6), [2, 3, 4])
  np.testing.assert_array_equal(_get_range("1:4", 6), [1, 2, 3])
