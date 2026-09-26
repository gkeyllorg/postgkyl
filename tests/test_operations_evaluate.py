"""Tests for the ``evaluate`` verb -- the RPN expression evaluator over datasets."""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, operations
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


def _field(value, grid=None):
  grid = grid if grid is not None else [np.linspace(0.0, 1.0, 5)]
  return _make(grid, np.full((4, 1), value))


# -------------------------------------------------------- parity with verbs
def test_add_two_datasets_matches_direct_arithmetic():
  """The grammar's dataset-index tokens are plain 'fN' (no brackets --
  'fN[c]' is the *component* selector, per the module docstring); this is
  the byte-compatible spelling for combining two whole datasets."""
  a, b = _field(2.0), _field(3.0)
  out = operations.evaluate("f0 f1 +", a, b)
  np.testing.assert_allclose(out.get_values().flatten(), 5.0)


def test_default_f_means_f0():
  a = _field(4.0)
  out = operations.evaluate("f 2 *", a)
  np.testing.assert_allclose(out.get_values().flatten(), 8.0)


def test_component_bracket_selects_a_component():
  a = _make([np.linspace(0.0, 1.0, 5)], np.tile([1.0, 2.0, 3.0], (4, 1)))
  out = operations.evaluate("f0[1] sq", a)
  np.testing.assert_allclose(out.get_values().flatten(), 4.0)


def test_ctx_key_token():
  a = _field(1.0)
  a.ctx["scale"] = 3.0
  out = operations.evaluate("f0 f0.scale *", a)
  np.testing.assert_allclose(out.get_values().flatten(), 3.0)


def test_unknown_ctx_key_raises():
  a = _field(1.0)
  with pytest.raises(ValueError, match="unknown ctx key"):
    operations.evaluate("f0.nope", a)


# -------------------------------------------------------------- operators
def test_sqrt_and_abs():
  a = _field(-4.0)
  out = operations.evaluate("f abs sqrt", a)
  np.testing.assert_allclose(out.get_values().flatten(), 2.0)


def test_min_max_mean():
  a = _make([np.linspace(0.0, 1.0, 5)],
            np.array([1.0, 2.0, 3.0, 4.0])[:, np.newaxis])
  assert operations.evaluate("f min",
                             a).get_values().flatten()[0] == pytest.approx(1.0)
  assert operations.evaluate("f max",
                             a).get_values().flatten()[0] == pytest.approx(4.0)
  assert operations.evaluate("f mean",
                             a).get_values().flatten()[0] == pytest.approx(2.5)


def test_numeric_literal_and_axis_slice_literal():
  a = _field(2.0)
  out = operations.evaluate("f 3.0 +", a)
  np.testing.assert_allclose(out.get_values().flatten(), 5.0)


# ------------------------------------------------------------------ result
def test_result_class_and_defaults():
  a, b = _field(2.0), _field(3.0)
  out = operations.evaluate("f0 f1 +", a, b)
  assert isinstance(out, GDataState)
  assert out.get_tag() == "default"
  assert out.get_label() == "f0 f1 +"


def test_tag_and_label_explicit():
  a, b = _field(2.0), _field(3.0)
  out = operations.evaluate("f0 f1 +", a, b, tag="t", label="sum")
  assert out.get_tag() == "t"
  assert out.get_label() == "sum"


def test_num_comps_reflects_the_actual_output_not_a_stale_operand_value():
  """A component-changing op (here 'dot', which reduces a vector to a
  scalar) must not have its output metadata clobbered by a stale
  'num_comps'/'cells' merged in from the (differently-shaped) operands."""
  a = _make([np.linspace(0.0, 1.0, 5)], np.tile([1.0, 0.0, 0.0], (4, 1)))
  b = _make([np.linspace(0.0, 1.0, 5)], np.tile([1.0, 0.0, 0.0], (4, 1)))
  out = operations.evaluate("f0 f1 dot", a, b)
  assert out.get_num_comps() == 1
  np.testing.assert_allclose(out.get_values().flatten(), 1.0)


def test_conflicting_ctx_keys_are_dropped_not_merged():
  a = _field(2.0)
  b = _field(3.0)
  a.ctx["note"] = "A"
  b.ctx["note"] = "B"
  out = operations.evaluate("f0 f1 +", a, b)
  assert "note" not in out.ctx


def test_bracket_literal_and_colon_axis_literal():
  a = _make([np.linspace(0.0, 1.0, 5)],
            np.array([1.0, 2.0, 3.0, 4.0])[:, np.newaxis])
  # a bare bracket literal (no leading 'f') exercises the eval() fallback
  out = operations.evaluate("[1,2,3] mean", a)
  np.testing.assert_allclose(out.get_values().flatten(), 2.0)
  # a bare colon axis spec exercises the str-literal fallback + 'int'
  out2 = operations.evaluate("f 0:1 int", a)
  assert isinstance(out2, GDataState)


# -------------------------------------------------------------------- errors
def test_empty_datasets_raises():
  with pytest.raises(ValueError, match="at least one dataset"):
    operations.evaluate("f 2 *")


def test_empty_expression_raises():
  a = _field(1.0)
  with pytest.raises(ValueError, match="produced no result"):
    operations.evaluate("", a)


def test_unrecognized_token_raises():
  a = _field(1.0)
  with pytest.raises(ValueError, match="neither data nor an operator"):
    operations.evaluate("f totally_bogus_token", a)


def test_operator_failure_is_wrapped_in_value_error():
  # 1D grid (num_dims=1) with 4 components: 'div' (num_in=1) refuses a
  # component count larger than the number of dimensions.
  a = _make([np.linspace(0.0, 1.0, 2)], np.tile([1.0, 2.0, 3.0, 4.0], (1, 1)))
  with pytest.raises(ValueError, match="ERROR in 'evaluate div'"):
    operations.evaluate("f div", a)


@needs_gkeyll
def test_modal_data_supported_ops_use_weak_kernels():
  """+ - * / and integer pow/sq have an exact DG meaning, so they run on raw
  modal coefficients via Gkeyll's own weak kernels -- the result stays
  native (gkyl-backed), never silently dropping to plain NumPy math."""
  d = pg.load(F1)
  assert d.backend == "gkyl"
  for chain in ("f sq", "f 2 *", "1 f /", "f0 f0 +"):
    result = operations.evaluate(chain, d)
    assert result.backend == "gkyl"


@needs_gkeyll
@pytest.mark.parametrize("chain", ["f sqrt", "f 0.5 pow"])
def test_modal_sqrt_preserves_constant_field_normalization(chain):
  d = pg.load(os.path.join(DATA, "generated", "fsimple.gkyl"))
  result = operations.evaluate(chain, d)
  assert result.backend == "gkyl"
  assert result.ctx["value_form"] == "modal"
  np.testing.assert_allclose(result.values, [[2 * np.sqrt(2), 0]], atol=1e-14)
  np.testing.assert_allclose(result.interpolate().values, 2.0, atol=1e-14)


@needs_gkeyll
@pytest.mark.parametrize("chain, exponent", [
    ("f sqrt", 0.5),
    ("f 0.5 pow", 0.5),
    ("f -0.5 pow", -0.5),
    ("f 0 pow", 0),
    ("f -1 pow", -1),
    ("f sq", 2),
    ("f 3 pow", 3),
])
def test_modal_evaluate_powers_match_arithmetic_on_varying_fields(
    chain, exponent):
  # Two positive fields with nonzero slopes exercise projection of every field.
  coeffs = np.array([[4 * np.sqrt(2), 0.2, 9 * np.sqrt(2), -0.3]])
  d = pg.GData(ctx=dict(
      cells=[1], basis_type="serendipity", poly_order=1, value_form="modal"))
  d.push([np.array([-1., 1.])], gpython.GkylArray.from_numpy(coeffs))
  result = operations.evaluate(chain, d)
  assert result.backend == "gkyl"
  assert result.ctx["value_form"] == "modal"
  np.testing.assert_allclose(result.values, (d**exponent).values, atol=1e-14)


@needs_gkeyll
@pytest.mark.parametrize("chain",
                         ["f sin", "f mean", "f0 f0 max2", "f 0 2 scale_comp"])
def test_modal_data_unsupported_ops_raise(chain):
  d = pg.load(F1)
  with pytest.raises(ValueError, match="not defined for modal"):
    operations.evaluate(chain, d)


@needs_gkeyll
@pytest.mark.parametrize("chain", ["f0 f1 * 2 /", "f1 f0 * 2 /"])
def test_modal_evaluate_preserves_phase_layout_in_chained_arithmetic(chain):
  conf = pg.load(os.path.join(DATA, "generated", "fsimple.gkyl"))
  phase = pg.load(os.path.join(DATA, "generated", "fsimple_hyb.gkyl"))
  result = operations.evaluate(chain, conf, phase)
  assert result.backend == "gkyl"
  assert result.ctx["value_form"] == "modal"
  assert result.ctx["basis_type"] == "hybrid"
  np.testing.assert_array_equal(result.num_cells, [1, 1])
  np.testing.assert_allclose(result.values, (2 * phase).values, atol=1e-14)


def test_available_operator_vocabulary_is_sorted_and_public():
  operators = pg.available_evaluate_operators()
  assert operators == sorted(operators)
  assert {"+", "pow", "sqrt", "int"} <= set(operators)


@needs_gkeyll
def test_modal_dataset_binary_operators_cover_each_weak_kernel_dispatch():
  d = _native_field(2.0, "modal")
  for token in ("-", "*", "/"):
    result = operations.evaluate(f"f0 f0 {token}", d, d)
    assert result.backend == "gkyl"

  with pytest.raises(ValueError, match="between two modal datasets"):
    operations.evaluate("f0 f0 pow", d, d)


@needs_gkeyll
def test_modal_dataset_scalar_operators_cover_operand_order_and_power():
  d = _native_field(2.0, "modal")
  for chain in ("f 2 +", "f 2 -", "2 f -", "f 2 pow"):
    result = operations.evaluate(chain, d)
    assert result.backend == "gkyl"

  with pytest.raises(ValueError, match="not defined for modal"):
    operations.evaluate("2 f pow", d)


@needs_gkeyll
def test_modal_dispatch_rejects_missing_or_mismatched_metadata():
  d = _native_field(2.0, "modal")
  missing = d.clone()
  missing.ctx.pop("basis_type")
  with pytest.raises(ValueError, match="no basis_type/poly_order"):
    operations.evaluate("f sq", missing)

  mismatched = d.clone()
  mismatched.ctx["basis_type"] = "tensor"
  with pytest.raises(ValueError, match="different DG bases"):
    operations.evaluate("f0 f1 +", d, mismatched)

  with pytest.raises(ValueError, match="plain array"):
    operations.evaluate("f [1] +", d)


def _native_field(value, value_form):
  grid = [np.linspace(0.0, 1.0, 5)]
  native = gpython.GkylArray.from_numpy(np.full((4, 1), value))
  return _make(grid,
               native,
               basis_type="serendipity",
               poly_order=0,
               cells=[4],
               num_quad=1,
               quad_rule="gauss",
               value_form=value_form)


@needs_gkeyll
@pytest.mark.parametrize("value_form", ["nodal", "quad"])
def test_native_pointwise_evaluation_stays_native(value_form):
  d = _native_field(4.0, value_form)
  result = operations.evaluate("f sqrt", d)
  assert result.backend == "gkyl"
  assert result.ctx["value_form"] == value_form
  np.testing.assert_allclose(result.values, 2.0)


@needs_gkeyll
def test_native_point_reduction_leaves_the_value_form_domain():
  d = _native_field(4.0, "nodal")
  result = operations.evaluate("f mean", d)
  assert result.backend == "numpy"
  assert result.ctx.get("value_form") is None
  assert result.ctx["interpolated"] is True


@needs_gkeyll
@pytest.mark.parametrize("value_form", ["modal", "quad"])
def test_mixed_native_value_forms_raise(value_form):
  nodal = _native_field(2.0, "nodal")
  other = _native_field(3.0, value_form)
  with pytest.raises(ValueError, match="different value_forms"):
    operations.evaluate("f0 f1 +", nodal, other)


@needs_gkeyll
def test_modal_dispatch_scalar_helpers_cover_scalar_shapes():
  from importlib import import_module
  evaluate_module = import_module("postgkyl.operations.evaluate")
  assert evaluate_module._as_scalar(np.int64(3)) == 3.0
  assert evaluate_module._as_scalar(np.array([3.0])) is None
  assert evaluate_module._modal_kernel("+", [None], [np.array([1.0])],
                                       [{}]) is None


@pytest.mark.parametrize(
    "expression", ["1 grad", "1 0 int", "1 0 avg", "1 0 len", "1 0 grad2"])
def test_physical_operators_reject_scalar_operands(expression):
  with pytest.raises(ValueError, match="requires a dataset with a grid"):
    operations.evaluate(expression, _field(1.0))


def test_gradient_selector_requires_at_least_one_axis():
  with pytest.raises(ValueError, match="grad2 needs at least one axis"):
    operations.evaluate("f 0:0 grad2", _field(1.0))


def test_domain_length_uses_selected_axis_endpoints():
  data = _make([np.array([-2., -1., 3.]),
                np.array([4., 5., 7., 9.])], np.zeros((2, 3, 1)))
  result = operations.evaluate("f 1 len", data)
  assert result.grid == []
  np.testing.assert_array_equal(result.values, [5.])


def test_average_rejects_nonpositive_domain_volume():
  data = _make([np.array([2., 1., 0.])], np.ones((2, 1)))
  with pytest.raises(ValueError, match="positive domain volume"):
    operations.evaluate("f 0 avg", data)


@pytest.mark.parametrize("expression, message", [
    ("f 0 len", "separable coordinate axis"),
    ("f 0,1 avg", "separable coordinates"),
])
def test_domain_queries_reject_curvilinear_coordinates(expression, message):
  # An ordinary Cartesian mesh stored as joint coordinates still has no
  # single separable coordinate array for len/avg.
  grid = list(np.meshgrid(np.arange(4.), np.arange(5.), indexing="ij"))
  data = _make(grid, np.ones((3, 4, 1)), mapped_axes={0: 0, 1: 0})
  with pytest.raises(ValueError, match=message):
    operations.evaluate(expression, data)
