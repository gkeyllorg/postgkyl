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
def test_modal_data_unsupported_op_warns_and_falls_back():
  """sqrt has no weak-kernel form: rather than hard-blocking (basis/
  value_form metadata can be wrong), evaluate warns and computes on the
  raw coefficient view, which is exact only when coefficient 0 already IS
  the point value."""
  d = pg.load(F1)
  with pytest.warns(UserWarning, match="weak-kernel"):
    result = operations.evaluate("f sqrt", d)
  assert result.backend == "numpy"
