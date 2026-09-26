"""Dataset arithmetic must combine values at the same physical locations."""

import operator

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, operations

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

ADD = [operator.add, np.add, lambda a, b: operations.evaluate("f0 f1 +", a, b)]
ADD_IDS = ["operator", "ufunc", "rpn"]
BACKENDS = ["numpy", pytest.param("gkyl", marks=needs_gkeyll)]


def _field(backend="numpy", value_form=None):
  # Two fields on three unequal cells; p0 nodal/quad values have one node/cell.
  values = np.array([[1., 2.], [3., 4.], [5., 6.]])
  ctx = {}
  if value_form is not None:
    ctx.update(basis_type="serendipity",
               poly_order=0,
               value_form=value_form,
               cells=[3])
    if value_form == "quad":
      ctx.update(quad_rule="gauss", num_quad=1)
  if backend == "gkyl":
    values = gpython.GkylArray.from_numpy(values)
  return pg.GData(ctx=ctx).push([np.array([0., 1., 3., 6.])], values)


@pytest.mark.parametrize("add", ADD, ids=ADD_IDS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("value_form", ["nodal", "quad"])
@pytest.mark.parametrize("mismatch", ["disjoint", "interior"])
def test_point_arithmetic_rejects_different_grids(add, backend, value_form,
                                                  mismatch):
  a, b = _field(backend, value_form), _field(backend, value_form)
  if mismatch == "disjoint":
    b.grid = [b.grid[0] + 100.]
  else:
    b.grid[0][1] = 0.5  # Same bounds and shape, different cell locations.
  for left, right in ((a, b), (b, a)):
    with pytest.raises(ValueError, match="different grids"):
      add(left, right)


@pytest.mark.parametrize("add", ADD, ids=ADD_IDS)
def test_plain_numpy_fields_reject_different_grids(add):
  a, b = _field(), _field()
  b.grid = [b.grid[0] + 100.]
  with pytest.raises(ValueError, match="different grids"):
    add(a, b)


@pytest.mark.parametrize("add", ADD, ids=ADD_IDS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("value_form", ["nodal", "quad"])
def test_compatible_point_arithmetic_preserves_values_and_layout(
    add, backend, value_form):
  a, b = _field(backend, value_form), _field(backend, value_form)
  result = add(a, b)
  assert isinstance(result, pg.GData)
  assert result.backend == backend
  assert result.ctx["value_form"] == value_form
  assert result.ctx["basis_type"] == "serendipity"
  np.testing.assert_array_equal(result.grid[0], [0., 1., 3., 6.])
  np.testing.assert_array_equal(result.values, [[2., 4.], [6., 8.], [10., 12.]])
  for source in (a, b):
    np.testing.assert_array_equal(source.values, [[1., 2.], [3., 4.], [5., 6.]])


@pytest.mark.parametrize("add", ADD, ids=ADD_IDS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("key, value, message", [
    ("basis_type", "tensor", "different DG bases"),
    ("poly_order", 1, "different DG bases"),
    ("value_form", "quad", "different value_forms"),
])
def test_same_shape_does_not_override_dg_metadata(add, backend, key, value,
                                                  message):
  a, b = _field(backend, "nodal"), _field(backend, "nodal")
  b.ctx[key] = value
  if key == "value_form" and value == "quad":
    b.ctx.update(num_quad=1, quad_rule="gauss")
  with pytest.raises(ValueError, match=message):
    add(a, b)


@pytest.mark.parametrize("add", ADD, ids=ADD_IDS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_same_shape_cannot_claim_an_unavailable_basis_quadrature(add, backend):
  a, b = _field(backend, "quad"), _field(backend, "quad")
  b.ctx["quad_rule"] = "gkeyll"
  # The p0 serendipity basis has no native quadrature table. A stored count
  # cannot make that rule valid merely because the array shapes match.
  with pytest.raises(ValueError, match="complete field blocks|quadrature"):
    add(a, b)


@needs_gkeyll
@pytest.mark.parametrize("add", ADD, ids=ADD_IDS)
def test_point_arithmetic_rejects_mixed_backends(add):
  a, b = _field("gkyl", "nodal"), _field("numpy", "nodal")
  for left, right in ((a, b), (b, a)):
    with pytest.raises(ValueError, match="different backends"):
      add(left, right)


@pytest.mark.parametrize("add", ADD, ids=ADD_IDS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_dataset_shapes_are_not_broadcast(add, backend):
  a, b = _field(backend, "nodal"), _field(backend, "nodal").select(comp=0)
  with pytest.raises(ValueError, match="incompatible shapes"):
    add(a, b)


@pytest.mark.parametrize("token", ["-", "*", "/", "pow", "min2", "max2", "dot"])
def test_rpn_binary_functions_check_the_operand_grids(token):
  a, b = _field(), _field()
  b.grid = [b.grid[0] + 100.]
  with pytest.raises(ValueError, match="different grids"):
    operations.evaluate(f"f0 f1 {token}", a, b)


def test_rpn_checks_intermediate_results_and_selected_components():
  a, b = _field(), _field()
  b.grid = [b.grid[0] + 100.]
  with pytest.raises(ValueError, match="different grids"):
    operations.evaluate("f0[0] 2 * f1[0] +", a, b)


def test_rpn_reduced_scalar_can_combine_with_a_field_on_another_grid():
  a, b = _field(), _field()
  b.grid = [b.grid[0] + 100.]
  result = operations.evaluate("f0 f1 max +", a, b)
  np.testing.assert_array_equal(result.grid[0], [0., 1., 3., 6.])
  np.testing.assert_array_equal(result.values,
                                [[7., 8.], [9., 10.], [11., 12.]])


@pytest.mark.parametrize("add", ADD, ids=ADD_IDS)
def test_interpolated_fields_do_not_require_their_original_bases_to_match(add):
  a, b = _field("numpy", "modal"), _field("numpy", "modal")
  a.ctx["interpolated"] = b.ctx["interpolated"] = True
  b.ctx.update(basis_type="tensor", poly_order=2, value_form="nodal")
  result = add(a, b)
  np.testing.assert_array_equal(result.values, [[2., 4.], [6., 8.], [10., 12.]])
