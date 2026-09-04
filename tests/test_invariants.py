"""Cross-cutting laws that should hold across data shapes and backends.

These deterministic, parametrized checks complement example-based unit tests:
they exercise algebraic, integration, representation, and state-copy contracts
over a small matrix of inputs without adding a property-testing dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython

GENERATED = Path(__file__).parent / "test_data" / "generated"
needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


def _field(cells: tuple[int, ...], num_comps: int = 2) -> pg.GData:
  """Build deterministic cell-average data on a nonuniform tensor grid."""
  grid = []
  for d, n in enumerate(cells):
    lower, upper = -0.2 * d, 1.0 + 0.3 * d
    fraction = np.linspace(0.0, 1.0, n + 1)**(d + 1)
    grid.append(lower + (upper - lower) * fraction)
  shape = (*cells, num_comps)
  values = np.arange(np.prod(shape), dtype=float).reshape(shape) / 7.0 + 0.25
  return pg.GData().push(grid, values)


@pytest.mark.parametrize("cells", [(7, ), (3, 5), (2, 3, 4)])
def test_numpy_arithmetic_obeys_affine_law_without_mutating_inputs(cells):
  left = _field(cells)
  right = _field(cells)
  right.values[...] = np.flip(right.values, axis=0) + 0.5
  left_before = left.values.copy()
  right_before = right.values.copy()

  scale = -1.75
  distributed = scale * (left + right)
  expanded = scale * left + scale * right

  np.testing.assert_allclose(distributed.values, expanded.values)
  np.testing.assert_array_equal(left.values, left_before)
  np.testing.assert_array_equal(right.values, right_before)
  assert distributed.backend == "numpy"
  assert distributed.num_cells.tolist() == list(cells)


@pytest.mark.parametrize("ufunc", [np.negative, np.absolute, np.square, np.exp])
def test_numpy_ufunc_result_matches_array_and_is_independent(ufunc):
  source = _field((4, 3), num_comps=1)
  before = source.values.copy()

  result = ufunc(source)

  np.testing.assert_allclose(result.values, ufunc(before))
  result.values.flat[0] += 100.0
  np.testing.assert_array_equal(source.values, before)


@pytest.mark.parametrize(("cells", "lower", "upper", "constants"), [
    ((7, ), (-2.0, ), (3.0, ), (1.25, -4.0)),
    ((3, 5), (-1.0, 2.0), (2.0, 6.0), (0.5, 3.0)),
    ((2, 3, 4), (0.0, -2.0, 1.0), (5.0, 1.0, 2.5), (2.0, -0.25)),
])
def test_integral_of_constant_is_constant_times_physical_volume(
    cells, lower, upper, constants):
  # Uneven edges ensure the result comes from cell widths, not cell count.
  grid = []
  for n, lo, hi in zip(cells, lower, upper):
    fraction = np.linspace(0.0, 1.0, n + 1)**1.7
    grid.append(lo + (hi - lo) * fraction)
  values = np.empty((*cells, len(constants)))
  values[...] = constants
  data = pg.GData().push(grid, values)

  expected = np.asarray(constants) * np.prod(np.asarray(upper) - lower)
  np.testing.assert_allclose(data.integrate(), expected)


def test_partial_integrals_commute_and_preserve_surviving_grid():
  data = _field((3, 4, 5))

  together = data.integrate(axis=(0, 2))
  first_then_last = data.integrate(axis=2).integrate(axis=0)

  np.testing.assert_allclose(together.values, first_then_last.values)
  np.testing.assert_array_equal(together.grid[0], data.grid[1])
  assert together.num_cells.tolist() == [4]
  assert data.num_cells.tolist() == [3, 4, 5]


@pytest.mark.parametrize("cells", [(6, ), (3, 4)])
def test_clone_is_a_by_value_state_copy(cells):
  original = _field(cells)
  original.ctx["provenance"] = "original"
  clone = original.clone()

  assert clone is not original
  assert clone.ctx is not original.ctx
  assert clone.ctx.keys() == original.ctx.keys()
  np.testing.assert_array_equal(clone.num_cells, original.num_cells)
  assert clone.values is not original.values
  assert all(a is not b for a, b in zip(clone.grid, original.grid))

  clone.values.flat[0] = -999.0
  clone.grid[0][0] = -999.0
  clone.ctx["provenance"] = "clone"
  assert original.values.flat[0] != -999.0
  assert original.grid[0][0] != -999.0
  assert original.ctx["provenance"] == "original"


@needs_gkeyll
@pytest.mark.parametrize(("filename", "num_quad"), [
    ("1d_ms_p1.gkyl", 2),
    ("1d_ms_p2.gkyl", 3),
    ("2d_ms_p1.gkyl", 2),
    ("2d_ms_p2.gkyl", 3),
])
@pytest.mark.parametrize("value_form", ["nodal", "quad"])
def test_native_representation_roundtrip_preserves_coefficients_and_source(
    filename, num_quad, value_form):
  source = pg.load(GENERATED / filename)
  before = source.values.copy()

  if value_form == "nodal":
    represented = source.to_nodal()
  else:
    represented = source.to_quad(num_quad=num_quad)
  restored = represented.to_modal()

  assert represented.ctx["value_form"] == value_form
  assert represented.backend == restored.backend == "gkyl"
  np.testing.assert_allclose(restored.values, before, rtol=2e-13, atol=2e-13)
  np.testing.assert_array_equal(source.values, before)


@needs_gkeyll
@pytest.mark.parametrize("value_form", ["modal", "nodal", "quad"])
def test_native_linear_arithmetic_commutes_with_representation(value_form):
  source = pg.load(GENERATED / "2d_ms_p1.gkyl")
  represented = {
      "modal": source,
      "nodal": source.to_nodal(),
      "quad": source.to_quad(),
  }[value_form]

  transformed = 2.5 * represented - represented + 0.75 * represented
  restored = transformed if value_form == "modal" else transformed.to_modal()
  expected = 2.25 * source

  np.testing.assert_allclose(restored.values,
                             expected.values,
                             rtol=2e-13,
                             atol=2e-13)
