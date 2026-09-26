"""Exact DG laws on independently generated polynomial files.

References use physical polynomials and their antiderivatives, never a
Postgkyl kernel or the coefficients returned by the operation under test.
The small, exactly representable fields allow roundoff-level tolerances.
"""

from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.polynomial import Polynomial
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.gdatastate import materialize_point_values

from generate_test_data import (POLYNOMIAL_CASES, polynomial_factors,
                                polynomial_grid)

pytestmark = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")
GEN = Path(__file__).parent / "test_data" / "generated"
ROUND_OFF = dict(rtol=3e-12, atol=3e-12)
SERENDIPITY_CASES = [c for c in POLYNOMIAL_CASES if c[2] == "serendipity"]


def _load_case(case):
  stem, ndim, basis, order = case
  data = pg.load(GEN / f"{stem}.gkyl")
  factors = [[Polynomial(c) for c in field]
             for field in polynomial_factors(ndim, basis, order)]
  return data, factors, polynomial_grid(ndim)


@pytest.fixture(params=POLYNOMIAL_CASES, ids=lambda case: case[0])
def polynomial_case(request):
  return _load_case(request.param)


@pytest.fixture(params=SERENDIPITY_CASES, ids=lambda case: case[0])
def integrable_case(request):
  # Full/partial native integration explicitly supports serendipity p1-p2.
  return _load_case(request.param)


def _centers(grid):
  return [(e[:-1] + e[1:]) / 2 for e in grid]


def _values(factors, axes):
  coords = np.meshgrid(*axes, indexing="ij")
  return np.stack([
      np.prod([factor(x) for factor, x in zip(field, coords)], axis=0)
      for field in factors
  ],
                  axis=-1)


def _integral(factor, edges):
  primitive = factor.integ()
  return primitive(edges[-1]) - primitive(edges[0])


@pytest.mark.parametrize("num_interp", [1, 3, 5])
def test_interpolation_recovers_every_field_and_cell(polynomial_case,
                                                     num_interp):
  data, factors, grid = polynomial_case
  before = data.values.copy()
  result = data.interpolate(num_interp=num_interp)
  expected_grid = [
      np.linspace(e[0], e[-1], (len(e) - 1) * num_interp + 1) for e in grid
  ]
  for actual, expected in zip(result.grid, expected_grid):
    np.testing.assert_allclose(actual, expected, **ROUND_OFF)
  np.testing.assert_allclose(result.values,
                             _values(factors, _centers(expected_grid)),
                             **ROUND_OFF)
  np.testing.assert_array_equal(data.values, before)


@pytest.mark.parametrize("num_quad", [3, 4])
def test_quadrature_values_use_physical_gauss_locations(polynomial_case,
                                                        num_quad):
  data, factors, grid = polynomial_case
  quad = data.represent(to="quad", num_quad=num_quad)
  samples = materialize_point_values(quad)
  # NumPy's Legendre roots are independent of Gkeyll's basis/quad tables.
  nodes, _ = np.polynomial.legendre.leggauss(num_quad)
  axes = [(c[:, None] + np.diff(e)[:, None] * nodes / 2).ravel()
          for c, e in zip(_centers(grid), grid)]
  for actual, expected in zip(samples.grid, axes):
    np.testing.assert_allclose(actual, expected, **ROUND_OFF)
  np.testing.assert_allclose(samples.values, _values(factors, axes),
                             **ROUND_OFF)


@pytest.mark.parametrize("representation", ["nodal", "quad"])
def test_conversion_preserves_the_polynomial_not_just_a_round_trip(
    polynomial_case, representation):
  data, factors, grid = polynomial_case
  converted = data.represent(to=representation)
  if representation == "nodal":
    # Gkeyll's p1/p2 interpolation nodes run x-fast over [-1,1]. The 2D
    # serendipity p2 element keeps the boundary nodes, omitting the center.
    nodes = np.array(
        list(
            product(np.linspace(-1., 1., data.ctx["poly_order"] + 1),
                    repeat=data.num_dims)))[:, ::-1]
    if data.num_dims == 2 and data.ctx["basis_type"] == "serendipity":
      nodes = nodes[np.any(nodes != 0, axis=1)]
    centers = np.stack(np.meshgrid(*_centers(grid), indexing="ij"), axis=-1)
    widths = np.array([e[1] - e[0] for e in grid])
    coords = centers[..., None, :] + nodes * widths / 2
    expected = np.stack([
        np.prod([p(coords[..., d]) for d, p in enumerate(field)], axis=0)
        for field in factors
    ],
                        axis=-2)
    np.testing.assert_allclose(converted.values,
                               expected.reshape(*data.num_cells, -1),
                               **ROUND_OFF)
  restored = converted.represent(to="modal")
  points = restored.interpolate(num_interp=4)
  np.testing.assert_allclose(points.values,
                             _values(factors, _centers(points.grid)),
                             **ROUND_OFF)


@pytest.mark.parametrize("point_values", [False, True],
                         ids=["modal", "sampled"])
def test_gradient_includes_all_directions_and_both_fields(
    polynomial_case, point_values):
  data, factors, _ = polynomial_case
  if point_values:
    data = data.interpolate(num_interp=4)
  result = data.differentiate()
  if not point_values:
    assert result.backend == "gkyl"
    assert result.ctx["value_form"] == "modal"
    result = result.interpolate(num_interp=4)
  # The finite-difference path is also exact for these degree <= 2 factors,
  # including its one-sided quadratic boundary stencil.
  axes = _centers(result.grid)
  expected = []
  for direction in range(data.num_dims):
    differentiated = [[
        p.deriv() if d == direction else p for d, p in enumerate(field)
    ] for field in factors]
    expected.append(_values(differentiated, axes))
  np.testing.assert_allclose(result.values, np.concatenate(expected, axis=-1),
                             **ROUND_OFF)


@pytest.mark.parametrize("op", ["none", "sq"])
def test_full_integral_matches_antiderivatives(integrable_case, op):
  data, factors, grid = integrable_case
  expected = [
      np.prod([
          _integral(p * p if op == "sq" else p, e) for p, e in zip(field, grid)
      ]) for field in factors
  ]
  np.testing.assert_allclose(data.integrate(op=op), expected, **ROUND_OFF)


@pytest.mark.parametrize("reduction", ["integrate", "average"])
@pytest.mark.parametrize("case,removed", [
    pytest.param(case, d, id=f"{case[0]}-axis{d}")
    for case in SERENDIPITY_CASES if case[1] > 1 for d in range(case[1])
])
def test_partial_reduction_retains_analytic_dependence(case, removed,
                                                       reduction):
  data, factors, grid = _load_case(case)
  result = (data.integrate(
      axis=removed) if reduction == "integrate" else data.average([removed]))
  assert result.backend == "gkyl"
  assert result.ctx["value_form"] == "modal"
  kept = [d for d in range(data.num_dims) if d != removed]
  assert result.num_dims == len(kept)
  for edges, d in zip(result.grid, kept):
    np.testing.assert_array_equal(edges, grid[d])
  reduced_factors = [[field[d] for d in kept] for field in factors]
  scale = np.array(
      [_integral(field[removed], grid[removed]) for field in factors])
  if reduction == "average":
    scale /= grid[removed][-1] - grid[removed][0]
  points = result.interpolate(num_interp=3)
  np.testing.assert_allclose(
      points.values,
      _values(reduced_factors, _centers(points.grid)) * scale, **ROUND_OFF)


def test_three_dimensional_reduction_over_nonadjacent_axes():
  data = pg.load(GEN / "polynomial_3d_ms_p1.gkyl")
  # f=(2+x)(2+2y)(2+3z), g=(4-x/4)(4-y/2)(4-3z/4).
  # Integrals in x and z are respectively (7.5,44) and (11.625,7).
  result = data.integrate(axis=(0, 2)).interpolate(num_interp=3)
  y, = _centers(result.grid)
  expected = np.stack([330 * (2 + 2 * y), 81.375 * (4 - y / 2)], axis=-1)
  np.testing.assert_allclose(result.values, expected, **ROUND_OFF)


@pytest.mark.parametrize("case,removed", [
    pytest.param(case, d, id=f"{case[0]}-axis{d}")
    for case in SERENDIPITY_CASES if case[1] > 1 for d in range(case[1])
])
def test_weighted_average_preserves_analytic_dependence(case, removed):
  data, factors, grid = _load_case(case)
  result = data.average([removed],
                        weight=data.select(comp=1)).interpolate(num_interp=3)
  kept = [d for d in range(data.num_dims) if d != removed]
  weight = factors[1][removed]
  scale = [
      _integral(field[removed] * weight, grid[removed]) /
      _integral(weight, grid[removed]) for field in factors
  ]
  expected = _values([[field[d] for d in kept]
                      for field in factors], _centers(result.grid)) * scale
  # Separability cancels the surviving weight in the projected weak solve;
  # the retained answer is again exactly in the donor's polynomial space.
  np.testing.assert_allclose(result.values, expected, **ROUND_OFF)


def _coordinate_projection(data, directions, coordinates, tmp_path):
  output = tmp_path / "projection.npz"
  # Regression protection: an output-stride bug used to corrupt native memory.
  # Keep crashes isolated so other analytic checks can still run.
  process = subprocess.run([
      sys.executable, "-c", """
import json
import sys
import numpy as np
import postgkyl as pg
data = pg.load(sys.argv[1])
result = data.eval_at_coord_proj(json.loads(sys.argv[2]), json.loads(sys.argv[3]))
assert result.backend == "gkyl"
assert result.ctx["value_form"] == "modal"
points = result.interpolate(num_interp=3)
np.savez(sys.argv[4], values=points.values,
         coefficients=result.values, poly_order=result.ctx["poly_order"],
         **{f"grid{d}": e for d, e in enumerate(points.grid)})
""", data.file_name,
      json.dumps(directions),
      json.dumps(coordinates),
      str(output)
  ],
                           capture_output=True,
                           text=True,
                           timeout=30)
  assert process.returncode == 0, (
      f"Coordinate projection exited {process.returncode}:\n{process.stderr}")
  return output


@pytest.mark.slow
@pytest.mark.parametrize("fraction", [0., 0.37, 1.])
@pytest.mark.parametrize("case,direction", [
    pytest.param(case, d, id=f"{case[0]}-axis{d}")
    for case in POLYNOMIAL_CASES if case[1] > 1 for d in range(case[1])
])
def test_coordinate_projection_evaluates_physical_polynomial(
    case, direction, fraction, tmp_path):
  data, factors, grid = _load_case(case)
  e = grid[direction]
  coord = e[0] + fraction * (e[-1] - e[0])
  output = _coordinate_projection(data, [direction], [coord], tmp_path)
  with np.load(output) as result:
    kept = [d for d in range(data.num_dims) if d != direction]
    axes = []
    for i, d in enumerate(kept):
      expected_grid = np.linspace(grid[d][0], grid[d][-1],
                                  3 * (len(grid[d]) - 1) + 1)
      np.testing.assert_allclose(result[f"grid{i}"], expected_grid, **ROUND_OFF)
      axes.append((expected_grid[:-1] + expected_grid[1:]) / 2)
    kept_factors = [[field[d] for d in kept] for field in factors]
    scale = [field[direction](coord) for field in factors]
    np.testing.assert_allclose(result["values"],
                               _values(kept_factors, axes) * scale, **ROUND_OFF)


@pytest.mark.slow
@pytest.mark.parametrize("case", POLYNOMIAL_CASES, ids=lambda case: case[0])
def test_full_coordinate_projection_preserves_modal_normalization(
    case, tmp_path):
  data, factors, grid = _load_case(case)
  coordinates = [
      e[0] + (0.2 + 0.2 * d) * (e[-1] - e[0]) for d, e in enumerate(grid)
  ]
  # Reverse the directions and coordinates together to exercise the pairing.
  output = _coordinate_projection(data, list(reversed(range(data.num_dims))),
                                  coordinates[::-1], tmp_path)
  expected = np.array([
      np.prod([p(x) for p, x in zip(field, coordinates)]) for field in factors
  ])
  with np.load(output) as result:
    assert result["poly_order"] == 0
    np.testing.assert_allclose(result["coefficients"],
                               np.sqrt(2) * expected[None, :], **ROUND_OFF)
    np.testing.assert_allclose(result["values"],
                               np.broadcast_to(expected, (3, 2)), **ROUND_OFF)


@pytest.mark.slow
@pytest.mark.parametrize("directions", [[0, 2], [2, 0]])
def test_coordinate_projection_pairs_nonadjacent_directions(
    directions, tmp_path):
  data, factors, _ = _load_case(("polynomial_3d_ms_p1", 3, "serendipity", 1))
  coordinates = {0: -0.4, 2: 4.1}
  output = _coordinate_projection(data, directions,
                                  [coordinates[d] for d in directions],
                                  tmp_path)
  with np.load(output) as result:
    y, = _centers([result["grid0"]])
    expected = np.stack(
        [field[0](-0.4) * field[1](y) * field[2](4.1) for field in factors],
        axis=-1)
    np.testing.assert_allclose(result["values"], expected, **ROUND_OFF)


@pytest.mark.slow
@pytest.mark.parametrize("direction", [1, 2])
@pytest.mark.parametrize("coordinate", [0., 1., 2.])
def test_coordinate_projection_chooses_the_upper_trace_at_internal_faces(
    direction, coordinate, tmp_path):
  data = pg.load(GEN / "gk_drift_3d_p1.gkyl")
  output = _coordinate_projection(data, [direction], [coordinate], tmp_path)
  with np.load(output) as result:
    axes = _centers([result["grid0"], result["grid1"]])
    kept_points = iter(np.meshgrid(*axes, indexing="ij"))
    coords = [
        np.full((6, 6), coordinate) if d == direction else next(kept_points)
        for d in range(3)
    ]
    # Each unit cell has F0=j+sum((d+1)*xi_d), with C-order cell index j.
    # An internal face uses the upper cell; the domain endpoint stays inside.
    indices = [np.minimum(x.astype(int), 1) for x in coords]
    value = 4 * indices[0] + 2 * indices[1] + indices[2]
    for d, (x, i) in enumerate(zip(coords, indices)):
      value = value + (d + 1) * (2 * (x - i) - 1)
    expected = np.empty((6, 6, 5))
    expected[..., 0] = value
    expected[..., 1:] = [2., 3., 4., 5.]
    np.testing.assert_allclose(result["values"], expected, **ROUND_OFF)


def test_scalar_arithmetic_and_field_selection_have_physical_meaning(
    polynomial_case):
  data, factors, _ = polynomial_case
  result = (3. - 2. * data).select(comp="1,0").interpolate(num_interp=3)
  expected = (3. - 2. * _values(factors, _centers(result.grid)))[..., ::-1]
  np.testing.assert_allclose(result.values, expected, **ROUND_OFF)


@pytest.mark.parametrize("operation", ["multiply", "power", "apply"])
def test_p1_square_is_its_l2_projection(operation):
  f = pg.load(GEN / "polynomial_1d_ms_p1.gkyl").select(comp=0)
  result = {
      "multiply": lambda: f * f,
      "power": lambda: f**2,
      "apply": lambda: f.apply(np.square, num_quad=3)
  }[operation]()
  # f=2+x=A+h*xi, h=1/2. Project xi^2 onto p1: its mean is 1/3.
  c = _centers(polynomial_grid(1))[0]
  mean = (2 + c)**2 + 1 / 12
  slope = 2 + c
  expected = np.stack([np.sqrt(2) * mean, np.sqrt(2 / 3) * slope], axis=-1)
  np.testing.assert_allclose(result.values, expected, **ROUND_OFF)


@pytest.mark.parametrize("inverse", [False, True])
def test_weak_division_solves_the_projected_equations(inverse):
  data = pg.load(GEN / "polynomial_1d_ms_p1.gkyl")
  f, g = data.select(comp=0), data.select(comp=1)
  result = 1. / g if inverse else f / g
  c = _centers(polynomial_grid(1))[0]
  # For g=a+b*xi and f=A+B*xi, require <g*u-f,1>=<g*u-f,xi>=0.
  # This gives a*u0+b*u1/3=A and b*u0+a*u1=B, not u=f(x)/g(x).
  a, b = 4 - c / 4, -1 / 8
  A, B = (np.ones_like(c), 0.) if inverse else (2 + c, 1 / 2)
  denominator = a * a - b * b / 3
  mean = (a * A - b * B / 3) / denominator
  slope = (a * B - b * A) / denominator
  expected = np.stack([np.sqrt(2) * mean, np.sqrt(2 / 3) * slope], axis=-1)
  np.testing.assert_allclose(result.values, expected, **ROUND_OFF)


def test_local_derivative_ignores_jumps_between_cells():
  data = pg.load(GEN / "gk_moments_p1.gkyl")
  result = data.differentiate().interpolate(num_interp=3)
  # xi=2*(x-i)-1; derivative is 2*b in each unit-width cell, regardless
  # of the jumps between the two prescribed means in gk_moments_p1.
  slopes = np.array([[0.7, -0.4, 1.2, 0.5], [-0.6, 0.3, -0.8, 1.1]])
  np.testing.assert_allclose(result.values, np.repeat(2 * slopes, 3, axis=0),
                             **ROUND_OFF)
