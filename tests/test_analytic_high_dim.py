"""File-backed analytic checks of 4D--6D DG data, including velocity modes."""

from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.gdatastate import materialize_point_values

from generate_test_data import ANALYTIC_CONFIGS, analytic_fields

pytestmark = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")
GEN = Path(__file__).parent / "test_data" / "generated"


@pytest.fixture(params=ANALYTIC_CONFIGS, ids=lambda config: config[0])
def analytic_data(request):
  stem, ndim, order, basis = request.param
  data = pg.load(GEN / f"{stem}.gkyl")
  assert data.num_dims == ndim
  assert data.backend == "gkyl"
  assert data.ctx["basis_type"] == basis
  assert data.ctx["poly_order"] == order
  assert data.ctx["value_form"] == "modal"
  assert data.num_comps == 2 * gpython.basis.num_basis(basis, ndim, order)
  return data


def _mesh(grid):
  return np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1)


def _expected(data, points):
  return analytic_fields(points, data.ctx["basis_type"], data.ctx["poly_order"])


@pytest.mark.parametrize("num_interp", [2, 4])
def test_interpolation_matches_polynomials_in_every_cell(
    analytic_data, num_interp):
  data = analytic_data
  before = data.values.copy()
  result = data.interpolate(num_interp=num_interp)
  assert result.backend == "numpy"
  assert result.values.shape == (*tuple(data.num_cells * num_interp), 2)
  points = _mesh([(edge[:-1] + edge[1:]) / 2 for edge in result.grid])
  np.testing.assert_allclose(result.values,
                             _expected(data, points),
                             rtol=2e-12,
                             atol=2e-12)
  np.testing.assert_array_equal(data.values, before)


@pytest.mark.parametrize("value_form", ["nodal", "quad"])
def test_representation_values_and_inverse_match_analytic_field(
    analytic_data, value_form):
  data = analytic_data
  basis, order = data.ctx["basis_type"], data.ctx["poly_order"]
  missing_transform = basis == "tensor" and data.num_dims == 5 and order == 2
  options = {
      "num_quad": 3
  } if missing_transform and value_form == "quad" else {}
  converted = data.represent(to=value_form, **options)
  assert converted.backend == "gkyl"
  assert converted.ctx["value_form"] == value_form
  if value_form == "quad":
    # Also check the scatter into the physical tensor grid. gkhybrid has
    # three v_parallel quadrature points and two in every other direction.
    points = materialize_point_values(converted)
    np.testing.assert_allclose(points.values,
                               _expected(data, _mesh(points.grid)),
                               rtol=2e-12,
                               atol=2e-12)
  else:
    nodes = gpython.basis.node_coords(basis, data.num_dims, order)
    centers = _mesh([(edge[:-1] + edge[1:]) / 2 for edge in data.grid])
    widths = np.array([edge[1] - edge[0] for edge in data.grid])
    points = centers[..., None, :] + nodes * widths / 2
    expected = np.swapaxes(_expected(data, points), -1, -2)
    np.testing.assert_allclose(converted.values,
                               expected.reshape(*data.num_cells, -1),
                               rtol=2e-12,
                               atol=2e-12)
  if missing_transform and value_form == "nodal":
    with pytest.raises(NotImplementedError,
                       match="no nodal-to-modal transform"):
      converted.represent(to="modal")
    return
  restored = converted.represent(to="modal")
  assert restored.backend == "gkyl"
  assert restored.ctx["value_form"] == "modal"
  np.testing.assert_allclose(restored.values,
                             data.values,
                             rtol=2e-12,
                             atol=2e-12)


def test_interpolated_gradient_matches_analytic_derivatives(analytic_data):
  data = analytic_data
  # Native local derivative kernels do not cover 4D--6D. After explicit
  # interpolation, the numerical gradient is exact for f's quadratic terms.
  result = data.interpolate(num_interp=3).differentiate()
  assert result.backend == "numpy"
  points = _mesh([(edge[:-1] + edge[1:]) / 2 for edge in result.grid])
  ndim = data.num_dims
  expected = np.broadcast_to(np.arange(1, ndim + 1, dtype=float),
                             points.shape).copy()
  expected[..., 0] += points[..., -1]
  expected[..., -1] += points[..., 0]
  if data.ctx["poly_order"] == 2 or data.ctx["basis_type"] == "gkhybrid":
    expected[..., ndim - 2] += points[..., ndim - 2]
  # Gradient fields are direction-major: [d0_f, d0_g, d1_f, d1_g, ...].
  np.testing.assert_allclose(result.values[..., ::2],
                             expected,
                             rtol=2e-11,
                             atol=2e-11)


def test_modal_linear_arithmetic_preserves_both_fields(analytic_data):
  data = analytic_data
  result = 2.5 * data - data + 0.75
  assert result.backend == "gkyl"
  assert result.ctx["value_form"] == "modal"
  interpolated = result.interpolate(num_interp=3)
  points = _mesh([(edge[:-1] + edge[1:]) / 2 for edge in interpolated.grid])
  np.testing.assert_allclose(interpolated.values,
                             1.5 * _expected(data, points) + 0.75,
                             rtol=2e-12,
                             atol=2e-12)
