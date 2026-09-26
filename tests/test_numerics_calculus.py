"""Tests for postgkyl.numerics.calculus -- integrate over a nodal grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from postgkyl.numerics import calculus

GEN = Path(__file__).parent / "test_data" / "generated"


class TestIntegrate1D:

  def test_uniform_ones_integrates_to_domain_length(self):
    grid = [np.linspace(0.0, 1.0, 6)]  # 5 cells, dx=0.2
    _, out = calculus.integrate(grid, np.ones((5, 1)), axis=0)
    np.testing.assert_allclose(out.flat[0], 1.0, rtol=1e-12)

  def test_linear_function_exact_integral(self):
    # Midpoint integration is exact for affine functions, even on 8 cells.
    with np.load(GEN / "midpoint_polynomials_8.npz") as data:
      _, out = calculus.integrate([data["edges"]],
                                  data["values"][:, 1:],
                                  axis=0)
    np.testing.assert_allclose(out, [[7.5]], rtol=0, atol=2e-14)

  @pytest.mark.parametrize("n", [8, 16, 32])
  def test_quadratic_midpoint_error_has_exact_size_and_sign(self, n):
    # Integral_{-1}^2 x^2 dx = 3. For uniform midpoint samples,
    # Q - I = -L*h^2/12 = -27/(12*n^2), exactly, since f''=2.
    with np.load(GEN / f"midpoint_polynomials_{n}.npz") as data:
      _, out = calculus.integrate([data["edges"]],
                                  data["values"][:, :1],
                                  axis=0)
    np.testing.assert_allclose(out - 3., [[-27 / (12 * n * n)]],
                               rtol=0,
                               atol=2e-14)

  def test_integer_axis(self):
    grid = [np.linspace(0.0, 2.0, 5)]  # 4 cells, dx=0.5
    _, out = calculus.integrate(grid, np.ones((4, 1)), axis=0)
    np.testing.assert_allclose(out.flat[0], 2.0, rtol=1e-12)

  def test_string_integer_axis(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    _, out = calculus.integrate(grid, np.ones((5, 1)), axis="0")
    np.testing.assert_allclose(out.flat[0], 1.0, rtol=1e-12)

  def test_tuple_axis(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    _, out = calculus.integrate(grid, np.ones((5, 1)), axis=(0, ))
    np.testing.assert_allclose(out.flat[0], 1.0, rtol=1e-12)

  def test_none_axis_integrates_all(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    _, out = calculus.integrate(grid, np.ones((5, 1)), axis=None)
    np.testing.assert_allclose(out.flat[0], 1.0, rtol=1e-12)

  def test_colon_slice_axis_string(self):
    """src_bak's colon-slice branch passed raw strings to ``range()``,
    which raises TypeError immediately -- a latent bug never exercised by
    any caller. Fixed here (cast to int) and proven by this test."""
    grid = [np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 5)]
    _, out = calculus.integrate(grid, np.ones((5, 4, 1)), axis="0:2")
    np.testing.assert_allclose(out.flat[0], 2.0, rtol=1e-12)

  def test_does_not_mutate_input_values(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 1))
    calculus.integrate(grid, values, axis=0)
    np.testing.assert_allclose(values, np.ones((5, 1)))

  def test_wrong_axis_type_raises(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    with pytest.raises(TypeError):
      calculus.integrate(grid, np.ones((5, 1)), axis=3.14)

  def test_output_shape_preserved_with_expand_dims(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    _, out = calculus.integrate(grid, np.ones((5, 2)), axis=0)
    assert out.shape == (1, 2)

  def test_multiple_components(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    values = np.column_stack([np.ones(5), 2.0 * np.ones(5)])
    _, out = calculus.integrate(grid, values, axis=0)
    np.testing.assert_allclose(out[0, 0], 1.0, rtol=1e-12)
    np.testing.assert_allclose(out[0, 1], 2.0, rtol=1e-12)


class TestIntegrate2D:

  def test_ones_integrates_to_area(self):
    grid = [np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 5)]  # 5x4 cells
    _, out = calculus.integrate(grid, np.ones((5, 4, 1)), axis=None)
    np.testing.assert_allclose(out.flat[0], 2.0, rtol=1e-12)

  def test_integrate_axis0_only(self):
    grid = [np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 4)]  # 5x3
    _, out = calculus.integrate(grid, np.ones((5, 3, 1)), axis=0)
    assert out.shape == (1, 3, 1)
    np.testing.assert_allclose(out[:, :, 0], 1.0, rtol=1e-12)

  def test_integrate_axis1_only(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 2.0, 5)]  # 3x4
    _, out = calculus.integrate(grid, np.ones((3, 4, 1)), axis=1)
    assert out.shape == (3, 1, 1)
    np.testing.assert_allclose(out[:, :, 0], 2.0, rtol=1e-12)

  def test_comma_separated_string_axes(self):
    grid = [np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 5)]
    _, out = calculus.integrate(grid, np.ones((5, 4, 1)), axis="0,1")
    np.testing.assert_allclose(out.flat[0], 2.0, rtol=1e-12)

  def test_nonuniform_grid(self):
    x = np.array([0.0, 0.1, 0.4, 1.0])
    _, out = calculus.integrate([x], np.ones((3, 1)), axis=0)
    np.testing.assert_allclose(out.flat[0], 1.0, rtol=1e-12)


class TestIntegrateCellCentered:

  def test_cell_centered_grid(self):
    # Point coordinates specify only [0.1,0.9], not extrapolated cell edges.
    x_cc = np.linspace(0.1, 0.9, 5)
    _, out = calculus.integrate([x_cc], np.ones((5, 1)), axis=0)
    np.testing.assert_allclose(out.flat[0], 0.8, rtol=0, atol=2e-15)

  def test_single_point_has_zero_measure(self):
    grid = [np.array([0.5]), np.linspace(0.0, 1.0, 4)]
    _, out = calculus.integrate(grid, np.ones((1, 3, 1)), axis=0)
    assert out.shape[0] == 1
    np.testing.assert_array_equal(out, np.zeros((1, 3, 1)))


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_nonuniform_cell_averages_obey_analytic_integrals(axis):
  with np.load(GEN / "nonuniform_polynomials.npz") as data:
    x, y, values = data["x"], data["y"], data["values"]
  _, result = calculus.integrate([x, y], values, axis=axis)
  # f=2+3x-4y+5xy; g=(1+x^2)(2-y), on [-1,2] x [1/4,9/4].
  if axis is None:
    expected = np.array([[[9.75, 9.]]])
  elif axis == 0:
    c = (y[:-1] + y[1:]) / 2
    expected = np.stack([10.5 - 4.5 * c, 6 * (2 - c)], axis=-1)[None, ...]
  else:
    c = (x[:-1] + x[1:]) / 2
    mean_x2 = (x[1:]**3 - x[:-1]**3) / (3 * np.diff(x))
    expected = np.stack([-6 + 18.5 * c, 1.5 * (1 + mean_x2)], axis=-1)[:,
                                                                       None, :]
  np.testing.assert_allclose(result, expected, rtol=2e-14, atol=2e-14)
