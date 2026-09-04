"""Tests for postgkyl.numerics.calculus -- integrate over a nodal grid."""

from __future__ import annotations

import numpy as np
import pytest

from postgkyl.numerics import calculus


class TestIntegrate1D:

  def test_uniform_ones_integrates_to_domain_length(self):
    grid = [np.linspace(0.0, 1.0, 6)]  # 5 cells, dx=0.2
    _, out = calculus.integrate(grid, np.ones((5, 1)), axis=0)
    np.testing.assert_allclose(out.flat[0], 1.0, rtol=1e-12)

  def test_linear_function_exact_integral(self):
    # integral of x from 0 to 1 = 0.5 (analytic, hand-computed)
    N = 100
    grid = [np.linspace(0.0, 1.0, N + 1)]
    x_cc = 0.5 * (grid[0][:-1] + grid[0][1:])
    values = x_cc[:, np.newaxis]
    _, out = calculus.integrate(grid, values, axis=0)
    np.testing.assert_allclose(out.flat[0], 0.5, rtol=1e-3)

  def test_quadratic_function_exact_integral(self):
    # integral of x^2 from 0 to 1 = 1/3 (analytic, hand-computed)
    N = 4000
    grid = [np.linspace(0.0, 1.0, N + 1)]
    x_cc = 0.5 * (grid[0][:-1] + grid[0][1:])
    values = (x_cc**2)[:, np.newaxis]
    _, out = calculus.integrate(grid, values, axis=0)
    np.testing.assert_allclose(out.flat[0], 1.0 / 3.0, rtol=1e-3)

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
    # len(coord) == values.shape[d] -> a last element is appended to dz
    x_cc = np.linspace(0.1, 0.9, 5)  # 5 cell centers, dx=0.2
    _, out = calculus.integrate([x_cc], np.ones((5, 1)), axis=0)
    np.testing.assert_allclose(out.flat[0], 1.0, rtol=1e-12)

  def test_single_cell_axis_uses_mean(self):
    grid = [np.array([0.5]), np.linspace(0.0, 1.0, 4)]
    _, out = calculus.integrate(grid, np.ones((1, 3, 1)), axis=0)
    assert out.shape[0] == 1
