"""Tests for postgkyl.numerics.ev_ops -- the RPN operator registry.

Every operator is ``f(in_grid, in_values) -> ([out_grid], [out_values])``
over plain lists / NumPy arrays. ``cmds`` maps each RPN token to its
arity and function.
"""

from __future__ import annotations

import numpy as np
import pytest

from postgkyl.numerics import ev_ops


def _arr(*vals):
  return np.array(vals, dtype=float)


class TestCmdsTable:

  def test_expected_keys_present(self):
    expected = {
        "+",
        "-",
        "*",
        "/",
        "dot",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "abs",
        "avg",
        "log",
        "log10",
        "max",
        "min",
        "max2",
        "min2",
        "mean",
        "len",
        "pow",
        "sq",
        "exp",
        "grad",
        "grad2",
        "int",
        "div",
        "curl",
        "scale_comp",
        "scale_zi_axis",
    }
    assert set(ev_ops.cmds) == expected

  def test_arities_are_ints(self):
    for tok, spec in ev_ops.cmds.items():
      assert isinstance(spec["num_in"], int)
      assert isinstance(spec["num_out"], int)
      assert callable(spec["func"])


class TestGetGrid:

  def test_both_none(self):
    assert ev_ops._get_grid(None, None) is None

  def test_first_none(self):
    g = [np.array([0.0, 1.0])]
    assert ev_ops._get_grid(None, g) is g

  def test_second_none(self):
    g = [np.array([0.0, 1.0])]
    assert ev_ops._get_grid(g, None) is g

  def test_prefers_longer_grid(self):
    g1 = [np.array([0.0, 1.0])]
    g2 = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    assert ev_ops._get_grid(g1, g2) is g2
    assert ev_ops._get_grid(g2, g1) is g2


class TestArithmetic:

  def test_add(self):
    out_grid, out_vals = ev_ops.add([None, None], [_arr(1.0), _arr(2.0)])
    np.testing.assert_allclose(out_vals[0], 3.0)

  def test_subtract_is_stack_order(self):
    # RPN: a b -  computes b - a (in_values[1] - in_values[0])
    _, out_vals = ev_ops.subtract([None, None], [_arr(1.0), _arr(5.0)])
    np.testing.assert_allclose(out_vals[0], 4.0)

  def test_mult_same_shape(self):
    _, out_vals = ev_ops.mult([None, None], [_arr(2.0), _arr(3.0)])
    np.testing.assert_allclose(out_vals[0], 6.0)

  def test_mult_broadcast_leading_axis(self):
    """Cross-basis (conf x phase) multiply: the conf-space field's leading
    axis matches the phase-space field's leading axis, so multiply via
    transpose-multiply-transpose instead of NumPy's trailing-axis rule."""
    conf = np.ones((3, 1))  # 3 conf cells, 1 comp
    phase = np.arange(12.0).reshape(3, 4)  # 3 conf cells x 4 vel cells
    _, out_vals = ev_ops.mult([None, None], [conf, phase])
    expected = (phase.transpose() * conf.transpose()).transpose()
    np.testing.assert_allclose(out_vals[0], expected)

  def test_divide_same_shape(self):
    _, out_vals = ev_ops.divide([None, None], [_arr(2.0), _arr(10.0)])
    np.testing.assert_allclose(out_vals[0], 5.0)

  def test_divide_broadcast_leading_axis(self):
    conf = np.full((3, 1), 2.0)
    phase = np.arange(1.0, 13.0).reshape(3, 4)
    _, out_vals = ev_ops.divide([None, None], [conf, phase])
    expected = (phase.transpose() / conf.transpose()).transpose()
    np.testing.assert_allclose(out_vals[0], expected)

  def test_dot(self):
    g = [np.array([0.0, 1.0])]
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[4.0, 5.0, 6.0]])
    out_grid, out_vals = ev_ops.dot([g, g], [a, b])
    np.testing.assert_allclose(out_vals[0], [[32.0]])


class TestUnaryMath:

  def test_sqrt(self):
    g = [np.array([0.0, 1.0])]
    _, out_vals = ev_ops.sqrt([g], [_arr(4.0)])
    np.testing.assert_allclose(out_vals[0], 2.0)

  def test_sin_cos_tan(self):
    g = [np.array([0.0, 1.0])]
    x = _arr(0.0)
    np.testing.assert_allclose(ev_ops.psin([g], [x])[1][0], 0.0)
    np.testing.assert_allclose(ev_ops.pcos([g], [x])[1][0], 1.0)
    np.testing.assert_allclose(ev_ops.ptan([g], [x])[1][0], 0.0)

  def test_absolute(self):
    g = [np.array([0.0, 1.0])]
    _, out_vals = ev_ops.absolute([g], [_arr(-3.0)])
    np.testing.assert_allclose(out_vals[0], 3.0)

  def test_log_and_log10(self):
    g = [np.array([0.0, 1.0])]
    np.testing.assert_allclose(ev_ops.log([g], [_arr(np.e)])[1][0], 1.0)
    np.testing.assert_allclose(ev_ops.log10([g], [_arr(100.0)])[1][0], 2.0)

  def test_sq(self):
    g = [np.array([0.0, 1.0])]
    _, out_vals = ev_ops.sq([g], [_arr(3.0)])
    np.testing.assert_allclose(out_vals[0], 9.0)

  def test_exp(self):
    g = [np.array([0.0, 1.0])]
    _, out_vals = ev_ops.exp([g], [_arr(0.0)])
    np.testing.assert_allclose(out_vals[0], 1.0)


class TestReductions:

  def test_minimum(self):
    _, out_vals = ev_ops.minimum([None], [np.array([3.0, 1.0, 2.0])])
    np.testing.assert_allclose(out_vals[0], [1.0])

  def test_minimum_ignores_nan(self):
    _, out_vals = ev_ops.minimum([None], [np.array([np.nan, 1.0, 2.0])])
    np.testing.assert_allclose(out_vals[0], [1.0])

  def test_maximum(self):
    _, out_vals = ev_ops.maximum([None], [np.array([3.0, 1.0, 2.0])])
    np.testing.assert_allclose(out_vals[0], [3.0])

  def test_mean(self):
    _, out_vals = ev_ops.mean([None], [np.array([1.0, 2.0, 3.0])])
    np.testing.assert_allclose(out_vals[0], [2.0])

  def test_minimum2(self):
    _, out_vals = ev_ops.minimum2(
        [None, None], [_arr(1.0, 5.0), _arr(3.0, 2.0)])
    np.testing.assert_allclose(out_vals[0], [1.0, 2.0])

  def test_maximum2(self):
    _, out_vals = ev_ops.maximum2(
        [None, None], [_arr(1.0, 5.0), _arr(3.0, 2.0)])
    np.testing.assert_allclose(out_vals[0], [3.0, 5.0])


class TestPower:

  def test_power_is_stack_order(self):
    # RPN: a b pow  computes b ** a  (in_values[1] ** in_values[0])
    _, out_vals = ev_ops.power([None, _arr(0.0)], [_arr(2.0), _arr(3.0)])
    np.testing.assert_allclose(out_vals[0], 9.0)


class TestLength:

  def test_nodal_grid_length(self):
    grid = [np.linspace(0.0, 4.0, 5)]  # nodal, 4 cells
    values = np.ones((4, 1))
    _, out_vals = ev_ops.length([None, grid], [0.0, values])
    np.testing.assert_allclose(out_vals[0], 4.0)

  def test_cell_centered_grid_length_adds_one_more_dz(self):
    """When ``len(coord) == values.shape[axis]`` (already-cell-centered
    grid), one extra spacing is added, matching the ``calculus.integrate``
    convention."""
    grid = [np.linspace(0.0, 3.0, 4)]  # 4 cell centers, dx=1
    values = np.ones((4, 1))
    _, out_vals = ev_ops.length([None, grid], [0.0, values])
    np.testing.assert_allclose(out_vals[0], 4.0)


class TestGrad:

  def test_grad_1d_matches_analytic_slope(self):
    grid = [np.linspace(0.0, 1.0, 11)]  # 10 cells
    zc = 0.5 * (grid[0][:-1] + grid[0][1:])
    values = (2.0 * zc)[:, np.newaxis]  # f(x) = 2x -> df/dx = 2
    _, out_vals = ev_ops.grad([grid], [values])
    np.testing.assert_allclose(out_vals[0][:, 0], 2.0, rtol=1e-8)

  def test_grad2_colon_range(self):
    grid = [np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11)]
    values = np.ones((10, 10, 1))
    _, out_vals = ev_ops.grad2([None, grid], ["0:2", values])
    assert out_vals[0].shape[-1] == 2

  def test_grad2_comma_list(self):
    grid = [np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11)]
    values = np.ones((10, 10, 1))
    _, out_vals = ev_ops.grad2([None, grid], ["0,1", values])
    assert out_vals[0].shape[-1] == 2

  def test_grad2_single_axis(self):
    grid = [np.linspace(0.0, 1.0, 11)]
    zc = 0.5 * (grid[0][:-1] + grid[0][1:])
    values = (3.0 * zc)[:, np.newaxis]
    _, out_vals = ev_ops.grad2([None, grid], [0, values])
    np.testing.assert_allclose(out_vals[0][:, 0], 3.0, rtol=1e-8)


class TestIntegrateAndAverage:

  def test_integrate_matches_calculus_integrate(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 1))
    _, out_vals = ev_ops.integrate([None, grid], [np.array(0.0), values])
    np.testing.assert_allclose(out_vals[0].flat[0], 1.0, rtol=1e-12)

  def test_integrate_axis_all_string(self):
    grid = [np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 5)]
    values = np.ones((5, 4, 1))
    _, out_vals = ev_ops.integrate([None, grid], ["all", values])
    np.testing.assert_allclose(out_vals[0].flat[0], 2.0, rtol=1e-12)

  def test_integrate_colon_slice_axis(self):
    """src_bak's colon-slice branch passed raw strings to ``range()``,
    a TypeError-raising latent bug never exercised by any caller. Fixed
    here (cast to int) and proven by this test."""
    grid = [np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 5)]
    values = np.ones((5, 4, 1))
    _, out_vals = ev_ops.integrate([None, grid], ["0:2", values])
    np.testing.assert_allclose(out_vals[0].flat[0], 2.0, rtol=1e-12)

  def test_integrate_ndarray_axis(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 1))
    _, out_vals = ev_ops.integrate([None, grid], [np.array(0.0), values])
    np.testing.assert_allclose(out_vals[0].flat[0], 1.0, rtol=1e-12)

  def test_integrate_bad_axis_type_raises(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 1))
    with pytest.raises(TypeError):
      ev_ops.integrate([None, grid], [3 + 4j, values])

  def test_average_divides_by_length(self):
    grid = [np.linspace(0.0, 2.0, 6)]  # length 2, 5 cells
    values = 3.0 * np.ones((5, 1))
    _, out_vals = ev_ops.average([None, grid], [np.array(0.0), values])
    np.testing.assert_allclose(out_vals[0].flat[0], 3.0, rtol=1e-10)

  def test_integrate_float_axis(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 1))
    _, out_vals = ev_ops.integrate([None, grid], [0.0, values])
    np.testing.assert_allclose(out_vals[0].flat[0], 1.0, rtol=1e-12)

  def test_integrate_tuple_axis(self):
    grid = [np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 5)]
    values = np.ones((5, 4, 1))
    _, out_vals = ev_ops.integrate([None, grid], [(0, 1), values])
    np.testing.assert_allclose(out_vals[0].flat[0], 2.0, rtol=1e-12)

  def test_integrate_comma_string_axis(self):
    grid = [np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 5)]
    values = np.ones((5, 4, 1))
    _, out_vals = ev_ops.integrate([None, grid], ["0,1", values])
    np.testing.assert_allclose(out_vals[0].flat[0], 2.0, rtol=1e-12)

  def test_integrate_single_int_string_axis(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 1))
    _, out_vals = ev_ops.integrate([None, grid], ["0", values])
    np.testing.assert_allclose(out_vals[0].flat[0], 1.0, rtol=1e-12)

  def test_integrate_cell_centered_grid_appends_last_spacing(self):
    """When ``len(coord) == values.shape[d]`` (an already-cell-centered
    grid), one extra spacing is appended to ``dz`` -- matching
    ``calculus.integrate``'s convention."""
    x_cc = np.linspace(0.1, 0.9, 5)  # 5 cell centers, dx=0.2
    _, out_vals = ev_ops.integrate(
        [None, [x_cc]], [np.array(0.0), np.ones((5, 1))])
    np.testing.assert_allclose(out_vals[0].flat[0], 1.0, rtol=1e-12)

  def test_average_cell_centered_grid_length(self):
    """``avg``'s length computation also has the cell-centered
    (``len(coord) == values.shape[axis]``) extra-spacing branch."""
    x_cc = np.linspace(0.1, 0.9, 5)  # total length 1.0
    values = 2.0 * np.ones((5, 1))
    _, out_vals = ev_ops.average([None, [x_cc]], [np.array(0.0), values])
    np.testing.assert_allclose(out_vals[0].flat[0], 2.0, rtol=1e-10)


class TestDivergence:

  def test_uniform_field_zero_divergence(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 1))
    _, out_vals = ev_ops.divergence([grid], [values])
    np.testing.assert_allclose(out_vals[0], 0.0, atol=1e-8)

  def test_linear_field_matches_analytic_divergence(self):
    grid = [np.linspace(0.0, 1.0, 21)]
    zc = 0.5 * (grid[0][:-1] + grid[0][1:])
    values = (2.0 * zc)[:, np.newaxis]  # d/dx (2x) = 2
    _, out_vals = ev_ops.divergence([grid], [values])
    np.testing.assert_allclose(out_vals[0][:, 0], 2.0, rtol=1e-8)

  def test_too_many_components_raises(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 3))  # 3 comps, 1 dim
    with pytest.raises(ValueError, match="longer than number of dimensions"):
      ev_ops.divergence([grid], [values])


class TestCurl:

  def test_1d_requires_3_components(self):
    grid = [np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 2))
    with pytest.raises(ValueError, match="requires 3-component"):
      ev_ops.curl([grid], [values])

  def test_1d_curl_matches_analytic(self):
    grid = [np.linspace(0.0, 1.0, 21)]
    zc = 0.5 * (grid[0][:-1] + grid[0][1:])
    values = np.zeros((20, 3))
    values[:, 1] = zc  # f_y = x -> curl_z = d(f_y)/dx = 1
    values[:, 2] = 2.0 * zc  # f_z = 2x -> curl_y = -d(f_z)/dx = -2
    _, out_vals = ev_ops.curl([grid], [values])
    np.testing.assert_allclose(out_vals[0][:, 1], -2.0, rtol=1e-8)
    np.testing.assert_allclose(out_vals[0][:, 2], 1.0, rtol=1e-8)

  def test_2d_too_few_components_raises(self):
    grid = [np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 5, 1))
    with pytest.raises(ValueError, match="smaller than number of dimensions"):
      ev_ops.curl([grid], [values])

  def test_2d_exactly_two_components_computes_scalar_curl(self):
    """The legacy code printed a misleading 'too long' WARNING for this
    exact-match (num_comps == num_dims == 2) case and then computed the
    standard 2D (in-plane) curl anyway -- a message bug, not a real
    anomaly, fixed here by dropping the false-positive message. This is
    the normal way to take the curl of a 2D vector field."""
    grid = [np.linspace(0.0, 1.0, 21), np.linspace(0.0, 1.0, 21)]
    zc = 0.5 * (grid[0][:-1] + grid[0][1:])
    X, Y = np.meshgrid(zc, zc, indexing="ij")
    values = np.zeros((20, 20, 2))
    values[..., 0] = -Y  # f_x = -y
    values[..., 1] = X  # f_y = x  -> curl_z = df_y/dx - df_x/dy = 1 - (-1) = 2
    _, out_vals = ev_ops.curl([grid], [values])
    assert out_vals[0].shape[-1] == 1
    np.testing.assert_allclose(out_vals[0][..., 0], 2.0, rtol=1e-6)

  def test_2d_three_components_computes_full_curl(self):
    grid = [np.linspace(0.0, 1.0, 21), np.linspace(0.0, 1.0, 21)]
    values = np.ones((20, 20, 3))
    _, out_vals = ev_ops.curl([grid], [values])
    assert out_vals[0].shape[-1] == 3

  def test_2d_too_many_components_raises(self):
    grid = [np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 6)]
    values = np.ones((5, 5, 4))
    with pytest.raises(ValueError, match="longer than number of dimensions"):
      ev_ops.curl([grid], [values])

  def test_3d_too_few_components_raises(self):
    grid = [np.linspace(0.0, 1.0, 6)] * 3
    values = np.ones((5, 5, 5, 2))
    with pytest.raises(ValueError, match="smaller than number of dimensions"):
      ev_ops.curl([grid], [values])

  def test_3d_too_many_components_raises(self):
    grid = [np.linspace(0.0, 1.0, 6)] * 3
    values = np.ones((5, 5, 5, 4))
    with pytest.raises(ValueError, match="longer than number of dimensions"):
      ev_ops.curl([grid], [values])

  def test_3d_curl_of_uniform_field_is_zero(self):
    grid = [np.linspace(0.0, 1.0, 6)] * 3
    values = np.ones((5, 5, 5, 3))
    _, out_vals = ev_ops.curl([grid], [values])
    np.testing.assert_allclose(out_vals[0], 0.0, atol=1e-10)


class TestScaleComp:

  def test_slice_spec_scales_selected_components(self):
    grid = [np.linspace(0.0, 1.0, 2)]
    data = np.array([[1.0, 2.0, 3.0, 4.0]])
    out_grid, out_vals = ev_ops.scale_comp([None, None, grid],
                                           [np.array(10.0), "1:3", data])
    np.testing.assert_allclose(out_vals[0], [[1.0, 20.0, 30.0, 4.0]])
    assert out_grid[0] is grid

  def test_int_array_spec(self):
    data = np.array([[1.0, 2.0, 3.0]])
    _, out_vals = ev_ops.scale_comp(
        [None, None, None], [np.array(2.0), np.array(1.0), data])
    np.testing.assert_allclose(out_vals[0], [[1.0, 4.0, 3.0]])

  def test_bare_int_spec(self):
    data = np.array([[1.0, 2.0, 3.0]])
    _, out_vals = ev_ops.scale_comp([None, None, None],
                                    [np.array(5.0), 2, data])
    np.testing.assert_allclose(out_vals[0], [[1.0, 2.0, 15.0]])

  def test_comma_list_spec(self):
    data = np.array([[1.0, 2.0, 3.0]])
    _, out_vals = ev_ops.scale_comp([None, None, None],
                                    [np.array(2.0), "0,2", data])
    np.testing.assert_allclose(out_vals[0], [[2.0, 2.0, 6.0]])

  def test_does_not_mutate_original(self):
    data = np.array([[1.0, 2.0, 3.0]])
    ev_ops.scale_comp([None, None, None], [np.array(10.0), "0:1", data])
    np.testing.assert_allclose(data, [[1.0, 2.0, 3.0]])


class TestScaleZiAxis:

  def test_scales_named_axis(self):
    axis = np.array([0.0, 1.0, 2.0])
    grid = [axis]
    data = np.array([[1.0], [2.0], [3.0]])
    out_grid, out_vals = ev_ops.scale_zi_axis(
        [None, None, grid],
        [np.array(10.0), np.array(0.0), data])
    np.testing.assert_allclose(out_grid[0][0], [0.0, 10.0, 20.0])
    np.testing.assert_allclose(out_vals[0], data)
