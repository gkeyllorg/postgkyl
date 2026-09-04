"""Tests for postgkyl.numerics.fit -- model functions, RPN parser, fit/auto_guess.

Ports the array-only subset of ``tests_bak/test_fit.py``: the model
functions, ``fit``/``fit_evaluate``, and the RPN expression machinery.
``FitTypeParam`` and the ``fit`` CLI command belong to the CLI/operations layers
and are not part of this leaf module -- they are not ported here.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# `postgkyl.numerics.fit` (the submodule) is shadowed by the `fit` FUNCTION
# that numerics/__init__.py re-exports under the same attribute name -- see
# the note in tests/test_coverage_leaf.py. importlib sidesteps the
# package's __init__ entirely and returns the actual submodule object.
fitmod = importlib.import_module("postgkyl.numerics.fit")

# ── model functions ──────────────────────────────────────────────────────────


class TestFitFunctions:

  def test_linear_evaluation(self):
    x = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(fitmod.linear(x, 3.0, -1.0), [-1.0, 2.0, 5.0])

  def test_quadratic_evaluation(self):
    x = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(fitmod.quadratic(x, 1.0, -2.0, 1.0),
                               [1.0, 0.0, 1.0, 4.0])

  def test_plane_evaluation(self):
    XY = np.array([[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(fitmod.plane(XY, 2.0, -1.0, 0.5), [0.5, 1.5])

  def test_quadratic2d_evaluation(self):
    XY = np.array([[1.0], [2.0]])
    result = fitmod.quadratic2d(XY, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0)
    np.testing.assert_allclose(result, [4.0])

  def test_exp_plateau_evaluation(self):
    x = np.array([0.0, 1.0])
    np.testing.assert_allclose(fitmod.exp_plateau(x, 2.0, 0.0, 1.0), [3.0, 3.0])

  def test_gaussian_evaluation(self):
    x = np.array([0.0])
    np.testing.assert_allclose(fitmod.gaussian(x, 3.0, 0.0, 1.0), [3.0])

  def test_power_evaluation(self):
    x = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(fitmod.power(x, 2.0, 3.0, 1.0),
                               [3.0, 17.0, 129.0])

  def test_sinusoid_evaluation(self):
    x = np.array([0.0, np.pi / 2])
    np.testing.assert_allclose(fitmod.sinusoid(x, 1.0, 1.0, 0.0, 0.5),
                               [0.5, 1.5],
                               atol=1e-14)

  def test_tanh_transition_evaluation(self):
    x = np.array([0.0])
    np.testing.assert_allclose(fitmod.tanh_transition(x, 2.0, 0.0, 1.0, -1.0),
                               [-1.0])

  def test_exp2_evaluation(self):
    np.testing.assert_allclose(fitmod.exp2(0.0, a=2.0, b=1.0), 2.0)
    x = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(fitmod.exp2(x, a=1.0, b=1.0), np.exp(2 * x))

  def test_fit_functions_and_ndim_consistent(self):
    assert set(fitmod.FIT_FUNCTIONS) == set(fitmod.FIT_NDIM)

  def test_fit_ndim_values(self):
    assert fitmod.FIT_NDIM["linear"] == 1
    assert fitmod.FIT_NDIM["quadratic"] == 1
    assert fitmod.FIT_NDIM["plane"] == 2
    assert fitmod.FIT_NDIM["quadratic2d"] == 2
    assert fitmod.FIT_NDIM["exp_plateau"] == 1
    assert fitmod.FIT_NDIM["gaussian"] == 1
    assert fitmod.FIT_NDIM["power"] == 1
    assert fitmod.FIT_NDIM["sinusoid"] == 1
    assert fitmod.FIT_NDIM["tanh_transition"] == 1
    assert fitmod.FIT_NDIM["exp2"] == 1

  def test_fit_evaluate_builtin(self):
    x = np.array([0.0, 1.0, 2.0])
    out = fitmod.fit_evaluate(x, "linear", [3.0, -1.0])
    np.testing.assert_allclose(out, [-1.0, 2.0, 5.0])

  def test_fit_evaluate_rpn(self):
    x = np.array([0.0, 1.0, 2.0])
    out = fitmod.fit_evaluate(x, "a x * b +", [3.0, -1.0])
    np.testing.assert_allclose(out, [-1.0, 2.0, 5.0])


# ── fit() -- 1-D models ──────────────────────────────────────────────────────


class TestFit1D:

  def test_linear_exact_data_recovers_params(self):
    x = np.linspace(0, 10, 50)
    y = 3.0 * x - 1.5
    params, _, R2 = fitmod.fit(x, y, "linear")
    np.testing.assert_allclose(params, [3.0, -1.5], rtol=1e-10)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_quadratic_exact_data_recovers_params(self):
    x = np.linspace(-2, 2, 60)
    y = 0.5 * x**2 - 1.0 * x + 2.0
    params, _, R2 = fitmod.fit(x, y, "quadratic")
    np.testing.assert_allclose(params, [0.5, -1.0, 2.0], rtol=1e-10)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_linear_noisy_data_high_R2_and_close_params(self):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 200)
    params, _, R2 = fitmod.fit(x, y, "linear")
    assert R2 > 0.999
    np.testing.assert_allclose(params[0], 2.0, atol=0.05)
    np.testing.assert_allclose(params[1], 1.0, atol=0.1)

  def test_returns_covariance_with_correct_shape(self):
    x = np.linspace(0, 5, 30)
    y = x + 1.0
    _, cov, _ = fitmod.fit(x, y, "linear")
    assert cov.shape == (2, 2)

  def test_initial_guess_does_not_change_result_on_exact_data(self):
    x = np.linspace(0, 10, 50)
    y = 5.0 * x + 3.0
    params_default, _, _ = fitmod.fit(x, y, "linear")
    params_guess, _, _ = fitmod.fit(x, y, "linear", p0=[10.0, 10.0])
    np.testing.assert_allclose(params_default, params_guess, rtol=1e-8)

  def test_exp_plateau_exact_data_recovers_params(self):
    x = np.linspace(0, 5, 80)
    true_params = [3.0, -1.5, 1.0]
    y = fitmod.exp_plateau(x, *true_params)
    params, _, R2 = fitmod.fit(x, y, "exp_plateau", p0=[1.0, -1.0, 0.0])
    np.testing.assert_allclose(params, true_params, rtol=1e-6)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_exp_plateau_noisy_data_high_R2(self):
    rng = np.random.default_rng(7)
    x = np.linspace(0, 5, 100)
    y = fitmod.exp_plateau(x, 3.0, -1.5, 1.0) + rng.normal(0, 0.05, 100)
    _, _, R2 = fitmod.fit(x, y, "exp_plateau", p0=[1.0, -1.0, 0.0])
    assert R2 > 0.99

  def test_invalid_fit_type_raises_value_error(self):
    x = np.linspace(0, 1, 10)
    y = x
    with pytest.raises(ValueError, match="not recognized"):
      fitmod.fit(x, y, "cubic")

  def test_gaussian_exact_data_recovers_params(self):
    x = np.linspace(-3, 3, 100)
    true_params = [2.0, 0.5, 0.8]
    y = fitmod.gaussian(x, *true_params)
    params, _, R2 = fitmod.fit(x, y, "gaussian", p0=[1.0, 0.0, 1.0])
    np.testing.assert_allclose(params, true_params, rtol=1e-6)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_power_exact_data_recovers_params(self):
    x = np.linspace(1, 5, 60)
    true_params = [3.0, 2.0, -1.0]
    y = fitmod.power(x, *true_params)
    params, _, R2 = fitmod.fit(x, y, "power", p0=[1.0, 1.5, 0.0])
    np.testing.assert_allclose(params, true_params, rtol=1e-6)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_sinusoid_exact_data_recovers_params(self):
    x = np.linspace(0, 4 * np.pi, 200)
    true_params = [2.0, 1.0, 0.3, 0.5]
    y = fitmod.sinusoid(x, *true_params)
    params, _, R2 = fitmod.fit(x, y, "sinusoid", p0=[1.5, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(params, true_params, rtol=1e-5)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_tanh_transition_exact_data_recovers_params(self):
    x = np.linspace(-5, 5, 100)
    true_params = [3.0, 1.0, 0.5, 2.0]
    y = fitmod.tanh_transition(x, *true_params)
    params, _, R2 = fitmod.fit(x, y, "tanh_transition", p0=[1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(params, true_params, rtol=1e-6)
    assert R2 == pytest.approx(1.0, abs=1e-8)


# ── RPN expression support ───────────────────────────────────────────────────


class TestRPN:

  def test_param_names_basic(self):
    assert fitmod.rpn_param_names("a x * b +") == ["a", "b"]

  def test_param_names_excludes_spatial_vars(self):
    assert "x" not in fitmod.rpn_param_names("a x * b +")
    assert "y" not in fitmod.rpn_param_names("a x * b y * + c +")

  def test_param_names_excludes_operators(self):
    assert "+" not in fitmod.rpn_param_names("a x * b +")
    assert "*" not in fitmod.rpn_param_names("a x * b +")

  def test_param_names_excludes_functions(self):
    assert "exp" not in fitmod.rpn_param_names("A b x * exp *")

  def test_param_names_excludes_numeric_literals(self):
    assert fitmod.rpn_param_names("2 x * 1 +") == []

  def test_param_names_preserves_order(self):
    assert fitmod.rpn_param_names("A b x * exp * C +") == ["A", "b", "C"]

  def test_param_names_empty_expression(self):
    assert fitmod.rpn_param_names("") == []

  def test_ndim_1d(self):
    assert fitmod.rpn_ndim("a x * b +") == 1

  def test_ndim_2d(self):
    assert fitmod.rpn_ndim("a x * b y * + c +") == 2

  def test_rpn_linear_recovers_params(self):
    x = np.linspace(0, 10, 50)
    y = 3.0 * x - 1.5
    params, _, R2 = fitmod.fit(x, y, "a x * b +", p0=[1.0, 0.0])
    np.testing.assert_allclose(params, [3.0, -1.5], rtol=1e-8)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_rpn_exp_recovers_params(self):
    x = np.linspace(0, 3, 80)
    true_A, true_b = 2.0, -0.5
    y = true_A * np.exp(true_b * x)
    params, _, R2 = fitmod.fit(x, y, "A b x * exp *", p0=[1.0, -1.0])
    np.testing.assert_allclose(params, [true_A, true_b], rtol=1e-6)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_rpn_plane_2d_recovers_params(self):
    X, Y = np.meshgrid(np.linspace(0, 5, 15),
                       np.linspace(0, 3, 10),
                       indexing="ij")
    xdata = np.array([X.flatten(), Y.flatten()])
    y = 2.0 * X.flatten() - 1.5 * Y.flatten() + 0.5
    params, _, R2 = fitmod.fit(xdata,
                               y,
                               "a x * b y * + c +",
                               p0=[1.0, 1.0, 0.0])
    np.testing.assert_allclose(params, [2.0, -1.5, 0.5], rtol=1e-8)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_rpn_literal_coefficients(self):
    x = np.linspace(1, 5, 40)
    y = 2.0 * x**2
    params, _, R2 = fitmod.fit(x, y, "a x 2 ** *", p0=[1.0])
    np.testing.assert_allclose(params, [2.0], rtol=1e-8)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_rpn_caret_power_operator(self):
    x = np.linspace(1, 5, 40)
    y = 2.0 * x**2
    params, _, R2 = fitmod.fit(x, y, "a x 2 ^ *", p0=[1.0])
    np.testing.assert_allclose(params, [2.0], rtol=1e-8)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_rpn_subtract_operator(self):
    func = fitmod._rpn_make_func("x a -")
    x = np.array([5.0, 10.0])
    np.testing.assert_allclose(func(x, 2.0), [3.0, 8.0])

  def test_rpn_divide_operator(self):
    func = fitmod._rpn_make_func("x a /")
    x = np.array([10.0, 20.0])
    np.testing.assert_allclose(func(x, 2.0), [5.0, 10.0])

  def test_rpn_pure_constant_expression_broadcasts_to_x_shape(self):
    """A scalar-only RPN expression (no free params, no spatial var used)
    still broadcasts its result to xdata's shape, since ``x`` is always
    bound in the evaluation namespace regardless of whether the
    expression actually references it."""
    func = fitmod._rpn_make_func("2 3 +")
    x = np.array([0.0, 1.0, 2.0])
    out = func(x)
    np.testing.assert_allclose(out, [5.0, 5.0, 5.0])

  def test_rpn_malformed_stack_raises(self):
    # Leading operator with empty stack causes IndexError inside curve_fit.
    x = np.linspace(0, 1, 10)
    y = x
    with pytest.raises((IndexError, Exception)):
      fitmod.fit(x, y, "* x a +", p0=[1.0])

  def test_rpn_bad_token_raises_value_error(self):
    """A token that is neither an operator, function, known parameter, nor
    a valid float literal (bad token edge case)."""
    x = np.linspace(0, 1, 10)
    y = x
    func = fitmod._rpn_make_func("a x * not_a_number +")
    with pytest.raises(ValueError):
      func(x, 1.0)

  def test_rpn_arity_mismatch_raises(self):
    """Requesting fewer parameter values than the expression's free
    parameters is an arity mismatch: the dict(zip(...)) call silently
    drops the excess names, so the *unbound* stray name looks up as
    missing from ``ns`` and falls through to ``float(tok)``, which raises
    ValueError on a non-numeric token."""
    func = fitmod._rpn_make_func("a b + x *")
    with pytest.raises(ValueError):
      func(np.array([1.0, 2.0]), 1.0)  # only 'a' bound, 'b' unresolved

  def test_fittype_param_accepts_rpn_via_fit(self):
    x = np.linspace(0, 10, 50)
    y = 3.0 * x - 1.5
    params, _, _ = fitmod.fit(x, y, "a x * b +", p0=[1.0, 0.0])
    assert len(params) == 2


# ── fit() -- 2-D models ──────────────────────────────────────────────────────


class TestFit2D:

  @staticmethod
  def _xdata(x, y):
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.array([X.flatten(), Y.flatten()])

  def test_plane_exact_data_recovers_params(self):
    xdata = self._xdata(np.linspace(0, 5, 20), np.linspace(0, 3, 15))
    zdata = fitmod.plane(xdata, 2.0, -1.5, 0.5)
    params, _, R2 = fitmod.fit(xdata, zdata, "plane")
    np.testing.assert_allclose(params, [2.0, -1.5, 0.5], rtol=1e-10)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_quadratic2d_exact_data_recovers_params(self):
    xdata = self._xdata(np.linspace(0, 4, 15), np.linspace(0, 3, 12))
    true_params = [0.3, 0.2, -0.1, 1.0, -0.5, 2.0]
    zdata = fitmod.quadratic2d(xdata, *true_params)
    params, _, R2 = fitmod.fit(xdata, zdata, "quadratic2d")
    np.testing.assert_allclose(params, true_params, rtol=1e-8)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_plane_noisy_data_high_R2(self):
    rng = np.random.default_rng(42)
    xdata = self._xdata(np.linspace(0, 5, 30), np.linspace(0, 3, 25))
    zdata = fitmod.plane(xdata, 2.0, -1.5, 0.5) + rng.normal(
        0, 0.05, xdata.shape[1])
    _, _, R2 = fitmod.fit(xdata, zdata, "plane")
    assert R2 > 0.999

  def test_plane_returns_correct_covariance_shape(self):
    xdata = self._xdata(np.linspace(0, 5, 10), np.linspace(0, 3, 8))
    zdata = fitmod.plane(xdata, 1.0, 2.0, 0.0)
    _, cov, _ = fitmod.fit(xdata, zdata, "plane")
    assert cov.shape == (3, 3)


# ── auto_guess ────────────────────────────────────────────────────────────────


class TestAutoGuess:

  def test_returns_none_for_all_nan(self):
    x = np.linspace(0, 1, 10)
    y = np.full(10, np.nan)
    assert fitmod.auto_guess("linear", x, y) is None

  def test_returns_none_for_rpn_expression(self):
    x = np.linspace(0, 1, 10)
    y = x
    assert fitmod.auto_guess("a x * b +", x, y) is None

  def test_linear_guess_is_reasonable(self):
    x = np.linspace(0, 10, 50)
    y = 2.0 * x + 1.0
    a, b = fitmod.auto_guess("linear", x, y)
    np.testing.assert_allclose([a, b], [2.0, 1.0], rtol=1e-6)

  def test_quadratic_guess_is_reasonable(self):
    x = np.linspace(-2, 2, 60)
    y = 0.5 * x**2 - 1.0 * x + 2.0
    guess = fitmod.auto_guess("quadratic", x, y)
    np.testing.assert_allclose(guess, [0.5, -1.0, 2.0], rtol=1e-6)

  def test_quadratic_guess_falls_back_when_polyfit_raises(self):
    """np.polyfit raises on an empty vector; auto_guess catches it and
    falls back to a fixed placeholder guess rather than propagating."""
    x = np.array([])
    y = np.array([1.0, 2.0])  # not all-NaN, so the finite-check passes
    guess = fitmod.auto_guess("quadratic", x, y)
    assert guess == [0.0, 1.0, pytest.approx(1.5)]

  def test_plane_guess_is_reasonable(self):
    X, Y = np.meshgrid(np.linspace(0, 5, 20),
                       np.linspace(0, 3, 15),
                       indexing="ij")
    xdata = np.array([X.flatten(), Y.flatten()])
    y = 2.0 * X.flatten() - 1.5 * Y.flatten() + 0.5
    guess = fitmod.auto_guess("plane", xdata, y)
    np.testing.assert_allclose(guess, [2.0, -1.5, 0.5], rtol=1e-6)

  def test_quadratic2d_guess_is_reasonable(self):
    X, Y = np.meshgrid(np.linspace(0, 4, 15),
                       np.linspace(0, 3, 12),
                       indexing="ij")
    xdata = np.array([X.flatten(), Y.flatten()])
    true_params = [0.3, 0.2, -0.1, 1.0, -0.5, 2.0]
    y = fitmod.quadratic2d(xdata, *true_params)
    guess = fitmod.auto_guess("quadratic2d", xdata, y)
    np.testing.assert_allclose(guess, true_params, rtol=1e-6)

  def test_exp_plateau_guess_seeds_a_working_fit(self):
    x = np.linspace(0, 5, 80)
    true_params = [3.0, -1.5, 1.0]
    y = fitmod.exp_plateau(x, *true_params)
    guess = fitmod.auto_guess("exp_plateau", x, y)
    params, _, R2 = fitmod.fit(x, y, "exp_plateau", p0=guess)
    np.testing.assert_allclose(params, true_params, rtol=1e-4)

  def test_gaussian_guess_seeds_a_working_fit(self):
    x = np.linspace(-3, 3, 100)
    true_params = [2.0, 0.5, 0.8]
    y = fitmod.gaussian(x, *true_params)
    guess = fitmod.auto_guess("gaussian", x, y)
    params, _, R2 = fitmod.fit(x, y, "gaussian", p0=guess)
    np.testing.assert_allclose(params, true_params, rtol=1e-4)

  def test_gaussian_guess_narrow_peak_uses_fallback_sigma(self):
    """When fewer than two points reach half-max, the FWHM estimate falls
    back to a quarter of the domain width."""
    x = np.linspace(-3, 3, 7)
    y = np.zeros_like(x)
    y[3] = 5.0  # single spike -> only one point at/above half-max
    guess = fitmod.auto_guess("gaussian", x, y)
    assert guess[2] == pytest.approx((x.max() - x.min()) / 4)

  def test_power_guess_seeds_a_working_fit(self):
    x = np.linspace(1, 5, 60)
    true_params = [3.0, 2.0, -1.0]
    y = fitmod.power(x, *true_params)
    guess = fitmod.auto_guess("power", x, y)
    params, _, R2 = fitmod.fit(x, y, "power", p0=guess)
    np.testing.assert_allclose(params, true_params, rtol=1e-4)

  def test_sinusoid_guess_seeds_a_working_fit(self):
    x = np.linspace(0, 4 * np.pi, 200)
    true_params = [2.0, 1.0, 0.3, 0.5]
    y = fitmod.sinusoid(x, *true_params)
    guess = fitmod.auto_guess("sinusoid", x, y)
    params, _, R2 = fitmod.fit(x, y, "sinusoid", p0=guess)
    assert R2 > 0.99

  def test_sinusoid_guess_single_point_omega_fallback(self):
    x = np.array([0.0])
    y = np.array([1.0])
    guess = fitmod.auto_guess("sinusoid", x, y)
    assert guess[1] == 1.0  # omega fallback for len(x) <= 1

  def test_tanh_transition_guess_seeds_a_working_fit(self):
    x = np.linspace(-5, 5, 100)
    true_params = [3.0, 1.0, 0.5, 2.0]
    y = fitmod.tanh_transition(x, *true_params)
    guess = fitmod.auto_guess("tanh_transition", x, y)
    params, _, R2 = fitmod.fit(x, y, "tanh_transition", p0=guess)
    np.testing.assert_allclose(params, true_params, rtol=1e-4)

  def test_unknown_fit_type_returns_none(self):
    x = np.linspace(0, 1, 10)
    y = x
    assert fitmod.auto_guess("not_a_real_model", x, y) is None

  def test_exp2_guess_seeds_a_working_fit(self):
    x = np.linspace(0, 5, 80)
    true_params = [1.0, 0.8]
    y = fitmod.exp2(x, *true_params)
    guess = fitmod.auto_guess("exp2", x, y)
    params, _, R2 = fitmod.fit(x, y, "exp2", p0=guess)
    np.testing.assert_allclose(params, true_params, rtol=1e-4)

  def test_exp2_guess_is_scale_invariant(self):
    """The log-linear guess should converge without needing x normalized
    to O(1) -- unlike a blind (1, 1) seed, it stays accurate as the time
    axis grows."""
    x = np.linspace(0, 500, 200)
    true_params = [2.0, 0.01]
    y = fitmod.exp2(x, *true_params)
    guess = fitmod.auto_guess("exp2", x, y)
    params, _, R2 = fitmod.fit(x, y, "exp2", p0=guess)
    np.testing.assert_allclose(params, true_params, rtol=1e-4)


# ── fit_best_window ───────────────────────────────────────────────────────────


class TestFitBestWindow:

  def test_recovers_known_growth_rate(self):
    x = np.linspace(0, 5, 60)
    true_a, true_b = 1.0, 0.8
    y = fitmod.exp2(x, true_a, true_b)
    params, cov, R2, n = fitmod.fit_best_window(x, y, "exp2")
    assert R2 > 0.99
    np.testing.assert_allclose(params[1], true_b, rtol=0.05)

  def test_returns_four_elements(self):
    x = np.linspace(0, 3, 30)
    y = fitmod.exp2(x, 1.0, 0.5)
    result = fitmod.fit_best_window(x, y, "exp2")
    assert len(result) == 4

  def test_best_n_is_within_bounds(self):
    x = np.linspace(0, 4, 40)
    y = fitmod.exp2(x, 1.0, 0.5)
    _, _, _, n = fitmod.fit_best_window(x, y, "exp2", min_n=5)
    assert 5 <= n <= len(x)

  def test_custom_min_n(self):
    x = np.linspace(0, 3, 30)
    y = fitmod.exp2(x, 1.0, 0.5)
    _, _, _, n = fitmod.fit_best_window(x, y, "exp2", min_n=10)
    assert n >= 10

  def test_curve_fit_failure_for_some_windows_is_skipped(self, monkeypatch):
    """A RuntimeError from curve_fit (non-convergence) for one fitting
    window is caught, not fatal -- the scan continues and still returns
    the best window that did converge."""
    x = np.linspace(0, 5, 30)
    y = fitmod.exp2(x, 1.0, 0.8)
    real_curve_fit = fitmod.opt.curve_fit
    calls = {"n": 0}

    def flaky_curve_fit(*args, **kwargs):
      calls["n"] += 1
      if calls["n"] == 1:
        raise RuntimeError("simulated non-convergence")
      return real_curve_fit(*args, **kwargs)

    monkeypatch.setattr(fitmod.opt, "curve_fit", flaky_curve_fit)
    _, _, R2, _ = fitmod.fit_best_window(x, y, "exp2", min_n=5)
    assert R2 > 0.9

  def test_all_windows_failing_to_converge_raises(self, monkeypatch):
    """If curve_fit never converges for any window in the scan range,
    fit_best_window must raise a clear domain error rather than crash."""
    x = np.linspace(0, 5, 30)
    y = fitmod.exp2(x, 1.0, 0.8)

    def always_fails(*args, **kwargs):
      raise RuntimeError("simulated non-convergence")

    monkeypatch.setattr(fitmod.opt, "curve_fit", always_fails)
    with pytest.raises(RuntimeError, match="failed to converge"):
      fitmod.fit_best_window(x, y, "exp2", min_n=5)

  def test_generic_over_fit_type(self):
    """fit_best_window is generic over any registered fit_type, not
    hard-wired to exp2 -- generalizing the old growth-specific scan."""
    x = np.linspace(0.1, 5, 40)
    y = 2.0 * x + 1.0
    params, _, R2, _ = fitmod.fit_best_window(x, y, "linear", p0=[1.0, 1.0])
    assert R2 > 0.99
    np.testing.assert_allclose(params, [2.0, 1.0], rtol=1e-6)
