"""Curve fitting: built-in model functions (including ``exp2``, the
growth-rate model), an RPN custom-model parser, ``scipy.optimize.curve_fit``
wrappers, and the leading-window search used for growth-rate-style fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.optimize as opt


def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
  """``f(x) = a*x + b``

  a: slope (change in f per unit x).
  b: intercept, f(0).
  """
  return a * x + b


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
  """``f(x) = a*x**2 + b*x + c``

  a: quadratic coefficient; the second derivative is 2*a.
  b: linear coefficient, the slope at x = 0.
  c: intercept, f(0).
  """
  return a * x**2 + b * x + c


def plane(XY: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
  """``f(x, y) = a*x + b*y + c``

  a: slope along x at fixed y.
  b: slope along y at fixed x.
  c: intercept, f(0, 0).
  """
  x, y = XY
  return a * x + b * y + c


def quadratic2d(XY: np.ndarray, a: float, b: float, c: float, d: float,
                e: float, f: float) -> np.ndarray:
  """``fitted(x, y) = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f``

  a: x-squared coefficient.
  b: y-squared coefficient.
  c: cross-term coefficient multiplying x*y.
  d: linear x coefficient.
  e: linear y coefficient.
  f: intercept, fitted(0, 0).
  """
  x, y = XY
  return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f


def exp_plateau(x: np.ndarray, A: float, b: float, C: float) -> np.ndarray:
  """``f(x) = A*exp(b*x) + C``

  A: initial offset from C; f(0) = A + C.
  b: exponential rate (inverse x units); b < 0 means decay toward C.
  C: plateau approached as b*x -> -infinity.
  """
  return A * np.exp(b * x) + C


def gaussian(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
  """``f(x) = A * exp(-0.5 * ((x - mu) / sigma)**2)``

  A: amplitude at the center, f(mu).
  mu: center position.
  sigma: width parameter; abs(sigma) is the standard deviation in x units.
  """
  return A * np.exp(-0.5 * ((x - mu) / sigma)**2)


def power(x: np.ndarray, a: float, n: float, b: float) -> np.ndarray:
  """``f(x) = a * x**n + b``

  a: amplitude multiplying the power law.
  n: power-law exponent.
  b: additive offset.
  """
  return a * x**n + b


def sinusoid(x: np.ndarray, A: float, omega: float, phi: float,
             C: float) -> np.ndarray:
  """``f(x) = A * sin(omega * x + phi) + C``

  A: signed oscillation amplitude.
  omega: angular frequency (radians per unit x).
  phi: phase at x = 0 (radians).
  C: mean level of the oscillation.
  """
  return A * np.sin(omega * x + phi) + C


def tanh_transition(x: np.ndarray, A: float, x0: float, w: float,
                    C: float) -> np.ndarray:
  """``f(x) = A * tanh((x - x0) / w) + C``

  A: signed half-difference between the two asymptotic levels C - A and C + A.
  x0: transition midpoint, where f(x0) = C.
  w: transition scale in x units; the slope at x0 is A/w.
  C: midpoint level.
  """
  return A * np.tanh((x - x0) / w) + C


def exp2(x: np.ndarray, a: float, b: float) -> np.ndarray:
  """``f(x) = a * exp(2*b*x)``

  a: initial value, f(0).
  b: amplitude growth rate (inverse x units); the fitted curve's rate is 2*b.
  Energy (a squared quantity) is typically used for growth-rate studies,
  hence the factor of 2 in the exponent.
  """
  return a * np.exp(2 * b * x)


RPN_OPERATORS: frozenset = frozenset({'+', '-', '*', '/', '**', '^'})

RPN_FUNCTIONS: dict[str, Callable] = {
    'exp': np.exp,
    'log': np.log,
    'ln': np.log,
    'log10': np.log10,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'sqrt': np.sqrt,
    'abs': np.abs,
    'tanh': np.tanh,
}

_SPATIAL_VARS: frozenset = frozenset({'x', 'y', 'z'})


def rpn_param_names(expression: str) -> list[str]:
  """Return the free parameter names from an RPN expression, in order of
  first appearance."""
  names = []
  for tok in expression.split():
    if tok in _SPATIAL_VARS or tok in RPN_OPERATORS or tok in RPN_FUNCTIONS:
      continue
    try:
      float(tok)
    except ValueError:
      if tok not in names:
        names.append(tok)
  return names


def rpn_ndim(expression: str) -> int:
  """Return 1 or 2 depending on whether ``y`` appears as a spatial variable."""
  return 2 if 'y' in expression.split() else 1


def _rpn_make_func(expression: str) -> Callable:
  """Build a ``curve_fit``-compatible callable from an RPN expression string."""
  tokens = expression.split()
  param_names = rpn_param_names(expression)
  ndim = rpn_ndim(expression)

  def _func(xdata, *param_values):
    ns: dict = dict(zip(param_names, param_values))
    if ndim == 1:
      ns['x'] = np.asarray(xdata, dtype=float)
    else:
      ns['x'] = np.asarray(xdata[0], dtype=float)
      ns['y'] = np.asarray(xdata[1], dtype=float)

    stack = []
    for tok in tokens:
      if tok in RPN_OPERATORS:
        b, a = stack.pop(), stack.pop()
        if tok == '+':
          stack.append(a + b)
        elif tok == '-':
          stack.append(a - b)
        elif tok == '*':
          stack.append(a * b)
        elif tok == '/':
          stack.append(a / b)
        else:
          stack.append(a**b)  # ** or ^
      elif tok in RPN_FUNCTIONS:
        stack.append(RPN_FUNCTIONS[tok](stack.pop()))
      elif tok in ns:
        stack.append(ns[tok])
      else:
        stack.append(float(tok))

    result = stack[0]
    ref = ns.get('x', ns.get('y'))
    if np.ndim(result) == 0 and ref is not None:
      result = np.full_like(ref, float(result))
    return np.asarray(result, dtype=float)

  return _func


FIT_FUNCTIONS: dict[str, Callable] = {
    "linear": linear,
    "quadratic": quadratic,
    "plane": plane,
    "quadratic2d": quadratic2d,
    "exp_plateau": exp_plateau,
    "gaussian": gaussian,
    "power": power,
    "sinusoid": sinusoid,
    "tanh_transition": tanh_transition,
    "exp2": exp2,
}

# Number of spatial dimensions each fit type operates on
FIT_NDIM: dict[str, int] = {
    "linear": 1,
    "quadratic": 1,
    "plane": 2,
    "quadratic2d": 2,
    "exp_plateau": 1,
    "gaussian": 1,
    "power": 1,
    "sinusoid": 1,
    "tanh_transition": 1,
    "exp2": 1,
}


@dataclass(frozen=True)
class FitResult:
  """Optimizer coefficients and covariance, with their coordinate transform.

  Evaluate with :func:`fit_evaluate`; :func:`fit_coefficients` converts these
  coefficients and their covariance to the original coordinate system.
  """

  params: np.ndarray
  cov: np.ndarray
  R2: float
  x_offset: float = 0.0
  x_scale: float = 1.0


def fit_evaluate(xdata: np.ndarray, fit_type: str,
                 params: np.ndarray | FitResult) -> np.ndarray:
  """Evaluate on the original grid, using coefficients or a stable fit result."""
  if isinstance(params, FitResult):
    xdata = (np.asarray(xdata) - params.x_offset) / params.x_scale
    params = params.params
  if fit_type in FIT_FUNCTIONS:
    return FIT_FUNCTIONS[fit_type](xdata, *params)
  return _rpn_make_func(fit_type)(xdata, *params)


def fit_coefficients(result: FitResult,
                     fit_type: str) -> tuple[np.ndarray, np.ndarray]:
  """Return original-coordinate coefficients and covariance.

  An exponential amplitude at x=0 can exceed floating-point range even when
  the fitted curve is finite. Its coefficient is then infinite and its
  covariance row/column unavailable (NaN); underflow gives zero with the same
  unavailable covariance. Use the FitResult for evaluation.
  """
  if fit_type != "exp_plateau":
    return result.params, result.cov
  A, k, C = result.params
  offset, scale = result.x_offset, result.x_scale
  with np.errstate(over="ignore", under="ignore", invalid="ignore"):
    factor = np.exp(-k * offset / scale)
    amplitude = (np.sign(A) *
                 np.exp(np.log(abs(A)) - k * offset / scale) if A != 0 else 0.)
    jacobian = np.array([[factor, -amplitude * offset / scale, 0.],
                         [0., 1. / scale, 0.], [0., 0., 1.]])
    cov = jacobian @ result.cov @ jacobian.T
  # An unrepresentable amplitude must not corrupt the rate/plateau errors.
  cov[1:, 1:] = result.cov[1:, 1:] / np.outer([scale, 1.], [scale, 1.])
  if not np.isfinite(amplitude) or (A != 0 and amplitude == 0):
    cov[0, :] = np.nan
    cov[:, 0] = np.nan
  return np.array([amplitude, k / scale, C]), cov


def fit_model(xdata: np.ndarray,
              ydata: np.ndarray,
              fit_type: str = "linear",
              p0: list | None = None) -> FitResult:
  """Fit a model, retaining stable coordinates for subsequent evaluation.

  Exponential plateau fits use u=(x-min(x))/ptp(x) internally. Explicit p0
  always uses original-coordinate coefficients. Other models retain their
  existing coordinates and initialization.
  """
  if fit_type in FIT_FUNCTIONS:
    func = FIT_FUNCTIONS[fit_type]
    n_params = func.__code__.co_argcount - 1
  else:
    toks = set(fit_type.split())
    if not (toks & (RPN_OPERATORS | set(RPN_FUNCTIONS))):
      raise ValueError(
          f"fit_type '{fit_type}' not recognized. Choose from: {list(FIT_FUNCTIONS)}"
      )
    func = _rpn_make_func(fit_type)
    n_params = len(rpn_param_names(fit_type))

  xdata = np.asarray(xdata, dtype=float)
  ydata = np.asarray(ydata, dtype=float)
  offset, scale = 0., 1.
  if fit_type == "exp_plateau":
    offset, scale = float(xdata.min()), float(np.ptp(xdata))
    if not np.isfinite(scale) or scale <= 0:
      raise ValueError("exp_plateau requires a finite, nonzero x range")
    xdata = (xdata - offset) / scale
    if p0 is None:
      p0 = auto_guess(fit_type, xdata, ydata)
    else:
      A, b, C = p0
      # Combine exponents before exponentiation to avoid intermediate overflow.
      amplitude = (np.sign(A) *
                   np.exp(np.log(abs(A)) + b * offset) if A != 0 else 0.)
      p0 = [amplitude, b * scale, C]
  if p0 is None:
    p0 = np.ones(n_params)

  params, cov = opt.curve_fit(func, xdata, ydata, p0=p0)
  residual = ydata - func(xdata, *params)
  ss_res = np.sum(residual**2)
  ss_tot = np.sum((ydata - np.mean(ydata))**2)
  R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
  return FitResult(params, cov, R2, offset, scale)


def fit(xdata: np.ndarray,
        ydata: np.ndarray,
        fit_type: str = "linear",
        p0: list | None = None) -> tuple[np.ndarray, np.ndarray, float]:
  """Return original-coordinate (params, covariance, R2) from curve fitting.

  ``fit_type`` is a built-in name or an RPN expression. ``p0`` contains
  original-coordinate coefficients; None uses a data-driven guess for
  exp_plateau and ones otherwise. For 1D fits xdata has shape (N,); for
  2D fits it has shape (2, N). ydata has shape (N,). Plateau fits normalize
  x internally. Unknown model names raise ValueError.
  For extreme offsets, the amplitude at x=0 may overflow or underflow;
  use :func:`fit_model` and :func:`fit_evaluate` for stable evaluation.
  """
  result = fit_model(xdata, ydata, fit_type, p0)
  params, cov = fit_coefficients(result, fit_type)
  return params, cov, result.R2


def auto_guess(fit_type: str, xdata: np.ndarray,
               ydata: np.ndarray) -> list | None:
  """Return data-driven initial parameter guesses for known fit types.

  Produces a sensible ``p0`` for :func:`fit` by inspecting the data (e.g. a
  least-squares seed for linear/polynomial models, peak location and FWHM
  for a gaussian, the dominant FFT frequency for a sinusoid). Returns
  ``None`` for RPN expressions, when the data has no finite values, or when
  a plateau amplitude at x=0 cannot represent the seed. :func:`fit` uses
  its default initialization in that case.

  Args:
    fit_type: A built-in model name (an RPN expression yields ``None``).
    xdata: Independent variable: shape ``(N,)`` for 1D models, ``(2, N)``
      for 2D.
    ydata: Dependent variable, shape ``(N,)``.

  Returns:
    A list of initial parameter guesses, or ``None`` when no heuristic
    applies.
  """
  y = np.asarray(ydata, dtype=float)
  finite = np.isfinite(y)
  if not np.any(finite):
    return None
  y_fin = y[finite]
  y_min, y_max = y_fin.min(), y_fin.max()
  y_mean = y_fin.mean()
  y_range = y_max - y_min

  if fit_type == "linear":
    x = np.asarray(xdata)
    dx = x.max() - x.min()
    a = y_range / dx if dx != 0 else 1.0
    b = y_mean - a * x.mean()
    return [a, b]

  if fit_type == "quadratic":
    x = np.asarray(xdata)
    try:
      return list(np.polyfit(x, y, 2))
    except Exception:
      return [0.0, 1.0, y_mean]

  if fit_type == "plane":
    x, yc = xdata[0], xdata[1]
    A = np.column_stack([x, yc, np.ones_like(x)])
    result, *_ = np.linalg.lstsq(A, y, rcond=None)
    return list(result)

  if fit_type == "quadratic2d":
    x, yc = xdata[0], xdata[1]
    A = np.column_stack([x**2, yc**2, x * yc, x, yc, np.ones_like(x)])
    result, *_ = np.linalg.lstsq(A, y, rcond=None)
    return list(result)

  if fit_type == "exp_plateau":
    x = np.asarray(xdata)
    n_tail = max(1, len(x) // 10)
    C = float(y[np.argsort(x)[-n_tail:]].mean())
    A = float(y[np.argmin(x)] - C) or 1.0
    x_span = x.max() - x.min()
    b = -1.0 / x_span if x_span > 0 else -1.0
    with np.errstate(over="ignore", under="ignore"):
      A = float(np.sign(A) * np.exp(np.log(abs(A)) - b * x.min()))
    # Let fit_model initialize in normalized coordinates when a physical
    # amplitude cannot represent the seed.
    return [A, b, C] if np.isfinite(A) and A != 0 else None

  if fit_type == "gaussian":
    x = np.asarray(xdata)
    A = float(y_max)
    mu = float(x[np.argmax(y)])
    above = x[y >= A / 2] if A != 0 else x
    if len(above) >= 2:
      sigma = float((above[-1] - above[0]) / (2 * np.sqrt(2 * np.log(2))))
    else:
      sigma = float((x.max() - x.min()) / 4)
    return [A, mu, max(abs(sigma), 1e-10)]

  if fit_type == "power":
    b_off = float(y_min)
    a = float(y_max - b_off) or 1.0
    return [a, 1.0, b_off]

  if fit_type == "sinusoid":
    x = np.asarray(xdata)
    A = float(y_range / 2) or 1.0
    C = float((y_max + y_min) / 2)
    sort_idx = np.argsort(x)
    x_s, y_s = x[sort_idx], y[sort_idx]
    if len(x_s) > 1:
      dx = np.mean(np.diff(x_s))
      freqs = np.fft.rfftfreq(len(y_s), d=dx)
      fft_amp = np.abs(np.fft.rfft(y_s - C))
      i_peak = np.argmax(fft_amp[1:]) + 1 if len(fft_amp) > 1 else 1
      omega = float(2 * np.pi * freqs[i_peak])
    else:
      omega = 1.0
    return [A, omega, 0.0, C]

  if fit_type == "tanh_transition":
    x = np.asarray(xdata)
    A = float(y_range / 2) or 1.0
    C = float((y_max + y_min) / 2)
    x0 = float(x[np.argmax(np.abs(np.gradient(y)))])
    w = float((x.max() - x.min()) / 4) or 1.0
    return [A, x0, w, C]

  if fit_type == "exp2":
    # log(y) = log(a) + 2*b*x is linear -- a log-linear regression gives a
    # scale-invariant guess without needing to normalize x for curve_fit.
    x = np.asarray(xdata, dtype=float)
    y_pos = np.clip(y, 1e-300, None)
    slope, intercept = np.polyfit(x, np.log(y_pos), 1)
    return [float(np.exp(intercept)), float(slope / 2)]

  return None


def fit_best_model(xdata: np.ndarray,
                   ydata: np.ndarray,
                   fit_type: str = "exp2",
                   min_n: int | None = None,
                   p0: list | None = None) -> tuple[FitResult, int]:
  """Fit ``fit_type`` to the best-scoring leading window of a 1D series.

  Scans windows ``xdata[:n]`` for ``n`` from ``min_n`` up to ``len(xdata)``,
  keeping the window with the best coefficient of determination (R^2).
  Plateau windows initialize independently in normalized coordinates; other
  models warm-start from the previous fit (or ``p0``/:func:`auto_guess`
  for the first). This generalizes a single
  full-domain :func:`fit` call to the common case of a time series whose
  early or late region should be excluded (e.g. growth-rate fits, which are
  only valid while the signal grows/decays continuously).

  Args:
    xdata: 1D independent variable (e.g. time).
    ydata: dependent variable, shape matching ``xdata``.
    fit_type: passed to :func:`fit`.
    min_n: minimum number of points in the fitted window. Defaults to
      ``len(xdata) // 10``.
    p0: initial guess for the first window (every window for exp_plateau);
      ``None`` uses data-driven initialization.

  Returns:
    ``(result, N)`` for the best-scoring window, retaining stable coordinates.

  Raises:
    RuntimeError: if ``curve_fit`` fails to converge for every window in
      the scan range.
  """
  xdata = np.asarray(xdata, dtype=float)
  ydata = np.asarray(ydata, dtype=float)
  if min_n is None:
    min_n = max(3 if fit_type == "exp_plateau" else 2, len(xdata) // 10)

  best_R2 = -np.inf
  best = None
  guess = p0
  for n in range(min_n, len(xdata) + 1):
    xn, yn = xdata[:n], ydata[:n]
    try:
      result = fit_model(
          xn,
          yn,
          fit_type,
          p0=guess if guess is not None else
          (None if fit_type == "exp_plateau" else auto_guess(fit_type, xn, yn)))
    except RuntimeError:
      continue
    # Reinitialize plateau windows in their own normalized coordinates;
    # their physical amplitudes may be unrepresentable.
    guess = p0 if fit_type == "exp_plateau" else list(result.params)
    if result.R2 > best_R2:
      best_R2, best = result.R2, (result, n)
  if best is None:
    raise RuntimeError(
        "fit_best_window: curve_fit failed to converge for every window in "
        f"[{min_n:d}, {len(xdata):d}]")
  return best


def fit_best_window(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fit_type: str = "exp2",
    min_n: int | None = None,
    p0: list | None = None) -> tuple[np.ndarray, np.ndarray, float, int]:
  """Return original-coordinate (params, covariance, R2, N) for the best window.

  See :func:`fit_best_model` for window selection and stable evaluation.
  """
  result, n = fit_best_model(xdata, ydata, fit_type, min_n, p0)
  params, cov = fit_coefficients(result, fit_type)
  return params, cov, result.R2, n
