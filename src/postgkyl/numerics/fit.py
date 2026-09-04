"""Curve fitting: built-in model functions (including ``exp2``, the
growth-rate model), an RPN custom-model parser, ``scipy.optimize.curve_fit``
wrappers, and the leading-window search used for growth-rate-style fits."""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.optimize as opt


def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
  """Linear model ``a*x + b``."""
  return a * x + b


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
  """Quadratic model ``a*x**2 + b*x + c``."""
  return a * x**2 + b * x + c


def plane(XY: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
  """Planar model ``a*x + b*y + c`` over two independent variables packed
  as ``(x, y)`` (e.g. shape ``(2, N)``)."""
  x, y = XY
  return a * x + b * y + c


def quadratic2d(XY: np.ndarray, a: float, b: float, c: float, d: float,
                e: float, f: float) -> np.ndarray:
  """``a*x^2 + b*y^2 + c*x*y + d*x + e*y + f``."""
  x, y = XY
  return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f


def exp_plateau(x: np.ndarray, A: float, b: float, C: float) -> np.ndarray:
  """``A*exp(b*x) + C`` (plateaus at ``C`` as ``b*x -> -inf``, or at
  ``A+C`` as ``b*x -> +inf``)."""
  return A * np.exp(b * x) + C


def gaussian(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
  """``A * exp(-0.5 * ((x - mu) / sigma)**2)``."""
  return A * np.exp(-0.5 * ((x - mu) / sigma)**2)


def power(x: np.ndarray, a: float, n: float, b: float) -> np.ndarray:
  """``a * x^n + b``."""
  return a * x**n + b


def sinusoid(x: np.ndarray, A: float, omega: float, phi: float,
             C: float) -> np.ndarray:
  """``A * sin(omega * x + phi) + C``."""
  return A * np.sin(omega * x + phi) + C


def tanh_transition(x: np.ndarray, A: float, x0: float, w: float,
                    C: float) -> np.ndarray:
  """``A * tanh((x - x0) / w) + C``."""
  return A * np.tanh((x - x0) / w) + C


def exp2(x: np.ndarray, a: float, b: float) -> np.ndarray:
  """``a * exp(2*b*x)`` -- the growth-rate model.

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


def fit_evaluate(xdata: np.ndarray, fit_type: str,
                 params: np.ndarray) -> np.ndarray:
  """Evaluate a fitted model at ``xdata`` given the optimized parameters."""
  if fit_type in FIT_FUNCTIONS:
    return FIT_FUNCTIONS[fit_type](xdata, *params)
  return _rpn_make_func(fit_type)(xdata, *params)


def fit(xdata: np.ndarray,
        ydata: np.ndarray,
        fit_type: str = "linear",
        p0: list | None = None) -> tuple[np.ndarray, np.ndarray, float]:
  """Fit data using ``scipy.optimize.curve_fit`` with the specified model.

  Args:
    xdata: For 1D fits, shape ``(N,)``. For 2D fits, shape ``(2, N)`` where
      rows are the two independent variables flattened.
    ydata: Dependent variable, shape ``(N,)``.
    fit_type: A key in :data:`FIT_FUNCTIONS`, or an RPN expression string
      (e.g. ``"a x * b +"``).
    p0: Initial guess for the fit parameters; defaults to all ones.

  Returns:
    ``(params, cov, R2)``.

  Raises:
    ValueError: If ``fit_type`` is neither a known model name nor a
      recognizable RPN expression.
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

  if p0 is None:
    p0 = np.ones(n_params)

  params, cov = opt.curve_fit(func, xdata, ydata, p0=p0)

  residual = ydata - func(xdata, *params)
  ss_res = np.sum(residual**2)
  ss_tot = np.sum((ydata - np.mean(ydata))**2)
  R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

  return params, cov, R2


def auto_guess(fit_type: str, xdata: np.ndarray,
               ydata: np.ndarray) -> list | None:
  """Return data-driven initial parameter guesses for known fit types.

  Produces a sensible ``p0`` for :func:`fit` by inspecting the data (e.g. a
  least-squares seed for linear/polynomial models, peak location and FWHM
  for a gaussian, the dominant FFT frequency for a sinusoid). Returns
  ``None`` for RPN expressions or when the data has no finite values, in
  which case :func:`fit` falls back to its default (ones).

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
    A = float(y_max - C) or 1.0
    x_span = x.max() - x.min()
    b = -1.0 / x_span if x_span > 0 else -1.0
    return [A, b, C]

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


def fit_best_window(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fit_type: str = "exp2",
    min_n: int | None = None,
    p0: list | None = None) -> tuple[np.ndarray, np.ndarray, float, int]:
  """Fit ``fit_type`` to the best-scoring leading window of a 1D series.

  Scans windows ``xdata[:n]`` for ``n`` from ``min_n`` up to ``len(xdata)``,
  keeping the window with the best coefficient of determination (R^2). Each
  window is warm-started from the previous window's fitted parameters (or
  ``p0``/:func:`auto_guess` for the first), so this generalizes a single
  full-domain :func:`fit` call to the common case of a time series whose
  early or late region should be excluded (e.g. growth-rate fits, which are
  only valid while the signal grows/decays continuously).

  Args:
    xdata: 1D independent variable (e.g. time).
    ydata: dependent variable, shape matching ``xdata``.
    fit_type: passed to :func:`fit`.
    min_n: minimum number of points in the fitted window. Defaults to
      ``len(xdata) // 10``.
    p0: initial guess for the first window; ``None`` uses :func:`auto_guess`.

  Returns:
    ``(params, cov, R2, N)`` for the best-scoring window.

  Raises:
    RuntimeError: if ``curve_fit`` fails to converge for every window in
      the scan range.
  """
  xdata = np.asarray(xdata, dtype=float)
  ydata = np.asarray(ydata, dtype=float)
  if min_n is None:
    min_n = max(2, len(xdata) // 10)

  best_R2 = -np.inf
  best = None
  guess = p0
  for n in range(min_n, len(xdata) + 1):
    xn, yn = xdata[:n], ydata[:n]
    try:
      params, cov, R2 = fit(
          xn,
          yn,
          fit_type,
          p0=guess if guess is not None else auto_guess(fit_type, xn, yn))
    except RuntimeError:
      continue
    guess = list(params)
    if R2 > best_R2:
      best_R2, best = R2, (params, cov, R2, n)
  if best is None:
    raise RuntimeError(
        "fit_best_window: curve_fit failed to converge for every window in "
        f"[{min_n:d}, {len(xdata):d}]")
  return best
