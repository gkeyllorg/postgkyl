"""The ``fit`` verb -- fit a model to data and return the fitted curve.

The result holds the fitted values on the data's grid; the per-component fit
parameters, 1-sigma uncertainties, and R^2 are stored in
``ctx['fit_params']``, ``ctx['fit_std']``, and ``ctx['fit_R2']``. ``fit_type``
is a model name (e.g. ``'linear'``, ``'gaussian'``, ``'exp2'`` for
growth-rate fits) or an RPN expression -- see :mod:`postgkyl.numerics.fit`.

``window=True`` restricts each component's fit to its best-scoring leading
window rather than the full domain -- the growth-rate use case, where only
a continuously growing/decaying leading region of a longer time series
should be fit (e.g. ``fit(d, 'exp2', window=True)``); see
:func:`postgkyl.numerics.fit_best_window`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from postgkyl import numerics

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def fit(data: "GDataState",
        fit_type: str,
        *,
        guess=None,
        window: bool = False,
        min_n: int | None = None,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None):
  """Fit a model to data and return the fitted curve.

  Fits the model named (or expressed) by ``fit_type`` to each component of
  ``data`` independently and returns the fitted values evaluated on the
  data's (cell-centered) grid. Axes collapsed to a single cell (e.g. after
  ``integrate`` or ``select``) are dropped, so 1D and 2D fits are supported.

  Args:
    data: the dataset to fit; must be NumPy-backed. Its grid provides the
      independent variable(s) and each component is fit separately.
    fit_type: the model to fit -- a key of ``numerics.FIT_FUNCTIONS``
      ('linear', 'quadratic', 'plane', 'quadratic2d', 'exp_plateau',
      'gaussian', 'power', 'sinusoid', 'tanh_transition', 'exp2'), or a
      custom RPN expression string (e.g. ``'x a * b +'``) whose free tokens
      (not the spatial variables 'x'/'y', operators, or numbers) become fit
      parameters.
    guess: initial guess for the fit parameters -- a comma-separated string
      (e.g. ``'1,0,2'``) or a sequence of floats. None derives a
      data-driven guess per component via ``numerics.auto_guess`` (for the
      first window, if ``window=True``).
    window: fit only the best-scoring leading window of the data (1D only)
      instead of the full domain -- see ``numerics.fit_best_window``.
    min_n: minimum window length when ``window=True``; ``None`` defaults to
      one tenth of the number of samples. Ignored otherwise.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset holding the fitted curve on the active grid, with
    ``ctx['fit_params']``, ``ctx['fit_std']``, and ``ctx['fit_R2']`` set.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed), if ``fit_type``
      is neither a recognized model name nor a valid RPN expression, if the
      data's active dimensionality does not match the model's, or if
      ``window=True`` and the data is not 1D.
  """
  if data.backend == "gkyl":
    raise ValueError(
        "fit operates on interpolated (NumPy) values; call .interpolate() first "
        "-- fitting raw DG coefficients would mix basis functions.")
  grid = data.grid
  values = data.values
  spatial_shape = values.shape[:-1]

  if any(grid[d].shape[0] == spatial_shape[d] + 1 for d in range(len(grid))):
    cc_grid = numerics.nodal_to_cell_centered_grid(grid, spatial_shape)
  else:
    cc_grid = list(grid)

  # Drop dimensions collapsed to a single cell (e.g. after integrate/select).
  active = [d for d in range(len(cc_grid)) if cc_grid[d].shape[0] > 1]
  if len(active) < len(cc_grid):
    idx = tuple(
        slice(None) if d in active else 0
        for d in range(len(spatial_shape))) + (slice(None), )
    cc_grid = [cc_grid[d] for d in active]
    values = values[idx]

  ndim_fit = numerics.FIT_NDIM.get(fit_type, numerics.rpn_ndim(fit_type))
  if len(cc_grid) != ndim_fit:
    raise ValueError(
        f"fit '{fit_type}' requires {ndim_fit:d} spatial dimension(s), but "
        f"data has {len(cc_grid):d}. Reduce it first (e.g. select or integrate)."
    )
  if window and len(cc_grid) != 1:
    raise ValueError(
        "fit: window=True is only supported for 1D (time-series-like) data, "
        f"but data has {len(cc_grid):d} active dimension(s).")

  if len(cc_grid) == 1:
    xdata = cc_grid[0]
  else:
    mesh = np.meshgrid(cc_grid[0], cc_grid[1], indexing="ij")
    xdata = np.array([mesh[0].flatten(), mesh[1].flatten()])

  guess_list = None
  if guess is not None:
    guess_list = ([float(v) for v in guess.split(",")] if isinstance(
        guess, str) else list(guess))

  active_shape = tuple(cg.shape[0] for cg in cc_grid)
  fit_values_list, all_params, all_std, all_r2 = [], [], [], []
  for comp in range(values.shape[-1]):
    ydata = values[..., comp].flatten()
    if window:
      params, cov, r2, _n = numerics.fit_best_window(xdata,
                                                     ydata,
                                                     fit_type,
                                                     min_n=min_n,
                                                     p0=guess_list)
    else:
      p0 = guess_list if guess_list is not None else numerics.auto_guess(
          fit_type, xdata, ydata)
      params, cov, r2 = numerics.fit(xdata, ydata, fit_type, p0=p0)
    y_fit = numerics.fit_evaluate(xdata, fit_type, params)
    fit_values_list.append(y_fit.reshape(active_shape + (1, )))
    all_params.append(params)
    all_std.append(np.sqrt(np.diag(cov)))
    all_r2.append(r2)

  fit_values = np.concatenate(fit_values_list, axis=-1)
  fit_grid = [grid[d] for d in active]
  return data._result(fit_grid,
                      fit_values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      fit_params=all_params,
                      fit_std=all_std,
                      fit_R2=all_r2)
