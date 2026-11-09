"""Measure growth or oscillation frequency from one-dimensional time traces.

Apply optional time normalization and window selection, fit an exponential or
sinusoid, and return the reconstructed curve. Frequency can also use a temporal
FFT and return its amplitude spectrum.
"""

from __future__ import annotations

import numpy as np

from postgkyl import numerics
from postgkyl.gdatastate import materialize_point_values
from postgkyl.gdatastate.guards import require_field_domain
from postgkyl.gdatastate.gdatastate import GDataState

from .fit import fit
from ._normalization import scale


def _trace(data: GDataState):
  require_field_domain(data, "rate", "raw modal coefficients are not a trace")
  data = materialize_point_values(data)
  if data.num_dims != 1:
    raise ValueError("growth/frequency requires a one-dimensional time trace")
  x = data.grid[0]
  if len(x) == len(data.values) + 1:
    x = (x[1:] + x[:-1]) / 2
  elif len(x) != len(data.values):
    raise ValueError("time coordinates do not match trace samples")
  x = np.asarray(x, dtype=float)
  if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
    raise ValueError("time coordinates must be finite and increasing")
  return data, x


def _selected_times(x, time_window):
  if time_window is None:
    return np.ones(len(x), dtype=bool)
  lo, hi = time_window
  if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
    raise ValueError("time_window needs increasing finite bounds")
  selected = (x >= lo) & (x <= hi)
  if selected.sum() < 3:
    raise ValueError("time_window contains fewer than three samples")
  return selected


def fitted_trace(data,
                 model,
                 *,
                 guess=None,
                 min_n=None,
                 time_norm=None,
                 time_window=None,
                 inplace=False,
                 tag=None,
                 label=None):
  source, x = _trace(data)
  time_scale = scale([source], time_norm, "time_norm")
  normalized = x / time_scale
  chosen = _selected_times(normalized, time_window)
  if model == "sinusoid" and chosen.sum() < 6:
    raise ValueError("frequency fit requires at least six time samples")
  temporary = GDataState(ctx={
      "interpolated": True
  }).push([normalized[chosen]], source.values[chosen])
  minimum = (max(6,
                 len(normalized) //
                 10) if model == "sinusoid" and min_n is None else min_n)
  if model == "sinusoid" and minimum is not None and minimum < 6:
    raise ValueError("frequency fit requires min_n >= 6")
  result = fit(temporary,
               model,
               guess=guess,
               window=time_window is None,
               min_n=minimum)
  predicted = np.column_stack([
      numerics.fit_evaluate(normalized, model, fit_model)
      for fit_model in result.ctx["fit_models"]
  ])
  updates = {
      key: result.ctx[key]
      for key in ("fit_models", "fit_params", "fit_std", "fit_R2")
  }
  updates["time_norm"] = time_scale
  if time_window is not None:
    updates["fit_time_window"] = (float(normalized[chosen][0]),
                                  float(normalized[chosen][-1]))
  return data._result(data.grid,
                      predicted,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      **updates)


def temporal_fft(data,
                 *,
                 time_norm=None,
                 time_window=None,
                 tag=None,
                 label=None):
  source, x = _trace(data)
  time_scale = scale([source], time_norm, "time_norm")
  normalized = x / time_scale
  chosen = _selected_times(normalized, time_window)
  times = normalized[chosen]
  if len(times) < 3 or not np.allclose(
      np.diff(times), np.diff(times)[0], rtol=1e-8, atol=0):
    raise ValueError("frequency --fft requires uniform time samples")
  values = source.values[chosen]
  omega = 2 * np.pi * np.fft.rfftfreq(len(times), d=times[1] - times[0])
  amplitudes = np.abs(np.fft.rfft(values, axis=0)) / len(times)
  if len(omega) < 2:
    raise ValueError("frequency --fft needs a positive frequency bin")
  peaks = np.argmax(amplitudes[1:], axis=0) + 1
  measured = omega[peaks]
  out = data._result([omega[1:]],
                     amplitudes[1:],
                     tag=tag,
                     label=label or "temporal frequency spectrum",
                     interpolated=True,
                     value_form=None,
                     dominant_frequency=measured.tolist(),
                     time_norm=time_scale)
  for key in ("time", "frame", "basis_type", "poly_order"):
    out.ctx.pop(key, None)
  return out, measured
