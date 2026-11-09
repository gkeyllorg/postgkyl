"""Measure growth and phase frequency across spatial Fourier modes.

Sample each timed field frame, Fourier transform one spatial direction, and
fit mode amplitudes or phases over a shared time window. The resulting dataset
contains one rate per positive wavenumber.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from postgkyl import numerics
from postgkyl.cli_spec import CliType, CommaTuple
from postgkyl.gdatastate import flatten_datasets, materialize_point_values
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.gdatastate.layout import dg_layout

from .interpolate import interpolate
from ._normalization import scale


def _samples(frame, interpolated):
  if frame.ctx.get("value_form") == "modal" and not interpolated:
    try:
      layout = dg_layout(frame)
    except RuntimeError as exc:
      if "bridge is unavailable" not in str(exc):
        raise
    else:
      if layout.num_fields != 1 or layout.surface_dir is not None:
        raise ValueError("growth spectrum: select one volume scalar field")
    # The constant basis function is 2**(-ndim/2), independently of p.
    return frame.values[..., 0] / 2.0**(frame.num_dims / 2.0)
  layout = dg_layout(frame)
  if layout is not None and layout.value_form == "modal":
    frame = interpolate(frame)
  else:
    frame = materialize_point_values(frame)
  if frame.values.shape[-1] != 1:
    raise ValueError("growth spectrum: select one scalar field")
  return frame.values[..., 0]


def mode_spectrum(
    *datasets: "GDataState",
    dir: int,
    quantity: str,
    time_norm: Annotated[float | str | None,
                         CliType(str | None)] = None,
    time_window: Annotated[tuple[float, float] | None,
                           CommaTuple()] = None,
    space_norm: Annotated[float | str | None,
                          CliType(str | None)] = None,
    interpolated: bool = False,
    tag: str | None = None,
    label: str | None = None,
) -> "GDataState":
  """Return one measured rate versus positive wavenumber for scalar frames.

  Args:
    *datasets: Frames with matching Cartesian grids and distinct time metadata.
    dir: Zero-based spatial direction to Fourier transform.
    time_norm: Positive scale or metadata expression; multiplies rates.
    time_window: Inclusive pair of times, normalized when time_norm is set.
    space_norm: Positive scale or metadata expression; multiplies k.
    interpolated: Evaluate modal fields at interpolation points before FFT.
    tag: Optional tag for the resulting spectrum.
    label: Optional label for the resulting spectrum.
  """
  frames = flatten_datasets(datasets)
  if len(frames) < 3:
    raise ValueError("growth spectrum: at least three frames are required")
  if any("time" not in frame.ctx for frame in frames):
    raise ValueError("growth spectrum: every frame needs time metadata")
  frames.sort(key=lambda frame: frame.ctx["time"])
  times = np.array([frame.ctx["time"] for frame in frames], dtype=float)
  if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
    raise ValueError("growth spectrum: frame times must be finite and distinct")
  first = frames[0]
  if dir < 0 or dir >= first.num_dims:
    raise ValueError(
        "growth spectrum: direction is outside the field dimensions")
  edges = first.grid[dir]
  if edges.ndim != 1 or len(edges) < 3 or not np.allclose(
      edges, np.linspace(edges[0], edges[-1], len(edges)), rtol=0, atol=1e-12):
    raise ValueError("growth spectrum: FFT direction needs a uniform grid")
  if edges[-1] <= edges[0]:
    raise ValueError("growth spectrum: FFT domain length must be positive")
  for frame in frames[1:]:
    if frame.num_dims != first.num_dims or not numerics.grids_compatible(
        frame.grid, first.grid):
      raise ValueError("growth spectrum: frames must have matching grids")
  tscale = scale(frames, time_norm, "time_norm")
  kscale = scale(frames, space_norm, "space_norm")
  t = times / tscale
  amplitudes, phases = [], []
  shape = None
  for frame in frames:
    samples = _samples(frame, interpolated)
    if not interpolated and len(frame.grid[dir]) != samples.shape[dir] + 1:
      raise ValueError("growth spectrum: FFT axis needs cell-edge coordinates")
    if shape is not None and samples.shape != shape:
      raise ValueError("growth spectrum: frames have incompatible samples")
    shape = samples.shape
    spectrum = np.fft.rfft(samples, axis=dir) / samples.shape[dir]
    other_axes = tuple(axis for axis in range(samples.ndim) if axis != dir)
    amplitudes.append(np.sqrt(np.mean(np.abs(spectrum)**2, axis=other_axes)))
    if quantity == "frequency":
      moved = np.moveaxis(spectrum, dir, 0)
      phases.append(
          [np.angle(mode.flat[np.argmax(np.abs(mode))]) for mode in moved])
  amp = np.asarray(amplitudes)
  if amp.shape[1] < 2:
    raise ValueError(
        "growth spectrum: FFT direction needs at least two samples")
  if time_window is None:
    dominant = 1 + np.argmax(amp[-1, 1:])
    _, count = numerics.fit_best_model(t,
                                       np.log(
                                           np.maximum(amp[:, dominant],
                                                      1e-300)),
                                       "linear",
                                       min_n=max(3,
                                                 len(t) // 10))
    chosen = np.arange(len(t)) < count
  else:
    lo, hi = time_window
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
      raise ValueError(
          "growth spectrum: time_window needs increasing finite bounds")
    chosen = (t >= lo) & (t <= hi)
    if chosen.sum() < 3:
      raise ValueError(
          "growth spectrum: time_window contains fewer than three frames")
  if quantity == "growth":
    rates = np.polyfit(t[chosen], np.log(np.maximum(amp[chosen], 1e-300)), 1)[0]
  elif quantity == "frequency":
    phase = np.unwrap(np.asarray(phases), axis=0)
    rates = -np.polyfit(t[chosen], phase[chosen], 1)[0]
  else:
    raise ValueError(f"unknown mode-spectrum quantity {quantity!r}")
  k = 2 * np.pi * np.arange(1, amp.shape[1]) / (edges[-1] - edges[0]) * kscale
  result = first._result([k],
                         rates[1:, None],
                         tag=tag,
                         label=label or f"{quantity} spectrum",
                         interpolated=True,
                         value_form=None,
                         growth_window=(float(t[chosen][0]),
                                        float(t[chosen][-1])))
  for key in ("time", "frame", "cells", "basis_type", "poly_order",
              "num_comps"):
    result.ctx.pop(key, None)
  return result
