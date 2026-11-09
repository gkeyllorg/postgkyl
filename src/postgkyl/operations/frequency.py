"""Oscillation frequencies from time traces or spatial Fourier modes."""

from __future__ import annotations

from typing import Annotated

from postgkyl.cli_spec import CliType, CommaTuple
from postgkyl.gdatastate import flatten_datasets
from postgkyl.gdatastate.gdatastate import GDataState

from ._rate_trace import fitted_trace, temporal_fft
from ._mode_spectrum import mode_spectrum


def frequency(*datasets: GDataState,
              spectrum: bool = False,
              dir: int | None = None,
              tag: str | None = None,
              fft: bool = False,
              guess: str | None = None,
              min_n: int | None = None,
              time_norm: Annotated[float | str | None,
                                   CliType(str | None)] = None,
              time_window: Annotated[tuple[float, float] | None,
                                     CommaTuple()] = None,
              space_norm: Annotated[float | str | None,
                                    CliType(str | None)] = None,
              interpolated: bool = False,
              inplace: bool = False,
              label: str | None = None) -> GDataState | list[GDataState]:
  """Print dominant angular frequencies and return curves or spectra.

  Args:
    *datasets: One-dimensional time traces, or field frames for a spectrum.
    spectrum: Measure frequency versus positive spatial wavenumber.
    dir: Zero-based FFT direction, required with spectrum.
    tag: Optional tag for resulting dataset(s).
    fft: Use a temporal FFT peak for plain time traces.
    guess: Initial sinusoid parameters ``amplitude,omega,phase,offset``.
    min_n: Minimum length of an automatic leading fit window.
    time_norm: Positive time scale, metadata name, or arithmetic expression.
    time_window: Inclusive normalized or unnormalized time bounds.
    space_norm: Positive length scale, metadata name, or arithmetic expression.
    interpolated: Interpolate modal fields before the spatial FFT.
    inplace: Mutate a plain fitted trace instead of creating one.
    label: Optional label for the returned dataset.
  """
  states = flatten_datasets(datasets)
  if not states:
    raise ValueError("frequency requires at least one dataset")
  tag = "frequency" if tag is None else tag
  if spectrum:
    if dir is None:
      raise ValueError("frequency --spectrum requires --dir")
    if guess is not None or min_n is not None or inplace:
      raise ValueError("guess, min_n and inplace apply only to plain frequency")
    return mode_spectrum(*states,
                         dir=dir,
                         quantity="frequency",
                         time_norm=time_norm,
                         time_window=time_window,
                         space_norm=space_norm,
                         interpolated=interpolated,
                         tag=tag,
                         label=label)
  if dir is not None or space_norm is not None or interpolated:
    raise ValueError("dir, space_norm and interpolated require --spectrum")
  if fft and (guess is not None or min_n is not None or inplace):
    raise ValueError("guess, min_n and inplace do not apply to --fft")
  results = []
  for data in states:
    if fft:
      result, measured = temporal_fft(data,
                                      time_norm=time_norm,
                                      time_window=time_window,
                                      tag=tag,
                                      label=label)
    else:
      result = fitted_trace(data,
                            "sinusoid",
                            guess=guess,
                            min_n=min_n,
                            time_norm=time_norm,
                            time_window=time_window,
                            inplace=inplace,
                            tag=tag,
                            label=label)
      measured = [abs(params[1]) for params in result.ctx["fit_params"]]
    for component, omega in enumerate(measured):
      print(f"frequency component {component}: {omega:.12g}")
    results.append(result)
  return results[0] if len(results) == 1 else results


__all__ = ["frequency"]
