"""Growth rates from time traces or spatial Fourier modes."""

from __future__ import annotations

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.gdatastate import flatten_datasets
from postgkyl.cli_spec import CliType, CommaTuple
from typing import Annotated

from ._rate_trace import fitted_trace
from .fit import fit
from ._mode_spectrum import mode_spectrum


def growth(*datasets: GDataState,
           spectrum: bool = False,
           dir: int | None = None,
           tag: str | None = None,
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
  """Print growth rates and return fitted curves, or a spatial rate spectrum.

  Args:
    *datasets: One-dimensional time traces, or field frames for a spectrum.
    spectrum: Compute growth rate versus positive spatial wavenumber.
    dir: Zero-based FFT direction, required with spectrum.
    tag: Optional tag for resulting dataset(s).
    guess: Initial ``amplitude,rate`` parameter guess.
    min_n: Minimum number of points in the fitted leading window.
    time_norm: Positive time scale, metadata name, or arithmetic expression.
    time_window: Inclusive normalized or unnormalized time bounds.
    space_norm: Positive length scale, metadata name, or arithmetic expression.
    interpolated: Interpolate modal fields before the spatial FFT.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    label: Optional label for the fitted curve.

  Returns:
    The fitted curve; rate, uncertainty, and R-squared are in its fit context.
  """
  states = flatten_datasets(datasets)
  if not states:
    raise ValueError("growth requires at least one dataset")
  tag = "growth" if tag is None else tag
  if spectrum:
    if dir is None:
      raise ValueError("growth --spectrum requires --dir")
    if guess is not None or min_n is not None or inplace:
      raise ValueError("guess, min_n and inplace apply only to plain growth")
    return mode_spectrum(*states,
                         dir=dir,
                         quantity="growth",
                         time_norm=time_norm,
                         time_window=time_window,
                         space_norm=space_norm,
                         interpolated=interpolated,
                         tag=tag,
                         label=label)
  if dir is not None or space_norm is not None or interpolated:
    raise ValueError("dir, space_norm and interpolated require --spectrum")
  results = []
  for data in states:
    if time_norm is None and time_window is None:
      result = fit(data,
                   "exp2",
                   guess=guess,
                   window=True,
                   min_n=min_n,
                   inplace=inplace,
                   tag=tag,
                   label=label)
    else:
      result = fitted_trace(data,
                            "exp2",
                            guess=guess,
                            min_n=min_n,
                            time_norm=time_norm,
                            time_window=time_window,
                            inplace=inplace,
                            tag=tag,
                            label=label)
    if isinstance(result, GDataState):
      for component, params in enumerate(result.ctx["fit_params"]):
        print(f"growth component {component}: {params[1]:.12g}")
    results.append(result)
  return results[0] if len(results) == 1 else results


__all__ = ["growth"]
