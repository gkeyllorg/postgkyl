"""Convenience composition for exponential growth-rate fits."""

from __future__ import annotations

from postgkyl.gdatastate.gdatastate import GDataState

from .fit import fit


def growth(data: GDataState,
           *,
           guess: str | None = None,
           min_n: int | None = None,
           inplace: bool = False,
           tag: str | None = None,
           label: str | None = None) -> GDataState:
  """Fit exponential growth on the best leading data window.

  Args:
    data: One-dimensional point-value dataset to fit.
    guess: Initial ``amplitude,rate`` parameter guess.
    min_n: Minimum number of points in the fitted leading window.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the fitted curve.
    label: Optional label for the fitted curve.

  Returns:
    The fitted curve; rate, uncertainty, and R-squared are in its fit context.
  """
  return fit(data,
             "exp2",
             guess=guess,
             window=True,
             min_n=min_n,
             inplace=inplace,
             tag=tag,
             label=label)


__all__ = ["growth"]
