"""The ``magsq`` verb -- magnitude squared of a vector field."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import numerics

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def magsq(data: "GDataState",
          *,
          coords: str = "0:3",
          inplace: bool = False,
          tag: str | None = None,
          label: str | None = None):
  """Magnitude squared of a vector field.

  Sums the squares of the selected components (``numerics.mag_sq``),
  returning a single-component field.

  Args:
    data: the dataset holding the vector field; must be NumPy-backed.
    coords: ``"start:end"`` slice of the component axis to sum the squares
      of. Defaults to the first three components.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the magnitude squared.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  if data.backend == "gkyl":
    raise ValueError(
        "magsq operates on interpolated (NumPy) values; call .interpolate() "
        "first -- summing squares of raw DG coefficients would mix basis functions."
    )
  grid, values = numerics.mag_sq(data.grid, data.values, coords=coords)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
