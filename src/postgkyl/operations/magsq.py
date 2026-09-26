"""The ``magsq`` verb -- magnitude squared of a vector field."""

from __future__ import annotations

from postgkyl.gdatastate import materialize_point_values
from postgkyl.gdatastate.guards import require_field_domain

from postgkyl import numerics

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
    data: the dataset holding the vector field; must contain point values.
    coords: ``"start:end"`` slice of the component axis to sum the squares
      of. Defaults to the first three components.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the magnitude squared.

  Raises:
    ValueError: if ``data`` is unevaluated modal coefficients.
  """
  require_field_domain(data, "magsq", "raw coefficients are not field values")
  data = materialize_point_values(data, inplace=inplace)
  grid, values = numerics.mag_sq(data.grid, data.values, coords=coords)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
