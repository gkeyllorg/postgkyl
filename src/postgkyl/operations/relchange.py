"""The ``relchange`` verb -- relative change between two datasets."""

from __future__ import annotations

from typing import Annotated
from postgkyl.cli_spec import CliType, DatasetRef

from postgkyl import numerics
from postgkyl.gdatastate import materialize_point_values
from postgkyl.gdatastate.guards import require_field_domain

from postgkyl.gdatastate.gdatastate import GDataState


def relchange(data0: Annotated[GDataState, DatasetRef()],
              data: Annotated[GDataState, DatasetRef()],
              *,
              comp: Annotated[int | str | None,
                              CliType(str | None)] = None,
              inplace: bool = False,
              tag: str | None = None,
              label: str | None = None):
  """Relative change of ``data`` with respect to the baseline ``data0``.

  Computes ``(data - data0) / data0`` component-wise (``numerics.rel_change``).
  Both datasets are assumed to share the same grid and component layout.

  Args:
    data0: the baseline ("before") dataset -- the denominator.
    data: the dataset whose relative change is computed; the returned
      dataset is built from this one (its grid/ctx are the base of the
      result).
    comp: when given, every numerator component is divided by this single
      baseline component instead of its own (e.g. normalize every energy
      component by the total energy component). None divides component-wise.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset of the relative change, built from ``data``.

  Raises:
    ValueError: if either operand is unevaluated modal coefficients.
  """
  require_field_domain(data0, "relchange",
                       "raw coefficients are not field values")
  require_field_domain(data, "relchange",
                       "raw coefficients are not field values")
  data0 = materialize_point_values(data0)
  data = materialize_point_values(data, inplace=inplace)
  grid, values = numerics.rel_change(data.grid, data0.values, data.values, comp)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
