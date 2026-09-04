"""The ``relchange`` verb -- relative change between two datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import numerics

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def _require_field_domain(data: "GDataState", who: str) -> None:
  if data.backend == "gkyl":
    raise ValueError(
        f"relchange operates on interpolated (NumPy) values; call .interpolate() "
        f"first on {who} -- dividing raw DG coefficients would mix basis functions."
    )


def relchange(data0: "GDataState",
              data: "GDataState",
              *,
              comp: int | str | None = None,
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
    ValueError: if either operand is native modal (gkyl-backed).
  """
  _require_field_domain(data0, "'data0'")
  _require_field_domain(data, "'data'")
  grid, values = numerics.rel_change(data.grid, data0.values, data.values, comp)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
