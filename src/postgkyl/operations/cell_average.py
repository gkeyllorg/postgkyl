"""The ``cell_average`` verb -- each cell's average of a modal DG field."""

from __future__ import annotations

from postgkyl import dg
from postgkyl.gdatastate.layout import require_dg_layout

from postgkyl.gdatastate.gdatastate import GDataState


def cell_average(data: "GDataState",
                 *,
                 inplace: bool = False,
                 tag: str | None = None,
                 label: str | None = None):
  """The average of each field over each cell: one value per cell.

  Every orthonormal basis function but the constant ``b_0 = 2^(-ndim/2)``
  integrates to zero over a cell, so the average is ``b_0`` times each
  field's coefficient 0, for any basis and polynomial order. (The cell-centre
  value, ``interpolate --num_interp 1``, equals it only for p = 1.)

  Args:
    data: Modal DG coefficients.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    Point values, one per cell and field, on the cells' grid; flagged
    ``interpolated`` so they are safe for element-wise math.

  Raises:
    ValueError: if ``data`` holds no modal volume coefficients.
  """
  layout = require_dg_layout(data)
  if layout.value_form != "modal" or layout.surface_dir is not None:
    raise ValueError(
        "cell_average needs modal volume coefficients; this "
        f"dataset holds {layout.value_form} values" +
        (" on a surface" if layout.surface_dir is not None else ""))
  values = dg.modal.cell_means(data.values, layout.ndim, layout.block_size)
  return data._result(data.grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True)
