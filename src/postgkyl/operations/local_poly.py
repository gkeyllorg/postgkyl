"""The ``local_poly`` verb -- modal DG coefficients -> a discontinuity-
preserving plotting mesh (see ``dg.local_poly``)."""

from __future__ import annotations

from postgkyl import dg
from postgkyl.gdatastate.layout import require_dg_layout

from postgkyl.gdatastate.gdatastate import GDataState


def local_poly(data: "GDataState",
               *,
               npoints: int = 2,
               inplace: bool = False,
               tag: str | None = None,
               label: str | None = None):
  """Evaluate the DG polynomial cell-by-cell onto a plotting mesh that keeps
  every inter-cell discontinuity visible, instead of the continuous refined
  mesh ``interpolate`` produces.

  ``npoints`` reference points span the whole cell (``[-1, 1]``, endpoints
  included) and a NaN is spliced in at every cell interface, so a plot breaks
  the curve there rather than drawing a spuriously smooth line across it.

  Basis, polynomial order, and value_form are properties of ``data`` itself,
  fixed at load time, same as ``interpolate``. The result is flagged
  ``interpolated=True`` so it becomes safe for element-wise math.

  Args:
    data: Dataset containing modal coefficients or cell-local DG nodal values.
    npoints: Evaluation points in each cell and direction.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.
  """
  if not data.ctx.get("basis_type"):
    raise ValueError(
        "dataset has no 'basis_type' metadata; set it at load time")
  layout = require_dg_layout(data)
  if layout.value_form == "quad":
    raise ValueError("quadrature input requires .represent(to='modal') first")

  grid, values = dg.local_poly(data.values,
                               data.grid,
                               poly_order=layout.poly_order,
                               basis_type=layout.basis_type,
                               nodal=(layout.value_form == "nodal"),
                               npoints=npoints,
                               **layout.basis_kwargs)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True)
