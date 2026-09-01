"""The ``local_poly`` verb -- modal DG coefficients -> a discontinuity-
preserving plotting mesh (see ``dg.local_poly``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import dg

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


def local_poly(data: "GDataState", *, npoints: int = 2, inplace: bool = False,
    tag: str | None = None, label: str | None = None):
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
    data: Dataset containing modal DG coefficients.
    npoints: Evaluation points in each cell and direction.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.
  """
  basis_type = data.ctx.get("basis_type")
  if not basis_type:
    raise ValueError(
        "dataset has no 'basis_type' metadata; set it at load time "
        "(pg.load(..., basis_type=...) or the CLI's -b/--basis).")
  # end

  poly_order = data.ctx.get("poly_order")
  if poly_order is None:
    raise ValueError(
        "dataset has no 'poly_order' metadata; set it at load time "
        "(pg.load(..., poly_order=...) or the CLI's -p/--poly-order).")
  # end

  value_form = data.ctx.get("value_form", "modal")
  if data.backend == "gkyl" and value_form != "modal":
    raise ValueError(
        f"local_poly expects the modal value_form, not "
        f"'{value_form}'; call .to_modal() first.")
  # end

  grid, values = dg.local_poly(data.values, data.grid, poly_order=poly_order,
      basis_type=basis_type, modal=(value_form == "modal"), npoints=npoints)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label,
      interpolated=True)
# end
