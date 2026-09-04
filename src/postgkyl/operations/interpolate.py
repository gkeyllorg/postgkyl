"""The ``interpolate`` verb -- DG coefficients -> values on a uniform mesh."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import dg

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


def interpolate(data: "GDataState", *, num_interp: int | None = None,
    inplace: bool = False, tag: str | None = None, label: str | None = None):
  """Interpolate DG (modal/nodal) data onto a uniform evaluation mesh.

  Basis, polynomial order, and value_form are properties of ``data`` itself,
  fixed at load time (``pg.load(..., basis_type=..., poly_order=...,
  value_form=...)`` or the CLI's ``-b``/``-p``/``-v``) -- this verb only
  ever reads them off ``data.ctx``. The result is flagged
  ``interpolated=True`` so it becomes safe for element-wise math.

  Args:
    data: Dataset containing DG coefficients.
    num_interp: Evaluation points per cell; use the basis default when omitted.
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
        "(pg.load(..., poly_order=...) or the CLI's -p/--poly_order).")
  # end

  value_form = data.ctx.get("value_form", "modal")
  if data.backend == "gkyl" and value_form != "modal":
    raise ValueError(
        f"interpolate expects the modal value_form, not "
        f"'{value_form}'; call .to_modal() first.")
  # end

  grid, values = dg.interpolate(data.values, data.grid, poly_order=poly_order,
      basis_type=basis_type, modal=(value_form == "modal"), num_interp=num_interp)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label,
      interpolated=True)
# end
