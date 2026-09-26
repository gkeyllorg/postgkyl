"""Evaluate DG fields on a refined mesh within their original cell edges."""

from __future__ import annotations

from postgkyl import dg
from postgkyl.gdatastate.layout import require_dg_layout

from postgkyl.gdatastate.gdatastate import GDataState


def interpolate(data: "GDataState",
                *,
                num_interp: int | None = None,
                inplace: bool = False,
                tag: str | None = None,
                label: str | None = None):
  """Interpolate DG (modal/nodal) data at uniformly spaced points in each cell.

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
  if not data.ctx.get("basis_type"):
    raise ValueError(
        "dataset has no 'basis_type' metadata; set it at load time")
  layout = require_dg_layout(data)
  if layout.value_form == "quad":
    raise ValueError("quadrature input requires .represent(to='modal') first")

  grid, values = dg.interpolate(data.values,
                                data.grid,
                                poly_order=layout.poly_order,
                                basis_type=layout.basis_type,
                                nodal=(layout.value_form == "nodal"),
                                num_interp=num_interp,
                                **layout.basis_kwargs)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True)
