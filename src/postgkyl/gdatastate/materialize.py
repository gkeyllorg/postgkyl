"""Materialize point-value state without changing its value form."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl.dg import rep

if TYPE_CHECKING:
  from .gdatastate import GDataState


def materialize_point_values(data: "GDataState") -> "GDataState":
  """Return a NumPy-backed view of nodal or quadrature point values.

  NumPy-backed data is already materialized. Native modal coefficients have
  no unique point-value interpretation and therefore require an explicit
  representation choice by the caller.
  """
  if data.backend != "gkyl":
    return data
  value_form = data.ctx.get("value_form", "modal")
  if value_form == "modal":
    raise ValueError(
        "modal DG coefficients are not plottable; choose explicitly: "
        ".interpolate() (uniform evaluation mesh), .to_nodal() or .to_quad() "
        "(plot at the basis/quadrature points).")
  grid, values = rep.materialize(str(data.ctx["basis_type"]), data.num_dims,
                                 int(data.ctx["poly_order"]),
                                 data.native, data.grid, value_form,
                                 data.ctx.get("num_quad"))
  return data._result(grid, values)


__all__ = ["materialize_point_values"]
