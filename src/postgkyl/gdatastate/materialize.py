"""Unpack DG point locations and physical fields into ordinary NumPy state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl.dg import rep
from .layout import dg_layout

if TYPE_CHECKING:
  from .gdatastate import GDataState


def materialize_point_values(data: "GDataState",
                             *,
                             inplace: bool = False) -> "GDataState":
  """Return ordinary NumPy point values with their actual coordinates.

  Already unpacked values pass through. Packed nodal/quad fields are scattered
  onto their tensor point grid and cease to carry a packed value_form. Modal
  coefficients always require an explicit evaluation choice.
  """
  data._require_operable()
  layout = dg_layout(data)
  if layout is None:
    return data
  grid, values = rep.materialize(*layout.basis_args, data.values, data.grid,
                                 layout.value_form, layout.num_quad,
                                 **layout.basis_kwargs)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      interpolated=True,
                      value_form=None,
                      num_quad=None,
                      quad_rule=None)


__all__ = ["materialize_point_values"]
