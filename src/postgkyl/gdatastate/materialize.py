"""Unpack DG point locations and physical fields into ordinary NumPy state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl.dg import interpolate, rep
from .layout import dg_layout

if TYPE_CHECKING:
  from .gdatastate import GDataState


def materialize_point_values(data: "GDataState",
                             *,
                             inplace: bool = False) -> "GDataState":
  """Return ordinary NumPy point values with their actual coordinates.

  Already unpacked values pass through. Packed nodal/quad fields are scattered
  onto their tensor point grid and cease to carry a packed value_form. Gkeyll
  stores 1-D surface values directly at the faces, so they become point values
  without a basis transform. Other modal coefficients require an explicit
  evaluation choice.
  """
  layout = dg_layout(data)
  if layout is None:
    data._require_operable()
    return data
  if layout.surface_dir is not None and layout.ndim == 1:
    grid, values = interpolate(data.values,
                               data.grid,
                               poly_order=layout.poly_order,
                               basis_type=layout.basis_type,
                               surface_dir=layout.surface_dir,
                               **layout.basis_kwargs)
  else:
    data._require_operable()
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
