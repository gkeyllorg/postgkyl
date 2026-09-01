"""The ``average`` verb -- weighted (or plain) average of a native DG field
over a subset of dimensions, via Gkeyll's ``gkyl_array_average``.

Terminal-adjacent (like ``represent``): unlike ``integrate`` (whose whole-grid
mode returns numbers), this produces a new, lower-dimensional dataset -- still
modal and gkyl-native -- so it composes with ``.to_nodal()``/``.interpolate()``/
further ``.average()`` calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from postgkyl import dg

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


def _native_basis(data: "GDataState", what: str):
  if data.backend != "gkyl":
    raise ValueError(
        f"average wraps gkyl_array_average and needs native modal data; "
        f"{what} is not available after .interpolate() or without the "
        "Gkeyll library.")
  # end
  if data.ctx.get("value_form", "modal") != "modal":
    raise ValueError(
        f"average expects the modal value_form, not "
        f"'{data.ctx['value_form']}' ({what}); call .to_modal() first.")
  # end
  basis_type = data.ctx.get("basis_type")
  poly_order = data.ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError(f"{what} has no basis_type/poly_order metadata")
  # end
  return str(basis_type), int(poly_order)
# end


def average(data: "GDataState", dims, *, weight: "GDataState | None" = None,
    inplace: bool = False, tag: str | None = None, label: str | None = None):
  """``int f w dx^dims / int w dx^dims`` over the directions in ``dims``.

  Args:
    data: gkyl-backed (native modal) dataset in the modal value_form.
    dims: iterable of 0-based direction indices to average over (e.g. the
      selected ``z0``-``z5`` flags at the CLI layer).
    weight: optional gkyl-backed dataset in the modal value_form, same
      ``num_dims``/``basis_type``/``poly_order`` as ``data`` and exactly one
      field (``gkyl_array_average`` takes no field-index argument) -- the
      plain average (dividing by volume) is computed when omitted.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A new dataset over the surviving dimensions -- or a single degenerate
    dimension (``grid=[0, 1]``) when every direction is averaged out,
    Gkeyll's own convention since there is no true 0-dimensional basis --
    still modal and gkyl-native.

  Raises:
    ValueError: ``data`` (or ``weight``) is NumPy-backed or non-modal, is
      missing basis metadata, or ``weight``'s grid/basis doesn't match
      ``data``'s.
  """
  basis_type, poly_order = _native_basis(data, "data")
  ndim = data.num_dims

  weight_native = None
  if weight is not None:
    w_basis_type, w_poly_order = _native_basis(weight, "weight")
    if weight.num_dims != ndim:
      raise ValueError(
          f"weight has {weight.num_dims} dims but the field has {ndim}")
    # end
    if w_basis_type != basis_type:
      raise ValueError(
          f"weight basis_type '{w_basis_type}' != field's '{basis_type}'")
    # end
    if w_poly_order != poly_order:
      raise ValueError(
          f"weight poly_order {w_poly_order} != field's {poly_order}")
    # end
    weight_native = weight.native
  # end

  grid = {
      "ndim": ndim,
      "lower": np.asarray(data.ctx["lower"]),
      "upper": np.asarray(data.ctx["upper"]),
      "cells": np.asarray(data.ctx["cells"]),
  }
  keep_dirs, cells_avg, out_native = dg.modal.average(grid, basis_type, ndim,
      poly_order, data.native, dims, weight=weight_native)

  if keep_dirs:
    new_grid = [np.asarray(data.grid[d]) for d in keep_dirs]
  # end
  else:
    new_grid = [np.array([0.0, 1.0])]
  # end

  return data._result(new_grid, out_native, inplace=inplace, tag=tag,
      label=label, cells=np.asarray(cells_avg))
# end
