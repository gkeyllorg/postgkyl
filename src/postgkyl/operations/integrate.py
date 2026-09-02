"""The ``integrate`` verb family -- grid integrals, two domains.

``integrate`` is a *terminal* verb (like ``info``): it returns numbers, not a
dataset. The integral runs entirely inside Gkeyll (``gkyl_array_integrate``)
on the native DG coefficients -- no interpolation involved, and exact for the
basis.

``integrate_axis`` is the NumPy trapezoidal counterpart (``postgkyl.tools.
calculus.integrate`` in the legacy tree, ported verbatim to ``numerics.
calculus.integrate`` and wired here): it collapses one or more axes of
point-value data and returns a new (reduced) dataset, like ``select``. It
never touches raw modal coefficients -- nodal/quad value_forms are
materialized to their true point locations first (the same bridge ``plot``
uses); modal data must be converted explicitly first.

A curvilinear axis -- part of a joint, non-separable ``.map(space="conf")``
block, whose grid arrays are multi-dimensional and carry no single 1-D
coordinate of their own -- has no meaningful per-axis trapezoidal width, so
it is reduced separately from the ordinary (separable) axes: the whole
block is collapsed at once via its physical cell volume (``numerics.
curvilinear.cell_volume``, the Jacobian-determinant change-of-variables
weight). Requesting only part of a curvilinear block raises -- holding the
rest of the block "fixed" while integrating one of its axes has no single
physical answer once the block's coordinates genuinely couple (see
``operations.select``'s analogous curvilinear guard).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from postgkyl import dg
from postgkyl.numerics import calculus, curvilinear

from ._curvilinear import curvilinear_blocks
from postgkyl.gdatastate import materialize_point_values

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


def integrate(data: "GDataState", *, op: str = "none"):
  """``int dx op(f)`` over the whole grid, one value per field component.

  Args:
    data: a gkyl-backed (native modal) dataset.
    op: ``"none"`` (plain integral), ``"abs"``, or ``"sq"``.

  Returns:
    A float for single-field data, else a ``(num_fields,)`` NumPy array.
  """
  if data.backend != "gkyl":
    raise ValueError(
        "integrate wraps gkyl_array_integrate and needs native modal data; "
        "it is not available after .interpolate() or without the Gkeyll library.")
  # end
  if data.ctx.get("value_form", "modal") != "modal":
    raise ValueError(
        f"integrate expects the modal value_form, not "
        f"'{data.ctx['value_form']}'; call .to_modal() first.")
  # end
  basis_type = data.ctx.get("basis_type")
  poly_order = data.ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError("dataset has no basis_type/poly_order metadata")
  # end
  grid = {
      "ndim": data.num_dims,
      "lower": np.asarray(data.ctx["lower"]),
      "upper": np.asarray(data.ctx["upper"]),
      "cells": np.asarray(data.ctx["cells"]),
  }
  result = dg.modal.integrate(grid, str(basis_type), int(poly_order),
      data.native, op=op)
  return float(result[0]) if result.size == 1 else result
# end


def integrate_axis(data: "GDataState", axis: int | tuple | str | None = None, *,
    inplace: bool = False, tag: str | None = None, label: str | None = None):
  """``int dz`` over one or more axes of point-value data (non-terminal).

  Args:
    data: point-value dataset -- already-interpolated (NumPy) data, or a
      native ``nodal``/``quad`` value_form (materialized to its true
      point locations first). Raw modal DG coefficients raise; convert
      explicitly first (``.interpolate()``, ``.to_nodal()``, ``.to_quad()``).
    axis: axis (or axes) to integrate over: an ``int``, a ``tuple`` of
      ``int``, a comma-separated string (``"0,1"``), a colon slice string
      (``"0:2"``), or ``None`` (integrate over every axis).
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A new dataset with the integrated axes collapsed to a single, grid-mean
    cell (shape retained, like ``select``). Always NumPy-backed, whatever the
    input's value_form (like ``.interpolate()``'s result): stamped
    ``interpolated=True`` and cleared of any stale ``value_form`` tag so
    ``info``/``repr`` don't keep describing collapsed values as "modal".

  Raises:
    ValueError: ``axis`` selects some but not all of a curvilinear
      (``.map(space="conf")``) block's dimensions.
  """
  data._require_operable()  # the one home for "is this point-value data"
  shadow = materialize_point_values(data)
  grid = list(shadow.grid)
  values = shadow.values
  axes = calculus.parse_axis(axis, len(grid))

  blocks = curvilinear_blocks(grid, data.ctx.get("mapped_axes", {}))
  requested = set(axes)
  curvilinear_runs = []
  handled = set()
  for off, dims in blocks.items():
    overlap = requested & set(dims)
    if not overlap:
      continue
    # end
    if overlap != set(dims):
      raise ValueError(
          f"integrate: axis/axes {sorted(overlap)} belong to a curvilinear "
          f"(mapped) block spanning dimensions {dims}; a partial reduction "
          "of the block has no single physical answer -- include every "
          "axis of the block together in the same call.")
    # end
    curvilinear_runs.append((off, dims))
    handled.update(dims)
  # end

  separable_axes = tuple(a for a in axes if a not in handled)
  if separable_axes:
    grid, values = calculus.integrate(grid, values, separable_axes)
  # end

  for _, dims in curvilinear_runs:
    m = len(dims)
    block_coords = [grid[d] for d in dims]
    vol = curvilinear.cell_volume(block_coords)
    vol = vol.reshape(vol.shape + (1,) * (values.ndim - m))
    moved = np.moveaxis(values, dims, range(m))
    reduced = np.sum(moved * vol, axis=tuple(range(m)), keepdims=True)
    values = np.moveaxis(reduced, range(m), dims)
    for d in dims:
      grid[d] = np.array([grid[d].mean()])
    # end
  # end

  return data._result(grid, values, inplace=inplace, tag=tag, label=label,
      interpolated=True, value_form=None)
# end
