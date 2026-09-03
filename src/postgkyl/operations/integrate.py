"""Full and partial grid integration under one ``integrate`` verb.

The return shape states which operation was requested:

* integrating every spatial direction is terminal and returns one number per
  field;
* integrating a strict subset returns a dataset over the surviving
  directions.

Modal data never leave Gkeyll. Full integration uses
``gkyl_array_integrate`` directly. Partial integration uses
``gkyl_array_average`` and scales its modal result by the physical volume of
the removed directions, which is exactly ``int f dx^axes`` rather than a
sampled/trapezoidal approximation. Point-value data use the NumPy integration
path at their true point locations.

A curvilinear axis -- part of a joint, non-separable ``.map(space="conf")``
block -- has no meaningful independent width. Point-value integration must
therefore remove the whole mapped block at once, using its physical cell
volumes (the Jacobian-determinant change-of-variables weight).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from postgkyl import dg
from postgkyl.gdatastate import materialize_point_values
from postgkyl.numerics import calculus, curvilinear

from ._curvilinear import curvilinear_blocks

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


def _parse_axes(axis: int | tuple | str | None, ndim: int) -> tuple[int, ...]:
  axes = tuple(int(a) for a in calculus.parse_axis(axis, ndim))
  if not axes:
    raise ValueError("integrate needs at least one axis")
  # end
  if len(set(axes)) != len(axes):
    raise ValueError(f"integrate axes must be distinct, got {axes}")
  # end
  if min(axes) < 0 or max(axes) >= ndim:
    raise ValueError(f"integrate axes {axes} out of range for a {ndim}D field")
  # end
  return tuple(sorted(axes))
# end


def _native_basis(data: "GDataState") -> tuple[str, int]:
  if data.backend != "gkyl":
    raise ValueError(
        "exact DG integration needs native modal data and is not available "
        "without the Gkeyll library")
  # end
  if data.ctx.get("value_form", "modal") != "modal":
    raise ValueError(
        f"exact DG integration expects the modal value_form, not "
        f"'{data.ctx['value_form']}'; call .to_modal() first")
  # end
  basis_type = data.ctx.get("basis_type")
  poly_order = data.ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError("dataset has no basis_type/poly_order metadata")
  # end
  return str(basis_type), int(poly_order)
# end


def _native_grid(data: "GDataState") -> dict:
  return {
      "ndim": data.num_dims,
      "lower": np.asarray(data.ctx["lower"]),
      "upper": np.asarray(data.ctx["upper"]),
      "cells": np.asarray(data.ctx["cells"]),
  }
# end


def _native_full(data: "GDataState", op: str):
  basis_type, poly_order = _native_basis(data)
  result = dg.modal.integrate(_native_grid(data), basis_type, poly_order,
      data.native, op=op)
  return float(result[0]) if result.size == 1 else result
# end


def _native_partial(data: "GDataState", axes: tuple[int, ...], *,
    inplace: bool, tag: str | None, label: str | None):
  basis_type, poly_order = _native_basis(data)
  grid = _native_grid(data)
  keep_dirs, cells, out = dg.modal.average(grid, basis_type, data.num_dims,
      poly_order, data.native, axes)

  # gkyl_array_average returns int(f dx^axes) / int(dx^axes). Scaling the
  # reduced modal coefficients recovers the integral exactly and stays native.
  lengths = grid["upper"] - grid["lower"]
  out = dg.modal.scale(out, float(np.prod(lengths[list(axes)])))
  new_grid = [np.asarray(data.grid[d]) for d in keep_dirs]
  return data._result(new_grid, out, inplace=inplace, tag=tag, label=label,
      cells=np.asarray(cells))
# end


def _point_integral(data: "GDataState", axes: tuple[int, ...]):
  data._require_operable()
  shadow = materialize_point_values(data)
  grid = list(shadow.grid)
  values = shadow.values

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
          "axis of the block together in the same call")
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
    volume = curvilinear.cell_volume(block_coords)
    volume = volume.reshape(volume.shape + (1,) * (values.ndim - m))
    moved = np.moveaxis(values, dims, range(m))
    reduced = np.sum(moved * volume, axis=tuple(range(m)), keepdims=True)
    values = np.moveaxis(reduced, range(m), dims)
    for d in dims:
      grid[d] = np.array([grid[d].mean()])
    # end
  # end
  return grid, values
# end


def _terminal_value(values: np.ndarray):
  result = np.asarray(values).reshape(-1, values.shape[-1])[0]
  return float(result[0]) if result.size == 1 else np.array(result, copy=True)
# end


def _remaining_mapped_axes(data: "GDataState", keep_dirs: list[int]) -> dict:
  old = data.ctx.get("mapped_axes", {})
  old_to_new = {old_dim: new_dim for new_dim, old_dim in enumerate(keep_dirs)}
  groups: dict[int, list[int]] = {}
  for old_dim, offset in old.items():
    if old_dim in old_to_new:
      groups.setdefault(offset, []).append(old_dim)
    # end
  # end

  result = {}
  for old_dims in groups.values():
    new_dims = [old_to_new[d] for d in old_dims]
    new_offset = min(new_dims)
    result.update({d: new_offset for d in new_dims})
  # end
  return result
# end


def _require_partial_options(*, inplace: bool, tag: str | None,
    label: str | None) -> None:
  if inplace or tag is not None or label is not None:
    raise ValueError(
        "inplace, tag, and label apply only to partial integration, which "
        "returns a dataset")
  # end
# end


def integrate(data: "GDataState", axis: int | tuple | str | None = None, *,
    op: str = "none", inplace: bool = False, tag: str | None = None,
    label: str | None = None):
  """Integrate over all or a subset of a dataset's spatial axes.

  Integrating every axis is terminal and returns one number per field.
  Integrating a strict subset returns a dataset over the surviving axes.
  Native modal input uses Gkeyll for both cases and remains modal/native after
  a partial integration; no interpolation or point sampling occurs. Nodal,
  quadrature, and NumPy-backed inputs are integrated numerically at their true
  point locations.

  Args:
    data: Dataset to integrate. Modal data are integrated exactly in DG space;
      point-value data use their physical point grid.
    axis: Axis or axes to integrate: an integer, tuple of integers,
      comma-separated string (``"0,1"``), colon slice string (``"0:2"``), or
      ``None`` for every spatial axis.
    op: Full native-DG integrand operation: ``"none"``, ``"abs"``, or
      ``"sq"``. Partial and point-value integration support ``"none"`` only.
    inplace: Mutate ``data`` for partial integration; unavailable for a full,
      terminal integration.
    tag: Optional tag for a partial-integration result.
    label: Optional label for a partial-integration result.

  Returns:
    A float (one field) or NumPy array (multiple fields) when every axis is
    integrated; otherwise a dataset over the surviving axes. A partial modal
    result is exact, modal, and Gkeyll-native.

  Raises:
    ValueError: If the axes are invalid; a partial curvilinear mapped block is
      requested; ``op`` is used outside a full native-DG integration; or
      dataset-only result options are used for a terminal integration.
  """
  axes = _parse_axes(axis, data.num_dims)
  full = len(axes) == data.num_dims
  modal = (data.backend == "gkyl"
      and data.ctx.get("value_form", "modal") == "modal")

  if full:
    _require_partial_options(inplace=inplace, tag=tag, label=label)
    if modal:
      return _native_full(data, op)
    # end
    if op != "none":
      raise ValueError("op is available only for full native-DG integration")
    # end
    _, values = _point_integral(data, axes)
    return _terminal_value(values)
  # end

  if op != "none":
    raise ValueError("op is available only for full native-DG integration")
  # end
  if modal:
    return _native_partial(data, axes, inplace=inplace, tag=tag, label=label)
  # end

  grid, values = _point_integral(data, axes)
  keep_dirs = [d for d in range(data.num_dims) if d not in axes]
  values = np.squeeze(values, axis=axes)
  new_grid = [grid[d] for d in keep_dirs]
  mapped_axes = _remaining_mapped_axes(data, keep_dirs)
  return data._result(new_grid, values, inplace=inplace, tag=tag, label=label,
      interpolated=True, value_form=None, mapped_axes=mapped_axes,
      grid_type="mapped" if mapped_axes else "uniform")
# end
