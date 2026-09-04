"""The ``map`` verb -- deform a dataset's grid by evaluating a coordinate map.

See ``MAPPING.md`` for the full design. A mapping file is a DG field whose
components hold the coefficients of the physical coordinates of each mapped
dimension; this verb evaluates those coefficients at the *target*'s own grid
points (:func:`postgkyl.dg.map_grid`) and splices the resulting arrays into
a copy of the target's grid. Only the grid changes -- the mapping's
coefficients are read straight from its native modal storage and are never
interpolated, and the target's values are passed through unchanged (no
copy: this verb never touches them).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import dg
from postgkyl.gdatastate.gdatastate import GDataState

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState as _GDataState


def map(data: "_GDataState",
        mapping: "str | _GDataState",
        *,
        space: str = "conf",
        basis_type: str | None = None,
        poly_order: int | None = None,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None) -> "_GDataState":
  """Replace a block of ``data``'s grid axes with mapped coordinates.

  Evaluates the mapping's DG coefficients at ``data``'s existing grid
  points (no resolution parameter, no alignment arithmetic -- the mapped
  axes always keep the shape of the axes they replace) and splices the
  result into a copy of ``data``'s grid.

  Args:
    data: The dataset whose grid is deformed; must be NumPy-backed
      (post-``interpolate()``), like ``select``.
    mapping: The coordinate-mapping field, as a filename or an
      already-loaded dataset. Read from its native modal coefficients --
      never interpolated. Its number of dimensions (``m``) sets how many
      of ``data``'s axes are replaced.
    space: ``'conf'`` deforms the leading ``m`` axes (offset 0),
      curvilinearly (every physical coordinate is evaluated over all ``m``
      mapped dimensions, so non-separable maps such as rotations work); its
      component count must be ``m * num_basis`` for an ``m``-D basis.
      ``'vel'`` deforms the trailing ``m`` axes (offset
      ``data.num_dims - m``); Gkeyll's velocity-space maps
      (``mapc2p_vel``) are diagonal -- each dimension is evaluated by its
      *own* 1-D map, so the component count must be ``m * num_basis`` for a
      *1-D* basis. For a combined map, apply the verb twice.
    basis_type: ``mapping``'s ``basis_type`` (long name, e.g.
      ``"serendipity"``), set at the moment this call loads it -- only
      takes effect when ``mapping`` is given as a filename (a property of
      already-loaded data can't be re-specified here). Velocity-space
      mapping files (``mapc2p_vel``) commonly carry no basis metadata at
      all, so this is typically required for ``space="vel"``.
    poly_order: ``mapping``'s ``poly_order``, set the same way as
      ``basis_type``.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset carrying the deformed grid; ``ctx["grid_type"]`` is set to
    ``"mapped"`` and ``ctx["mapped_axes"]`` records, for every absolute
    dimension touched so far (by this call and any earlier one), the
    ``offset`` of the mapped block it belongs to -- ``select``'s
    curvilinear guard needs this to convert an absolute dimension index
    back to the curvilinear grid array's own (relative) axis. The values
    array is untouched.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed); if ``space`` is
      neither ``'conf'`` nor ``'vel'``; if the map does not fit ``data``'s
      dimensionality; if the mapping has no ``basis_type``/``poly_order``
      metadata (and none was given at load time); or if its component
      count does not match the expected ``m * num_basis``.
  """
  if data.backend == "gkyl":
    raise ValueError(
        "map operates on interpolated (NumPy) target grids; call .interpolate() "
        "first -- deforming a native modal grid has no basis-space meaning.")

  # basis_type/poly_order are load-time properties of the mapping dataset;
  # they only apply here when this call is the one loading it (a filename),
  # threaded straight into GDataState's own override mechanism -- the one
  # home for "correct a file's missing/mislabeled basis metadata."
  map_data = (mapping if isinstance(mapping, GDataState) else GDataState(
      mapping, basis_type=basis_type, poly_order=poly_order))
  m = map_data.num_dims
  num_dims = data.num_dims

  if space == "conf":
    offset = 0
  elif space == "vel":
    offset = num_dims - m
  else:
    raise ValueError(f"map: 'space' must be 'conf' or 'vel', got {space!r}.")

  if offset < 0 or offset + m > num_dims:
    raise ValueError(
        f"map: a {m}D {space} map does not fit a {num_dims}D dataset.")

  resolved_basis_type = map_data.ctx.get("basis_type")
  resolved_poly_order = map_data.ctx.get("poly_order")
  if resolved_basis_type is None or resolved_poly_order is None:
    raise ValueError(
        "map: the mapping dataset has no 'basis_type'/'poly_order' "
        "metadata; pass basis_type=.../poly_order=... (mapping as a "
        "filename), or load it explicitly first with those set.")

  # Velocity-space maps (mapc2p_vel) are diagonal: each mapped dimension is
  # its own separate 1-D map, so its basis is 1-D regardless of m; a
  # configuration-space map (mapc2p/mc2nu) is one joint m-D curvilinear map.
  basis_dim = 1 if space == "vel" else m
  num_basis = dg.num_basis(basis_dim, resolved_poly_order, resolved_basis_type)
  if map_data.num_comps != m * num_basis:
    raise ValueError(
        f"map: mapping has {map_data.num_comps} component(s), expected "
        f"m * num_basis = {m} * {num_basis} = {m * num_basis} for a "
        f"{basis_dim}D {resolved_basis_type} p{resolved_poly_order} map" +
        (" per velocity dimension." if space == "vel" else "."))

  target_axes = list(data.grid[offset:offset + m])
  map_ctx = {
      "lower": map_data.ctx["lower"],
      "upper": map_data.ctx["upper"],
      "cells": map_data.ctx["cells"],
      "basis_type": resolved_basis_type,
      "poly_order": resolved_poly_order,
      "value_form": map_data.ctx.get("value_form", "modal"),
  }
  if space == "vel":
    new_axes = dg.map_grid_separable(map_data.get_values(), map_ctx,
                                     target_axes)
  else:
    new_axes = dg.map_grid(map_data.get_values(), map_ctx, target_axes)

  grid = list(data.grid)
  for d in range(m):
    grid[offset + d] = new_axes[d]

  # Record, per absolute dimension, the offset of the mapped block it
  # belongs to -- a curvilinear (m > 1) grid array's own axis k corresponds
  # to absolute dimension offset + k, not to the array's position in
  # `grid`, so `select`'s curvilinear guard needs this to convert back.
  # Merge with any prior block (e.g. a separate `space="vel"` map applied
  # after a `space="conf"` one) rather than overwrite it.
  mapped_axes = dict(data.ctx.get("mapped_axes", {}))
  mapped_axes.update({offset + d: offset for d in range(m)})

  return data._result(grid,
                      data.values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      grid_type="mapped",
                      mapped_axes=mapped_axes)
