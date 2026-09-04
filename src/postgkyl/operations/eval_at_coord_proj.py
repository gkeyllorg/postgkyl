"""The ``eval_at_coord_proj`` verb -- evaluate a native DG field at physical
coordinates in a subset of directions, projecting onto the lower-dimensional
target basis for the survivors, via Gkeyll's ``gkyl_dg_eval_at_coord_proj``.

Terminal-adjacent (like ``average``): produces a new, lower-dimensional
dataset -- still modal and gkyl-native -- so it composes with further
``.to_nodal()``/``.interpolate()``/``.eval_at_coord_proj()`` calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from postgkyl import dg

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def _native_basis(data: "GDataState"):
  if data.backend != "gkyl":
    raise ValueError(
        "eval_at_coord_proj wraps gkyl_dg_eval_at_coord_proj and needs "
        "native modal data; it is not available after .interpolate() or "
        "without the Gkeyll library.")
  if data.ctx.get("value_form", "modal") != "modal":
    raise ValueError(f"eval_at_coord_proj expects the modal value_form, not "
                     f"'{data.ctx['value_form']}'; call .to_modal() first.")
  basis_type = data.ctx.get("basis_type")
  poly_order = data.ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError("eval_at_coord_proj: data has no basis_type/poly_order "
                     "metadata")
  return str(basis_type), int(poly_order)


def eval_at_coord_proj(data: "GDataState",
                       eval_dirs,
                       eval_coords,
                       *,
                       inplace: bool = False,
                       tag: str | None = None,
                       label: str | None = None):
  """Evaluate ``data`` at ``eval_coords`` in ``eval_dirs`` and project onto
  the surviving directions' target basis.

  Args:
    data: gkyl-backed (native modal) dataset in the modal value_form.
    eval_dirs: 0-based direction indices to evaluate away (e.g. the selected
      ``z0``-``z5`` flags at the CLI layer).
    eval_coords: physical coordinates, one per entry of ``eval_dirs`` (same
      order); in the dataset's own computational grid sense (the same
      convention every other native-modal verb here, e.g. ``average``,
      uses -- not a separately mapped/deformed physical grid, which this
      architecture only ever produces post-``interpolate()``).
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A new dataset over the surviving dimensions -- or a single degenerate
    dimension (``grid=[0, 1]``) when every direction is evaluated away,
    Gkeyll's own convention since there is no true 0-dimensional basis --
    still modal and gkyl-native. Its basis metadata (``basis_type``,
    ``poly_order``, ``num_cdim``, ``num_vdim``) reflects the TARGET basis
    Gkeyll picked, which can differ in type from the donor's (e.g.
    eliminating a gkhybrid velocity direction can yield a plain serendipity
    target).

  Raises:
    ValueError: ``data`` is NumPy-backed or non-modal, is missing basis
      metadata, or ``eval_dirs``/``eval_coords`` don't match in length or
      range.
  """
  basis_type, poly_order = _native_basis(data)
  ndim = data.num_dims

  grid = {
      "ndim": ndim,
      "lower": np.asarray(data.ctx["lower"]),
      "upper": np.asarray(data.ctx["upper"]),
      "cells": np.asarray(data.ctx["cells"]),
  }
  keep_dirs, cells_tar, out_native, btype_tar, poly_order_tar, cdim_tar, \
      vdim_tar = dg.modal.eval_at_coord_proj(grid, basis_type, ndim,
          poly_order, data.native, eval_dirs, eval_coords)

  if keep_dirs:
    new_grid = [np.asarray(data.grid[d]) for d in keep_dirs]
  else:
    new_grid = [np.array([0.0, 1.0])]

  return data._result(new_grid,
                      out_native,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      cells=np.asarray(cells_tar),
                      basis_type=btype_tar,
                      poly_order=poly_order_tar,
                      num_cdim=cdim_tar,
                      num_vdim=vdim_tar)
