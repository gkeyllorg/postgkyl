"""Shared physical collocation contract for multi-dataset operations."""

import numpy as np

from postgkyl import numerics
from postgkyl.gdatastate.guards import require_same_quadrature
from postgkyl.gdatastate.layout import dg_layout, require_kernel_basis


def uniform_cartesian_grid(data) -> dict:
  """Resolve actual cell edges for native kernels with a uniform-grid ABI.

  Bounds come from the coordinates used by the dataset. A cached context
  bound cannot override its geometry, and a nonuniform mesh cannot silently
  enter a kernel whose grid descriptor only stores bounds and cell counts.
  """
  cells = data.num_cells
  message = "native DG operations require uniform Cartesian cell edges"
  if len(data.grid) != len(cells):
    raise ValueError(message)
  for coord, count in zip(data.grid, cells):
    if (count < 1 or coord.ndim != 1 or len(coord) != count + 1
        or not np.all(np.isfinite(coord)) or np.any(np.diff(coord) <= 0)):
      raise ValueError(message)
    # Subtracting large coordinates can make equal mathematical widths differ
    # by several ulps. Compare against reconstructed edges at coordinate
    # precision, rather than requiring those subtractions to agree exactly.
    expected = np.linspace(coord[0], coord[-1], count + 1)
    if not numerics.grids_compatible([coord], [expected], rtol=0):
      raise ValueError(message)
  return {
      "ndim": len(cells),
      "lower": np.array([coord[0] for coord in data.grid]),
      "upper": np.array([coord[-1] for coord in data.grid]),
      "cells": np.array(cells),
  }


def require_same_backend(left, right) -> None:
  if left.backend != right.backend:
    raise ValueError(
        "operands have different backends (gkyl vs numpy); call .interpolate() "
        "on the native operand to combine them.")


def require_collocated_layout(left, right) -> None:
  """Require the same geometry and representation, allowing field counts to differ.

  A scalar weight can apply to multiple fields, but it must describe values at
  the same physical locations with the same basis and point ordering.
  """
  require_same_backend(left, right)
  if not numerics.grids_compatible(left.grid, right.grid):
    raise ValueError("operands live on different grids")
  left_layout, right_layout = dg_layout(left), dg_layout(right)
  left_rep = left_layout.value_form if left_layout else None
  right_rep = right_layout.value_form if right_layout else None
  if left_rep != right_rep:
    raise ValueError(
        f"operands are in different value_forms ({left_rep} vs {right_rep}); "
        "convert one explicitly with .represent(to=...).")
  if left_layout is not None:
    if left_layout.basis_identity != right_layout.basis_identity:
      raise ValueError("operands have different DG bases")
    if left_rep == "quad":
      require_same_quadrature(left, right)
  if tuple(left.num_cells) != tuple(right.num_cells):
    raise ValueError("operands have different cell layouts")


def _native_modal_basis(data, what: str, verb: str) -> tuple[str, int]:
  if data.backend != "gkyl":
    raise ValueError(
        f"{verb} wraps gkyl_array_average and needs native modal data; "
        f"{what} is not available after .interpolate() or without the "
        "Gkeyll library.")
  if data.ctx.get("value_form", "modal") != "modal":
    raise ValueError(f"{verb} expects the modal value_form, not "
                     f"'{data.ctx['value_form']}' ({what}); "
                     "call .represent(to='modal') first.")
  basis_type = data.ctx.get("basis_type")
  poly_order = data.ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError(f"{what} has no basis_type/poly_order metadata")
  layout = require_kernel_basis(data)
  return layout.basis_type, layout.poly_order


def average_operands(data, weight, verb: str):
  """Validate the operands of a ``gkyl_array_average``-backed verb.

  ``data`` must be native modal data with basis metadata; ``weight``, when
  given, must be native modal data on the same grid, basis, and cell layout
  (the kernel takes a single-field weight over the donor layout).

  Returns:
    ``(basis_type, poly_order, grid, weight_native)`` -- the kernel basis,
    the donor grid dict for the native kernels (see
    :func:`uniform_cartesian_grid`), and the weight's native array (``None``
    without a weight).

  Raises:
    ValueError: ``data`` (or ``weight``) is NumPy-backed or non-modal, is
      missing basis metadata, or ``weight``'s grid/basis doesn't match
      ``data``'s.
  """
  basis_type, poly_order = _native_modal_basis(data, "data", verb)
  ndim = data.num_dims

  weight_native = None
  if weight is not None:
    w_basis_type, w_poly_order = _native_modal_basis(weight, "weight", verb)
    if weight.num_dims != ndim:
      raise ValueError(
          f"weight has {weight.num_dims} dims but the field has {ndim}")
    if w_basis_type != basis_type:
      raise ValueError(
          f"weight basis_type '{w_basis_type}' != field's '{basis_type}'")
    if w_poly_order != poly_order:
      raise ValueError(
          f"weight poly_order {w_poly_order} != field's {poly_order}")
    require_collocated_layout(data, weight)
    weight_native = weight.native

  return basis_type, poly_order, uniform_cartesian_grid(data), weight_native
