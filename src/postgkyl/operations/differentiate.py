"""The ``differentiate`` verb -- local DG derivatives or field gradients.

Native modal data uses Gkeyll's exact derivative of the polynomial within
each cell, with no inter-cell stencil. NumPy point values use ``np.gradient``.

For NumPy point values on a separable axis (including a nonuniform/stretched
grid), this is a plain per-axis ``np.gradient`` against that axis' own 1-D
coordinate array. On a curvilinear axis -- part of a joint, non-separable
``.map(space="conf")`` block, whose grid arrays are multi-dimensional and
have no single 1-D coordinate of their own -- the physical derivative is
computed via the chain rule instead (``numerics.curvilinear.
physical_gradient``): the whole block's Jacobian is inverted once and reused
for every direction/component request that touches it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from postgkyl import dg
from postgkyl.numerics import curvilinear

from ._curvilinear import block_for_axis, curvilinear_blocks

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def differentiate(data: "GDataState",
                  *,
                  direction: int | None = None,
                  inplace: bool = False,
                  tag: str | None = None,
                  label: str | None = None):
  """Differentiate within DG cells or take a numerical gradient of point values.

  Native modal inputs stay modal and use local polynomial derivatives on a
  uniform Cartesian grid. This does not include jumps between DG cells.
  Interpolated inputs use a finite-difference gradient across sample points.

  With ``direction=None``, differentiates along every spatial axis and
  stacks the results in the component axis (``num_comps`` becomes
  ``num_comps * num_dims``, grouped ``[d0_comp0..d0_compN, d1_comp0.., ...]``).
  With an explicit ``direction``, differentiates along that one axis only
  (``num_comps`` unchanged). A separable axis requires a nodal (edge) grid
  one entry longer than the value count along that axis; a mismatched axis
  silently returns a wrong result -- a caveat inherited unchanged from the
  legacy tool. A curvilinear axis (part of a joint ``.map(space="conf")``
  block) has no such per-axis length convention of its own; its block's
  grid arrays carry it instead.

  Args:
    data: Native modal data or NumPy point values to differentiate.
    direction: 0-based axis to differentiate along; None differentiates
      along every axis.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset of the gradient, on ``data``'s (unchanged) grid.

  Raises:
    ValueError: if native data is non-modal or lacks uniform cell edges.
  """
  if data.backend == "gkyl":
    out = _differentiate_modal(data, direction)
    return data._result(data.grid, out, inplace=inplace, tag=tag, label=label)
  data._require_operable()
  grid = data.grid
  values = data.values
  num_dims = data.num_dims
  nc = values.shape[-1]

  blocks = curvilinear_blocks(grid, data.ctx.get("mapped_axes", {}))
  block_grad_cache: dict = {}

  def grad_along(d: int) -> np.ndarray:
    info = block_for_axis(blocks, d)
    if info is None:
      zc = 0.5 * (grid[d][1:] + grid[d][:-1])  # cell centered values
      return np.gradient(values, zc, edge_order=2, axis=d)
    off, dims = info
    if off not in block_grad_cache:
      block_coords = [grid[dd] for dd in dims]
      block_grad_cache[off] = curvilinear.physical_gradient(
          block_coords, values, tuple(dims))
    return block_grad_cache[off][..., dims.index(d)]

  if direction is None:
    out_shape = list(values.shape)
    out_shape[-1] = nc * num_dims
    out_values = np.zeros(out_shape)
    for d in range(num_dims):
      out_values[..., d * nc:(d + 1) * nc] = grad_along(d)
  else:
    out_values = grad_along(int(direction))
  return data._result(grid, out_values, inplace=inplace, tag=tag, label=label)


def _differentiate_modal(data: "GDataState", direction: int | None):
  """Apply the local native derivative, retaining the modal representation."""
  if data.ctx.get("value_form", "modal") != "modal":
    raise ValueError("differentiate needs modal coefficients for native data; "
                     "call .to_modal() first.")
  basis_type = data.ctx.get("basis_type")
  poly_order = data.ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError("differentiate needs basis_type/poly_order metadata")
  directions = range(data.num_dims) if direction is None else [int(direction)]
  results = []
  for d in directions:
    if not 0 <= d < data.num_dims:
      raise ValueError(f"differentiate direction {d} out of range")
    edges = data.grid[d]
    if edges.ndim != 1 or len(edges) != data.num_cells[d] + 1:
      raise ValueError(
          "modal differentiate requires uniform Cartesian cell edges")
    widths = np.diff(edges)
    if widths[0] <= 0 or not np.allclose(widths, widths[0], rtol=1e-12, atol=0):
      raise ValueError(
          "modal differentiate requires uniform Cartesian cell edges")
    results.append(
        dg.modal.differentiate(str(basis_type), data.num_dims, int(poly_order),
                               data.native, d, 1, float(widths[0])))
  if len(results) == 1:
    return results[0]
  return dg.rep.wrap(np.concatenate([out.view() for out in results], axis=-1))
