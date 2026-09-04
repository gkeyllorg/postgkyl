"""The ``differentiate`` verb -- numerical gradient of field-domain data.

Per ``.claude/migration/notes/differentiate-decision.md`` (layer 03): an
*exact* modal derivative would need a ``gpython_basis_eval_grad`` addition to the
compiled shim (``gkeyll/core/zero/gkyl_gpython.h``/``gpython.c`` +
``gpython/csrc/_gpythonmodule.c``), out of scope for every layer above
``gpython``. This
verb instead differentiates *after* ``.interpolate()``, with ``np.gradient`` on the
plain NumPy field values -- a numerical (second-order accurate, cell-centered), not
exact, derivative. Exactness on the modal polynomial is unnecessary here precisely
because the data have already been interpolated to a uniform mesh.

On a separable axis (the ordinary case, including a nonuniform/stretched
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
  """Numerical gradient of field-domain data.

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
    data: the dataset to differentiate; must be NumPy-backed (call
      ``.interpolate()`` first on native modal data).
    direction: 0-based axis to differentiate along; None differentiates
      along every axis.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset of the gradient, on ``data``'s (unchanged) grid.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  if data.backend == "gkyl":
    raise ValueError(
        "differentiate operates on interpolated (NumPy) values; call "
        ".interpolate() first -- np.gradient has no basis-space meaning for raw "
        "DG coefficients.")
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
