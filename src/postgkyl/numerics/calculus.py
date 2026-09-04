"""Trapezoidal-style integration over a nodal grid (pure NumPy).

``grad``/``div``/``curl`` are deliberately absent: the ``src_bak`` originals
are unimplemented placeholders (``...`` bodies, no arguments) -- there is no
real numerics to port. The vector-calculus operators that *are* implemented
live in :mod:`postgkyl.numerics.ev_ops` (``divergence``/``curl``/``grad``),
expressed the same way, over ``(grid, values)`` pairs.
"""

from __future__ import annotations

import numpy as np


def _split_axis_string(axis: str) -> tuple:
  """Parse a comma-separated (``"0,1"``) or colon-sliced (``"0:2"``) axis
  string, or a bare integer string, into a tuple of integer axes.

  Shared with :func:`postgkyl.numerics.ev_ops._parse_axis`, whose outer
  type-dispatch differs (it also accepts ``float``/``np.ndarray``/``"all"``)
  but delegates this exact string-parsing branch here, so the comma/colon
  grammar has one home (Doctrine V) instead of two copies that could drift.
  """
  if len(axis.split(",")) > 1:
    return tuple(int(a) for a in axis.split(","))
  if len(axis.split(":")) == 2:
    lo, hi = axis.split(":")
    return tuple(range(int(lo), int(hi)))
  return (int(axis), )


def parse_axis(axis: int | tuple | str | None, num_dims: int) -> tuple:
  """Turn an axis selector into a tuple of integer axes."""
  if axis is None:
    return tuple(range(num_dims))
  if isinstance(axis, int):
    return (axis, )
  if isinstance(axis, tuple):
    return axis
  if isinstance(axis, str):
    return _split_axis_string(axis)
  raise TypeError(
      "'axis' needs to be integer, tuple, string of comma separated "
      "integers, or a slice ('int:int')")


def integrate(
    grid: list[np.ndarray],
    values: np.ndarray,
    axis: int | tuple | str | None = None
) -> tuple[list[np.ndarray], np.ndarray]:
  """Integrate cell-centered-average data over one or more axes.

  Uses the NumPy dot product against the cell widths (trapezoidal for
  nodal/edge grids, exact for cell-centered-average data); works for
  nonuniform meshes. True DG integration is not implemented here -- this
  mirrors the legacy behaviour exactly.

  Args:
    grid: Nodal (edge) coordinate arrays, one per spatial dimension.
    values: Data array; the last axis is components, the rest are spatial.
    axis: Axis (or axes) to integrate over: an ``int``, a ``tuple`` of
      ``int``, a comma-separated string (``"0,1"``), a colon slice string
      (``"0:2"``), or ``None`` (integrate over every spatial axis).

  Returns:
    ``(grid, values)`` with the integrated axes collapsed to a single,
    grid-mean cell and ``values`` reduced accordingly (shape retained via
    ``expand_dims``).

  Raises:
    TypeError: If ``axis`` is not an int, tuple, or string.
  """
  grid = list(grid)
  values = np.copy(values)
  axis = parse_axis(axis, len(grid))

  # Get dz elements
  dz = []
  for d, coord in enumerate(grid):
    dz.append(coord[1:] - coord[:-1])
    if len(coord) > 1 and len(coord) == values.shape[d]:
      dz[-1] = np.append(dz[-1], dz[-1][-1])

  # Integration assuming values are cell centered averages
  # Should work for nonuniform meshes
  for ax in sorted(axis, reverse=True):
    if len(grid[ax]) > 1:
      values = np.moveaxis(values, ax, -1)
      values = np.dot(values, dz[ax])
    else:
      values = values.mean(axis=ax)

  for ax in sorted(axis):
    grid[ax] = np.array([grid[ax].mean()])
    values = np.expand_dims(values, ax)

  return grid, values
