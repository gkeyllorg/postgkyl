"""Integrate cell averages or point samples on separable grids (pure NumPy).

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
  """Integrate cell averages or point samples over one or more axes.

  An axis with one more coordinate than values contains cell edges: sum
  each cell's average times its width. An axis matching the value count
  contains point samples: use the composite trapezoidal rule between the
  supplied points. Both rules support nonuniform meshes. A single point
  spans zero length. Packed DG data must be integrated before materializing
  it as a point mesh, since that mesh can lose cell-local polynomial meaning.

  Args:
    grid: Edge or point coordinate arrays, one per spatial dimension.
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

  for ax in sorted(axis, reverse=True):
    coord = np.asarray(grid[ax])
    if coord.ndim != 1:
      raise ValueError("separable integration requires one-dimensional axes")
    count = values.shape[ax]
    if len(coord) == count + 1:
      values = np.tensordot(values, np.diff(coord), axes=(ax, 0))
    elif len(coord) == count:
      values = np.trapezoid(values, x=coord, axis=ax)
    else:
      raise ValueError(
          f"axis {ax}: coordinate count {len(coord)} must match {count} "
          f"point samples or {count + 1} cell edges")

  for ax in sorted(axis):
    grid[ax] = np.array([grid[ax].mean()])
    values = np.expand_dims(values, ax)

  return grid, values
