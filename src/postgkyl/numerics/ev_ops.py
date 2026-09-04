"""RPN operator registry for the ``ev`` verb (pure ``(grid, values)`` functions).

This is the numeric core behind the ``ev`` expression evaluator. Each
operator is a pure function ``f(in_grid, in_values) -> ([out_grid], [out_values])``
over plain Python lists / NumPy arrays -- no ``GData`` dependency. The
``cmds`` table maps each RPN token to its arity (``num_in``/``num_out``)
and function; the stack machine that drives them lives in the ``operations``
layer's ``ev`` verb (layer 07), which can consume this table unchanged.

Every operator here is expressible over plain arrays; none needed a
``NotImplementedError`` GData-only placeholder.
"""

from __future__ import annotations

import numpy as np

from .calculus import _split_axis_string
from .idx_parser import idx_parser


def _get_grid(grid0, grid1):
  if grid0 is not None and grid1 is not None:
    return grid0 if len(grid0) > len(grid1) else grid1
  if grid0 is not None:
    return grid0
  if grid1 is not None:
    return grid1
  return None


def add(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  out_values = in_values[0] + in_values[1]
  return [out_grid], [out_values]


def subtract(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  out_values = in_values[1] - in_values[0]
  return [out_grid], [out_values]


def mult(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  a, b = in_values[1], in_values[0]
  if np.array_equal(a.shape, b.shape) or len(a.shape) == 0 or len(b.shape) == 0:
    out_values = a * b
  else:
    # When multiplying a phase-space and a conf-space field, the
    # dimensions do not match. NumPy broadcasting requires the *trailing*
    # indices to match, which is the opposite of what we have here (the
    # *leading* indices match) -- so transpose, multiply, transpose back.
    out_values = (a.transpose() * b.transpose()).transpose()
  return [out_grid], [out_values]


def dot(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  out_values = np.sum(in_values[1] * in_values[0], axis=-1)[..., np.newaxis]
  return [out_grid], [out_values]


def divide(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  a, b = in_values[1], in_values[0]
  if np.array_equal(a.shape, b.shape) or len(a.shape) == 0 or len(b.shape) == 0:
    out_values = a / b
  else:
    # See the 'mult' comment above.
    out_values = (a.transpose() / b.transpose()).transpose()
  return [out_grid], [out_values]


def sqrt(in_grid, in_values):
  return [in_grid[0]], [np.sqrt(in_values[0])]


def psin(in_grid, in_values):
  return [in_grid[0]], [np.sin(in_values[0])]


def pcos(in_grid, in_values):
  return [in_grid[0]], [np.cos(in_values[0])]


def ptan(in_grid, in_values):
  return [in_grid[0]], [np.tan(in_values[0])]


def absolute(in_grid, in_values):
  return [in_grid[0]], [np.abs(in_values[0])]


def log(in_grid, in_values):
  return [in_grid[0]], [np.log(in_values[0])]


def log10(in_grid, in_values):
  return [in_grid[0]], [np.log10(in_values[0])]


def minimum(in_grid, in_values):
  out_values = np.atleast_1d(np.nanmin(in_values[0]))
  return [[]], [out_values]


def minimum2(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  out_values = np.fmin(in_values[0], in_values[1])
  return [out_grid], [out_values]


def maximum(in_grid, in_values):
  out_values = np.atleast_1d(np.nanmax(in_values[0]))
  return [[]], [out_values]


def maximum2(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  out_values = np.fmax(in_values[0], in_values[1])
  return [out_grid], [out_values]


def mean(in_grid, in_values):
  out_values = np.atleast_1d(np.mean(in_values[0]))
  return [[]], [out_values]


def power(in_grid, in_values):
  out_grid = in_grid[1]
  out_values = np.power(in_values[1], in_values[0])
  return [out_grid], [out_values]


def sq(in_grid, in_values):
  return [in_grid[0]], [in_values[0]**2]


def exp(in_grid, in_values):
  return [in_grid[0]], [np.exp(in_values[0])]


def length(in_grid, in_values):
  ax = int(in_values[0])
  ln = in_grid[1][ax][-1] - in_grid[1][ax][0]
  if len(in_grid[1][ax]) == in_values[1].shape[ax]:
    ln += in_grid[1][ax][1] - in_grid[1][ax][0]
  return [[]], [ln]


def grad(in_grid, in_values):
  out_grid = in_grid[0]
  nd = len(in_values[0].shape) - 1
  out_shape = list(in_values[0].shape)
  nc = in_values[0].shape[-1]
  out_shape[-1] = nc * nd
  out_values = np.zeros(out_shape)

  for d in range(nd):
    zc = 0.5 * (in_grid[0][d][1:] + in_grid[0][d][:-1])  # cell centered values
    out_values[..., d * nc:(d + 1) * nc] = np.gradient(in_values[0],
                                                       zc,
                                                       edge_order=2,
                                                       axis=d)
  return [out_grid], [out_values]


def grad2(in_grid, in_values):
  out_grid = in_grid[1]
  ax = in_values[0]
  if isinstance(ax, str) and ":" in ax:
    lo, up = ax.split(":")
    rng = range(int(lo), int(up))
  elif isinstance(ax, str):
    rng = tuple(int(i) for i in ax.split(","))
  else:
    rng = range(int(ax), int(ax + 1))

  num_dims = len(rng)
  out_shape = list(in_values[1].shape)
  num_comps = in_values[1].shape[-1]
  out_shape[-1] = out_shape[-1] * num_dims
  out_values = np.zeros(out_shape)

  for cnt, d in enumerate(rng):
    zc = 0.5 * (in_grid[1][d][1:] + in_grid[1][d][:-1])  # cell centered values
    out_values[..., cnt * num_comps:(cnt + 1) * num_comps] = np.gradient(
        in_values[1], zc, edge_order=2, axis=d)
  return [out_grid], [out_values]


def _parse_axis(axis) -> tuple:
  if isinstance(axis, float):
    return (int(axis), )
  if isinstance(axis, tuple):
    return axis
  if isinstance(axis, np.ndarray):
    return (int(axis), )
  if isinstance(axis, str):
    if axis == "all":
      return None  # resolved against num_dims by the caller
    return _split_axis_string(axis)
  raise TypeError(
      "'axis' needs to be integer, tuple, string of comma separated "
      "integers, or a slice ('int:int')")


def integrate(in_grid, in_values, avg=False):
  grid = in_grid[1].copy()
  values = np.array(in_values[1])

  axis = _parse_axis(in_values[0])
  if axis is None:
    axis = tuple(range(len(grid)))

  dz = []
  for d, coord in enumerate(grid):
    dz.append(coord[1:] - coord[:-1])
    if len(coord) == values.shape[d]:
      dz[-1] = np.append(dz[-1], dz[-1][-1])

  # Integration assuming values are cell centered averages
  # Should work for nonuniform meshes
  for ax in sorted(axis, reverse=True):
    values = np.moveaxis(values, ax, -1)
    values = np.dot(values, dz[ax])
  for ax in sorted(axis):
    grid[ax] = np.array([0])
    values = np.expand_dims(values, ax)
    if avg:
      ln = in_grid[1][ax][-1] - in_grid[1][ax][0]
      if len(in_grid[1][ax]) == in_values[1].shape[ax]:
        ln += in_grid[1][ax][1] - in_grid[1][ax][0]
      values = values / ln
  return [grid], [values]


def average(in_grid, in_values):
  return integrate(in_grid, in_values, True)


def divergence(in_grid, in_values):
  out_grid = in_grid[0]
  num_dims = len(in_grid[0])
  num_comps = in_values[0].shape[-1]
  if num_comps > num_dims:
    # src_bak warned and computed a partial result (using only the first
    # num_dims components) here; this raises instead per PYTHON_PRINCIPLES §10.
    raise ValueError(
        f"ERROR in 'evaluate div': Length of the provided vector ({num_comps:d}) "
        f"is longer than number of dimensions ({num_dims:d}).")
  out_shape = list(in_values[0].shape)
  out_shape[-1] = 1
  out_values = np.zeros(out_shape)
  for d in range(num_dims):
    zc = 0.5 * (in_grid[0][d][1:] + in_grid[0][d][:-1])  # cell centered values
    out_values[..., 0] = out_values[..., 0] + np.gradient(
        in_values[0][..., d], zc, edge_order=2, axis=d)
  return [out_grid], [out_values]


def curl(in_grid, in_values):
  out_grid = in_grid[0]
  num_dims = len(in_grid[0])
  num_comps = in_values[0].shape[-1]

  out_shape = list(in_values[0].shape)

  if num_dims == 1:
    if num_comps != 3:
      raise ValueError(
          f"ERROR in 'evaluate curl': Curl in 1D requires 3-component input and "
          f"{num_comps:d}-component field was provided.")
    zc0 = 0.5 * (in_grid[0][0][1:] + in_grid[0][0][:-1])
    out_values = np.zeros(out_shape)
    out_values[..., 1] = -np.gradient(
        in_values[0][..., 2], zc0, edge_order=2, axis=0)
    out_values[..., 2] = np.gradient(in_values[0][..., 1],
                                     zc0,
                                     edge_order=2,
                                     axis=0)
  elif num_dims == 2:
    zc0 = 0.5 * (in_grid[0][0][1:] + in_grid[0][0][:-1])
    zc1 = 0.5 * (in_grid[0][1][1:] + in_grid[0][1][:-1])
    if num_comps < 2:
      raise ValueError(
          f"ERROR in 'evaluate curl': Length of the provided vector ({num_comps:d}) "
          f"is smaller than number of dimensions ({num_dims:d}). Curl can't "
          f"be calculated.")
    elif num_comps == 2:
      # A 2D vector field: curl reduces to the single in-plane (z) component.
      # This is the normal, expected input for 2D curl -- not an anomaly.
      out_shape[-1] = 1
      out_values = np.zeros(out_shape)
      out_values[..., 0] = np.gradient(
          in_values[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(
              in_values[0][..., 0], zc1, edge_order=2, axis=1)
    else:
      if num_comps > 3:
        # src_bak warned and computed a partial result (using only the
        # first 3 components) here; this raises instead per
        # PYTHON_PRINCIPLES §10.
        raise ValueError(
            f"ERROR in 'evaluate curl': Length of the provided vector "
            f"({num_comps:d}) is longer than number of dimensions "
            f"({num_dims:d}).")
      out_values = np.zeros(out_shape)
      out_values[..., 0] = np.gradient(in_values[0][..., 2],
                                       zc1,
                                       edge_order=2,
                                       axis=1)
      out_values[..., 1] = -np.gradient(
          in_values[0][..., 2], zc0, edge_order=2, axis=0)
      out_values[..., 2] = np.gradient(
          in_values[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(
              in_values[0][..., 0], zc1, edge_order=2, axis=1)
  else:  # 3D
    if num_comps > 3:
      # src_bak warned and computed a partial result (using only the
      # first 3 components) here; this raises instead per
      # PYTHON_PRINCIPLES §10.
      raise ValueError(
          f"ERROR in 'evaluate curl': Length of the provided vector ({num_comps:d}) "
          f"is longer than number of dimensions ({num_dims:d}).")
    elif num_comps < 3:
      raise ValueError(
          f"ERROR in 'evaluate curl': Length of the provided vector ({num_comps:d}) "
          f"is smaller than number of dimensions ({num_dims:d}). Curl can't "
          f"be calculated.")
    zc0 = 0.5 * (in_grid[0][0][1:] + in_grid[0][0][:-1])
    zc1 = 0.5 * (in_grid[0][1][1:] + in_grid[0][1][:-1])
    zc2 = 0.5 * (in_grid[0][2][1:] + in_grid[0][2][:-1])
    out_values = np.zeros(out_shape)
    out_values[..., 0] = np.gradient(
        in_values[0][..., 2], zc1, edge_order=2, axis=1) - np.gradient(
            in_values[0][..., 1], zc2, edge_order=2, axis=2)
    out_values[..., 1] = np.gradient(
        in_values[0][..., 0], zc2, edge_order=2, axis=2) - np.gradient(
            in_values[0][..., 2], zc0, edge_order=2, axis=0)
    out_values[..., 2] = np.gradient(
        in_values[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(
            in_values[0][..., 0], zc1, edge_order=2, axis=1)
  return [out_grid], [out_values]


def scale_comp(in_grid, in_values):
  """Scale specific components of the data.

  RPN stack order: ``f comp_spec scale_factor scale_comp`` -- usage
  ``f 2:4 1000 scale_comp`` scales components 2 and 3 by 1000.

  Args:
    in_values[0]: Scaling factor.
    in_values[1]: Component specification (a string like ``"2:4"``, or a
      number).
    in_values[2]: Original data array (``f``).
  """
  out_grid = in_grid[2]  # grid from the original data (f)
  original_data = in_values[2].copy()
  comp_spec = in_values[1]
  scale_factor = in_values[0]

  scale_factor = scale_factor.item()
  if isinstance(comp_spec, str):
    comp_idx = idx_parser(comp_spec)
  elif isinstance(comp_spec, np.ndarray) and comp_spec.size == 1:
    comp_idx = int(comp_spec.item())
  else:
    comp_idx = int(comp_spec)

  if isinstance(comp_idx, slice):
    original_data[..., comp_idx] *= scale_factor
  elif isinstance(comp_idx, tuple):
    for idx in comp_idx:
      original_data[..., idx] *= scale_factor
  else:
    original_data[..., comp_idx] *= scale_factor

  return [out_grid], [original_data]


def scale_zi_axis(in_grid, in_values):
  """Scale the ``z_i`` axis of the grid.

  RPN stack order: ``f axis scale_factor scale_zi_axis`` -- usage
  ``f 0 1000 scale_zi_axis`` scales the x-axis (axis 0) by 1000.

  Args:
    in_values[0]: Scaling factor.
    in_values[1]: Axis direction (``0``-``5``).
    in_values[2]: Original data array (``f``).
  """
  out_grid = in_grid[2]  # grid from the original data (f)
  original_data = in_values[2].copy()
  idx_scale = in_values[1].item()
  scale_factor = in_values[0].item()

  # NB: mutates the referenced axis array in place (matches src_bak exactly,
  # including its aliasing with the caller's original grid list).
  out_grid[int(idx_scale)] *= scale_factor

  return [out_grid], [original_data]


cmds = {
    "+": {
        "num_in": 2,
        "num_out": 1,
        "func": add
    },
    "-": {
        "num_in": 2,
        "num_out": 1,
        "func": subtract
    },
    "*": {
        "num_in": 2,
        "num_out": 1,
        "func": mult
    },
    "/": {
        "num_in": 2,
        "num_out": 1,
        "func": divide
    },
    "dot": {
        "num_in": 2,
        "num_out": 1,
        "func": dot
    },
    "sqrt": {
        "num_in": 1,
        "num_out": 1,
        "func": sqrt
    },
    "sin": {
        "num_in": 1,
        "num_out": 1,
        "func": psin
    },
    "cos": {
        "num_in": 1,
        "num_out": 1,
        "func": pcos
    },
    "tan": {
        "num_in": 1,
        "num_out": 1,
        "func": ptan
    },
    "abs": {
        "num_in": 1,
        "num_out": 1,
        "func": absolute
    },
    "avg": {
        "num_in": 2,
        "num_out": 1,
        "func": average
    },
    "log": {
        "num_in": 1,
        "num_out": 1,
        "func": log
    },
    "log10": {
        "num_in": 1,
        "num_out": 1,
        "func": log10
    },
    "max": {
        "num_in": 1,
        "num_out": 1,
        "func": maximum
    },
    "min": {
        "num_in": 1,
        "num_out": 1,
        "func": minimum
    },
    "max2": {
        "num_in": 2,
        "num_out": 1,
        "func": maximum2
    },
    "min2": {
        "num_in": 2,
        "num_out": 1,
        "func": minimum2
    },
    "mean": {
        "num_in": 1,
        "num_out": 1,
        "func": mean
    },
    "len": {
        "num_in": 2,
        "num_out": 1,
        "func": length
    },
    "pow": {
        "num_in": 2,
        "num_out": 1,
        "func": power
    },
    "sq": {
        "num_in": 1,
        "num_out": 1,
        "func": sq
    },
    "exp": {
        "num_in": 1,
        "num_out": 1,
        "func": exp
    },
    "grad": {
        "num_in": 1,
        "num_out": 1,
        "func": grad
    },
    "grad2": {
        "num_in": 2,
        "num_out": 1,
        "func": grad2
    },
    "int": {
        "num_in": 2,
        "num_out": 1,
        "func": integrate
    },
    "div": {
        "num_in": 1,
        "num_out": 1,
        "func": divergence
    },
    "curl": {
        "num_in": 1,
        "num_out": 1,
        "func": curl
    },
    "scale_comp": {
        "num_in": 3,
        "num_out": 1,
        "func": scale_comp
    },
    "scale_zi_axis": {
        "num_in": 3,
        "num_out": 1,
        "func": scale_zi_axis
    },
}
