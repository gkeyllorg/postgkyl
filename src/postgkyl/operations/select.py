"""The ``select`` verb -- subselect coordinates and components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from postgkyl import dg
from postgkyl.numerics import idx_parser

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def _curvilinear_coord_curve(grid_arr: np.ndarray, rel: int, d: int,
                             offset: int, values_shape: tuple,
                             touched: set) -> np.ndarray:
  """A 1-D coordinate curve along ``grid_arr``'s relative axis ``rel``.

  Holds every other axis fixed at cell 0. That is exact only if
  ``grid_arr`` doesn't actually vary along those other axes -- true for the
  diagonal/separable maps this codebase's real mapped grids tend to produce
  (e.g. field-aligned ``mc2nu`` coordinates, each depending on only one
  computational axis even though stored jointly), false for a genuinely
  coupled map (e.g. a rotation). Raises rather than silently picking an
  arbitrary cross-section when a still-*unresolved* other axis of the
  block demonstrably varies the coordinate. An axis is resolved -- and
  skipped by the check -- either because ``values_shape`` (the dataset's
  own values array; the one true record of what a prior, possibly separate,
  ``select()`` call already narrowed) already holds a single cell on its
  absolute dimension ``offset + k``, or because a selector targeted it
  earlier in this same call (``touched``, before ``values`` itself was
  re-sliced at the end of that call). Either way the caller has made a
  deliberate choice of cross-section that this search takes as given.
  """
  fixed = tuple(0 if k != rel else slice(None) for k in range(grid_arr.ndim))
  full_range = np.ptp(grid_arr)
  for k in range(grid_arr.ndim):
    if k == rel or k in touched or values_shape[offset + k] == 1:
      continue
    if np.ptp(grid_arr, axis=k).max() > 1e-9 * full_range:
      raise ValueError(
          f"select: z{d}'s physical coordinate also varies along another "
          "axis of the same mapped (curvilinear) block, so a coordinate "
          "value or slice string has no single answer -- select an "
          "integer index for that other axis first (narrowing it to one "
          f"cell), or pass an integer index for z{d} itself.")
  return grid_arr[fixed]


def select(data: "GDataState",
           *,
           comp=None,
           z0=None,
           z1=None,
           z2=None,
           z3=None,
           z4=None,
           z5=None,
           inplace: bool = False,
           tag: str | None = None,
           label: str | None = None):
  """Select part of a dataset by coordinate (``z0``-``z5``) and/or component.

  Each selector accepts an int index, a float coordinate value, or a slice
  string ``"start:end"``; ``comp`` additionally accepts ``"a,b"``. Unspecified
  axes are kept in full. The selected dimension is retained (length-1), matching
  the legacy behaviour.

  A curvilinear axis (a multi-dimensional grid array, produced by ``.map()``
  with ``space="conf"``) has no single 1-D coordinate array of its own to
  search -- a coordinate value or slice string is resolved against a 1-D
  cross-section instead (:func:`_curvilinear_coord_curve`), holding every
  other axis of the same mapped block at cell 0, *unless* that axis was
  already narrowed to a single cell by an earlier selector -- either in
  this same call, or in a prior ``select()`` call in the chain (recorded
  the only place it needs to be: the dataset's own values shape). Selecting
  one axis of a block also narrows every sibling axis' grid array along
  that same relative axis, so the block stays internally consistent and a
  later selector on a sibling sees the narrowed cross-section rather than
  the original full extent. If the array still varies along an
  as-yet-unresolved sibling axis (a genuinely non-separable map, e.g. a
  rotation), the coordinate/slice selector has no single answer and raises
  -- pick an integer index for that sibling axis first (in the same call,
  or an earlier one in the chain).

  Args:
    data: Dataset whose point values are selected.
    comp: Component selector such as ``"0"``, ``"0:3"``, or ``"0,2"``.
    z0: Selector for coordinate direction 0.
    z1: Selector for coordinate direction 1.
    z2: Selector for coordinate direction 2.
    z3: Selector for coordinate direction 3.
    z4: Selector for coordinate direction 4.
    z5: Selector for coordinate direction 5.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Raises:
    ValueError: if ``data`` holds native modal DG coefficients (nodal/quad
      value_forms of gkyl-backed data are point values and slice fine),
      or a coordinate/slice selector targets a curvilinear axis whose
      physical coordinate still varies along an unresolved sibling axis.
  """
  if data.backend == "gkyl" and data.ctx.get("value_form", "modal") == "modal":
    raise ValueError(
        "select operates on interpolated (NumPy) values, or on gkyl-native "
        "nodal/quad value_forms; call .interpolate()/.to_nodal()/"
        ".to_quad() first -- slicing raw modal DG coefficients would mix "
        "basis functions.")
  zs = (z0, z1, z2, z3, z4, z5)
  grid = list(data.grid)
  values = data.values
  num_dims = data.num_dims
  values_idx = [slice(0, values.shape[d]) for d in range(num_dims + 1)]

  # ctx["mapped_axes"] records, for every dimension touched by a .map()
  # call, the offset of the mapped block it belongs to; group them back
  # into blocks so selecting one axis can keep every sibling's grid array
  # (they all share the block's tensor shape) in sync.
  mapped_axes = data.ctx.get("mapped_axes", {})
  block_dims: dict[int, list[int]] = {}
  for dd, off in mapped_axes.items():
    block_dims.setdefault(off, []).append(dd)
  # per-block set of relative axes already given a selector earlier in this
  # same call -- lets a later axis's coordinate search skip the
  # separability check on a sibling the caller has deliberately pinned.
  touched: dict[int, set] = {}

  for d, z in enumerate(zs):
    if d >= num_dims or z is None:
      continue
    grid_arr = grid[d]
    curvilinear = grid_arr.ndim > 1  # a .map()-deformed grid axis
    # a curvilinear array's own axis k corresponds to absolute dimension
    # `offset + k` (map.py's mapped block), not to axis d of `grid` itself
    # -- ctx["mapped_axes"] records each absolute dimension's block offset
    # so the N-D array can be indexed on its own relative axis.
    offset = mapped_axes.get(d, 0)
    rel = d - offset if curvilinear else d
    len_grid = grid_arr.shape[rel] if curvilinear else grid_arr.shape[0]
    is_matching = values.shape[d] == len_grid
    if curvilinear and isinstance(z, int):
      idx = z
    elif curvilinear:
      coord_curve = _curvilinear_coord_curve(grid_arr, rel, d, offset,
                                             values.shape,
                                             touched.get(offset, set()))
      idx = idx_parser(z, coord_curve, is_matching)
    else:
      idx = idx_parser(z, grid_arr, is_matching)
    if isinstance(idx, int):
      if idx < 0:
        idx = values.shape[d] + idx
      v_idx = slice(idx, idx + 1)
      g_idx = slice(idx, idx + 1) if is_matching else slice(idx, idx + 2)
    elif isinstance(idx, slice):
      v_idx = idx
      g_idx = idx if is_matching else slice(idx.start, idx.stop + 1)
    else:
      raise TypeError("Coordinate selector must be a single index or a slice.")
    if curvilinear:
      # every axis in the same mapped block shares this array shape, so
      # slicing relative axis `rel` in lockstep keeps them all consistent
      # -- a later selector on a sibling axis then resolves against the
      # already-narrowed cross-section instead of the original full extent.
      for dd in block_dims.get(offset, [d]):
        arr = grid[dd]
        grid[dd] = arr[tuple(g_idx if k == rel else slice(None)
                             for k in range(arr.ndim))]
      touched.setdefault(offset, set()).add(rel)
    else:
      grid[d] = grid_arr[g_idx]
    values_idx[d] = v_idx

  if comp is not None:
    values_idx[-1] = idx_parser(comp)

  values_out = values[tuple(values_idx)]
  if num_dims == values_out.ndim:  # restore the squeezed component axis
    values_out = values_out[..., np.newaxis]

  ctx_updates = {}
  if data.backend == "gkyl":
    # A nodal/quad value_form stays gkyl-native (REFACTOR_GKEYLL_FFI.md
    # §3b): ``values`` above was only a read-only NumPy *view* of the native
    # array for slicing purposes -- wrap the sliced result back into a
    # native GkylArray so the dataset doesn't silently fall out of the gkyl
    # backend (and lose its value_form) just for having been selected.
    # Cell layout isn't derivable from the flat native array (see
    # ``GDataState.set_values``), so it must be threaded through explicitly,
    # the same way ``average``/``eval_at_coord_proj`` do.
    ctx_updates["cells"] = np.array(values_out.shape[:-1], dtype=np.int64)
    values_out = dg.rep.wrap(values_out)

  return data._result(grid,
                      values_out,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      **ctx_updates)
