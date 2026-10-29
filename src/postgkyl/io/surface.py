"""Reconstruct Gkeyll's component-packed surface geometry fields."""

from __future__ import annotations

import numpy as np

from postgkyl import gpython


def surface_direction(value_form: object, ndim: int) -> int | None:
  """Return the perpendicular axis encoded by ``modal_surf#``."""
  if not isinstance(value_form, str) or not value_form.startswith("modal_surf"):
    return None
  suffix = value_form[len("modal_surf"):]
  if not suffix.isascii() or not suffix.isdecimal():
    raise ValueError(f"invalid surface value_form {value_form!r}")
  direction = int(suffix)
  if direction >= ndim:
    raise ValueError(f"surface direction {direction} is out of range for "
                     f"{ndim}D data")
  return direction


def unpack_surface(grid: list, values, ctx: dict):
  """Move the final face from the last cell's second component half."""
  direction = surface_direction(ctx.get("value_form"), len(grid))
  if direction is None:
    return grid, values
  cells = np.asarray(ctx["cells"], dtype=np.int64)
  raw = (values.view(cells)
         if isinstance(values, gpython.GkylArray) else np.asarray(values))
  ncomp = raw.shape[-1]
  if ncomp < 2 or ncomp % 2:
    raise ValueError("surface component count must have two equal halves")
  if raw.shape[:-1] != tuple(cells):
    raise ValueError("surface payload does not match the stored cell counts")
  half = ncomp // 2
  out_shape = list(raw.shape)
  out_shape[direction] += 1
  out_shape[-1] = half
  out = np.empty(out_shape, dtype=raw.dtype)
  first = tuple(
      slice(None, -1) if axis == direction else slice(None)
      for axis in range(len(grid)))
  last = tuple(-1 if axis == direction else slice(None)
               for axis in range(len(grid)))
  out[first + (slice(None), )] = raw[..., :half]
  out[last + (slice(None), )] = raw[last + (slice(half, None), )]
  ctx["cells"] = np.asarray(out_shape[:-1], dtype=np.int64)
  ctx["num_comps"] = half
  return grid, (gpython.GkylArray.from_numpy(out) if isinstance(
      values, gpython.GkylArray) else out)
