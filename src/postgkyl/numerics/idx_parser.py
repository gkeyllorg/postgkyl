"""Parse index / value / slice selectors into NumPy indices (pure)."""

from __future__ import annotations

import numpy as np


def _find_nearest_index(array, value):
  if array is None:
    raise TypeError(
        "Float selector given but no coordinate array to match against.")
  if not np.isfinite(value):
    raise ValueError("Coordinate selector must be finite.")
  if len(array) == 0:
    raise ValueError("Cannot select from an empty coordinate array.")
  return int(np.argmin(np.abs(np.asarray(array) - value)))


def _string_to_index(value: str, array: np.ndarray | None) -> int:
  if not isinstance(value, str):
    raise TypeError("Value is not a string")
  if value.lstrip("-").isdigit():
    return int(value)
  return _find_nearest_index(array, float(value))


def idx_parser(value: int | float | str,
               array: np.ndarray | None = None,
               nodal: bool = False) -> int | slice | tuple:
  """Turn an int/float/str selector into an int index, ``slice``, or tuple.

  - int -> used as-is
  - float -> nearest sample, or nearest cell center for an edge array
  - ``"a,b,c"`` -> tuple of indices
  - ``"a:b"`` -> ``slice``
  - ``"a"`` -> single index

  ``nodal=True`` means the array contains sample positions, not cell edges.
  Equal distances choose the first index; outside coordinates choose an endpoint.
  """
  if array is not None and not nodal:
    array = 0.5 * (np.asarray(array[:-1]) + np.asarray(array[1:]))
  if isinstance(value, (int, np.integer)):
    return int(value)
  if isinstance(value, (float, np.floating)):
    return _find_nearest_index(array, value)
  if isinstance(value, str):
    if len(value.split(",")) > 1:
      return tuple(_string_to_index(i, array) for i in value.split(","))
    if len(value.split(":")) == 2:
      lo, hi = value.split(":")
      if lo == "":
        lo = "0"
      if hi == "":
        hi = str(len(array)) if array is not None else ""
      return slice(_string_to_index(lo, array),
                   _string_to_index(hi, array) if hi else None)
    return _string_to_index(value, array)
  raise TypeError(f"Unsupported selector type: {type(value)!r}")
