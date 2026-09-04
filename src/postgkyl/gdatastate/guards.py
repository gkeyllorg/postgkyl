"""The shared field-domain guard used by field-only verbs and diagnostics.

Centralizes the check-and-raise boilerplate that was independently retyped
across several ``operations`` physics verbs (moved to ``diagnostics`` by layer 10):
each caller keeps its own ``reason`` clause (why *this* function's math has
no meaning on raw modal coefficients), but the check itself --
``backend == "gkyl"`` -> raise with the standard ".interpolate() first" message
shape -- has one home. This is a state-invariant helper, not a verb, so it
lives on ``gdatastate`` (which stays verb-less) rather than ``operations``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .gdatastate import GDataState


def require_field_domain(data: "GDataState", who: str, reason: str) -> None:
  """Raise if ``data`` is native modal (gkyl-backed) DG coefficients.

  Args:
    data: The dataset to check.
    who: The verb (or argument) name to name in the error message.
    reason: The clause explaining why raw coefficients are unusable here,
      e.g. ``"rotating raw DG coefficients would mix basis functions"``.

  Raises:
    ValueError: if ``data.backend == "gkyl"``.
  """
  if data.backend == "gkyl":
    raise ValueError(
        f"{who} operates on interpolated (NumPy) values; call .interpolate() "
        f"first -- {reason}.")
