"""Element-wise difference metrics between two same-shaped arrays."""

from __future__ import annotations

import numpy as np


def max_abs_rel_diff(values0: np.ndarray,
                      values: np.ndarray) -> tuple[float, float]:
  """The largest absolute and relative element-wise differences.

  Mirrors the combined-tolerance metric Gkeyll's regression runner computes
  over two files (``gkeyll/lua/Tool/runregression.lua``'s ``compareFiles``):
  the maximum of ``|values - values0|`` and the maximum of
  ``|values - values0| / |values0|``, each taken over every element. Callers
  that want a pass/fail verdict combine the two (e.g. fail only when *both*
  exceed a tolerance, since a tiny absolute difference can have an enormous
  relative one near zero, and a tiny relative difference can still carry a
  large absolute one far from zero); callers that want a ranking (as
  :mod:`postgkyl.diagnostics.compare` does) use one or the other directly.

  Elements where ``values0`` is exactly zero produce ``inf`` in the
  per-element relative difference (unless the corresponding ``values``
  element is also exactly zero, which gives ``nan``, treated as the diff
  no-op it is). Neither is special-cased away: a caller discarding such a
  file as "no signal to normalize by" is a pass/fail or ranking policy
  decision, out of scope for this pure metric.

  Args:
    values0: Reference ("before"/baseline) array.
    values: Array to compare against the reference.

  Returns:
    ``(max_abs_diff, max_rel_diff)``, both ``float``.

  Raises:
    ValueError: if the two arrays' shapes differ.
  """
  values0 = np.asarray(values0)
  values = np.asarray(values)
  if values0.shape != values.shape:
    raise ValueError(
        f"max_abs_rel_diff: shape mismatch {values0.shape} vs {values.shape}")
  abs_diff = np.abs(values - values0)
  with np.errstate(divide="ignore", invalid="ignore"):
    rel_diff = abs_diff / np.abs(values0)
  max_abs_diff = float(np.max(abs_diff)) if abs_diff.size else 0.0
  # nanmax over an all-NaN array (every element identically zero, so 0/0
  # everywhere) warns and returns nan; that case has zero relative
  # difference by construction (values == values0), not an undefined one.
  finite_rel = rel_diff[~np.isnan(rel_diff)]
  max_rel_diff = float(np.max(finite_rel)) if finite_rel.size else 0.0
  return max_abs_diff, max_rel_diff


__all__ = ["max_abs_rel_diff"]
