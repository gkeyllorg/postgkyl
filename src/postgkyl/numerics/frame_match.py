"""Match the frames of two runs by time (pure NumPy)."""

from __future__ import annotations

import numpy as np


def match_frames(t_a: np.ndarray, t_b: np.ndarray) -> np.ndarray:
  """For each of run a's frame times, the index of run b's nearest frame.

  ``-1`` where b has no frame within half of the coarser of the two runs'
  frame spacings (each run's median spacing): before b starts, after it
  ends, or in a gap of b's frames. ``t_b`` need not be sorted; ties go to
  the earlier time.

  Args:
    t_a: Run a's frame times.
    t_b: Run b's frame times.

  Returns:
    An integer array the length of ``t_a``: indices into ``t_b``, or -1.
  """
  t_a, t_b = np.asarray(t_a, float), np.asarray(t_b, float)
  if not t_a.size or not t_b.size:
    return np.full(t_a.size, -1)
  order = np.argsort(t_b, kind="stable")
  tb = t_b[order]
  i = np.clip(np.searchsorted(tb, t_a), 1, max(tb.size - 1, 1))
  lo = np.clip(i - 1, 0, tb.size - 1)
  hi = np.clip(i, 0, tb.size - 1)
  near = np.where(np.abs(t_a - tb[lo]) <= np.abs(t_a - tb[hi]), lo, hi)
  spacings = [
      float(np.median(np.diff(t))) for t in (np.sort(t_a), tb) if t.size > 1
  ]
  tol = 0.5 * max(spacings) * (1 + 1e-9) if spacings else np.inf
  return np.where(np.abs(tb[near] - t_a) <= tol, order[near], -1)
