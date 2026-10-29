"""Locate toroidal angles in field-aligned coordinates and sample surfaces.

All coordinate information is supplied explicitly.
"""
from collections.abc import Callable

import numpy as np


def binormal_fraction(phi_tor: float | np.ndarray, phi_lo: np.ndarray,
                      span: np.ndarray) -> np.ndarray:
  """Fraction of the binormal period at which toroidal angle ``phi_tor`` lies.

  At each point the binormal domain covers toroidal angles from ``phi_lo``
  across ``span`` (negative when the angle decreases along y). An angle
  inside that wedge, modulo 2*pi, is located exactly. Beyond it, the
  periodic binormal domain repeats with period ``span``; unless 2*pi is a
  whole number of spans, the copies meet with a discontinuity. Arguments
  broadcast together; results lie in ``[0, 1)``.
  """
  offset = np.mod(phi_tor - phi_lo, 2.0 * np.pi * np.sign(span))
  return np.mod(offset / span, 1.0)


def wedge_surface(sample: Callable[[np.ndarray, np.ndarray], np.ndarray],
                  rows: np.ndarray, phi_tor: np.ndarray) -> np.ndarray:
  """Sample the closed boundary of a toroidal wedge as one surface.

  The wedge spans radial ``rows``, ordered inner to outer, and increasing
  toroidal angles ``phi_tor``; its parallel axis is periodic. ``sample(rows,
  angles)`` receives angles shaped ``(k, 1, 1)`` and returns arrays shaped
  ``(k, len(rows), nz, ...)``.

  Returns ``(2 * len(rows) + 2 * len(phi_tor) - 3, nz + 1, ...)`` samples: the
  first cap outward, the outer flux surface forward, the last cap inward, and
  the inner flux surface back. The last row and column repeat the first ones,
  so the surface closes in both directions.
  """
  caps = sample(rows, phi_tor[[0, -1], None, None])
  walls = sample(rows[[0, -1]], phi_tor[:, None, None])
  ring = np.concatenate(
      [caps[0], walls[1:, -1], caps[1, -2::-1], walls[-2::-1, 0]])
  return np.concatenate([ring, ring[:, :1]], axis=1)
