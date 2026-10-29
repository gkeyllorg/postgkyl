"""Gyrokinetic moments through Gkeyll's optional native shim."""

from __future__ import annotations

import numpy as np

from . import _lib
from .array import GkylArray
from .basis import get_basis


def mom_gyrokinetic(cdim: int,
                    vdim: int,
                    poly_order: int,
                    distribution: GkylArray,
                    bmag: GkylArray,
                    lower,
                    upper,
                    cells,
                    *,
                    mass: float,
                    moment: str = "M0",
                    charge: float = 0.0,
                    phi: GkylArray | None = None) -> GkylArray:
  """Compute native velocity moments on a uniform physical velocity grid.

  Configuration basis: serendipity. Phase basis: gkhybrid at p1,
  serendipity at p2. Supported splits: 1x1v, 1x2v, 2x2v (p1/p2), 3x2v
  (p1). Perpendicular moments require 2v; HamiltonianMoments requires phi.

  Retains Gkeyll's normalization: 2v moments integrate the supplied stored
  distribution with measure (2*pi/m) dv_parallel dmu. A spatial Jacobian
  already contained in the input remains in the output. No implicit B or
  geometry-Jacobian multiplication/division is performed.
  """
  lib = _lib.require_module("gyrokinetic", "mom_gyrokinetic")
  if (cdim, vdim) not in ((1, 1), (1, 2), (2, 2),
                          (3, 2)) or poly_order not in (1, 2):
    raise ValueError("unsupported gyrokinetic basis split or order")
  if cdim == 3 and poly_order != 1:
    raise ValueError("3x2v gyrokinetic moments support p1 only")
  cb = get_basis("serendipity", cdim, poly_order)
  pb = get_basis("gkhybrid" if poly_order == 1 else "serendipity",
                 cdim + vdim,
                 poly_order,
                 cdim=cdim,
                 vdim=vdim)
  cells = np.asarray(cells, dtype=np.int32)
  return GkylArray(
      lib.mom_gyrokinetic(cb._cap, pb._cap, lower, upper, cells, mass, charge,
                          distribution._cap, bmag._cap,
                          None if phi is None else phi._cap, moment))
