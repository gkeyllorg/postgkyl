"""Vlasov collision operators through the compiled, optional module shim."""

from __future__ import annotations

import numpy as np

from . import _lib
from .array import GkylArray
from .basis import get_basis


def coulomb_log(ns, nr, ts, tr, bmag, *, mass_self: float, mass_other: float,
                charge_self: float, charge_other: float, epsilon_0: float,
                hbar: float):
  """Native species-ordered Coulomb logarithm on broadcast point values.

  Temperatures are energies, so T/m = v_t^2. All arguments must use one
  consistent unit system. The collision updater averages this expression
  with the expression obtained by interchanging the two species.
  """
  lib = _lib.require_module("vlasov", "coulomb_log")
  samples = np.stack(np.broadcast_arrays(ns, nr, ts, tr, bmag), axis=-1)
  out = lib.coulomb_log(samples, mass_self, mass_other, charge_self,
                        charge_other, epsilon_0, hbar)
  return out.item() if out.ndim == 0 else out


def spitzer_coll_freq(basis_type: str,
                      ndim: int,
                      poly_order: int,
                      moms_self: GkylArray,
                      moms_other: GkylArray,
                      bmag: GkylArray,
                      *,
                      mass_self: float,
                      mass_other: float,
                      charge_self: float,
                      charge_other: float,
                      epsilon_0: float,
                      hbar: float,
                      nufrac: float = 1.0,
                      vt_sq_min_self: float = 0.0,
                      vt_sq_min_other: float = 0.0,
                      num_quad: int | None = None) -> GkylArray:
  """Project Spitzer frequency from modal [n, u..., T/m] moment blocks.

  Gkeyll computes the symmetric Coulomb log from cell averages, evaluates
  the density/thermal-speed factor at Gauss points, and projects it back.
  Velocity floors are squared speeds. Quadrature supports 1..64 points per
  direction; the default is poly_order + 2. Inputs must be CPU modal arrays.
  """
  lib = _lib.require_module("vlasov", "spitzer_coll_freq")
  if basis_type not in ("serendipity", "tensor") or not 1 <= ndim <= 3:
    raise ValueError("Spitzer frequency needs a 1..3D configuration basis")
  basis = get_basis(basis_type, ndim, poly_order)
  out = GkylArray.alloc(basis.num_basis, moms_self.size)
  lib.spitzer_coll_freq(basis._cap, moms_self._cap, moms_other._cap, bmag._cap,
                        mass_self, mass_other, charge_self, charge_other,
                        epsilon_0, hbar, nufrac, vt_sq_min_self,
                        vt_sq_min_other,
                        poly_order + 2 if num_quad is None else num_quad,
                        out._cap)
  return out
