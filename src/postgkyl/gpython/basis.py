"""Gkeyll basis objects + evaluation matrices, through the gpython shim.

``struct gkyl_basis`` carries the basis functions themselves; the shim
dispatches its function pointers in compiled C (``gpython_basis_eval`` & co.), so
the interpolation matrix is assembled by evaluating Gkeyll's own basis at the
interpolation points -- a few hundred calls, cached per basis -- and NumPy
applies it at array speed. The matrices are therefore bit-consistent with the
kernels the simulation used, with zero layout knowledge in Python.

Interpolation points follow the historical postgkyl convention: ``num_interp``
subcell centers per cell, ``z_i = -(n-1)/n + 2 i/n`` on [-1, 1], with
multi-dimensional points ordered Fortran-style (dimension 0 fastest) to match
the per-cell scatter in ``dg/interpolate.py``.
"""

from __future__ import annotations

import numpy as np

from . import _lib


class Basis:
  """A cached Gkeyll basis: opaque handle + the descriptors postgkyl reads."""

  def __init__(self, cap, ndim: int, poly_order: int, num_basis: int, id: str):
    self._cap = cap
    self.ndim = ndim
    self.poly_order = poly_order
    self.num_basis = num_basis
    self.id = id

  def __repr__(self) -> str:
    return (f"<Basis {self.id} ndim={self.ndim} p={self.poly_order} "
            f"N={self.num_basis}>")


_basis_cache: dict[tuple, Basis] = {}
_matrix_cache: dict[tuple, np.ndarray] = {}

# Highest poly_order each basis supports per ndim, mirroring the fixed-size
# `ev[4]` function-pointer tables in gkeyll's
# core/zero/gkyl_cart_modal_{serendip,tensor}_priv.h. Those tables have NO
# runtime bounds checking: gkyl_cart_modal_serendip/tensor assert
# `ndim>0 && ndim<=6` (a process abort on failure, not a Python exception),
# and index poly_order into the 4-slot array with no check at all, so an
# out-of-range poly_order is undefined behavior, not a clean failure. This
# guard must run before every call into the shim; keep it in sync with those
# two headers if Gkeyll ever adds higher-order kernels.
_MAX_POLY_ORDER = {
    "serendipity": {
        1: 3,
        2: 3,
        3: 3,
        4: 3,
        5: 2,
        6: 1
    },
    "tensor": {
        1: 3,
        2: 3,
        3: 2,
        4: 2,
        5: 2,
        6: 1
    },
}

# hybrid/gkhybrid are fixed-poly_order (=1) bases parameterized by
# (cdim, vdim) rather than (ndim, poly_order) -- see
# gkeyll/core/zero/gkyl_cart_modal_{hybrid,gkhybrid}.c. A .gkyl file only
# records the total ndim, not the cdim/vdim split, so this table recovers it
# from the one configuration Gkeyll actually produces for each: PKPM hybrid
# always carries a single parallel-velocity direction (vdim=1, cdim=ndim-1);
# gyrokinetic gkhybrid always carries (vpar, mu) (vdim=2), except the 1x1v
# case, which has no mu direction (vdim=1) -- mirroring the legacy postgkyl
# convention (src_bak/postgkyl/data/{dg.py,computeInterpolationMatrices.py})
# and matching gkeyll/core/unit/ctest_basis.c's own (cdim, vdim) choices.
# Gkeyll's gkhybrid kernel tables are indexed by ndim alone (poly_order fixed
# at 1), so any (cdim, vdim) pair summing to the same ndim would dispatch to
# the identical compiled basis; this table simply names the one physical
# configuration that split corresponds to.
_HYBRID_CDIM_VDIM = {
    "hybrid": {
        2: (1, 1),
        3: (2, 1),
        4: (3, 1)
    },
    "gkhybrid": {
        2: (1, 1),
        3: (1, 2),
        4: (2, 2),
        5: (3, 2)
    },
}


def get_basis(basis_type: str, ndim: int, poly_order: int) -> Basis:
  """A cached, fully-initialized Gkeyll basis object.

  Args:
    basis_type: ``"serendipity"``, ``"tensor"``, ``"hybrid"``, or
      ``"gkhybrid"`` (case-insensitive).
    ndim: number of dimensions. 1..6 for serendipity/tensor; the hybrid
      bases only exist for the ``(cdim, vdim)`` combinations Gkeyll actually
      generates kernels for -- see :data:`_HYBRID_CDIM_VDIM`.
    poly_order: polynomial order for serendipity/tensor (ceiling depends on
      ``(basis_type, ndim)``, see :data:`_MAX_POLY_ORDER`); must be ``1``
      for hybrid/gkhybrid, which have no other order.

  Returns:
    The cached :class:`Basis` (the same object for repeated requests with
    the same arguments).

  Raises:
    ValueError: unknown ``basis_type``, or ``(ndim, poly_order)`` outside
      what Gkeyll's compiled kernel tables support for it. Checked here
      because the C constructors have no such guard themselves (see above).
  """
  basis_type = basis_type.lower()
  key = (basis_type, ndim, poly_order)
  if key in _basis_cache:
    return _basis_cache[key]

  if basis_type in _HYBRID_CDIM_VDIM:
    if poly_order != 1:
      raise ValueError(f"Gkeyll's {basis_type} basis only exists at "
                       f"poly_order 1, got {poly_order}")
    cdim_vdim = _HYBRID_CDIM_VDIM[basis_type].get(ndim)
    if cdim_vdim is None:
      raise ValueError(f"Gkeyll's {basis_type} basis supports ndim "
                       f"{sorted(_HYBRID_CDIM_VDIM[basis_type])}, got {ndim}")
    cdim, vdim = cdim_vdim
    cap = _lib.require().basis_new_hybrid(basis_type, cdim, vdim)
  else:
    limits = _MAX_POLY_ORDER.get(basis_type)
    if limits is None:
      raise ValueError(
          f"unknown basis_type '{basis_type}'; expected one of "
          f"{sorted(set(_MAX_POLY_ORDER) | set(_HYBRID_CDIM_VDIM))}")
    max_p = limits.get(ndim)
    if max_p is None:
      raise ValueError(f"Gkeyll's {basis_type} basis supports ndim 1..6, "
                       f"got {ndim}")
    if not 0 <= poly_order <= max_p:
      raise ValueError(f"Gkeyll's {basis_type} basis in {ndim}D supports "
                       f"poly_order 0..{max_p}, got {poly_order}")
    cap = _lib.require().basis_new(basis_type, ndim, poly_order)

  nd, p, nb, bid = _lib.require().basis_info(cap)
  _basis_cache[key] = Basis(cap, nd, p, nb, bid)
  return _basis_cache[key]


def num_basis(basis_type: str, ndim: int, poly_order: int) -> int:
  """Number of DG basis functions, straight from Gkeyll."""
  return get_basis(basis_type, ndim, poly_order).num_basis


def cdim_vdim(basis_type: str, ndim: int) -> tuple[int, int]:
  """``(cdim, vdim)`` for a basis: the hybrid/gkhybrid split from
  :data:`_HYBRID_CDIM_VDIM`, or ``(ndim, 0)`` for a plain configuration-space
  basis (serendipity/tensor have no velocity-space concept).

  Raises:
    ValueError: ``basis_type`` is hybrid/gkhybrid and ``ndim`` is not one of
      the ``(cdim, vdim)`` combinations Gkeyll generates kernels for.
  """
  basis_type = basis_type.lower()
  if basis_type in _HYBRID_CDIM_VDIM:
    cdim_vdim_ = _HYBRID_CDIM_VDIM[basis_type].get(ndim)
    if cdim_vdim_ is None:
      raise ValueError(f"Gkeyll's {basis_type} basis supports ndim "
                       f"{sorted(_HYBRID_CDIM_VDIM[basis_type])}, got {ndim}")
    return cdim_vdim_
  return (ndim, 0)


def interpolation_points_1d(num_interp: int) -> np.ndarray:
  """Subcell-center evaluation points on [-1, 1] (legacy postgkyl convention)."""
  n = num_interp
  return np.array([-(n - 1.0) / n + 2.0 * i / n for i in range(n)])


def tensor_points(pts_1d: np.ndarray, ndim: int) -> np.ndarray:
  """``(len(pts_1d)**ndim, ndim)`` tensor-product point set, dimension 0
  fastest (Fortran multi-index order -- the convention every consumer uses)."""
  n = len(pts_1d)
  shape = (n, ) * ndim
  out = np.empty((n**ndim, ndim))
  for i in range(n**ndim):
    idx = np.unravel_index(i, shape, order="F")
    out[i, :] = [pts_1d[idx[d]] for d in range(ndim)]
  return out


def eval_matrix(basis_type: str, ndim: int, poly_order: int,
                points: np.ndarray) -> np.ndarray:
  """``(npts, num_basis)`` matrix ``M[i, j] = b_j(z_i)`` at arbitrary points
  in the reference cell [-1, 1]^ndim -- built by evaluating Gkeyll's own basis
  through the shim. The workhorse behind every value_form change *and*
  the plotting bridge."""
  g0 = _lib.require()
  basis = get_basis(basis_type, ndim, poly_order)
  points = np.atleast_2d(np.asarray(points, dtype=np.float64))
  mat = np.empty((points.shape[0], basis.num_basis))
  for i, pt in enumerate(points):
    mat[i, :] = g0.basis_eval(basis._cap, pt)
  return mat


def _cached(key, build):
  if key not in _matrix_cache:
    mat = build()
    mat.flags.writeable = False
    _matrix_cache[key] = mat
  return _matrix_cache[key]


def interpolation_matrix(basis_type: str, ndim: int, poly_order: int,
                         num_interp: int) -> np.ndarray:
  """Evaluation matrix at ``num_interp`` subcell centers per dimension.

  Row ``i`` corresponds to the point with multi-index
  ``np.unravel_index(i, [num_interp]*ndim, order="F")`` -- dimension 0 fastest,
  matching the consumer in ``dg/interpolate.py``.
  """
  return _cached(("interpolation", basis_type, ndim, poly_order, num_interp),
                 lambda: eval_matrix(
                     basis_type, ndim, poly_order,
                     tensor_points(interpolation_points_1d(num_interp), ndim)))


# ------------------------------------------------- nodal <-> modal (exact)
def node_coords(basis_type: str, ndim: int, poly_order: int) -> np.ndarray:
  """``(num_basis, ndim)`` node coordinates from the basis ``node_list``."""
  basis = get_basis(basis_type, ndim, poly_order)
  return _lib.require().basis_node_list(basis._cap)


def nodal_to_modal_matrix(basis_type: str, ndim: int,
                          poly_order: int) -> np.ndarray:
  """Exact N×N change of basis, from Gkeyll's ``nodal_to_modal``
  (columns = images of the nodal unit vectors)."""

  def build():
    g0 = _lib.require()
    basis = get_basis(basis_type, ndim, poly_order)
    nb = basis.num_basis
    mat = np.empty((nb, nb))
    for j in range(nb):
      fin = np.zeros(nb)
      fin[j] = 1.0
      mat[:, j] = g0.basis_nodal_to_modal(basis._cap, fin)
    return mat

  return _cached(("n2m", basis_type, ndim, poly_order), build)


def modal_to_nodal_matrix(basis_type: str, ndim: int,
                          poly_order: int) -> np.ndarray:
  """Evaluation at the basis nodes -- the exact inverse of ``nodal_to_modal``."""
  return _cached(("m2n", basis_type, ndim, poly_order), lambda: eval_matrix(
      basis_type, ndim, poly_order, node_coords(basis_type, ndim, poly_order)))


# ------------------------------------------- quadrature <-> modal (projection)
def gauss_quad(ndim: int, num_quad: int):
  """Tensor-product Gauss–Legendre rule on [-1, 1]^ndim:
  ``(points (nq**ndim, ndim), weights (nq**ndim,))``, dimension 0 fastest."""
  p1, w1 = np.polynomial.legendre.leggauss(num_quad)
  pts = tensor_points(p1, ndim)
  shape = (num_quad, ) * ndim
  w = np.empty(num_quad**ndim)
  for i in range(w.size):
    idx = np.unravel_index(i, shape, order="F")
    w[i] = np.prod([w1[idx[d]] for d in range(ndim)])
  return pts, w


def modal_to_quad_matrix(basis_type: str, ndim: int, poly_order: int,
                         num_quad: int) -> np.ndarray:
  """``(nq**ndim, num_basis)`` -- evaluate the expansion at the Gauss points."""
  return _cached(("m2q", basis_type, ndim, poly_order, num_quad),
                 lambda: eval_matrix(basis_type, ndim, poly_order,
                                     gauss_quad(ndim, num_quad)[0]))


def quad_to_modal_matrix(basis_type: str, ndim: int, poly_order: int,
                         num_quad: int) -> np.ndarray:
  """``(num_basis, nq**ndim)`` quadrature projection ``c_j = sum_i w_i b_j(z_i) f_i``.

  Exact whenever the integrand ``f·b_j`` has degree ≤ 2·num_quad−1 (the bases
  are orthonormal on the reference cell, so no mass-matrix solve is needed).
  ``quad_to_modal @ modal_to_quad == I`` for ``num_quad >= p+1``.
  """

  def build():
    pts, w = gauss_quad(ndim, num_quad)
    B = eval_matrix(basis_type, ndim, poly_order, pts)
    return B.T * w  # (N, npts): rows b_j(z_i), scaled by the weights

  return _cached(("q2m", basis_type, ndim, poly_order, num_quad), build)
