"""Thin wrappers over Gkeyll's compiled operators (weak algebra & reductions).

Each function takes :class:`~postgkyl.gpython.array.GkylArray` operands plus the
basis descriptor and calls one shim entry point; the per-field loop for
``ncomp == nfields * num_basis`` arrays and all transient C resources
(``gkyl_dg_bin_op_mem``, integrate updaters) live inside the compiled shim.

Python-side capability guards mirror Gkeyll's own limits (which are C
``assert``s -- letting them fire would abort the process).
"""

from __future__ import annotations

import numpy as np

from . import _lib
from .array import GkylArray
from .basis import get_basis

_WEAK_BASES = ("serendipity", "tensor")  # dg_bin_ops: assert(false) otherwise

# enum gkyl_array_op / gkyl_array_integrate_op ordinals used by the shim
REDUCE_OPS = {"min": 0, "max": 1, "sum": 2}
GKYL_MIN, GKYL_MAX, GKYL_SUM = 0, 1, 2
INTEGRATE_OPS = {"none": 0, "abs": 1, "sq": 2}

# Weak mul/div kernel tables (gkyl_dg_bin_ops_priv.h ser_mul_list/ten_mul_list/
# ser_div_set_list/ten_div_set_list) are fixed-size [ndim][poly_order] arrays
# covering ONLY ndim 1..3 -- narrower than the basis module's own eval range.
# ndim >= 4 hits `assert(dim < 4)` in choose_ser_mul_kern (a process abort);
# an out-of-table poly_order for tensor (p3 at ndim 2-3) returns a NULL
# kernel pointer that gkyl_dg_mul_op/div_op call with NO null check at all
# (a segfault, not an assert). Both must be refused here.
_WEAK_MAX_POLY_ORDER = {
    "serendipity": {
        1: 3,
        2: 3,
        3: 3
    },
    "tensor": {
        1: 3,
        2: 2,
        3: 2
    },
}
# gkyl_dg_inv_op's kernel table (ser_inv_list) has no dim bound check
# whatsoever (a raw out-of-bounds array read for ndim >= 4) and only fills
# poly_order == 1 for ndim 1..3.
_WEAK_INV_DIMS = (1, 2, 3)


def _check_weak(basis_type: str, ndim: int, poly_order: int, *arrays:
                GkylArray):
  basis_type = basis_type.lower()
  limits = _WEAK_MAX_POLY_ORDER.get(basis_type)
  if limits is None:
    raise NotImplementedError(
        f"Gkeyll weak ops support {_WEAK_BASES}, not '{basis_type}'")
  max_p = limits.get(ndim)
  if max_p is None:
    raise NotImplementedError(
        f"Gkeyll's weak (DG) mul/div kernels support ndim 1..3, got {ndim}")
  if not 0 <= poly_order <= max_p:
    raise NotImplementedError(
        f"Gkeyll's weak {basis_type} mul/div kernels in {ndim}D support "
        f"poly_order 0..{max_p}, got {poly_order}")
  first = arrays[0]
  for a in arrays[1:]:
    if (a.ncomp, a.size) != (first.ncomp, first.size):
      raise ValueError(f"operand shape mismatch: {a.ncomp}x{a.size} vs "
                       f"{first.ncomp}x{first.size}")


def _fields(arr: GkylArray, num_basis: int) -> int:
  if arr.ncomp % num_basis:
    raise ValueError(f"ncomp {arr.ncomp} is not a multiple of "
                     f"num_basis {num_basis}")
  return arr.ncomp // num_basis


def weak_mul(basis_type: str, ndim: int, poly_order: int, a: GkylArray,
             b: GkylArray) -> GkylArray:
  """Weak (DG) product ``a * b``, field by field, via ``gkyl_dg_mul_op``."""
  _check_weak(basis_type, ndim, poly_order, a, b)
  basis = get_basis(basis_type, ndim, poly_order)
  _fields(a, basis.num_basis)
  out = GkylArray.alloc(a.ncomp, a.size)
  _lib.require().dg_mul(basis._cap, out._cap, a._cap, b._cap)
  return out


def weak_div(basis_type: str, ndim: int, poly_order: int, a: GkylArray,
             b: GkylArray) -> GkylArray:
  """Weak (DG) quotient ``a / b`` via ``gkyl_dg_div_op`` (per-cell solve)."""
  _check_weak(basis_type, ndim, poly_order, a, b)
  basis = get_basis(basis_type, ndim, poly_order)
  _fields(a, basis.num_basis)
  out = GkylArray.alloc(a.ncomp, a.size)
  _lib.require().dg_div(basis._cap, out._cap, a._cap, b._cap)
  return out


def weak_inv(basis_type: str, ndim: int, poly_order: int,
             a: GkylArray) -> GkylArray:
  """Weak reciprocal ``1 / a`` via ``gkyl_dg_inv_op`` (Gkeyll: ser p=1, ndim<=3 only)."""
  if basis_type.lower() != "serendipity" or poly_order != 1:
    raise NotImplementedError(
        "gkyl_dg_inv_op supports serendipity p=1 only (a Gkeyll limit); "
        "use weak division instead.")
  if ndim not in _WEAK_INV_DIMS:
    raise NotImplementedError(
        f"gkyl_dg_inv_op supports ndim {_WEAK_INV_DIMS} only, got {ndim} "
        "(a Gkeyll limit; its kernel table has no bounds check at all, so "
        "this guard is load-bearing, not decorative)")
  basis = get_basis(basis_type, ndim, poly_order)
  _fields(a, basis.num_basis)
  out = GkylArray.alloc(a.ncomp, a.size)
  _lib.require().dg_inv(basis._cap, out._cap, a._cap)
  return out


# -------------------------------------------------- conf-space x phase-space
# gkyl_dg_mul_conf_phase_op_range picks its kernel from the PHASE basis type
# alone (choose_mul_conf_phase_kern in gkyl_dg_bin_ops_priv.h); for
# hybrid/gkhybrid the conf poly_order it reads is unused by that branch, so
# the only real requirement (Gkeyll's own PKPM/GK convention) is a
# serendipity conf basis. Every (cdim, vdim) split our own basis.py
# convention (_HYBRID_CDIM_VDIM) derives from a valid hybrid/gkhybrid ndim
# already has a populated cross_mul_list entry -- verified by hand against
# hyb_cross_mul_list/gkhyb_cross_mul_list, so no extra table is needed there.
# serendipity/tensor phase bases have a genuinely holey (cdim, pdim,
# poly_order) kernel table -- unlike same-basis weak_mul, most combinations
# a valid same-basis object could have are simply absent (NULL function
# pointer, no null check in gkyl_dg_mul_conf_phase_op_range) -- so those are
# guarded explicitly below, transcribed from ser_cross_mul_list /
# ten_cross_mul_list in gkyl_dg_bin_ops_priv.h.
_CROSS_MUL_SER = {
    2: {
        1: {1, 2, 3}
    },
    3: {
        1: {1, 2, 3},
        2: {1, 2, 3}
    },
    4: {
        1: {1, 2, 3},
        2: {1, 2, 3},
        3: {1, 2, 3}
    },
    5: {
        2: {1, 2},
        3: {1, 2}
    },
    6: {
        3: {1}
    },
}
_CROSS_MUL_TEN = {
    2: {
        1: {1, 2}
    },
    3: {
        1: {1, 2},
        2: {1, 2}
    },
    4: {
        1: {1, 2},
        2: {1, 2},
        3: {1}
    },
    5: {
        2: {1, 2},
        3: {1}
    },
    6: {
        3: {1}
    },
}
_CROSS_MUL_TABLES = {"serendipity": _CROSS_MUL_SER, "tensor": _CROSS_MUL_TEN}


def _check_mul_conf_phase(conf_basis_type: str, phase_basis_type: str,
                          conf_ndim: int, phase_ndim: int, poly_order: int):
  conf_basis_type = conf_basis_type.lower()
  phase_basis_type = phase_basis_type.lower()
  if phase_ndim <= conf_ndim:
    raise ValueError(
        f"phase_ndim ({phase_ndim}) must exceed conf_ndim ({conf_ndim})")
  if phase_basis_type in ("hybrid", "gkhybrid"):
    if conf_basis_type != "serendipity":
      raise NotImplementedError(
          "Gkeyll pairs a hybrid/gkhybrid phase basis with a serendipity "
          f"conf basis only (its own PKPM/GK convention), not "
          f"'{conf_basis_type}'")
    return
  if phase_basis_type in ("serendipity", "tensor"):
    if conf_basis_type != phase_basis_type:
      raise NotImplementedError(
          "gkyl_dg_mul_conf_phase_op_range picks its kernel from the phase "
          f"basis type alone ('{phase_basis_type}'); pair it with a conf "
          f"basis of the same type, not '{conf_basis_type}'")
    valid = _CROSS_MUL_TABLES[phase_basis_type].get(phase_ndim,
                                                    {}).get(conf_ndim)
    if not valid or poly_order not in valid:
      raise NotImplementedError(
          f"Gkeyll has no {phase_basis_type} conf*phase cross-mul kernel "
          f"for conf_ndim={conf_ndim}, phase_ndim={phase_ndim}, "
          f"poly_order={poly_order}")
    return
  raise NotImplementedError(
      "Gkeyll's conf*phase cross-mul supports serendipity, tensor, hybrid, "
      f"gkhybrid phase bases, not '{phase_basis_type}'")


def weak_mul_conf_phase(conf_basis_type: str, conf_ndim: int,
                        phase_basis_type: str, phase_ndim: int, poly_order: int,
                        conf_cells, phase_cells, cop: GkylArray,
                        pop: GkylArray) -> GkylArray:
  """Conf-space x phase-space weak product ``cop * pop`` via
  ``gkyl_dg_mul_conf_phase_op_range`` -- e.g. a density (conf-space) times a
  distribution function (phase-space) in PKPM/gyrokinetic post-processing.

  Unlike :func:`weak_mul`, this is single-field only on both sides (the
  underlying kernel takes no field-index arguments): ``cop.ncomp`` must
  equal the conf basis's ``num_basis`` and ``pop.ncomp`` the phase basis's.

  ``conf_cells``/``phase_cells`` are each grid's per-dimension cell counts
  (e.g. ``rio``'s ``grid["cells"]``) -- Gkeyll maps each phase cell to its
  conf cell by dropping the velocity-space indices, so both cell counts are
  needed to build matching index ranges; ``conf_cells`` must equal the
  leading ``conf_ndim`` entries of ``phase_cells``.

  The dispatch is asymmetric: Gkeyll chooses the kernel from the PHASE
  basis type alone, so ``conf_basis_type`` must be ``"serendipity"`` when
  pairing with hybrid/gkhybrid, or match ``phase_basis_type`` exactly for
  serendipity/tensor.
  """
  _check_mul_conf_phase(conf_basis_type, phase_basis_type, conf_ndim,
                        phase_ndim, poly_order)
  cbasis = get_basis(conf_basis_type, conf_ndim, poly_order)
  pbasis = get_basis(phase_basis_type, phase_ndim, poly_order)
  if cop.ncomp != cbasis.num_basis:
    raise ValueError(
        f"cop.ncomp ({cop.ncomp}) must equal the conf basis's num_basis "
        f"({cbasis.num_basis}); mul_conf_phase is single-field only")
  if pop.ncomp != pbasis.num_basis:
    raise ValueError(
        f"pop.ncomp ({pop.ncomp}) must equal the phase basis's num_basis "
        f"({pbasis.num_basis}); mul_conf_phase is single-field only")
  conf_cells = np.asarray(conf_cells, dtype=np.int32)
  phase_cells = np.asarray(phase_cells, dtype=np.int32)
  out = GkylArray.alloc(pop.ncomp, pop.size)
  _lib.require().dg_mul_conf_phase(cbasis._cap, pbasis._cap, out._cap, cop._cap,
                                   pop._cap, conf_cells, phase_cells)
  return out


# ------------------------------------------------------- linear coefficient ops
def lincomb(ca: float, a: GkylArray, cb: float, b: GkylArray) -> GkylArray:
  """``ca*a + cb*b`` on the DG coefficients (gkyl_array_set + accumulate)."""
  if (a.ncomp, a.size) != (b.ncomp, b.size):
    raise ValueError("operand shape mismatch in lincomb")
  g0 = _lib.require()
  out = GkylArray.alloc(a.ncomp, a.size)
  g0.array_set(out._cap, ca, a._cap)
  g0.array_accumulate(out._cap, cb, b._cap)
  return out


def scale(a: GkylArray, factor: float) -> GkylArray:
  """``factor * a`` (gkyl_array_scale on a clone; the input is untouched)."""
  out = a.clone()
  _lib.require().array_scale(out._cap, factor)
  return out


def shiftc(a: GkylArray, val: float, comp: int) -> GkylArray:
  """Add ``val`` to component ``comp`` of every cell (gkyl_array_shiftc)."""
  out = a.clone()
  _lib.require().array_shiftc(out._cap, float(val), comp)
  return out


# ---------------------------------------------------------------- reductions
def reduce(a: GkylArray, op: int) -> np.ndarray:
  """Per-component MIN/MAX/SUM over all cells (gkyl_array_reduce).

  This reduces the raw DG **coefficients**: exact for ``"sum"`` (the sum of
  coefficients over cells is linear), but NOT the field's true min/max -- a
  DG expansion can exceed its coefficient values between nodes. Use
  :func:`dg_reduce` for the field-aware version.
  """
  return _lib.require().array_reduce(a._cap, op)


def dg_reduce(basis_type: str, ndim: int, poly_order: int, a: GkylArray,
              comp: int, op: str) -> float:
  """MIN/MAX/SUM of the field ``comp`` actually represents (gkyl_array_dg_reducec).

  Evaluates the DG expansion at each cell's Gauss-Legendre quadrature nodes
  and reduces those values -- the true min/max/sum of the represented field,
  exact for ``"sum"`` and correct (not merely coefficient-bounded) for
  ``"min"``/``"max"`` to quadrature precision (exact for polynomials the
  quadrature integrates exactly, i.e. always for a basis's own degree).

  Args:
    basis_type: ``"serendipity"`` or ``"tensor"``.
    ndim: number of dimensions the basis was built for.
    poly_order: polynomial order the basis was built for.
    a: array whose ``ncomp`` is a multiple of the basis's ``num_basis``.
    comp: 0-based field index (NOT a coefficient offset).
    op: one of ``"min"``, ``"max"``, ``"sum"``.

  Returns:
    The reduced scalar.

  Raises:
    ValueError: unknown ``op``, or ``comp`` out of range for ``a``'s fields.
  """
  if op not in REDUCE_OPS:
    raise ValueError(f"dg_reduce op '{op}' not in {sorted(REDUCE_OPS)}")
  basis = get_basis(basis_type, ndim, poly_order)
  nfields = _fields(a, basis.num_basis)
  if not 0 <= comp < nfields:
    raise ValueError(f"comp {comp} out of range for {nfields} field(s)")
  return float(_lib.require().array_dg_reduce(basis._cap, a._cap, comp,
                                              REDUCE_OPS[op]))


def array_average(grid: dict,
                  basis_type: str,
                  poly_order: int,
                  ndim_avg: int,
                  cells_avg,
                  avg_dim,
                  a: GkylArray,
                  weight: GkylArray | None = None) -> GkylArray:
  """Single-field weighted (or plain) average of ``a`` via ``gkyl_array_average``:
  ``int f w dx^avg / int w dx^avg`` (or ``int f dx^avg / int dx^avg`` when
  ``weight`` is omitted).

  ``grid`` is the donor grid dict (ndim/lower/upper/cells, e.g. from ``rio``);
  ``avg_dim`` flags (length ``grid["ndim"]``, 1 = averaged, 0 = kept) which
  donor dims are reduced. ``ndim_avg``/``cells_avg`` describe the target: the
  surviving dims' cell counts, or ``ndim_avg=1``/``cells_avg=[1]`` for a full
  reduction (Gkeyll's own convention -- there is no true 0-dimensional
  basis). Single-field only: ``a``/``weight`` must carry exactly one donor
  basis's worth of coefficients (``a.ncomp == num_basis``); a multi-field
  caller loops field by field (:func:`postgkyl.dg.modal.average`).

  Guarded to the kernel set compiled into libg0core (serendipity p1-p2,
  donor ndim 1-3 -- ``gkyl_array_average_new`` asserts ``poly_order <= 2``
  and its kernel-choice table has no bound check past that).
  """
  if basis_type.lower() != "serendipity" or poly_order not in (1, 2):
    raise NotImplementedError(
        "gkyl_array_average kernels in libg0core cover serendipity p1-p2")
  ndim = int(grid["ndim"])
  if ndim not in (1, 2, 3):
    raise NotImplementedError(
        f"gkyl_array_average kernels in libg0core cover ndim 1-3, got {ndim}")
  basis = get_basis(basis_type, ndim, poly_order)
  basis_avg = get_basis(basis_type, ndim_avg, poly_order)
  if a.ncomp != basis.num_basis:
    raise ValueError(
        f"average: a.ncomp ({a.ncomp}) must equal the donor basis's "
        f"num_basis ({basis.num_basis}); average is single-field only")
  if weight is not None and (weight.ncomp, weight.size) != (basis.num_basis,
                                                            a.size):
    raise ValueError(
        f"average: weight must be single-field ({basis.num_basis} comps) "
        f"and share the donor array's size ({a.size} cells)")
  lower = np.asarray(grid["lower"], dtype=np.float64)
  upper = np.asarray(grid["upper"], dtype=np.float64)
  cells = np.asarray(grid["cells"], dtype=np.int32)
  if int(np.prod(cells)) != a.size:
    raise ValueError(f"grid cells {tuple(cells)} do not cover the array "
                     f"({int(np.prod(cells))} vs {a.size} cells)")
  cells_avg = np.asarray(cells_avg, dtype=np.int32)
  avg_dim = np.asarray(avg_dim, dtype=np.int32)
  out = GkylArray.alloc(basis_avg.num_basis, int(np.prod(cells_avg)))
  _lib.require().array_average(lower, upper, cells, basis._cap, basis_avg._cap,
                               cells_avg, avg_dim,
                               weight._cap if weight is not None else None,
                               a._cap, out._cap)
  return out


def integrate(grid: dict,
              basis_type: str,
              poly_order: int,
              a: GkylArray,
              op: str = "none",
              factor: float = 1.0) -> np.ndarray:
  """``int dx op(f)`` per field via ``gkyl_array_integrate`` -- one value per field.

  ``grid`` is the dict from ``rio`` (ndim/lower/upper/cells). Guarded to the
  kernel set compiled into libg0core (serendipity p1-p2, ndim 1-3, for
  none/abs/sq) -- ``gkyl_array_integrate_choose_kernel`` indexes its kernel
  table by ``ndim-1``/``poly_order-1`` with no bound past an
  ``assert(up->kernel)`` that a genuinely out-of-table ndim can dodge (an
  out-of-bounds array read that happens to be non-NULL), so ndim is checked
  here rather than left to that assert.
  """
  if op not in INTEGRATE_OPS:
    raise ValueError(f"integrate op '{op}' not in {sorted(INTEGRATE_OPS)}")
  if basis_type.lower() != "serendipity" or poly_order not in (1, 2):
    raise NotImplementedError(
        "gkyl_array_integrate kernels in libg0core cover serendipity p1-p2")
  ndim = int(grid["ndim"])
  if ndim not in (1, 2, 3):
    raise NotImplementedError(
        f"gkyl_array_integrate kernels in libg0core cover ndim 1-3, got {ndim}")
  basis = get_basis(basis_type, ndim, poly_order)
  nfields = _fields(a, basis.num_basis)
  lower = np.asarray(grid["lower"], dtype=np.float64)
  upper = np.asarray(grid["upper"], dtype=np.float64)
  cells = np.asarray(grid["cells"], dtype=np.int32)
  if int(np.prod(cells)) != a.size:
    raise ValueError(f"grid cells {tuple(cells)} do not cover the array "
                     f"({int(np.prod(cells))} vs {a.size} cells)")
  return _lib.require().array_integrate(lower, upper, cells,
                                        basis._cap, nfields, INTEGRATE_OPS[op],
                                        float(factor), a._cap)


# -------------------------------------------------------------------- powsqrt
def powsqrt(basis_type: str,
            ndim: int,
            poly_order: int,
            cells,
            a: GkylArray,
            exponent: float,
            num_quad: int | None = None) -> GkylArray:
  """Single-field ``pow(sqrt(a), exponent)`` (i.e. ``a ** (exponent/2)``) via
  ``gkyl_proj_powsqrt_on_basis`` -- a Gauss-Legendre-quadrature projection,
  not a fixed per-(basis_type, ndim, poly_order) kernel table like every
  other function in this module: the real updater works off the basis's own
  ``eval`` callback, so there is no coverage guard here beyond the shape
  check below.

  A negative value at a quadrature node is clamped to ``1e-40`` by the
  updater itself (Gkeyll's own convention), which can differ from the DG
  coefficients' own sign between nodes -- this is not re-validated here.

  ``cells`` is the grid's per-dimension cell count (e.g. ``ctx["cells"]``);
  no physical extent is needed, only cell indexing. ``num_quad`` defaults to
  ``poly_order + 1``, matching the gyrokinetic app's own use of this
  updater.
  """
  basis = get_basis(basis_type, ndim, poly_order)
  if a.ncomp != basis.num_basis:
    raise ValueError(
        f"powsqrt: a.ncomp ({a.ncomp}) must equal the basis's num_basis "
        f"({basis.num_basis}); powsqrt is single-field only")
  if num_quad is None:
    num_quad = poly_order + 1
  if num_quad < 1:
    raise ValueError(f"num_quad must be >= 1, got {num_quad}")
  cells = np.asarray(cells, dtype=np.int32)
  if int(np.prod(cells)) != a.size:
    raise ValueError(f"cells {tuple(cells)} do not cover the array "
                     f"({int(np.prod(cells))} vs {a.size} cells)")
  out = GkylArray.alloc(basis.num_basis, a.size)
  _lib.require().powsqrt(basis._cap, int(num_quad), float(exponent), cells,
                         out._cap, a._cap)
  return out


# --------------------------------------------------------------- differentiate
# gkyl_dg_differentiate_op_local's kernel tables (gkyl_dg_differentiate_priv.h
# ser_differentiate_list/ten_differentiate_list) cover only serendipity and
# tensor, with NO bounds check at all in the dispatch (an unconditional
# `assert(diff_op)` on a NULL table entry -- a process abort, not a Python
# exception), so this guard must run before every call.
_DIFFERENTIATE_MAX_POLY_ORDER = {
    "serendipity": {
        1: 2,
        2: 2,
        3: 1
    },
    "tensor": {
        1: 2,
        2: 2
    },  # ndim 3: no tensor differentiate kernels at all
}


def _check_differentiate(basis_type: str, ndim: int, poly_order: int, dir: int,
                         diff_order: int):
  basis_type = basis_type.lower()
  limits = _DIFFERENTIATE_MAX_POLY_ORDER.get(basis_type)
  if limits is None:
    raise NotImplementedError(
        f"gkyl_dg_differentiate_op_local supports serendipity/tensor, not "
        f"'{basis_type}'")
  max_p = limits.get(ndim)
  if max_p is None:
    raise NotImplementedError(
        f"Gkeyll's {basis_type} differentiate kernels support ndim "
        f"{sorted(limits)}, got {ndim}")
  if not 1 <= poly_order <= max_p:
    raise NotImplementedError(
        f"Gkeyll's {basis_type} differentiate kernels in {ndim}D support "
        f"poly_order 1..{max_p}, got {poly_order}")
  if not 0 <= dir < ndim:
    raise ValueError(f"differentiate dir {dir} out of range for a {ndim}D "
                     "field")
  if diff_order not in (1, 2):
    raise ValueError(f"differentiate order must be 1 or 2, got {diff_order}")


def weak_differentiate(basis_type: str, ndim: int, poly_order: int, dir: int,
                       diff_order: int, dx: float, a: GkylArray) -> GkylArray:
  """Local DG derivative ``d^diff_order/dx_dir^diff_order a`` via
  ``gkyl_dg_differentiate_op_local``, field by field.

  Differentiates the DG expansion independently in every cell (no
  inter-cell stencil) -- an exact derivative of the polynomial each cell
  already represents, not a finite-difference approximation across cells.
  Serendipity/tensor only (a Gkeyll limit); ``dx`` is the cell length along
  ``dir``.
  """
  _check_differentiate(basis_type, ndim, poly_order, dir, diff_order)
  basis = get_basis(basis_type, ndim, poly_order)
  out = GkylArray.alloc(a.ncomp, a.size)
  _lib.require().dg_differentiate(basis._cap, dir, diff_order, float(dx),
                                  out._cap, a._cap)
  return out


# ------------------------------------------------------- evaluate-and-project
# gkyl_dg_eval_at_coord_proj's own dispatch (gkyl_dg_eval_at_coord_proj_priv.h)
# covers serendipity (ndim 1-4 at p1-p2, ndim 5-6 at p1 only), tensor (ndim
# 1 and 3 at p1 only, ndim 2 at p1-p2), and gkhybrid (p1 only, at the same
# (cdim, vdim) combinations basis.py's own hybrid table already recognizes).
# Plain "hybrid" is not in that dispatch's switch at all. Like the tables
# above, an out-of-coverage combination is a process abort (an unconditional
# `assert(kers->ev_ker)` on a NULL table entry), not a Python exception.
_EVAL_AT_COORD_PROJ_MAX_POLY_ORDER = {
    "serendipity": {
        1: 2,
        2: 2,
        3: 2,
        4: 2,
        5: 1,
        6: 1
    },
    "tensor": {
        1: 1,
        2: 2,
        3: 1
    },
}

# gkyl_basis_type ordinals (gkeyll/core/zero/gkyl_basis.h) -- the target
# basis gpython_eval_at_coord_proj reports can differ in TYPE from the donor
# (e.g. eliminating a gkhybrid velocity direction can yield a plain
# serendipity target), so its ordinal must be translated back to postgkyl's
# string vocabulary here.
_BASIS_TYPE_ORDINALS = {
    0: "serendipity",
    1: "tensor",
    2: "hybrid",
    3: "gkhybrid",
    4: "gkhybrid_vel",
}


def _check_eval_at_coord_proj(basis_type: str, ndim: int, poly_order: int,
                              eval_dirs):
  basis_type = basis_type.lower()
  if basis_type == "gkhybrid":
    if poly_order != 1:
      raise NotImplementedError(
          "gkyl_dg_eval_at_coord_proj's gkhybrid kernels exist at "
          f"poly_order 1 only, got {poly_order}")
  else:
    limits = _EVAL_AT_COORD_PROJ_MAX_POLY_ORDER.get(basis_type)
    if limits is None:
      raise NotImplementedError(
          "gkyl_dg_eval_at_coord_proj supports serendipity/tensor/gkhybrid, "
          f"not '{basis_type}'")
    max_p = limits.get(ndim)
    if max_p is None:
      raise NotImplementedError(
          f"Gkeyll's {basis_type} eval_at_coord_proj kernels support ndim "
          f"{sorted(limits)}, got {ndim}")
    if not 1 <= poly_order <= max_p:
      raise NotImplementedError(
          f"Gkeyll's {basis_type} eval_at_coord_proj kernels in {ndim}D "
          f"support poly_order 1..{max_p}, got {poly_order}")
  eval_dirs = sorted(set(int(d) for d in eval_dirs))
  if not eval_dirs or eval_dirs[0] < 0 or eval_dirs[-1] >= ndim:
    raise ValueError(f"eval_dirs {eval_dirs} out of range for a {ndim}D "
                     "field")
  return eval_dirs


def eval_at_coord_proj(basis_type: str, ndim: int, poly_order: int,
                       cdim_do: int, grid: dict, eval_dirs, eval_coords,
                       ndim_tar: int, cells_tar, a: GkylArray):
  """Evaluate ``a`` at ``eval_coords`` in ``eval_dirs`` and project onto the
  lower-dimensional target basis Gkeyll picks for that elimination, via
  ``gkyl_dg_eval_at_coord_proj``.

  ``grid`` is the donor grid dict (``ndim``/``lower``/``upper``/``cells``,
  e.g. from ``rio``). ``cdim_do`` is the donor's configuration-space
  dimension count (equal to ``ndim`` for serendipity/tensor; the (cdim,
  vdim) split for gkhybrid -- see ``basis._HYBRID_CDIM_VDIM``).
  ``ndim_tar``/``cells_tar`` describe the target's rectangular index range
  with the same convention :func:`array_average` uses: the surviving donor
  dims' cell counts in donor order, or ``ndim_tar=1``/``cells_tar=[1]`` for
  a full reduction (every donor direction evaluated away).

  Returns:
    ``(out, target_basis_type, target_poly_order, target_cdim,
    target_vdim)`` -- the target array, field by field like the donor
    (``ncomp`` scaled to the donor's field count), and the target basis's
    metadata (which can differ in TYPE from the donor's, e.g. eliminating a
    gkhybrid velocity direction can yield a plain serendipity target).
  """
  eval_dirs = _check_eval_at_coord_proj(basis_type, ndim, poly_order, eval_dirs)
  basis = get_basis(basis_type, ndim, poly_order)
  lower = np.asarray(grid["lower"], dtype=np.float64)
  upper = np.asarray(grid["upper"], dtype=np.float64)
  cells = np.asarray(grid["cells"], dtype=np.int32)
  if int(np.prod(cells)) != a.size:
    raise ValueError(f"grid cells {tuple(cells)} do not cover the array "
                     f"({int(np.prod(cells))} vs {a.size} cells)")
  eval_dirs_arr = np.asarray(eval_dirs, dtype=np.int32)
  eval_coords_arr = np.asarray(eval_coords, dtype=np.float64)
  if eval_coords_arr.shape != eval_dirs_arr.shape:
    raise ValueError("eval_dirs and eval_coords must have the same length")
  cells_tar = np.asarray(cells_tar, dtype=np.int32)
  out_cap, btype, poly_order_tar, cdim_tar, vdim_tar = (
      _lib.require().eval_at_coord_proj(basis._cap, int(cdim_do), lower, upper,
                                        cells, eval_dirs_arr, eval_coords_arr,
                                        int(ndim_tar), cells_tar, a._cap))
  return (GkylArray(out_cap), _BASIS_TYPE_ORDINALS[btype], poly_order_tar,
          cdim_tar, vdim_tar)
