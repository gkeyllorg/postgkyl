"""Modal (DG-coefficient) operations -- thin orchestration over Gkeyll kernels.

Everything here acts on native :class:`~postgkyl.gpython.array.GkylArray` data and
returns native data (or plain numbers for reductions): the modal domain never
leaves Gkeyll's memory. The only logic this layer adds over ``gpython.kernels`` is
DG bookkeeping -- e.g. what "add a scalar" means for modal coefficients.
"""

from __future__ import annotations

import functools

import numpy as np

from postgkyl import gpython
from postgkyl.gpython.array import GkylArray

# Weak algebra and coefficient linear combinations -- direct kernel calls.
weak_mul = gpython.kernels.weak_mul
weak_div = gpython.kernels.weak_div
weak_mul_conf_phase = gpython.kernels.weak_mul_conf_phase
lincomb = gpython.kernels.lincomb
scale = gpython.kernels.scale
integrate = gpython.kernels.integrate
reduce = gpython.kernels.reduce


def is_native(value) -> bool:
  """True if ``value`` is a native (gkyl-backed) array, not a plain NumPy one.

  The one place outside ``gdatastate``/``gpython`` that needs to tell modal
  data apart from plain arrays without importing ``gpython`` directly (an
  import-contract boundary; see ``operations.evaluate``).
  """
  return isinstance(value, GkylArray)


def shift_mean(basis_type: str,
               ndim: int,
               poly_order: int,
               a: GkylArray,
               val: float,
               *,
               cdim: int | None = None,
               vdim: int | None = None) -> GkylArray:
  """``f + val`` for a modal field: only the mean coefficient moves.

  The normalized constant basis function is ``b_0 = 2^(-ndim/2)``, so a shift
  of the field by ``val`` is a shift of coefficient 0 by ``val * 2^(ndim/2)``,
  applied per field (``gkyl_array_shiftc`` on each field's coefficient 0).
  """
  nb = gpython.basis.num_basis(basis_type,
                               ndim,
                               poly_order,
                               cdim=cdim,
                               vdim=vdim)
  coeff_shift = float(val) * 2.0**(ndim / 2.0)
  out = a
  for f in range(a.ncomp // nb):
    out = gpython.kernels.shiftc(out, coeff_shift, f * nb)
  return out


def weak_inv(basis_type: str, ndim: int, poly_order: int,
             a: GkylArray) -> GkylArray:
  """Weak reciprocal ``1 / a``, field by field, robust to the field's scale.

  ``gkyl_dg_inv_op`` is cell-local but raises the coefficients to high
  powers (in 3-D p1 a constant field of 1e-39 or 1e40 already returns
  inf/nan), which SI products such as ``n T^(3/2) (dT/dx)^2`` reach. Each
  field of each cell is therefore scaled by the power of two ``2**-e`` that
  brings its largest coefficient to order one, inverted, and scaled back:
  ``1/f = 2**-e * inv(2**-e f)``. Power-of-two scaling is exact in floating
  point, so results are unchanged wherever the unscaled kernel was finite.
  """
  nb = gpython.basis.num_basis(basis_type, ndim, poly_order)
  if a.ncomp % nb:
    raise ValueError(f"ncomp {a.ncomp} is not a multiple of num_basis {nb}")
  blocks = a.view().reshape(a.size, a.ncomp // nb, nb)
  peak = np.abs(blocks).max(axis=-1, keepdims=True)
  exponent = np.where(np.isfinite(peak) & (peak > 0.0), np.frexp(peak)[1], 0)
  scaled = GkylArray.from_numpy(
      np.ldexp(blocks, -exponent).reshape(a.size, a.ncomp))
  inverse = gpython.kernels.weak_inv(basis_type, ndim, poly_order, scaled)
  return GkylArray.from_numpy(
      np.ldexp(inverse.view().reshape(blocks.shape),
               -exponent).reshape(a.size, a.ncomp))


def shift_all(a: GkylArray, val: float) -> GkylArray:
  """``values + val`` for point-value representations (nodal/quad): every
  component of every cell is a field value, so shift them all."""
  out = a.clone()
  for k in range(a.ncomp):
    out = gpython.kernels.shiftc(out, float(val), k)
  return out


def average(grid: dict,
            basis_type: str,
            ndim: int,
            poly_order: int,
            a: GkylArray,
            avg_dirs,
            weight: GkylArray | None = None):
  """``int f w dx^avg / int w dx^avg`` (or the plain average) of a modal
  field over ``avg_dirs``, field by field.

  ``gkyl_array_average`` has no field-index argument (unlike the weak ops
  above, which loop inside the compiled shim), so a multi-field ``a``
  (``ncomp == nfields * num_basis``) is split into single-field slices here
  and averaged one at a time, then reassembled.

  Args:
    grid: donor grid dict (``ndim``/``lower``/``upper``/``cells``, e.g. from
      ``rio``).
    avg_dirs: 0-based donor directions to average over.
    weight: optional single-field ``GkylArray`` over the same donor
      grid/basis as ``a`` (the plain average, dividing by volume, is used
      when omitted).

  Returns:
    ``(keep_dirs, cells_avg, result)`` -- the surviving donor directions (in
    order), the target's per-dimension cell counts, and the averaged array
    (``ncomp`` scaled to the same field count as ``a``). ``keep_dirs``/
    ``cells_avg`` are empty/``[1]`` for a full reduction: Gkeyll always
    keeps at least one target dimension, collapsing to a single cell when
    every donor direction is averaged out.
  """
  avg_dirs = sorted(set(int(d) for d in avg_dirs))
  if not avg_dirs or avg_dirs[0] < 0 or avg_dirs[-1] >= ndim:
    raise ValueError(
        f"average dirs {avg_dirs} out of range for a {ndim}D field")
  keep_dirs = [d for d in range(ndim) if d not in avg_dirs]
  ndim_avg = len(keep_dirs) if keep_dirs else 1
  cells = np.asarray(grid["cells"])
  cells_avg = [int(cells[d]) for d in keep_dirs] if keep_dirs else [1]
  avg_dim = [1 if d in avg_dirs else 0 for d in range(ndim)]

  nb = gpython.basis.num_basis(basis_type, ndim, poly_order)
  if a.ncomp % nb:
    raise ValueError(f"ncomp {a.ncomp} is not a multiple of num_basis {nb}")
  nfields = a.ncomp // nb
  if weight is not None and weight.ncomp != nb:
    raise ValueError(f"average weight ncomp ({weight.ncomp}) must equal "
                     f"the donor basis's num_basis ({nb})")

  if nfields == 1:
    out = gpython.kernels.array_average(grid,
                                        basis_type,
                                        poly_order,
                                        ndim_avg,
                                        cells_avg,
                                        avg_dim,
                                        a,
                                        weight=weight)
  else:
    a_view = a.view().reshape(a.size, nfields, nb)
    fields_out = []
    for f in range(nfields):
      a_f = GkylArray.from_numpy(np.ascontiguousarray(a_view[:, f, :]))
      out_f = gpython.kernels.array_average(grid,
                                            basis_type,
                                            poly_order,
                                            ndim_avg,
                                            cells_avg,
                                            avg_dim,
                                            a_f,
                                            weight=weight)
      fields_out.append(out_f.view())
    out = GkylArray.from_numpy(np.concatenate(fields_out, axis=-1))

  if not keep_dirs and weight is None:
    # Full reduction (every donor dim averaged), unweighted only: Gkeyll's
    # own kernels for this corner case (gkyl_array_average_NxYY_avg<all
    # dirs>) write a single raw VALUE into coefficient 0 -- there is no real
    # target dimension to normalize against, unlike every other path here
    # (a partial reduction, or ANY weighted reduction, which both go through
    # a genuine per-mode contraction/weak-division and so already come out
    # as a properly b0-normalized coefficient). Rescale so this dataset's
    # coefficient 0 means the same thing ("value = coeff0 * b0") as every
    # other modal dataset in the system -- verified against
    # gkyl_array_integrate on a constant field (see test_dg_modal_average).
    out = gpython.kernels.scale(out, 2.0**(ndim_avg / 2.0))
  return keep_dirs, cells_avg, out


@functools.lru_cache(maxsize=None)
def _lift_matrix(basis_type: str, ndim: int, poly_order: int,
                 keep_dirs: tuple[int, ...]) -> np.ndarray:
  """``(num_basis_full, num_basis_reduced)`` matrix ``T[j, k] = int B_j b_k``.

  ``B_j`` are the full ``ndim`` basis functions and ``b_k`` the reduced basis
  functions of the kept directions, both Gkeyll's own (via
  ``gpython.basis.eval_matrix``) and orthonormal on the reference cell, so
  ``T`` is the L2 projection of the reduced basis onto the full one. The
  tensor Gauss rule with ``poly_order + 1`` points per direction is exact for
  these products (degree at most ``2 poly_order`` per variable for the
  serendipity and tensor bases), so the projection is exact, not approximate.

  With no kept direction the reduced field is Gkeyll's one-cell 1-D stand-in
  whose coefficient 0 carries the mean (``b_0 = 1/sqrt(2)``); the single
  column lifts that one constant mode.
  """
  pts, w = gpython.basis.gauss_quad(ndim, poly_order + 1)
  full = gpython.basis.eval_matrix(basis_type, ndim, poly_order, pts)
  if keep_dirs:
    reduced = gpython.basis.eval_matrix(basis_type, len(keep_dirs), poly_order,
                                        pts[:, list(keep_dirs)])
  else:
    reduced = np.full((pts.shape[0], 1), 2.0**-0.5)
  matrix = full.T @ (w[:, None] * reduced)
  matrix.flags.writeable = False
  return matrix


def lift(basis_type: str, ndim: int, poly_order: int, reduced: GkylArray,
         avg_dirs, cells) -> GkylArray:
  """Lift a field over the kept directions back onto the full ``ndim``
  basis and cell layout, constant along ``avg_dirs`` -- the right inverse of
  :func:`average` (``average(lift(g)) == g``).

  ``reduced`` is the output of :func:`average` over ``avg_dirs``: modal
  coefficients on the reduced basis (same ``basis_type``/``poly_order``,
  ``ndim - len(avg_dirs)`` dimensions) over the kept directions' cells, or
  for a full reduction the one-cell 1-D field whose coefficient 0 carries
  the mean. A function of the kept variables in the reduced serendipity
  (or tensor) space lies in the full space, so the lift is exact: the
  returned field takes the same values as ``reduced`` at every point.
  Multi-field arrays (``ncomp == nfields * num_basis``) are lifted field by
  field, as :func:`average` reduces them.

  Args:
    reduced: reduced modal coefficients, sized for the kept directions'
      cells (``cells[d]`` for ``d`` not in ``avg_dirs``; one cell when every
      direction was averaged).
    avg_dirs: 0-based directions the field is constant along -- the ones
      :func:`average` reduced.
    cells: the full per-dimension cell counts of the target layout.

  Returns:
    A ``GkylArray`` with ``nfields * num_basis(full)`` components over
    ``prod(cells)`` cells.

  Raises:
    ValueError: ``avg_dirs`` is empty or out of range, or ``reduced``'s
      component count is not a multiple of the reduced basis size.
  """
  avg_dirs = sorted(set(int(d) for d in avg_dirs))
  if not avg_dirs or avg_dirs[0] < 0 or avg_dirs[-1] >= ndim:
    raise ValueError(f"lift dirs {avg_dirs} out of range for a {ndim}D field")
  keep_dirs = tuple(d for d in range(ndim) if d not in avg_dirs)
  cells = [int(c) for c in cells]
  matrix = _lift_matrix(basis_type, ndim, poly_order, keep_dirs)
  nb_full, nb_red = matrix.shape

  # Coefficient layout of ``reduced``: the full reduction is stored on a
  # one-cell 1-D basis of which only the constant mode is meaningful.
  cells_red = [cells[d] for d in keep_dirs]
  nb_stored = nb_red if keep_dirs else gpython.basis.num_basis(
      basis_type, 1, poly_order)
  if reduced.ncomp % nb_stored:
    raise ValueError(f"reduced ncomp {reduced.ncomp} is not a multiple of "
                     f"the reduced basis's num_basis {nb_stored}")
  nfields = reduced.ncomp // nb_stored
  coeffs = reduced.view(cells_red or [1]).reshape(*cells_red, nfields,
                                                  nb_stored)[..., :nb_red]

  lifted = (coeffs @ matrix.T).reshape(*cells_red, nfields * nb_full)
  for d in avg_dirs:
    lifted = np.expand_dims(lifted, axis=d)
  return GkylArray.from_numpy(
      np.broadcast_to(lifted, (*cells, nfields * nb_full)))


def differentiate(basis_type: str, ndim: int, poly_order: int, a: GkylArray,
                  dir: int, diff_order: int, dx: float) -> GkylArray:
  """``d^diff_order/dx_dir^diff_order a``, field by field
  (``gkyl_dg_differentiate_op_local`` -- exact on the polynomial each cell
  already represents; no inter-cell stencil). The field loop lives in the
  shim (like :func:`weak_mul`), so this is a direct pass-through."""
  return gpython.kernels.weak_differentiate(basis_type, ndim, poly_order, dir,
                                            diff_order, dx, a)


def eval_at_coord_proj(grid: dict, basis_type: str, ndim: int, poly_order: int,
                       a: GkylArray, eval_dirs, eval_coords):
  """Evaluate a modal field at ``eval_coords`` in ``eval_dirs`` and project
  onto the surviving directions' target basis (``gkyl_dg_eval_at_coord_proj``).

  ``grid`` is the donor grid dict (``ndim``/``lower``/``upper``/``cells``,
  e.g. from ``rio``). The donor's configuration-space dimension count (needed
  by the underlying updater) is derived from ``basis_type``/``ndim`` via
  ``gpython.basis.cdim_vdim``.

  Returns:
    ``(keep_dirs, cells_tar, out, target_basis_type, target_poly_order,
    target_cdim, target_vdim)`` -- ``keep_dirs``/``cells_tar`` follow the
    same full-reduction convention :func:`average` uses (empty/``[1]`` when
    every donor direction is evaluated away, since Gkeyll always keeps at
    least one target dimension).
  """
  # Keep directions paired with their coordinates. The native boundary checks
  # the request and sorts both together for Gkeyll's kernel convention.
  eval_dirs = tuple(eval_dirs)
  keep_dirs = [d for d in range(ndim) if d not in eval_dirs]
  cells = np.asarray(grid["cells"])
  ndim_tar = len(keep_dirs) if keep_dirs else 1
  cells_tar = [int(cells[d]) for d in keep_dirs] if keep_dirs else [1]
  cdim_do, _vdim_do = gpython.basis.cdim_vdim(basis_type, ndim)

  out, btype, poly_order_tar, cdim_tar, vdim_tar = (
      gpython.kernels.eval_at_coord_proj(basis_type, ndim, poly_order, cdim_do,
                                         grid, eval_dirs, eval_coords, ndim_tar,
                                         cells_tar, a))
  return (keep_dirs, cells_tar, out, btype, poly_order_tar, cdim_tar, vdim_tar)


def power(basis_type: str,
          ndim: int,
          poly_order: int,
          a: GkylArray,
          exponent,
          cells=None) -> GkylArray:
  """``f ** n``.

  A positive integer ``n`` takes the cheap, exact path: repeated weak
  multiplies. Any other exponent (0, negative, or fractional) falls
  through to :func:`powsqrt` (``f ** n == pow(sqrt(f), 2n)``), which needs
  ``cells`` (the grid's per-dimension cell count, e.g. ``ctx["cells"]``) to
  build Gkeyll's index range -- required whenever this fallback fires.
  """
  n = exponent
  if isinstance(n, (int, np.integer)) and n >= 1:
    out = a.clone()
    for _ in range(int(n) - 1):
      out = weak_mul(basis_type, ndim, poly_order, out, a)
    return out
  if cells is None:
    raise ValueError(
        f"modal power with exponent {n!r} (not a positive integer) needs "
        "cells= to build the powsqrt kernel's index range.")
  return powsqrt(basis_type, ndim, poly_order, cells, a, 2.0 * float(n))


def powsqrt(basis_type: str,
            ndim: int,
            poly_order: int,
            cells,
            a: GkylArray,
            exponent: float,
            num_quad: int | None = None) -> GkylArray:
  """``pow(sqrt(f), exponent)`` (i.e. ``f ** (exponent/2)``), field by field.

  ``gkyl_proj_powsqrt_on_basis`` has no field-index argument (like
  :func:`average`'s ``gkyl_array_average``), so a multi-field ``a``
  (``ncomp == nfields * num_basis``) is split into single-field slices
  here and processed one at a time, then reassembled.
  """
  nb = gpython.basis.num_basis(basis_type, ndim, poly_order)
  if a.ncomp % nb:
    raise ValueError(f"ncomp {a.ncomp} is not a multiple of num_basis {nb}")
  nfields = a.ncomp // nb
  if nfields == 1:
    return gpython.kernels.powsqrt(basis_type,
                                   ndim,
                                   poly_order,
                                   cells,
                                   a,
                                   exponent,
                                   num_quad=num_quad)
  a_view = a.view().reshape(a.size, nfields, nb)
  fields_out = []
  for f in range(nfields):
    a_f = GkylArray.from_numpy(np.ascontiguousarray(a_view[:, f, :]))
    out_f = gpython.kernels.powsqrt(basis_type,
                                    ndim,
                                    poly_order,
                                    cells,
                                    a_f,
                                    exponent,
                                    num_quad=num_quad)
    fields_out.append(out_f.view())
  return GkylArray.from_numpy(np.concatenate(fields_out, axis=-1))
