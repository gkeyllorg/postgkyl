"""Modal (DG-coefficient) operations -- thin orchestration over Gkeyll kernels.

Everything here acts on native :class:`~postgkyl.gpython.array.GkylArray` data and
returns native data (or plain numbers for reductions): the modal domain never
leaves Gkeyll's memory. The only logic this layer adds over ``gpython.kernels`` is
DG bookkeeping -- e.g. what "add a scalar" means for modal coefficients.
"""

from __future__ import annotations

import numpy as np

from postgkyl import gpython
from postgkyl.gpython.array import GkylArray

# Weak algebra and coefficient linear combinations -- direct kernel calls.
weak_mul = gpython.kernels.weak_mul
weak_div = gpython.kernels.weak_div
weak_inv = gpython.kernels.weak_inv
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


def shift_mean(basis_type: str, ndim: int, poly_order: int, a: GkylArray,
               val: float) -> GkylArray:
  """``f + val`` for a modal field: only the mean coefficient moves.

  The normalized constant basis function is ``b_0 = 2^(-ndim/2)``, so a shift
  of the field by ``val`` is a shift of coefficient 0 by ``val * 2^(ndim/2)``,
  applied per field (``gkyl_array_shiftc`` on each field's coefficient 0).
  """
  nb = gpython.basis.num_basis(basis_type, ndim, poly_order)
  coeff_shift = float(val) * 2.0**(ndim / 2.0)
  out = a
  for f in range(a.ncomp // nb):
    out = gpython.kernels.shiftc(out, coeff_shift, f * nb)
  return out


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
  eval_dirs = sorted(set(int(d) for d in eval_dirs))
  if not eval_dirs or eval_dirs[0] < 0 or eval_dirs[-1] >= ndim:
    raise ValueError(f"eval_dirs {eval_dirs} out of range for a {ndim}D field")
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
