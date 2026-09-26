"""Arithmetic / NumPy-ufunc backend for the fluent operators.

Defined here (in ``operations``) -- not on the container -- so the computing operators
follow the same one-way layering as every other verb (HIERARCHY_3.md).

Dispatch is on the container's ``backend`` (the two-domain lifecycle of
REFACTOR_GKEYLL_FFI.md):

- **gkyl-backed (modal) operands** run inside Gkeyll: ``*``/``/`` are the weak
  kernels (``gkyl_dg_mul_op``/``div_op``), ``+``/``-`` are coefficient linear
  combinations (``gkyl_array_set``/``accumulate``), scalar multiply is
  ``gkyl_array_scale``, scalar add shifts the mean coefficient, positive
  integer powers are repeated weak multiplies, and any other power (0,
  negative, or fractional) is ``gkyl_proj_powsqrt_on_basis`` (a
  quadrature projection of ``pow(sqrt(f), 2*exponent)``). Results stay
  modal (gkyl-backed).
  Two modal operands of *different* dimensionality (e.g. a conf-space density
  times a phase-space distribution) automatically route ``*`` through
  ``gkyl_dg_mul_conf_phase_op_range`` instead -- whichever operand has fewer
  dimensions is the conf side, independent of call order.
- **numpy-backed operands** take the unchanged NumPy path.
- **Mixing the domains** in one expression is an error naming the fix.
"""

from __future__ import annotations

import operator

import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.gdatastate.guards import require_same_quadrature
from postgkyl import dg, numerics


def binary(op, a, b):
  """``a <op> b`` where at least one operand is a dataset; result copies its grid."""
  pa = a if isinstance(a, GDataState) else None
  pb = b if isinstance(b, GDataState) else None
  if (pa is not None and pa.backend == "gkyl") or (pb is not None
                                                   and pb.backend == "gkyl"):
    return _modal_binary(op, a, b, pa, pb)
  return _numpy_binary(op, a, b, pa, pb)


# --------------------------------------------------------------- numpy domain
def _numpy_binary(op, a, b, pa, pb):
  primary = pa if pa is not None else pb
  primary._require_operable()
  if pa is not None and pb is not None:
    pb._require_operable()
    require_compatible_operands(pa, pb)
  va = pa.values if pa is not None else np.asarray(a)
  vb = pb.values if pb is not None else np.asarray(b)
  return primary._result(primary.grid, op(va, vb))


# --------------------------------------------------------------- modal domain
def _basis_of(data: GDataState):
  """(basis_type, ndim, poly_order) from ctx -- the modal ops' dispatch key."""
  basis_type = data.ctx.get("basis_type")
  poly_order = data.ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError("modal operand has no basis_type/poly_order metadata")
  return str(basis_type), data.num_dims, int(poly_order)


def _modal_binary(op, a, b, pa, pb):
  if pa is not None and pb is not None:
    return _modal_dataset_pair(op, pa, pb)
  primary = pa if pa is not None else pb
  other = b if pa is not None else a
  if not isinstance(other, (int, float, np.integer, np.floating)):
    raise ValueError(
        "cannot mix native modal data with arrays; call .interpolate() on the "
        "modal operand first (or use scalars / another modal dataset).")
  return _modal_scalar(op, primary, float(other), scalar_first=pa is None)


def _rep_of(data: GDataState) -> str | None:
  """DG layout, or None for fields whose grid directly locates their values.

  Interpolation retains source basis metadata as provenance; it no longer
  describes the output's value layout. Storage alone does not decide this:
  DG nodal/quad data can also be loaded into NumPy arrays.
  """
  if data.backend == "numpy" and (data.ctx.get("interpolated", False)
                                  or not data.ctx.get("basis_type")):
    return None
  return data.ctx.get("value_form", "modal")


def _require_same_backend(left: GDataState, right: GDataState) -> None:
  if left.backend != right.backend:
    raise ValueError(
        "operands have different backends (gkyl vs numpy); call .interpolate() "
        "on the native operand to combine them.")


def require_compatible_operands(left: GDataState, right: GDataState) -> None:
  """Require matching physical locations and value layouts for arithmetic.

  Operators, ufuncs, and RPN point-value operations share this contract.
  Scalars/arrays have no dataset coordinates to compare. The modal conf/phase
  product has a separate grid-prefix contract in ``_modal_conf_phase_mul``.
  """
  _require_same_backend(left, right)
  if not numerics.grids_compatible(left.grid, right.grid):
    raise ValueError("operands live on different grids")
  rep = _rep_of(left)
  if rep != _rep_of(right):
    raise ValueError(
        f"operands are in different value_forms ({rep} vs {_rep_of(right)}); "
        "convert one explicitly with .represent(to=...).")
  if rep is not None and _basis_of(left) != _basis_of(right):
    raise ValueError("operands have different DG bases")
  if rep == "quad":
    require_same_quadrature(left, right)
  if left.values.shape != right.values.shape:
    raise ValueError(
        f"incompatible shapes {left.values.shape} vs {right.values.shape}")


def _modal_dataset_pair(op, pa: GDataState, pb: GDataState):
  if pa.num_dims != pb.num_dims:
    _require_same_backend(pa, pb)
    return _modal_conf_phase_mul(op, pa, pb)
  require_compatible_operands(pa, pb)
  basis = _basis_of(pa)
  rep = _rep_of(pa)
  A, B = pa.native, pb.native
  if op is operator.add:  # linear: valid in any rep
    out = dg.modal.lincomb(1.0, A, 1.0, B)
  elif op is operator.sub:
    out = dg.modal.lincomb(1.0, A, -1.0, B)
  elif rep != "modal":
    # Point values (nodal/quad): every pointwise operation is exact -- compute
    # with NumPy on the views, wrap back native, stay in-value_form.
    out = dg.rep.wrap(op(np.asarray(pa.values), np.asarray(pb.values)))
  elif op in (operator.mul, operator.truediv):
    out = (dg.modal.weak_mul if op is operator.mul else dg.modal.weak_div)(
        *basis, A, B)
  else:
    raise ValueError(
        f"operation {getattr(op, '__name__', op)} is not defined between two "
        "modal datasets; .represent(to='nodal')/.represent(to='quad') "
        "for pointwise math.")
  return pa._result(pa.grid, out)


def _modal_conf_phase_mul(op, pa: GDataState, pb: GDataState):
  """``conf * phase`` (either order): the operands have different ``num_dims``,
  so Gkeyll's per-cell same-basis ``weak_mul`` cannot apply -- this is the
  cross-basis ``gkyl_dg_mul_conf_phase_op_range`` path
  (``dg.modal.weak_mul_conf_phase``), which multiplies every phase-space cell
  by its corresponding lower-dimensional conf-space cell (e.g. a density
  times a distribution function). Automatic: whichever operand has fewer
  dimensions is the conf side, regardless of call order (``a * b == b * a``).
  """
  if op is not operator.mul:
    raise ValueError(
        f"operands have different dimensionality ({pa.num_dims}D vs "
        f"{pb.num_dims}D); only '*' is defined between a lower-dimensional "
        "conf-space field and a higher-dimensional phase-space field "
        "(Gkeyll has no cross-basis weak divide/add).")
  conf, phase = (pa, pb) if pa.num_dims < pb.num_dims else (pb, pa)
  for d in (conf, phase):
    if _rep_of(d) != "modal":
      raise ValueError(
          "conf-space x phase-space multiplication is defined for modal DG "
          "coefficients only; .represent(to='modal') first.")
  if not numerics.grid_is_prefix(conf.grid, phase.grid):
    raise ValueError(
        "the lower-dimensional operand's grid is not the leading dimensions "
        "of the higher-dimensional operand's grid; they are not the same "
        "simulation's conf-space and phase-space grids.")
  conf_type, conf_ndim, conf_p = _basis_of(conf)
  phase_type, phase_ndim, _ = _basis_of(phase)
  out = dg.modal.weak_mul_conf_phase(conf_type, conf_ndim, phase_type,
                                     phase_ndim, conf_p, conf.num_cells,
                                     phase.num_cells, conf.native, phase.native)
  return phase._result(phase.grid, out)


def _modal_scalar(op, data: GDataState, s: float, *, scalar_first: bool):
  basis = _basis_of(data)
  rep = _rep_of(data)
  A = data.native
  # Adding/subtracting a *scalar* only shifts the mean (constant) DG
  # coefficient -- a constant has no projection onto the higher-order basis
  # functions, so gkyl_array_shiftc touches just coefficient 0 (shift_mean,
  # dg/modal.py). This is unrelated to array + array (lincomb, above), which
  # runs Gkeyll's own accumulate over every coefficient, higher orders
  # included. In point-value forms (nodal/quad) there's no separate mean
  # coefficient to single out, so a scalar shift moves every component.
  shift = (dg.modal.shift_all if rep != "modal" else
           lambda a, v: dg.modal.shift_mean(*basis, a, v))
  if op is operator.mul:  # linear: valid in any rep
    out = dg.modal.scale(A, s)
  elif op is operator.truediv and not scalar_first:
    out = dg.modal.scale(A, 1.0 / s)  # f / s: linear, any rep
  elif op is operator.add:
    out = shift(A, s)
  elif op is operator.sub:
    if scalar_first:  # s - f
      out = shift(dg.modal.scale(A, -1.0), s)
    else:  # f - s
      out = shift(A, -s)
  elif rep != "modal":
    # Point values: any remaining scalar operation is exact pointwise.
    args = (s, np.asarray(data.values)) if scalar_first else (np.asarray(
        data.values), s)
    out = dg.rep.wrap(op(*args))
  elif op is operator.truediv:  # s / f -- weak reciprocal
    out = dg.modal.scale(dg.modal.weak_inv(*basis, A), s)
  elif op is operator.pow and not scalar_first:
    out = dg.modal.power(*basis,
                         A,
                         s if not float(s).is_integer() else int(s),
                         cells=data.ctx.get("cells"))
  else:
    raise ValueError(
        f"operation {getattr(op, '__name__', op)} is not defined for modal "
        "data and a scalar; .represent(to='nodal')/.represent(to='quad') "
        "for pointwise math.")
  return data._result(data.grid, out)


# ------------------------------------------------------------------- ufuncs
def apply_ufunc(ufunc, method, *inputs, **kwargs):
  """Backend for ``GData.__array_ufunc__``.

  Ufuncs are pointwise, so they are valid wherever the data are point values:
  the NumPy field domain, and the nodal/quad value_forms (computed on the
  views, wrapped back native, staying in-value_form). Modal coefficients
  refuse (via ``_require_operable``): a ufunc has no basis-space meaning.

  Pointwise calls keep the result as a dataset. Reductions return the NumPy
  scalar/array produced by the ufunc: after an arbitrary axis reduction the
  original spatial grid no longer necessarily describes the result. This
  supports NumPy's reduction helpers (``max``, ``min``, ``sum``, ``prod``,
  ``all``, and ``any``), which dispatch here as ``ufunc.reduce``.
  """
  if method == "reduce":
    if len(inputs) != 1 or not isinstance(inputs[0], GDataState):
      return NotImplemented
    data = inputs[0]
    data._require_operable()
    return ufunc.reduce(np.asarray(data.values), **kwargs)
  if method != "__call__" or "out" in kwargs:
    return NotImplemented
  datasets = [x for x in inputs if isinstance(x, GDataState)]
  primary = datasets[0]
  raw = []
  for x in inputs:
    if isinstance(x, GDataState):
      x._require_operable()
      raw.append(np.asarray(x.values))
    elif isinstance(x, GDataState._HANDLED_TYPES):
      raw.append(x)
    else:
      return NotImplemented
  for other in datasets[1:]:
    require_compatible_operands(primary, other)
  result = ufunc(*raw, **kwargs)
  if primary.backend == "gkyl":
    result = dg.rep.wrap(result)
  return primary._result(primary.grid, result)
