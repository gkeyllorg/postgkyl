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
from postgkyl import dg, numerics


def _unpack(x):
  """(values, grid, dataset|None) for a dataset; (array, None, None) otherwise."""
  if isinstance(x, GDataState):
    return x.values, x.grid, x
  # end
  return np.asarray(x), None, None
# end


def binary(op, a, b):
  """``a <op> b`` where at least one operand is a dataset; result copies its grid."""
  pa = a if isinstance(a, GDataState) else None
  pb = b if isinstance(b, GDataState) else None
  if (pa is not None and pa.backend == "gkyl") or (
      pb is not None and pb.backend == "gkyl"):
    return _modal_binary(op, a, b, pa, pb)
  # end
  return _numpy_binary(op, a, b, pa, pb)
# end


# --------------------------------------------------------------- numpy domain
def _numpy_binary(op, a, b, pa, pb):
  va, ga, _ = _unpack(a)
  vb, gb, _ = _unpack(b)
  primary = pa if pa is not None else pb
  primary._require_operable()
  if pa is not None and pb is not None:
    pb._require_operable()
    if not numerics.grids_compatible(ga, gb):
      raise ValueError("operands live on different grids")
    # end
    if va.shape != vb.shape:
      raise ValueError(f"incompatible shapes {va.shape} vs {vb.shape}")
    # end
  # end
  return primary._result(primary.grid, op(va, vb))
# end


# --------------------------------------------------------------- modal domain
def _basis_of(data: GDataState):
  """(basis_type, ndim, poly_order) from ctx -- the modal ops' dispatch key."""
  basis_type = data.ctx.get("basis_type")
  poly_order = data.ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError("modal operand has no basis_type/poly_order metadata")
  # end
  return str(basis_type), data.num_dims, int(poly_order)
# end


def _modal_binary(op, a, b, pa, pb):
  if pa is not None and pb is not None:
    return _modal_dataset_pair(op, pa, pb)
  # end
  primary = pa if pa is not None else pb
  other = b if pa is not None else a
  if not isinstance(other, (int, float, np.integer, np.floating)):
    raise ValueError(
        "cannot mix native modal data with arrays; call .interpolate() on the "
        "modal operand first (or use scalars / another modal dataset).")
  # end
  return _modal_scalar(op, primary, float(other), scalar_first=pa is None)
# end


def _rep_of(data: GDataState) -> str:
  return data.ctx.get("value_form", "modal")
# end


def _modal_dataset_pair(op, pa: GDataState, pb: GDataState):
  if pb.backend != "gkyl" or pa.backend != "gkyl":
    raise ValueError(
        "one operand is modal (gkyl-native) and the other is interpolated; "
        "call .interpolate() on the modal operand to combine them.")
  # end
  if pa.num_dims != pb.num_dims:
    return _modal_conf_phase_mul(op, pa, pb)
  # end
  if not numerics.grids_compatible(pa.grid, pb.grid):
    raise ValueError("operands live on different grids")
  # end
  basis = _basis_of(pa)
  if _basis_of(pb) != basis:
    raise ValueError("operands have different DG bases")
  # end
  rep = _rep_of(pa)
  if rep != _rep_of(pb):
    raise ValueError(
        f"operands are in different value_forms ({rep} vs {_rep_of(pb)}); "
        "convert one explicitly (.to_modal()/.to_nodal()/.to_quad()).")
  # end
  A, B = pa.native, pb.native
  if op is operator.add:                       # linear: valid in any rep
    out = dg.modal.lincomb(1.0, A, 1.0, B)
  # end
  elif op is operator.sub:
    out = dg.modal.lincomb(1.0, A, -1.0, B)
  # end
  elif rep != "modal":
    # Point values (nodal/quad): every pointwise operation is exact -- compute
    # with NumPy on the views, wrap back native, stay in-value_form.
    out = dg.rep.wrap(op(np.asarray(pa.values), np.asarray(pb.values)))
  # end
  elif op in (operator.mul, operator.truediv):
    out = (dg.modal.weak_mul if op is operator.mul
           else dg.modal.weak_div)(*basis, A, B)
  # end
  else:
    raise ValueError(
        f"operation {getattr(op, '__name__', op)} is not defined between two "
        "modal datasets; .to_nodal()/.to_quad() for pointwise math.")
  # end
  return pa._result(pa.grid, out)
# end


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
  # end
  conf, phase = (pa, pb) if pa.num_dims < pb.num_dims else (pb, pa)
  for d in (conf, phase):
    if _rep_of(d) != "modal":
      raise ValueError(
          "conf-space x phase-space multiplication is defined for modal DG "
          "coefficients only; .to_modal() first.")
    # end
  # end
  if not numerics.grid_is_prefix(conf.grid, phase.grid):
    raise ValueError(
        "the lower-dimensional operand's grid is not the leading dimensions "
        "of the higher-dimensional operand's grid; they are not the same "
        "simulation's conf-space and phase-space grids.")
  # end
  conf_type, conf_ndim, conf_p = _basis_of(conf)
  phase_type, phase_ndim, _ = _basis_of(phase)
  out = dg.modal.weak_mul_conf_phase(conf_type, conf_ndim, phase_type,
      phase_ndim, conf_p, conf.num_cells, phase.num_cells, conf.native,
      phase.native)
  return phase._result(phase.grid, out)
# end


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
  shift = (dg.modal.shift_all if rep != "modal"
           else lambda a, v: dg.modal.shift_mean(*basis, a, v))
  if op is operator.mul:                       # linear: valid in any rep
    out = dg.modal.scale(A, s)
  # end
  elif op is operator.truediv and not scalar_first:
    out = dg.modal.scale(A, 1.0 / s)           # f / s: linear, any rep
  # end
  elif op is operator.add:
    out = shift(A, s)
  # end
  elif op is operator.sub:
    if scalar_first:  # s - f
      out = shift(dg.modal.scale(A, -1.0), s)
    # end
    else:             # f - s
      out = shift(A, -s)
    # end
  # end
  elif rep != "modal":
    # Point values: any remaining scalar operation is exact pointwise.
    args = (s, np.asarray(data.values)) if scalar_first else (
        np.asarray(data.values), s)
    out = dg.rep.wrap(op(*args))
  # end
  elif op is operator.truediv:                 # s / f -- weak reciprocal
    out = dg.modal.scale(dg.modal.weak_inv(*basis, A), s)
  # end
  elif op is operator.pow and not scalar_first:
    out = dg.modal.power(*basis, A, s if not float(s).is_integer() else int(s),
        cells=data.ctx.get("cells"))
  # end
  else:
    raise ValueError(
        f"operation {getattr(op, '__name__', op)} is not defined for modal "
        "data and a scalar; .to_nodal()/.to_quad() for pointwise math.")
  # end
  return data._result(data.grid, out)
# end


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
    # end
    data = inputs[0]
    data._require_operable()
    return ufunc.reduce(np.asarray(data.values), **kwargs)
  # end
  if method != "__call__" or "out" in kwargs:
    return NotImplemented
  # end
  primary = next(x for x in inputs if isinstance(x, GDataState))
  primary._require_operable()
  rep = (_rep_of(primary) if primary.backend == "gkyl" else None)
  raw = []
  for x in inputs:
    if isinstance(x, GDataState):
      x._require_operable()
      if x.backend == "gkyl" and _rep_of(x) != rep or (
          x.backend != "gkyl" and rep is not None):
        raise ValueError(
            "operands are in different value_forms; convert one "
            "explicitly (.to_modal()/.to_nodal()/.to_quad()).")
      # end
      if x.values.shape != primary.values.shape:
        raise ValueError(
            f"incompatible shapes {x.values.shape} vs {primary.values.shape}")
      # end
      raw.append(np.asarray(x.values))
    # end
    elif isinstance(x, GDataState._HANDLED_TYPES):
      raw.append(x)
    # end
    else:
      return NotImplemented
    # end
  # end
  result = ufunc(*raw, **kwargs)
  if rep is not None:
    return primary._result(primary.grid, dg.rep.wrap(result))
  # end
  return primary._result(primary.grid, result)
# end
