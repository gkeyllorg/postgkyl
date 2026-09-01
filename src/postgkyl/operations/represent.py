"""The value_form verbs -- explicit modal · nodal · quad changes + ``apply``.

Conversions are **never implicit** (REFACTOR_GKEYLL_FFI.md §3b): these verbs are
the only way a dataset changes value_form, and each one stamps
``ctx["value_form"]`` (and ``ctx["num_quad"]`` for quad data) so ``info``
always shows what the numbers mean. All of them keep the data gkyl-native.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import dg

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end

VALUE_FORMS = ("modal", "nodal", "quad")


def _native_basis(data: "GDataState"):
  """(basis_type, ndim, poly_order) for a gkyl-backed dataset, or raise."""
  if data.backend != "gkyl":
    raise ValueError(
        "value_form changes act on native (gkyl-backed) DG data; "
        "this dataset is NumPy-backed (already interpolated, or loaded "
        "without the Gkeyll library).")
  # end
  basis_type = data.ctx.get("basis_type")
  poly_order = data.ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError("dataset has no basis_type/poly_order metadata")
  # end
  return str(basis_type), data.num_dims, int(poly_order)
# end


def represent(data: "GDataState", *, to: str, num_quad: int | None = None,
    inplace: bool = False, tag: str | None = None, label: str | None = None):
  """Convert a native dataset to the ``to`` value_form (explicitly).

  ``modal`` <-> ``nodal`` is exact; ``modal`` -> ``quad`` evaluates at
  ``num_quad`` (default ``p+1``) Gauss–Legendre points per dimension;
  ``quad`` -> ``modal`` projects back with the rule the data was made with.
  ``nodal`` <-> ``quad`` composes through modal.

  Args:
    data: Native dataset to convert.
    to: Target value representation.
    num_quad: Gauss points per direction for quadrature representation.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.
  """
  if to not in VALUE_FORMS:
    raise ValueError(f"unknown value_form '{to}'; "
                     f"choices: {VALUE_FORMS}")
  # end
  basis_type, ndim, poly_order = _native_basis(data)
  cur = data.ctx.get("value_form", "modal")
  arr = data.native

  if cur != to:
    if cur == "nodal":  # leave nodal (exact)
      arr = dg.rep.nodal_to_modal(basis_type, ndim, poly_order, arr)
    # end
    elif cur == "quad":  # leave quad (projection, with the data's own rule)
      nq = data.ctx.get("num_quad")
      if nq is None:
        raise ValueError("quad-represented dataset lost its 'num_quad' ctx")
      # end
      arr = dg.rep.quad_to_modal(basis_type, ndim, poly_order, arr, int(nq))
    # end
    # arr is now modal
    if to == "nodal":
      arr = dg.rep.modal_to_nodal(basis_type, ndim, poly_order, arr)
    # end
    elif to == "quad":
      nq = int(num_quad) if num_quad else poly_order + 1
      arr = dg.rep.modal_to_quad(basis_type, ndim, poly_order, arr, nq)
    # end
  # end
  else:
    arr = arr.clone()
  # end

  return data._result(data.grid, arr, inplace=inplace, tag=tag, label=label,
      value_form=to,
      num_quad=(int(num_quad) if num_quad else poly_order + 1)
               if to == "quad" else None)
# end


def apply(data: "GDataState", fn, *, num_quad: int | None = None,
    inplace: bool = False, tag: str | None = None, label: str | None = None):
  """Apply ``fn`` pointwise via quadrature: modal -> quad -> fn -> modal.

  The explicit spelling of nonlinear pointwise operations on DG data (e.g.
  ``d.apply(np.sqrt)``): evaluate at ``num_quad`` (default ``p+1``) Gauss
  points, apply ``fn`` to the values, project back onto the basis. The result
  stays modal and gkyl-native; the projection is exact when ``fn(f)·b_j`` has
  degree ≤ 2·num_quad−1 -- raise ``num_quad`` to de-alias.
  """
  basis_type, ndim, poly_order = _native_basis(data)
  if data.ctx.get("value_form", "modal") != "modal":
    raise ValueError("apply() expects modal data; call .to_modal() first.")
  # end
  nq = int(num_quad) if num_quad else poly_order + 1
  out = dg.rep.apply_pointwise(basis_type, ndim, poly_order, data.native,
      fn, nq)
  return data._result(data.grid, out, inplace=inplace, tag=tag, label=label,
      applied=getattr(fn, "__name__", str(fn)), applied_num_quad=nq)
# end
