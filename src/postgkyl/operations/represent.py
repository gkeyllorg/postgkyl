"""The value_form verbs -- explicit modal · nodal · quad changes + ``apply``.

Conversions are **never implicit** (REFACTOR_GKEYLL_FFI.md §3b): these verbs are
the only way a dataset changes value_form, and each one stamps
``ctx["value_form"]`` (and ``ctx["num_quad"]`` for quad data) so ``info``
always shows what the numbers mean. All of them keep the data gkyl-native.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import dg
from postgkyl.gdatastate.guards import quadrature_order

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState

VALUE_FORMS = ("modal", "nodal", "quad")


def _native_basis(data: "GDataState"):
  """(basis_type, ndim, poly_order) for a gkyl-backed dataset, or raise."""
  if data.backend != "gkyl":
    raise ValueError(
        "value_form changes act on native (gkyl-backed) DG data; "
        "this dataset is NumPy-backed (already interpolated, or loaded "
        "without the Gkeyll library).")
  basis_type = data.ctx.get("basis_type")
  poly_order = data.ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError("dataset has no basis_type/poly_order metadata")
  return str(basis_type), data.num_dims, int(poly_order)


def represent(data: "GDataState",
              *,
              to: str,
              num_quad: int | None = None,
              inplace: bool = False,
              tag: str | None = None,
              label: str | None = None):
  """Convert a native dataset to the ``to`` value_form (explicitly).

  By default, ``quad`` uses Gkeyll's basis-owned quadrature nodes and
  modal/quadrature transforms. A modal -> quad -> modal round trip preserves
  the expansion. Explicit ``num_quad`` selects a uniform Gauss rule instead;
  projection back uses the rule and ordering stored with the data.
  ``nodal`` <-> ``quad`` composes through modal.

  Args:
    data: Native dataset to convert.
    to: Target value representation.
    num_quad: Optional Gauss points per direction; omitted uses Gkeyll's rule.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.
  """
  if to not in VALUE_FORMS:
    raise ValueError(f"unknown value_form '{to}'; "
                     f"choices: {VALUE_FORMS}")
  basis_type, ndim, poly_order = _native_basis(data)
  cur = data.ctx.get("value_form", "modal")
  arr = data.native
  same_quad = cur == to == "quad" and num_quad is None
  nq = data.ctx.get("num_quad") if same_quad else None
  rule = data.ctx.get("quad_rule", "gauss") if same_quad else None

  if cur != to or (cur == "quad" and num_quad is not None):
    if cur == "nodal":  # leave nodal (exact)
      arr = dg.rep.nodal_to_modal(basis_type, ndim, poly_order, arr)
    elif cur == "quad":  # leave quad (projection, with the data's own rule)
      arr = dg.rep.quad_to_modal(basis_type, ndim, poly_order, arr,
                                 quadrature_order(data))
    # arr is now modal
    if to == "nodal":
      arr = dg.rep.modal_to_nodal(basis_type, ndim, poly_order, arr)
    elif to == "quad":
      modal_ncomp = arr.ncomp
      arr = dg.rep.modal_to_quad(basis_type, ndim, poly_order, arr, num_quad)
      nb = dg.num_basis(ndim, poly_order, basis_type)
      nq = arr.ncomp // (modal_ncomp // nb) if num_quad is None else num_quad
      rule = "gkeyll" if num_quad is None else "gauss"
  else:
    arr = arr.clone()

  return data._result(data.grid,
                      arr,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      value_form=to,
                      num_quad=nq,
                      quad_rule=rule)


def apply(data: "GDataState",
          fn,
          *,
          num_quad: int | None = None,
          inplace: bool = False,
          tag: str | None = None,
          label: str | None = None):
  """Apply ``fn`` pointwise via quadrature: modal -> quad -> fn -> modal.

  The explicit spelling of nonlinear pointwise operations on DG data (e.g.
  ``d.apply(np.sqrt)``): evaluate at Gkeyll's basis quadrature nodes, apply
  ``fn`` to the values, and project back. The result stays modal and
  gkyl-native. Explicit ``num_quad`` selects a uniform Gauss rule with that
  many points per direction for de-aliasing.

  Args:
    data: Native modal dataset to transform.
    fn: Python callable applied to the quadrature-point values.
    num_quad: Optional Gauss points per direction; omitted uses Gkeyll's rule.
    inplace: Mutate and return ``data`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A modal dataset containing the projected pointwise result.

  Raises:
    ValueError: If ``data`` is not native modal data or lacks basis metadata.
  """
  basis_type, ndim, poly_order = _native_basis(data)
  if data.ctx.get("value_form", "modal") != "modal":
    raise ValueError(
        "apply() expects modal data; call .represent(to='modal') first.")
  out = dg.rep.apply_pointwise(basis_type, ndim, poly_order, data.native, fn,
                               num_quad)
  return data._result(data.grid,
                      out,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      applied=getattr(fn, "__name__", str(fn)),
                      applied_num_quad=num_quad)
