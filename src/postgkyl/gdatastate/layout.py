"""Validated current DG layout, independent of the array's storage backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from postgkyl.gpython import basis
from .guards import quadrature_order

if TYPE_CHECKING:
  from .gdatastate import GDataState


@dataclass(frozen=True)
class DGLayout:
  """One packed cell layout: complete node/coefficient blocks per field."""

  basis_type: str
  ndim: int
  poly_order: int
  value_form: str
  cdim: int
  vdim: int
  block_size: int
  num_fields: int
  num_quad: int | None = None

  @property
  def basis_args(self):
    return self.basis_type, self.ndim, self.poly_order

  @property
  def basis_kwargs(self):
    return {"cdim": self.cdim, "vdim": self.vdim}

  @property
  def basis_identity(self):
    return (*self.basis_args, self.cdim, self.vdim)


def resolve_basis_split(ctx: dict, ndim: int, ncomp: int) -> tuple[int, int]:
  """Resolve hybrid metadata without guessing between valid phase spaces.

  Legacy files may omit the split. Infer it only when the packed field count
  admits one producer basis; ambiguous files need explicit load context.
  """
  kind = ctx["basis_type"]
  cdim = ctx.get("num_cdim", ctx.get("cdim"))
  vdim = ctx.get("num_vdim", ctx.get("vdim"))
  if kind != "hybrid" or cdim is not None or vdim is not None:
    return basis.cdim_vdim(kind, ndim, cdim=cdim, vdim=vdim)
  candidates = []
  for c in range(1, 4):
    v = ndim - c
    if not 1 <= v <= 3:
      continue
    b = basis.get_basis(kind, ndim, int(ctx["poly_order"]), cdim=c, vdim=v)
    if ctx.get("value_form") == "quad":
      block = (int(ctx["num_quad"])**ndim
               if ctx.get("quad_rule", "gauss") == "gauss" else b.num_quad)
    else:
      block = b.num_basis
    if ncomp > 0 and ncomp % block == 0:
      candidates.append((c, v))
  if len(candidates) != 1:
    raise ValueError("hybrid basis split is ambiguous or incompatible with the "
                     "component count; specify num_cdim and num_vdim in ctx")
  return candidates[0]


def dg_layout(data: "GDataState") -> DGLayout | None:
  """Resolve and validate packed DG state; ordinary point arrays return None.

  Source basis metadata remains useful after interpolation, but does not
  describe that result's component axis. A nodal array on coordinates that
  match its samples is likewise already unpacked (including load defaults).
  """
  ctx = data.ctx
  kind = ctx.get("basis_type")
  if data.backend == "gkyl" and (kind is None or ctx.get("poly_order") is None):
    raise ValueError("dataset has no basis_type/poly_order metadata")
  if not kind or ctx.get("interpolated", False):
    return None
  form = ctx.get("value_form", "modal")
  if form not in ("modal", "nodal", "quad"):
    raise ValueError(f"unknown value_form '{form}'")
  values = data.values
  if values is None:
    raise ValueError("dataset has no values")
  if form == "nodal" and data.backend == "numpy" and all(
      g.ndim == 1 and len(g) == values.shape[d]
      for d, g in enumerate(data.grid)):
    return None
  order = ctx.get("poly_order")
  if order is None:
    raise ValueError("dataset has no basis_type/poly_order metadata")
  ndim = data.num_dims
  cdim, vdim = resolve_basis_split(ctx, ndim, values.shape[-1])
  b = basis.get_basis(kind, ndim, int(order), cdim=cdim, vdim=vdim)
  nq = quadrature_order(data) if form == "quad" else None
  block = (
      b.num_quad if nq is None else nq**ndim) if form == "quad" else b.num_basis
  if block <= 0 or values.shape[-1] == 0 or values.shape[-1] % block:
    raise ValueError(f"{form} component count {values.shape[-1]} must contain "
                     f"complete field blocks of {block}")
  return DGLayout(kind, ndim, int(order), form, cdim, vdim, block,
                  values.shape[-1] // block, nq)


def require_dg_layout(data: "GDataState") -> DGLayout:
  """Require coefficients or cell-local nodes, rather than evaluated points."""
  layout = dg_layout(data)
  if layout is None:
    raise ValueError("dataset contains point values, not packed DG data; "
                     "interpolation cannot be repeated")
  return layout


def require_kernel_basis(data: "GDataState") -> DGLayout:
  """Reject basis splits unsupported by the legacy native-kernel signatures."""
  layout = require_dg_layout(data)
  if (layout.cdim, layout.vdim) != basis.cdim_vdim(layout.basis_type,
                                                   layout.ndim):
    raise NotImplementedError(
        "this native operation does not support the "
        f"{layout.cdim}x{layout.vdim}v {layout.basis_type} basis")
  return layout
