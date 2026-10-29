"""Project a complete vector using explicit, collocated basis geometry."""
from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Literal

import numpy as np

from postgkyl import dg
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.gdatastate.layout import dg_layout
from ._compatibility import require_collocated_layout
from .arithmetic import binary
from .represent import apply
from .select import select

REPRESENTATIONS = ("covar", "contra", "cyl", "cart", "norm")
UNIT_VECTOR_REPRESENTATIONS = ("cyl", "cart", "norm")
BASIS_REPRESENTATION = {
    "covariant": "covar",
    "contravariant": "contra",
    "cartesian": "cart"
}
_METRIC_COMP = ((0, 1, 2), (1, 3, 4), (2, 4, 5))


def vector_direction(value) -> int:
  """Parse a numeric vector index; named directions now require rep plus dir."""
  if str(value) not in ("0", "1", "2"):
    raise ValueError(
        "Choose direction=0,1,2 and rep=covar,contra,cyl,cart,norm; "
        "replace R/phi/Z with rep=cyl, dir=0/1/2 and "
        "n0/n1/n2 with rep=norm, dir=0/1/2.")
  return int(value)


def _require_fields(data, count, template):
  """Require collocated data with the stated number of complete fields."""
  require_collocated_layout(template, data)
  layout = dg_layout(data)
  actual = layout.num_fields if layout else data.num_comps
  if actual != count:
    raise ValueError(f"Expected {count} fields, got {actual}.")


def _dot(left, right):
  """Contract scalar fields with the backend's weak or pointwise products."""
  products = [binary(operator.mul, a, b) for a, b in zip(left, right)]
  return binary(operator.add, binary(operator.add, products[0], products[1]),
                products[2])


def _cylindrical_unit(positions, selector, position_system):
  """Evaluate R/phi unit vectors pointwise, projecting back for modal data."""
  if position_system not in ("cartesian", "cylindrical"):
    raise ValueError("position_system must be 'cartesian' or 'cylindrical'.")

  def unit(values):
    """Apply cylindrical geometry to samples, never modal coefficients."""
    shape = values.shape
    values = values.reshape(-1, 3, shape[-1] // 3).swapaxes(1, 2)
    if position_system == "cartesian":
      Xcart, Ycart = values[..., 0], values[..., 1]
      Rmajor = np.hypot(Xcart, Ycart)
      if np.any(~np.isfinite(Rmajor) | (Rmajor <= 0)):
        raise ValueError("Cylindrical projections require finite R > 0.")
      cosphi, sinphi = Xcart / Rmajor, Ycart / Rmajor
    else:
      Rmajor, phi = values[..., 0], values[..., 2]
      if np.any(~np.isfinite(Rmajor) | (Rmajor <= 0) | ~np.isfinite(phi)):
        raise ValueError(
            "Cylindrical projections require finite R > 0 and phi.")
      cosphi, sinphi = np.cos(phi), np.sin(phi)
    result = np.zeros_like(values)
    if selector == "R":
      result[..., 0], result[..., 1] = cosphi, sinphi
    else:
      result[..., 0], result[..., 1] = -sinphi, cosphi
    return result.swapaxes(1, 2).reshape(shape)

  layout = dg_layout(positions)
  if layout and layout.value_form == "modal":
    return apply(positions, unit)
  positions._require_operable()
  values = unit(positions.values)
  if positions.backend == "gkyl":
    values = dg.rep.wrap(values)
  return positions._result(positions.grid, values)


def project_vector(
    components: Sequence[GDataState],
    *,
    basis: Literal["contravariant", "covariant", "cartesian"],
    rep: Literal["covar", "contra", "cyl", "cart", "norm"],
    direction: int,
    metric_cov: GDataState | None = None,
    metric_contra: GDataState | None = None,
    tangents: GDataState | None = None,
    duals: GDataState | None = None,
    normals: GDataState | None = None,
    positions: GDataState | None = None,
    position_system: Literal["cartesian", "cylindrical"] | None = None,
) -> GDataState:
  """Return one component of a complete vector in the requested representation.

  Coordinate conversion uses V_i = g_ij V^j or V^i = g^ij V_j.
  Cartesian reconstruction uses V = sum(V^i e_i) = sum(V_i grad(x^i)).
  All contractions use native weak products or pointwise products according
  to the input representation; no interpolation or geometry discovery occurs.
  In 2x all three vector components and the full 0,1,2 indexing remain present.

  Args:
    components: Three collocated scalar component datasets.
    basis: Input component basis.
    rep: Output representation: covar, contra, cyl, cart, or norm.
    direction: Index 0,1,2. Cylindrical order is R,phi,Z; Cartesian is X,Y,Z;
      norm selects the increasing-coordinate unit normal.
    metric_cov: Six covariant metric fields packed 00,01,02,11,12,22.
    metric_contra: Six contravariant metric fields in the same packing.
    tangents: Nine Cartesian fields packed e_0[X,Y,Z], e_1[X,Y,Z], e_2[X,Y,Z].
    duals: Nine reciprocal-basis fields with the same vector packing.
    normals: Nine increasing-coordinate unit-normal fields with that packing.
    positions: Three position fields, needed only for cylindrical R/phi.
    position_system: Position layout: cartesian (X,Y,Z) or cylindrical (R,Z,phi).

  Returns:
    A fresh scalar dataset in the same backend and value representation.

  Raises:
    ValueError: Invalid selectors, missing or incompatible geometry, unsupported
      conversion dimensions, or undefined cylindrical R/phi at R <= 0.
  """
  if len(components) != 3:
    raise ValueError("project_vector requires three scalar components.")
  if basis not in BASIS_REPRESENTATION:
    raise ValueError("basis must be contravariant, covariant, or cartesian.")
  if rep not in REPRESENTATIONS:
    raise ValueError("Choose rep from " + ", ".join(REPRESENTATIONS))
  comp = vector_direction(direction)
  template = components[0]
  for component in components:
    _require_fields(component, 1, template)
  if rep == BASIS_REPRESENTATION[basis]:
    return components[comp].clone()
  if template.num_dims not in (2, 3):
    raise ValueError("Vector representation conversions require 2x or 3x data.")
  if rep in ("covar", "contra"):
    if basis == "cartesian":
      vectors = tangents if rep == "covar" else duals
      if vectors is None:
        raise ValueError(
            "Cartesian-to-coordinate conversion requires tangents or duals.")
      _require_fields(vectors, 9, template)
      return _dot(components,
                  [select(vectors, comp=3 * comp + j) for j in range(3)])
    metric = metric_cov if rep == "covar" else metric_contra
    if metric is None:
      required = "metric_cov" if rep == "covar" else "metric_contra"
      raise ValueError(f"rep={rep} requires {required}.")
    _require_fields(metric, 6, template)
    return _dot(components,
                [select(metric, comp=j) for j in _METRIC_COMP[comp]])
  if basis == "cartesian":
    cart = components
  else:
    vectors = tangents if basis == "contravariant" else duals
    if vectors is None:
      required = "tangents" if basis == "contravariant" else "duals"
      raise ValueError(f"{basis} reconstruction requires {required}.")
    _require_fields(vectors, 9, template)
    cart = [
        _dot(components, [select(vectors, comp=3 * i + j) for i in range(3)])
        for j in range(3)
    ]
  if rep == "cart":
    return cart[comp].clone()
  if rep == "cyl" and comp == 2:
    return cart[2].clone()
  if rep == "norm":
    if normals is None:
      raise ValueError("Normal projections require normals.")
    _require_fields(normals, 9, template)
    unit = [select(normals, comp=3 * comp + j) for j in range(3)]
  else:
    if positions is None:
      raise ValueError("Cylindrical projections require positions.")
    _require_fields(positions, 3, template)
    target = _cylindrical_unit(positions, ("R", "phi")[comp], position_system)
    unit = [select(target, comp=j) for j in range(3)]
  return _dot(cart, unit)
