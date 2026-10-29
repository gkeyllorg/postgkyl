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

PHYSICAL_DIRECTIONS = ("n0", "n1", "n2", "R", "Z", "phi")


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
    direction: Literal["n0", "n1", "n2", "R", "Z", "phi"],
    tangents: GDataState | None = None,
    duals: GDataState | None = None,
    normals: GDataState | None = None,
    positions: GDataState | None = None,
    position_system: Literal["cartesian", "cylindrical"] | None = None,
) -> GDataState:
  """Return a physical projection of three coordinate or Cartesian components.

  Contravariant components reconstruct ``V = sum(V^i e_i)`` using tangents;
  covariant components reconstruct ``V = sum(V_i grad(x^i))`` using duals.
  Geometry packs each vector's Cartesian components consecutively, e.g.
  ``e_0[X,Y,Z], e_1[X,Y,Z], e_2[X,Y,Z]``. In 2x, all three components and
  geometry vectors remain present, with the full 0,1,2 component numbering.
  Products retain the inputs' DG weak-product or point-value semantics.
  No interpolation, geometry discovery, or physical-unit inference is done.

  Args:
    components: Three scalar datasets on the same 2x or 3x grid.
    basis: Representation of the supplied vector components.
    direction: Increasing-coordinate unit normal n0/n1/n2, or R/Z/phi.
    tangents: Nine Cartesian fields for the coordinate tangent vectors.
    duals: Nine Cartesian fields for the reciprocal basis vectors.
    normals: Nine Cartesian fields for increasing-coordinate unit normals.
    positions: Three fields for positions, required for R/phi projections.
    position_system: Explicit position layout: cartesian (X,Y,Z) or
      cylindrical (R,Z,phi), with phi in radians.

  Returns:
    A scalar dataset with a physical component in the vector's physical units.

  Raises:
    ValueError: Invalid selectors, missing geometry, incompatible layouts,
      or an undefined cylindrical basis (nonfinite positions or R <= 0).
  """
  if len(components) != 3:
    raise ValueError("project_vector requires three scalar components.")
  if basis not in ("contravariant", "covariant", "cartesian"):
    raise ValueError("basis must be contravariant, covariant, or cartesian.")
  if direction not in PHYSICAL_DIRECTIONS:
    raise ValueError("Choose direction from " + ", ".join(PHYSICAL_DIRECTIONS))
  template = components[0]
  if template.num_dims not in (2, 3):
    raise ValueError("Physical vector projections require 2x or 3x data.")
  for component in components:
    _require_fields(component, 1, template)
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
  if direction == "Z":
    return cart[2].clone()
  if direction.startswith("n"):
    if normals is None:
      raise ValueError("Normal projections require normals.")
    _require_fields(normals, 9, template)
    offset = 3 * int(direction[1])
    unit = [select(normals, comp=offset + j) for j in range(3)]
  else:
    if positions is None:
      raise ValueError("Cylindrical projections require positions.")
    _require_fields(positions, 3, template)
    target = _cylindrical_unit(positions, direction, position_system)
    unit = [select(target, comp=j) for j in range(3)]
  return _dot(cart, unit)
