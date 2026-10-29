"""GK geometry discovery, units, and component assembly for vector conversion."""
from __future__ import annotations

import os

from postgkyl import operations
from postgkyl.gdata import GData
from postgkyl.io.geometry import is_geo_mapc2p
from postgkyl.operations.vector_projection import (BASIS_REPRESENTATION,
                                                   UNIT_VECTOR_REPRESENTATIONS,
                                                   REPRESENTATIONS,
                                                   vector_direction)


def load_vector_geometry(path: str, name: str, direction: int, basis: str,
                         rep: str) -> dict:
  """Load only the geometry needed to convert the selected vector component."""
  if rep == BASIS_REPRESENTATION[basis]:
    return {}
  fields = {}
  if rep in ("covar", "contra"):
    fields["metric_cov" if rep == "covar" else "metric_contra"] = (
        "g_ij" if rep == "covar" else "gij")
  else:
    fields["tangents" if basis == "contravariant" else "duals"] = (
        "dxdz" if basis == "contravariant" else "dzdx")
    if rep == "norm":
      fields["normals"] = "normals"
    elif rep == "cyl" and direction != 2:
      fields["positions"] = "mapc2p"
  result = {}
  for argument, suffix in fields.items():
    filename = os.path.join(path, f"{name}-geo_int_{suffix}.gkyl")
    if not os.path.isfile(filename):
      raise FileNotFoundError(f"rep={rep}, dir={direction} requires {filename}")
    data = GData(filename)
    if data.num_dims not in (2, 3):
      raise ValueError(
          "Vector representation conversions require 2x or 3x geometry.")
    result[argument] = data
  if "positions" in result:
    result["position_system"] = ("cartesian" if is_geo_mapc2p(
        result["positions"].ctx) else "cylindrical")
  return result


def fetch_vector(fetch_component, gdatas, quantity, **kwargs):
  """Compute in the quantity's native basis before converting the final vector."""
  if quantity.packed_vector and kwargs.get("dir") is None and kwargs.get(
      "rep") is None:
    return fetch_component(gdatas, **kwargs)
  direction = vector_direction(kwargs.get("dir"))
  native_rep = BASIS_REPRESENTATION[quantity.component_basis]
  rep = kwargs.get("rep") or native_rep
  if rep not in REPRESENTATIONS:
    raise ValueError("Choose rep from " + ", ".join(REPRESENTATIONS))
  packed = fetch_component(gdatas, **kwargs) if quantity.packed_vector else None

  def component(i):
    """Read a packed component or evaluate a native scalar component."""
    if packed is not None:
      return operations.select(packed, comp=i)
    return fetch_component(gdatas, **dict(kwargs, dir=i))

  if rep == native_rep:
    out = component(direction)
  else:
    out = operations.project_vector([component(i) for i in range(3)],
                                    basis=quantity.component_basis,
                                    rep=rep,
                                    direction=direction,
                                    **kwargs.get("vector_geometry", {}))
  if rep in UNIT_VECTOR_REPRESENTATIONS:
    units = quantity.unit_vector_units
  else:
    basis = "covariant" if rep == "covar" else "contravariant"
    units = f"{basis} component units"
    if quantity.unit_vector_units == "m/s" and rep == "contra":
      units = "coordinate units/s"
      if out.ctx.get("geometry_type") == 1:
        units = ("Wb/(rad s)", "rad/s", "rad/s")[direction]
  out.ctx["projection_units"] = units
  out.ctx["projection_direction"] = direction
  out.ctx["vector_representation"] = rep
  return out
