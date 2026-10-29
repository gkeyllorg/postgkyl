"""GK geometry discovery and labels for the general vector projector."""
from __future__ import annotations

import os

from postgkyl import operations
from postgkyl.gdata import GData
from postgkyl.io.geometry import is_geo_mapc2p
from postgkyl.operations.vector_projection import PHYSICAL_DIRECTIONS

VECTOR_DIRECTIONS = ("0", "1", "2") + PHYSICAL_DIRECTIONS


def vector_direction(value) -> str:
  """Parse a coordinate component, coordinate normal, or cylindrical selector."""
  selector = str(value)
  if selector not in VECTOR_DIRECTIONS:
    raise ValueError("Choose direction= from " + ", ".join(VECTOR_DIRECTIONS))
  return selector


def load_vector_geometry(path: str, name: str, selector: str,
                         basis: str) -> dict:
  """Load only the geometry required for a registered vector's projection."""
  if selector in ("0", "1", "2"):
    return {}
  fields = {}
  if basis == "contravariant":
    fields["tangents"] = "dxdz"
  elif basis == "covariant":
    fields["duals"] = "dzdx"
  if selector.startswith("n"):
    fields["normals"] = "normals"
  elif selector != "Z":
    fields["positions"] = "mapc2p"
  result = {}
  for argument, suffix in fields.items():
    filename = os.path.join(path, f"{name}-geo_int_{suffix}.gkyl")
    if not os.path.isfile(filename):
      raise FileNotFoundError(
          f"Vector direction {selector!r} requires {filename}")
    data = GData(filename)
    if data.num_dims not in (2, 3):
      raise ValueError("Physical vector projections require 2x or 3x geometry.")
    result[argument] = data
  if "positions" in result:
    result["position_system"] = ("cartesian" if is_geo_mapc2p(
        result["positions"].ctx) else "cylindrical")
  return result


def fetch_vector(fetch_component, gdatas, quantity, **kwargs):
  """Compute numeric components or project the full vector using registry metadata."""
  selector = vector_direction(kwargs.get("dir"))
  if selector in PHYSICAL_DIRECTIONS:
    components = [
        fetch_component(gdatas, **dict(kwargs, dir=i)) for i in range(3)
    ]
    out = operations.project_vector(components,
                                    basis=quantity.component_basis,
                                    direction=selector,
                                    **kwargs.get("vector_geometry", {}))
    units = quantity.physical_units
  else:
    out = fetch_component(gdatas, **dict(kwargs, dir=int(selector)))
    units = f"{quantity.component_basis} component units"
    if quantity.physical_units == "m/s":
      units = "coordinate units/s"
      if out.ctx.get("geometry_type") == 1:
        units = ("Wb/(rad s)", "rad/s", "rad/s")[int(selector)]
  out.ctx["projection_units"] = units
  out.ctx["projection_direction"] = selector
  return out
