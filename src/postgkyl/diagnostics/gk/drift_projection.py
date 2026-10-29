"""Physical projections of GK coordinate-rate drifts using saved geometry."""
from __future__ import annotations

import os
import numpy as np

from postgkyl import operations
from postgkyl.gdata import GData
from postgkyl.io.geometry import is_geo_mapc2p

DRIFT_QUANTITIES = frozenset(("ExB_vel", "gradB_vel", "diamag_vel"))
DRIFT_DIRECTIONS = ("0", "1", "2", "n0", "n1", "n2", "R", "Z", "phi")


def drift_direction(value) -> str:
  """Parse a coordinate rate, coordinate normal, or cylindrical selector."""
  selector = str(value)
  if selector not in DRIFT_DIRECTIONS:
    raise ValueError("Choose direction= from " + ", ".join(DRIFT_DIRECTIONS))
  return selector


def load_drift_geometry(path: str, name: str, selector: str) -> dict:
  """Load only the geometry needed by a physical projection, once per load."""
  if selector in ("0", "1", "2"):
    return {}
  fields = {"tangents": "dxdz"}
  if selector.startswith("n"):
    fields["normals"] = "normals"
  elif selector != "Z":
    fields["positions"] = "mapc2p"
  result = {}
  for argument, suffix in fields.items():
    filename = os.path.join(path, f"{name}-geo_int_{suffix}.gkyl")
    if not os.path.isfile(filename):
      raise FileNotFoundError(
          f"Drift direction {selector!r} requires {filename}")
    data = GData(filename)
    if data.num_dims not in (2, 3):
      raise ValueError("Physical drift projections require 2x or 3x geometry.")
    result[argument] = data
  return result


def _component(data, comp):
  """Select one complete DG field from packed Cartesian geometry vectors."""
  return operations.select(data, comp=comp)


def _cylindrical_unit(positions, selector):
  """Project a cylindrical unit vector from pointwise physical positions."""
  cartesian = is_geo_mapc2p(positions.ctx)

  def unit(values):
    """Evaluate unit-vector components at quadrature points, never coefficients."""
    shape = values.shape
    values = values.reshape(shape[0], 3, -1).swapaxes(1, 2)
    if cartesian:
      Xcart, Ycart = values[..., 0], values[..., 1]
      Rmajor = np.hypot(Xcart, Ycart)
      if np.any(~np.isfinite(Rmajor) | (Rmajor <= 0)):
        raise ValueError("Cylindrical drift projections require finite R > 0.")
      cosphi, sinphi = Xcart / Rmajor, Ycart / Rmajor
    else:
      Rmajor, phi = values[..., 0], values[..., 2]
      if np.any(~np.isfinite(Rmajor) | (Rmajor <= 0) | ~np.isfinite(phi)):
        raise ValueError(
            "Cylindrical drift projections require finite R > 0 and phi.")
      cosphi, sinphi = np.cos(phi), np.sin(phi)
    result = np.zeros_like(values)
    if selector == "R":
      result[..., 0], result[..., 1] = cosphi, sinphi
    else:
      result[..., 0], result[..., 1] = -sinphi, cosphi
    return result.swapaxes(1, 2).reshape(shape)

  return positions.apply(unit)


def project_drift(component,
                  selector,
                  template,
                  *,
                  tangents=None,
                  normals=None,
                  positions=None):
  """Return a coordinate rate or a physical projection of the complete drift.

  ``component(i)`` returns v^i. Geometry packs each vector's Cartesian
  components consecutively: e_0[X,Y,Z], e_1[X,Y,Z], e_2[X,Y,Z].
  Coordinate normals have the same layout and point toward increasing x^i.
  In 2x, the grid retains x^0 and x^2, but all three vector components
  and geometry vectors remain available; the binormal velocity need not vanish.
  """
  selector = drift_direction(selector)
  if selector in ("0", "1", "2"):
    out = component(int(selector))
    units = "coordinate units/s"
    if template.ctx.get("geometry_type") == 1:
      units = ("Wb/(rad s)", "rad/s", "rad/s")[int(selector)]
  else:
    if template.num_dims not in (2, 3):
      raise ValueError("Physical drift projections require 2x or 3x data.")
    if tangents is None:
      raise ValueError(
          "Physical drift projections require geo_int_dxdz (tangents).")
    velocity = [component(i) for i in range(3)]
    cart = [
        sum((velocity[i] * _component(tangents, 3 * i + j) for i in range(3)))
        for j in range(3)
    ]
    if selector == "Z":
      out = cart[2]
    elif selector.startswith("n"):
      if normals is None:
        raise ValueError("Normal drift projections require geo_int_normals.")
      offset = 3 * int(selector[1])
      out = sum(cart[j] * _component(normals, offset + j) for j in range(3))
    else:
      if positions is None:
        raise ValueError(
            "Cylindrical drift projections require geo_int_mapc2p.")
      unit = _cylindrical_unit(positions, selector)
      out = sum(cart[j] * _component(unit, j) for j in range(3))
    units = "m/s"
  out.ctx["drift_units"] = units
  out.ctx["drift_direction"] = selector
  return out
