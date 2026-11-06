"""Map one gyrokinetic computational coordinate to a physical coordinate."""

from __future__ import annotations

import os
import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.io import read
from postgkyl.numerics.grid_centering import sample_coordinates

# Mirrors GKYL_GEOMETRY_TOKAMAK_IWL in gkyl_tok_geo_type.
_TOKAMAK_IWL = 20


def _flux_reference(file_name: str, geqdsk_file: str | None,
                    psi_file: str | None) -> tuple[float, float]:
  """Read the axis and boundary flux from this run's companion psi field."""
  if psi_file is None:
    if not file_name or not geqdsk_file:
      raise ValueError("map_gk_dir requires geqdsk_file metadata or psi_file")
    stem = os.path.splitext(os.path.basename(geqdsk_file))[0]
    psi_file = os.path.join(os.path.dirname(file_name), f"{stem}-psi.gkyl")
  ctx: dict = {}
  read(psi_file, ctx)
  if ctx.get("geometry_type") != 1 or "tokamak_geo_type" not in ctx:
    raise ValueError(
        f"'{psi_file}' requires tokamak geometry_type and tokamak_geo_type metadata"
    )
  boundary_key = ("psi_lcfs"
                  if ctx["tokamak_geo_type"] == _TOKAMAK_IWL else "psi_sep")
  try:
    axis, boundary = float(ctx["simag"]), float(ctx[boundary_key])
  except (KeyError, TypeError, ValueError) as err:
    raise ValueError(
        f"'{psi_file}' requires numeric simag and {boundary_key} metadata"
    ) from err
  if not np.isfinite(axis) or not np.isfinite(boundary) or axis == boundary:
    raise ValueError(f"simag and {boundary_key} must be finite and distinct")
  return axis, boundary


def map_gk_dir(data: GDataState,
               *,
               dir: int,
               coord: str,
               psi_file: str | None = None,
               inplace: bool = False,
               tag: str | None = None,
               label: str | None = None) -> GDataState:
  """Map interpolated psi samples to normalized poloidal flux radius.

  The first stage supports ``dir=0, coord='rho_pol'``. The companion
  ``<geqdsk_file stem>-psi.gkyl`` supplies ``simag`` and either
  ``psi_lcfs`` (inner-wall limited) or ``psi_sep`` (diverted);
  ``psi_file`` overrides its inferred path.

  Args:
    data: Interpolated gyrokinetic dataset.
    dir: Coordinate direction to map; currently 0.
    coord: Destination coordinate; currently ``rho_pol``.
    psi_file: Optional path to Gkeyll's psi field.
    inplace: Mutate and return ``data`` when true.
    tag: Optional tag for the result.
    label: Optional display label for the result.
  """
  if dir != 0 or coord != "rho_pol":
    raise ValueError(
        "map_gk_dir currently supports only dir=0, coord='rho_pol'")
  if not data.ctx.get("interpolated") or data.backend != "numpy":
    raise ValueError(
        "map_gk_dir requires interpolated NumPy data; call interp first")
  axis = np.asarray(data.grid[0])
  if axis.ndim != 1:
    raise ValueError("map_gk_dir requires a separable psi axis")
  psi = sample_coordinates(axis, data.values.shape[0])
  psi_axis, psi_sep = _flux_reference(data.file_name,
                                      data.ctx.get("geqdsk_file"), psi_file)
  normalized = (psi - psi_axis) / (psi_sep - psi_axis)
  if not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
    raise ValueError(
        "rho_pol is undefined for psi outside the axis-side domain")
  grid = list(data.grid)
  grid[0] = np.sqrt(normalized)
  labels = dict(data.ctx.get("coordinate_labels", {}))
  labels[0] = r"$\rho_{\mathrm{pol}}$"
  return data._result(grid,
                      data.values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      coordinate_labels=labels)
