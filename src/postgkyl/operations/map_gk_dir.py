"""Map one gyrokinetic computational coordinate to a physical coordinate."""

from __future__ import annotations

import os
from typing import Annotated
import numpy as np

from postgkyl.cli_spec import CliType
from postgkyl.dg import eval_at_points
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.io import parse_output_name, read
from postgkyl.numerics.grid_centering import sample_coordinates

# Mirrors GKYL_GEOMETRY_TOKAMAK_IWL in gkyl_tok_geo_type.
_TOKAMAK_IWL = 20


def _psi_source(file_name: str, psi_file: str | None):
  """Read the companion psi field and its geometry metadata once."""
  if psi_file is None:
    name = parse_output_name(file_name)
    if name is None:
      raise ValueError("map_gk_dir requires a source filename or psi_file")
    psi_file = os.path.join(name.directory, f"{name.sim}-psi.gkyl")
  ctx: dict = {}
  grid, values = read(psi_file, ctx)
  if ctx.get("geometry_type") != 1 or "tokamak_geo_type" not in ctx:
    raise ValueError(
        f"'{psi_file}' requires tokamak geometry_type and tokamak_geo_type metadata"
    )
  return psi_file, grid, values, ctx


def _flux_reference(file_name: str,
                    psi_file: str | None) -> tuple[float, float]:
  """Read the axis and boundary flux from this run's companion psi field."""
  psi_file, _, _, ctx = _psi_source(file_name, psi_file)
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


def _outboard_radius(
    psi: np.ndarray,
    file_name: str,
    psi_file: str | None,
    *,
    relative_to_boundary: bool = False) -> tuple[np.ndarray, str | None]:
  """Find the first outboard root of the stored piecewise polynomial psi."""
  path, grid, values, ctx = _psi_source(file_name, psi_file)
  if relative_to_boundary:
    boundary_key = ("psi_lcfs"
                    if ctx["tokamak_geo_type"] == _TOKAMAK_IWL else "psi_sep")
    try:
      boundary_psi = float(ctx[boundary_key])
    except (KeyError, TypeError, ValueError) as err:
      raise ValueError(
          f"'{path}' requires numeric {boundary_key} metadata") from err
    if not np.isfinite(boundary_psi):
      raise ValueError(f"'{path}' requires finite {boundary_key} metadata")
    targets = np.append(psi, boundary_psi)
  else:
    targets = psi
  try:
    r_axis = float(ctx["rmaxis"])
    z_axis = float(ctx["zmaxis"])
    order = int(ctx["poly_order"])
    basis_type = ctx["basis_type"]
  except (KeyError, TypeError, ValueError) as err:
    raise ValueError(
        f"'{path}' requires rmaxis, zmaxis, poly_order and basis_type metadata"
    ) from err
  lower = np.asarray(ctx["lower"], dtype=float)
  upper = np.asarray(ctx["upper"], dtype=float)
  cells = np.asarray(ctx["cells"], dtype=int)
  if (len(grid) != 2 or order < 1 or not np.isfinite(r_axis)
      or not np.isfinite(z_axis) or not lower[0] <= r_axis < upper[0]
      or not lower[1] <= z_axis <= upper[1]):
    raise ValueError(
        f"'{path}' requires a 2-D psi field with its axis inside the grid")
  coeffs = (values.view(cells) if hasattr(values, "view")
            and not isinstance(values, np.ndarray) else np.asarray(values))
  if coeffs.shape[:-1] != tuple(cells):
    raise ValueError(f"'{path}' psi coefficients do not match the file grid")

  # At fixed Z, a tensor DG field is a degree-p polynomial in each R cell.
  # Interior Gauss nodes avoid selecting the adjacent cell at a DG face.
  eta, _ = np.polynomial.legendre.leggauss(order + 1)
  edges = np.asarray(grid[0], dtype=float)
  polynomials = []
  for lo, hi in zip(edges[:-1], edges[1:]):
    if hi <= r_axis:
      continue
    radii = (lo + hi) / 2 + (hi - lo) * eta / 2
    points = np.column_stack((radii, np.full_like(radii, z_axis)))
    flux = eval_at_points(coeffs,
                          lower,
                          upper,
                          cells,
                          points,
                          basis_type=basis_type,
                          poly_order=order,
                          nodal=ctx.get("value_form") == "nodal")
    if not np.all(np.isfinite(flux)):
      raise ValueError(f"'{path}' has non-finite psi on the outboard midplane")
    polynomials.append(
        (max(lo, r_axis), hi, np.polynomial.Polynomial.fit(radii, flux, order)))

  result = np.empty_like(targets, dtype=float)
  for index, target in np.ndenumerate(targets):
    if not np.isfinite(target):
      raise ValueError("R_omp requires finite computational psi samples")
    for lo, hi, polynomial in polynomials:
      roots = (polynomial - target).roots()
      tol = 1e-10 * max(1.0, abs(lo), abs(hi))
      candidates = [
          float(root.real) for root in roots
          if abs(root.imag) <= tol and lo - tol <= root.real <= hi + tol
      ]
      if candidates:
        result[index] = max(lo, min(candidates))
        break
    else:
      raise ValueError(f"No outboard R_omp root for psi={target:g} in '{path}'")
  if relative_to_boundary:
    return result[:-1] - result[-1], boundary_key
  return result, None


def map_gk_dir(data: GDataState,
               *,
               dir: Annotated[int | tuple[int, ...] | list[int] | str,
                              CliType(str)],
               coord: Annotated[str | tuple[str, ...] | list[str],
                                CliType(str)],
               psi_file: str | None = None,
               inplace: bool = False,
               tag: str | None = None,
               label: str | None = None) -> GDataState:
  """Map interpolated psi samples to a flux or outboard-midplane coordinate.

  Supports ``dir=0`` with ``coord='rho_pol'``, ``'R_omp'``, or
  ``'R_omp_R_sep'``, and ``dir=1, coord='rho_s'``. Pair comma-separated
  directions and coordinates to map several axes in one call. ``rho_s`` maps the
  binormal angle to ``Cy * alpha / rho_s`` using run metadata. ``R_omp_R_sep``
  subtracts the outboard radius at
  ``psi_lcfs`` for inner-wall limited geometry (including Miller), or at
  ``psi_sep`` for diverted geometry. The companion ``<simulation_name>-psi.gkyl``
  supplies the geometry and is shared by all blocks. ``psi_file`` overrides
  this lookup.

  Args:
    data: Interpolated gyrokinetic dataset.
    dir: Computational directions, e.g. ``0`` or ``0,1`` at the CLI.
    coord: Matching destinations, e.g. ``R_omp,rho_s`` at the CLI.
    psi_file: Optional path to Gkeyll's psi field.
    inplace: Mutate and return ``data`` when true.
    tag: Optional tag for the result.
    label: Optional display label for the result.
  """
  try:
    directions = (tuple(int(item.strip())
                        for item in dir.split(",")) if isinstance(dir, str) else
                  (dir, ) if isinstance(dir, int) else tuple(dir))
  except (TypeError, ValueError) as err:
    raise ValueError("map_gk_dir requires integer directions") from err
  coordinates = (tuple(
      item.strip()
      for item in coord.split(",")) if isinstance(coord, str) else tuple(coord))
  if not directions or len(directions) != len(coordinates):
    raise ValueError("map_gk_dir needs one coordinate per direction")
  if len(set(directions)) != len(directions):
    raise ValueError("map_gk_dir directions must be distinct")
  allowed = {0: {"rho_pol", "R_omp", "R_omp_R_sep"}, 1: {"rho_s"}}
  if any(not isinstance(d, int) or d not in allowed or c not in allowed[d]
         for d, c in zip(directions, coordinates)):
    raise ValueError("map_gk_dir supports dir=0 with 'rho_pol', 'R_omp', or "
                     "'R_omp_R_sep', and dir=1 with 'rho_s'")
  if not data.ctx.get("interpolated") or data.backend != "numpy":
    raise ValueError(
        "map_gk_dir requires interpolated NumPy data; call interp first")
  grid = list(data.grid)
  labels = dict(data.ctx.get("coordinate_labels", {}))
  for direction, destination in zip(directions, coordinates):
    if len(data.grid) <= direction:
      raise ValueError(
          f"map_gk_dir requires direction {direction} in the data grid")
    axis = np.asarray(data.grid[direction])
    if axis.ndim != 1:
      raise ValueError("map_gk_dir requires a separable coordinate axis")
    samples = sample_coordinates(axis, data.values.shape[direction])
    if destination == "rho_s":
      if data.ctx.get("geometry_type") != 1:
        raise ValueError("rho_s requires tokamak geometry metadata")
      try:
        cy = float(data.ctx["Cy"])
        rho_s = float(data.ctx["rho_s"])
      except (KeyError, TypeError, ValueError) as err:
        raise ValueError(
            "rho_s requires numeric Cy and rho_s metadata") from err
      if not np.isfinite(cy) or cy == 0 or not np.isfinite(rho_s) or rho_s <= 0:
        raise ValueError("rho_s requires finite nonzero Cy and positive rho_s")
      mapped = samples * cy / rho_s
      axis_label = r"$y/\rho_s$"
    elif destination == "rho_pol":
      psi_axis, psi_sep = _flux_reference(data.file_name, psi_file)
      normalized = (samples - psi_axis) / (psi_sep - psi_axis)
      if not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
        raise ValueError(
            "rho_pol is undefined for psi outside the axis-side domain")
      mapped = np.sqrt(normalized)
      axis_label = r"$\rho_{\mathrm{pol}}$"
    else:
      mapped, boundary_key = _outboard_radius(
          samples,
          data.file_name,
          psi_file,
          relative_to_boundary=destination == "R_omp_R_sep")
      if boundary_key == "psi_lcfs":
        axis_label = r"$R_{\mathrm{omp}} - R_{\mathrm{LCFS}}$ (m)"
      elif boundary_key == "psi_sep":
        axis_label = r"$R_{\mathrm{omp}} - R_{\mathrm{sep}}$ (m)"
      else:
        axis_label = r"$R_{\mathrm{OMP}}$ (m)"
    grid[direction] = mapped
    labels[direction] = axis_label
  return data._result(grid,
                      data.values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      coordinate_labels=labels)
