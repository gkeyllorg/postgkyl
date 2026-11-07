"""Map one gyrokinetic computational coordinate to a physical coordinate."""

from __future__ import annotations

import os
import numpy as np

from postgkyl.dg import eval_at_points
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.io import read
from postgkyl.numerics.grid_centering import sample_coordinates

# Mirrors GKYL_GEOMETRY_TOKAMAK_IWL in gkyl_tok_geo_type.
_TOKAMAK_IWL = 20


def _psi_source(file_name: str, geqdsk_file: str | None, psi_file: str | None):
  """Read the companion psi field and its geometry metadata once."""
  if psi_file is None:
    if not file_name or not geqdsk_file:
      raise ValueError("map_gk_dir requires geqdsk_file metadata or psi_file")
    stem = os.path.splitext(os.path.basename(geqdsk_file))[0]
    psi_file = os.path.join(os.path.dirname(file_name), f"{stem}-psi.gkyl")
  ctx: dict = {}
  grid, values = read(psi_file, ctx)
  if ctx.get("geometry_type") != 1 or "tokamak_geo_type" not in ctx:
    raise ValueError(
        f"'{psi_file}' requires tokamak geometry_type and tokamak_geo_type metadata"
    )
  return psi_file, grid, values, ctx


def _flux_reference(file_name: str, geqdsk_file: str | None,
                    psi_file: str | None) -> tuple[float, float]:
  """Read the axis and boundary flux from this run's companion psi field."""
  psi_file, _, _, ctx = _psi_source(file_name, geqdsk_file, psi_file)
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


def _outboard_radius(psi: np.ndarray, file_name: str, geqdsk_file: str | None,
                     psi_file: str | None) -> np.ndarray:
  """Find the first outboard root of the stored piecewise polynomial psi."""
  path, grid, values, ctx = _psi_source(file_name, geqdsk_file, psi_file)
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

  result = np.empty_like(psi, dtype=float)
  for index, target in np.ndenumerate(psi):
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
  return result


def map_gk_dir(data: GDataState,
               *,
               dir: int,
               coord: str,
               psi_file: str | None = None,
               inplace: bool = False,
               tag: str | None = None,
               label: str | None = None) -> GDataState:
  """Map interpolated psi samples to normalized flux radius or OMP radius.

  Supports ``dir=0`` with ``coord='rho_pol'`` or ``coord='R_omp'``. The companion
  ``<geqdsk_file stem>-psi.gkyl`` supplies ``simag`` and either
  ``psi_lcfs`` (inner-wall limited) or ``psi_sep`` (diverted);
  ``psi_file`` overrides its inferred path.

  Args:
    data: Interpolated gyrokinetic dataset.
    dir: Coordinate direction to map; currently 0.
    coord: Destination coordinate: ``rho_pol`` or ``R_omp``.
    psi_file: Optional path to Gkeyll's psi field.
    inplace: Mutate and return ``data`` when true.
    tag: Optional tag for the result.
    label: Optional display label for the result.
  """
  if dir != 0 or coord not in ("rho_pol", "R_omp"):
    raise ValueError(
        "map_gk_dir supports only dir=0 with coord='rho_pol' or 'R_omp'")
  if not data.ctx.get("interpolated") or data.backend != "numpy":
    raise ValueError(
        "map_gk_dir requires interpolated NumPy data; call interp first")
  axis = np.asarray(data.grid[0])
  if axis.ndim != 1:
    raise ValueError("map_gk_dir requires a separable psi axis")
  psi = sample_coordinates(axis, data.values.shape[0])
  if coord == "rho_pol":
    psi_axis, psi_sep = _flux_reference(data.file_name,
                                        data.ctx.get("geqdsk_file"), psi_file)
    normalized = (psi - psi_axis) / (psi_sep - psi_axis)
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
      raise ValueError(
          "rho_pol is undefined for psi outside the axis-side domain")
    mapped = np.sqrt(normalized)
    axis_label = r"$\rho_{\mathrm{pol}}$"
  else:
    mapped = _outboard_radius(psi, data.file_name, data.ctx.get("geqdsk_file"),
                              psi_file)
    axis_label = r"$R_{\mathrm{OMP}}$ (m)"
  grid = list(data.grid)
  grid[0] = mapped
  labels = dict(data.ctx.get("coordinate_labels", {}))
  labels[0] = axis_label
  return data._result(grid,
                      data.values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      coordinate_labels=labels)
