"""Map one or multiple coordinates of an interpolated dataset."""

from __future__ import annotations

import os
from typing import Annotated
import numpy as np

from postgkyl.cli_spec import CliType, KeyValue
from postgkyl.dg import eval_at_points
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.io import parse_output_name, read
from postgkyl.numerics.grid_centering import sample_coordinates

# Mirrors GKYL_GEOMETRY_TOKAMAK_IWL in gkyl_tok_geo_type.
_TOKAMAK_IWL = 20


def _normalization_options(optional: dict[str, str] | None) -> dict[str, str]:
  """Expand comma-separated CLI entries into the two normalization inputs."""
  result: dict[str, str] = {}
  for key, value in (optional or {}).items():
    parts = str(value).split(",")
    entries = [f"{key}={parts[0]}", *parts[1:]]
    for entry in entries:
      name, separator, source = entry.partition("=")
      name, source = name.strip(), source.strip()
      if not separator or not source or name not in {"Cy", "rho_s"}:
        raise ValueError("map_interp optional expects Cy or rho_s as key=value")
      if name in result:
        raise ValueError(f"map_interp optional repeats {name}")
      result[name] = source
  return result


def _normalization_value(ctx: dict, name: str, options: dict[str,
                                                             str]) -> float:
  source = options.get(name, name)
  try:
    return float(source)
  except (TypeError, ValueError):
    pass
  try:
    return float(ctx[source])
  except (KeyError, TypeError, ValueError) as err:
    raise ValueError(f"rho_s requires numeric {name} from {source!r}") from err


def _psi_source(file_name: str, psi_file: str | None):
  """Read the companion psi field and its geometry metadata once."""
  if psi_file is None:
    name = parse_output_name(file_name)
    if name is None:
      raise ValueError("map_interp requires a source filename or psi_file")
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


def _psi_pol_to_rho_pol(samples, data, psi_file, options):
  psi_axis, psi_sep = _flux_reference(data.file_name, psi_file)
  normalized = (samples - psi_axis) / (psi_sep - psi_axis)
  if not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
    raise ValueError(
        "rho_pol is undefined for psi outside the axis-side domain")
  return np.sqrt(normalized), r"$\rho_{\mathrm{pol}}$"


def _psi_pol_to_R_omp(samples, data, psi_file, options):
  mapped, _ = _outboard_radius(samples, data.file_name, psi_file)
  return mapped, r"$R_{\mathrm{OMP}}$ (m)"


def _psi_pol_to_R_omp_R_sep(samples, data, psi_file, options):
  mapped, boundary_key = _outboard_radius(samples,
                                          data.file_name,
                                          psi_file,
                                          relative_to_boundary=True)
  label = (r"$R_{\mathrm{omp}} - R_{\mathrm{LCFS}}$ (m)" if boundary_key
           == "psi_lcfs" else r"$R_{\mathrm{omp}} - R_{\mathrm{sep}}$ (m)")
  return mapped, label


def _alpha_to_rho_s(samples, data, psi_file, options):
  if data.ctx.get("geometry_type") != 1:
    raise ValueError("rho_s requires tokamak geometry metadata")
  cy = _normalization_value(data.ctx, "Cy", options)
  rho_s = _normalization_value(data.ctx, "rho_s", options)
  if not np.isfinite(cy) or cy == 0 or not np.isfinite(rho_s) or rho_s <= 0:
    raise ValueError("rho_s requires finite nonzero Cy and positive rho_s")
  return samples * cy / rho_s, r"$y/\rho_s$"


_MAPPINGS = {
    (0, "psi_pol", "rho_pol"):
    (_psi_pol_to_rho_pol, "Poloidal flux to rho poloidal."),
    (0, "psi_pol", "R_omp"):
    (_psi_pol_to_R_omp, "Poloidal flux to major radius at OMP."),
    (0, "psi_pol", "R_omp_R_sep"):
    (_psi_pol_to_R_omp_R_sep,
     "Poloidal flux to R_omp minus separatrix/LCFS major radius."),
    (1, "alpha", "rho_s"):
    (_alpha_to_rho_s, "Binormal alpha to ion sound gyroradius units."),
}
_DEFAULT_SOURCES = {0: "psi_pol", 1: "alpha"}


def available_maps() -> list[tuple[str, str]]:
  """Return the supported map options and their descriptions."""
  return [(f"* -a {axis}, -s {source}, -d {dest}", description)
          for (axis, source, dest), (_, description) in _MAPPINGS.items()]


def _names(value):
  return tuple(
      item.strip()
      for item in value.split(",")) if isinstance(value, str) else tuple(value)


def map_interp(
    data: GDataState,
    *,
    axes: Annotated[int | tuple[int, ...] | list[int] | str,
                    CliType(str)],
    dest: Annotated[str | tuple[str, ...] | list[str],
                    CliType(str)],
    source: Annotated[str | tuple[str, ...] | list[str] | None,
                      CliType(str)] = None,
    psi_file: str | None = None,
    inplace: bool = False,
    tag: str | None = None,
    label: str | None = None,
    optional: Annotated[dict[str, str] | None,
                        KeyValue()] = None) -> GDataState:
  """Map one or multiple coordinates of an interpolated dataset.

  See ``--avail_maps`` for a list of currently available maps.

  Args:
    data: Interpolated gyrokinetic dataset.
    axes: Computational axes, e.g. ``0`` or ``0,1`` at the CLI.
    dest: Matching destinations, e.g. ``R_omp,rho_s`` at the CLI.
    source: Incoming coordinates, e.g. ``psi_pol,alpha``; defaults by axis.
    psi_file: Optional path to Gkeyll's psi field.
    optional: Normalization inputs as metadata names or numeric values. At the
      CLI, use ``-o rho_s=rho_s0,Cy=0.2`` or repeat the option as
      ``-o rho_s=rho_s0 -o Cy=0.2``; ``-o rho_s=1.1`` uses a literal value.
    inplace: Mutate and return ``data`` when true.
    tag: Optional tag for the result.
    label: Optional display label for the result.
  """
  try:
    directions = (tuple(
        int(item.strip())
        for item in axes.split(",")) if isinstance(axes, str) else
                  (axes, ) if isinstance(axes, int) else tuple(axes))
  except (TypeError, ValueError) as err:
    raise ValueError("map_interp requires integer axes") from err
  coordinates = _names(dest)
  if not directions or len(directions) != len(coordinates):
    raise ValueError("map_interp needs one destination per axis")
  if len(set(directions)) != len(directions):
    raise ValueError("map_interp axes must be distinct")
  sources = (tuple(_DEFAULT_SOURCES.get(d)
                   for d in directions) if source is None else _names(source))
  if len(sources) != len(directions):
    raise ValueError("map_interp needs one source per axis")
  if any((d, s, c) not in _MAPPINGS
         for d, s, c in zip(directions, sources, coordinates)):
    raise ValueError("map_interp supports axes=0 from psi_pol to rho_pol, "
                     "R_omp, or R_omp_R_sep, and axes=1 from alpha to rho_s")
  if not data.ctx.get("interpolated") or data.backend != "numpy":
    raise ValueError(
        "map_interp requires interpolated NumPy data; call interp first")
  options = _normalization_options(optional)
  grid = list(data.grid)
  labels = dict(data.ctx.get("coordinate_labels", {}))
  for direction, incoming, outgoing in zip(directions, sources, coordinates):
    if len(data.grid) <= direction:
      raise ValueError(
          f"map_interp requires direction {direction} in the data grid")
    axis = np.asarray(data.grid[direction])
    if axis.ndim != 1:
      raise ValueError("map_interp requires a separable coordinate axis")
    samples = sample_coordinates(axis, data.values.shape[direction])
    mapping, _ = _MAPPINGS[(direction, incoming, outgoing)]
    mapped, axis_label = mapping(samples, data, psi_file, options)
    grid[direction] = mapped
    labels[direction] = axis_label
  return data._result(grid,
                      data.values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      coordinate_labels=labels)
