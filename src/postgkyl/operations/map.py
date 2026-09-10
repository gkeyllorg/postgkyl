"""Coordinate transformations with explicit mapping inputs.

``map_to_rz`` reconstructs a poloidal slice; ``extract_flux_surface`` samples
a toroidal surface. Both receive precomputed geometry and never discover
auxiliary files. Three-dimensional R-Z reconstruction assumes periodic
field-aligned coordinates with twist-and-shift boundary conditions.

The ``map`` verb deforms a dataset's grid by evaluating a coordinate map.

See ``MAPPING.md`` for the full design. A mapping file is a DG field whose
components hold the coefficients of the physical coordinates of each mapped
dimension; this verb evaluates those coefficients at the *target*'s own grid
points (:func:`postgkyl.dg.map_grid`) and splices the resulting arrays into
a copy of the target's grid. Only the grid changes -- the mapping's
coefficients are read straight from its native modal storage and are never
interpolated, and the target's values are passed through unchanged (no
copy: this verb never touches them).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

from postgkyl.numerics.resample import resample_grid
from postgkyl.numerics.rz import fft_poloidal_project
from .geometry import (
    Geometry,
    RzProjection,
    FluxSurfaceGrid,
    _interpolate_component,
    _interpolation_grid,
    validate_mapping_grid,
    _validate_component,
    _validate_geometry,
    _validate_modal_data,
    _validate_positive_int,
)

from postgkyl import dg
from postgkyl.gdatastate.gdatastate import GDataState

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState as _GDataState


def map(data: "_GDataState",
        mapping: "str | _GDataState",
        *,
        space: str = "conf",
        basis_type: str | None = None,
        poly_order: int | None = None,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None) -> "_GDataState":
  """Replace a block of ``data``'s grid axes with mapped coordinates.

  Evaluates the mapping's DG coefficients at ``data``'s existing grid
  points (no resolution parameter, no alignment arithmetic -- the mapped
  axes always keep the shape of the axes they replace) and splices the
  result into a copy of ``data``'s grid.

  Args:
    data: The dataset whose grid is deformed; must be NumPy-backed
      (post-``interpolate()``), like ``select``.
    mapping: The coordinate-mapping field, as a filename or an
      already-loaded dataset. Read from its native modal coefficients --
      never interpolated. Its number of dimensions (``m``) sets how many
      of ``data``'s axes are replaced.
    space: ``'conf'`` deforms the leading ``m`` axes (offset 0),
      curvilinearly (every physical coordinate is evaluated over all ``m``
      mapped dimensions, so non-separable maps such as rotations work); its
      component count must be ``m * num_basis`` for an ``m``-D basis.
      ``'vel'`` deforms the trailing ``m`` axes (offset
      ``data.num_dims - m``); Gkeyll's velocity-space maps
      (``mapc2p_vel``) are diagonal -- each dimension is evaluated by its
      *own* 1-D map, so the component count must be ``m * num_basis`` for a
      *1-D* basis. For a combined map, apply the verb twice.
    basis_type: ``mapping``'s ``basis_type`` (long name, e.g.
      ``"serendipity"``), set at the moment this call loads it -- only
      takes effect when ``mapping`` is given as a filename (a property of
      already-loaded data can't be re-specified here). Velocity-space
      mapping files (``mapc2p_vel``) commonly carry no basis metadata at
      all, so this is typically required for ``space="vel"``.
    poly_order: ``mapping``'s ``poly_order``, set the same way as
      ``basis_type``.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset carrying the deformed grid; ``ctx["grid_type"]`` is set to
    ``"mapped"`` and ``ctx["mapped_axes"]`` records, for every absolute
    dimension touched so far (by this call and any earlier one), the
    ``offset`` of the mapped block it belongs to -- ``select``'s
    curvilinear guard needs this to convert an absolute dimension index
    back to the curvilinear grid array's own (relative) axis. The values
    array is untouched.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed); if ``space`` is
      neither ``'conf'`` nor ``'vel'``; if the map does not fit ``data``'s
      dimensionality; if the mapping has no ``basis_type``/``poly_order``
      metadata (and none was given at load time); or if its component
      count does not match the expected ``m * num_basis``.
  """
  if data.backend == "gkyl":
    raise ValueError(
        "map operates on interpolated (NumPy) target grids; call .interpolate() "
        "first -- deforming a native modal grid has no basis-space meaning.")

  # basis_type/poly_order are load-time properties of the mapping dataset;
  # they only apply here when this call is the one loading it (a filename),
  # threaded straight into GDataState's own override mechanism -- the one
  # home for "correct a file's missing/mislabeled basis metadata."
  map_data = (mapping if isinstance(mapping, GDataState) else GDataState(
      mapping, basis_type=basis_type, poly_order=poly_order))
  m = map_data.num_dims
  num_dims = data.num_dims

  if space == "conf":
    offset = 0
  elif space == "vel":
    offset = num_dims - m
  else:
    raise ValueError(f"map: 'space' must be 'conf' or 'vel', got {space!r}.")

  if offset < 0 or offset + m > num_dims:
    raise ValueError(
        f"map: a {m}D {space} map does not fit a {num_dims}D dataset.")

  resolved_basis_type = map_data.ctx.get("basis_type")
  resolved_poly_order = map_data.ctx.get("poly_order")
  if resolved_basis_type is None or resolved_poly_order is None:
    raise ValueError(
        "map: the mapping dataset has no 'basis_type'/'poly_order' "
        "metadata; pass basis_type=.../poly_order=... (mapping as a "
        "filename), or load it explicitly first with those set.")

  # Velocity-space maps (mapc2p_vel) are diagonal: each mapped dimension is
  # its own separate 1-D map, so its basis is 1-D regardless of m; a
  # configuration-space map (mapc2p/mc2nu) is one joint m-D curvilinear map.
  basis_dim = 1 if space == "vel" else m
  num_basis = dg.num_basis(basis_dim, resolved_poly_order, resolved_basis_type)
  if map_data.num_comps != m * num_basis:
    raise ValueError(
        f"map: mapping has {map_data.num_comps} component(s), expected "
        f"m * num_basis = {m} * {num_basis} = {m * num_basis} for a "
        f"{basis_dim}D {resolved_basis_type} p{resolved_poly_order} map" +
        (" per velocity dimension." if space == "vel" else "."))

  target_axes = list(data.grid[offset:offset + m])
  map_ctx = {
      "lower": map_data.ctx["lower"],
      "upper": map_data.ctx["upper"],
      "cells": map_data.ctx["cells"],
      "basis_type": resolved_basis_type,
      "poly_order": resolved_poly_order,
      "value_form": map_data.ctx.get("value_form", "modal"),
  }
  if space == "vel":
    new_axes = dg.map_grid_separable(map_data.get_values(), map_ctx,
                                     target_axes)
  else:
    new_axes = dg.map_grid(map_data.get_values(), map_ctx, target_axes)

  grid = list(data.grid)
  for d in range(m):
    grid[offset + d] = new_axes[d]

  # Record, per absolute dimension, the offset of the mapped block it
  # belongs to -- a curvilinear (m > 1) grid array's own axis k corresponds
  # to absolute dimension offset + k, not to the array's position in
  # `grid`, so `select`'s curvilinear guard needs this to convert back.
  # Merge with any prior block (e.g. a separate `space="vel"` map applied
  # after a `space="conf"` one) rather than overwrite it.
  mapped_axes = dict(data.ctx.get("mapped_axes", {}))
  mapped_axes.update({offset + d: offset for d in range(m)})

  return data._result(grid,
                      data.values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      grid_type="mapped",
                      mapped_axes=mapped_axes)


def resolve_rz_projection(first: "GDataState",
                          geo: Geometry,
                          *,
                          z_axis: float = 0.0,
                          nz_interp: int = 8) -> RzProjection:
  """Build an R-Z projection for ``first``'s grid and ``geo``.

  Only the computational grid and DG metadata are read from ``first``;
  projection construction never evaluates or selects its field values.
  ``z_axis`` is the magnetic-axis vertical position in meters.
  """
  _validate_modal_data(first, "map_to_rz", (2, 3))
  nz_interp = _validate_positive_int(nz_interp, "nz_interp")
  _validate_geometry(geo, first.num_dims)
  edges, centers = _interpolation_grid(first)
  vert_z = geo.vert_z + float(z_axis)

  if first.num_dims == 2:
    r = resample_grid(geo.major_r, geo.coords, edges)
    z = resample_grid(vert_z, geo.coords, edges)
    return RzProjection(num_dims=2,
                        r=r,
                        z=z,
                        computational_grid=tuple(
                            np.array(axis, copy=True) for axis in edges))

  if geo.phi is None:
    raise ValueError(
        "The geometry has no toroidal-angle component; 3-D map_to_rz requires one."
    )

  xc, yc, zc = centers
  nx, ny, nz = xc.size, yc.size, zc.size
  if ny < 2 or nz < 2:
    raise ValueError(
        "3-D map_to_rz requires at least two interpolated y and z points.")

  phi = np.unwrap(np.unwrap(np.unwrap(geo.phi, axis=2), axis=1), axis=0)
  phi_field = resample_grid(phi, geo.coords, centers)
  box_estimate = np.mean(np.diff(phi_field[nx // 2, :, nz // 2])) * ny
  if not np.isfinite(box_estimate) or np.isclose(box_estimate, 0.0):
    raise ValueError(
        "Toroidal geometry has a zero or non-finite binormal angular span.")
  n0 = max(1, int(round(abs(2.0 * np.pi / box_estimate))))
  box = np.sign(box_estimate) * 2.0 * np.pi / n0
  wind = phi_field[:, 0, -1] - phi_field[:, 0, 0]

  xn, _, zn = edges
  zf_edges = np.linspace(zn[0], zn[-1], nz_interp * nz + 1)
  zf = 0.5 * (zf_edges[:-1] + zf_edges[1:])
  phi0_zf = np.array(
      [np.interp(zf, zc, phi_field[ix, 0, :]) for ix in range(nx)])

  gx, _, gz = geo.coords
  r2d = geo.major_r[:, 0, :]
  z2d = vert_z[:, 0, :]
  gz_rz = gz
  if geo.corner is not None:
    corner_coords, corner_r, corner_z = geo.corner
    if len(corner_coords) != 3:
      raise ValueError(
          "Corner geometry must be three-dimensional for 3-D map_to_rz.")
    cx, cz = corner_coords[0], corner_coords[2]
    corner_r = corner_r[:, 0, :]
    corner_z = corner_z[:, 0, :] + float(z_axis)
    r2d = np.concatenate([
        np.interp(gx, cx, corner_r[:, 0])[:, None], r2d,
        np.interp(gx, cx, corner_r[:, -1])[:, None]
    ],
                         axis=1)
    z2d = np.concatenate([
        np.interp(gx, cx, corner_z[:, 0])[:, None], z2d,
        np.interp(gx, cx, corner_z[:, -1])[:, None]
    ],
                         axis=1)
    gz_rz = np.concatenate([[cz[0]], gz, [cz[-1]]])

  r = resample_grid(r2d, [gx, gz_rz], [xn, zf_edges])
  z = resample_grid(z2d, [gx, gz_rz], [xn, zf_edges])
  return RzProjection(num_dims=3,
                      r=r,
                      z=z,
                      zc=zc,
                      zf=zf,
                      box=box,
                      wind=wind,
                      phi0_zf=phi0_zf,
                      computational_grid=tuple(
                          np.array(axis, copy=True) for axis in edges))


def _validate_projection(data: "GDataState", projection: RzProjection) -> None:
  if projection.num_dims not in (2, 3):
    raise ValueError(
        f"R-Z projection has invalid dimensionality {projection.num_dims}; expected 2 or 3."
    )
  if data.num_dims != projection.num_dims:
    raise ValueError(
        "Incompatible R-Z projection: projection dimensionality does not match the data."
    )
  edges, _ = _interpolation_grid(data)
  validate_mapping_grid(data, projection.computational_grid)
  if projection.r.shape != projection.z.shape or projection.r.ndim != 2:
    raise ValueError(
        "Incompatible R-Z projection: R and Z grids must be matching 2-D arrays."
    )

  if projection.num_dims == 2:
    expected = (edges[0].size, edges[1].size)
    if projection.r.shape != expected:
      raise ValueError(
          f"Incompatible R-Z projection: expected grid shape {expected}, "
          f"got {projection.r.shape}.")
    return

  required = (projection.zc, projection.zf, projection.box, projection.wind,
              projection.phi0_zf)
  if any(value is None for value in required):
    raise ValueError(
        "Incompatible R-Z projection: 3-D projection metadata is incomplete.")
  if not np.isfinite(projection.box) or np.isclose(projection.box, 0.0):
    raise ValueError(
        "Incompatible R-Z projection: toroidal angular span must be finite and nonzero."
    )
  nx, _, nz = (axis.size - 1 for axis in edges)
  if (projection.zc.shape != (nz, ) or projection.wind.shape != (nx, )
      or projection.phi0_zf.shape != (nx, projection.zf.size)
      or projection.r.shape != (nx + 1, projection.zf.size + 1)):
    raise ValueError(
        "Incompatible R-Z projection: projection and data grid shapes differ.")


def map_to_rz(data: "GDataState",
              *,
              projection: RzProjection,
              phi_tor: float = 0.0,
              comp: int = 0,
              inplace: bool = False,
              tag: str | None = None,
              label: str | None = None) -> "GDataState":
  """Map one component of ``data`` with a reusable ``projection``.

  ``phi_tor`` is in radians and is used only for 3-D field-aligned input.
  The source and projection computational grids must be identical.
  """
  _validate_modal_data(data, "map_to_rz", (2, 3))
  _validate_component(data, comp)
  _validate_projection(data, projection)
  _, _, values = _interpolate_component(data, comp)

  if projection.num_dims == 2:
    out = values[..., np.newaxis]
  else:
    out = fft_poloidal_project(values, projection.zc, projection.box,
                               projection.wind, projection.phi0_zf,
                               projection.zf, float(phi_tor))[..., np.newaxis]

  return data._result([projection.r, projection.z],
                      out,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True)


def resolve_flux_surface_grid(first: "GDataState",
                              geo: Geometry,
                              *,
                              x_idx: int = 0,
                              nphi: int = 128,
                              nz_interp: int = 8) -> FluxSurfaceGrid:
  """Precompute a flux-surface sampling grid for compatible 3-D fields."""
  _validate_modal_data(first, "extract_flux_surface", (3, ))
  nphi = _validate_positive_int(nphi, "nphi")
  nz_interp = _validate_positive_int(nz_interp, "nz_interp")
  _validate_geometry(geo, 3)
  if geo.phi is None:
    raise ValueError(
        "The geometry has no toroidal-angle component; cannot extract a flux surface."
    )
  if isinstance(x_idx, bool) or not isinstance(x_idx, (int, np.integer)):
    raise ValueError("x_idx must be an integer radial index.")

  edges, centers = _interpolation_grid(first)
  xc, yc, zc = centers
  x_idx = int(x_idx)
  if not 0 <= x_idx < xc.size:
    raise ValueError(
        f"x_idx {x_idx} is out of bounds for data with Nx={xc.size}.")
  if zc.size < 2 or yc.size < 2:
    raise ValueError(
        "extract_flux_surface requires at least two interpolated y and z points."
    )

  zf_edges = np.linspace(edges[2][0], edges[2][-1], nz_interp * zc.size + 1)
  zf = 0.5 * (zf_edges[:-1] + zf_edges[1:])
  phi = np.unwrap(np.unwrap(np.unwrap(geo.phi, axis=2), axis=1), axis=0)
  phi_grid = resample_grid(phi, geo.coords, [xc, yc, zf])
  phi_2d = phi_grid[x_idx, :, :]
  phi_tor_list = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
  return FluxSurfaceGrid(x_idx=x_idx,
                         zc=zc,
                         zf=zf,
                         phi_tor_list=phi_tor_list,
                         phi_2d=phi_2d,
                         computational_grid=tuple(
                             np.array(axis, copy=True) for axis in edges))


def extract_flux_surface(data: "GDataState",
                         *,
                         fs_grid: FluxSurfaceGrid,
                         comp: int = 0,
                         inplace: bool = False,
                         tag: str | None = None,
                         label: str | None = None) -> "GDataState":
  """Extract component ``comp`` using a reusable ``fs_grid``."""
  _validate_modal_data(data, "extract_flux_surface", (3, ))
  _validate_component(data, comp)
  edges, _ = _interpolation_grid(data)
  validate_mapping_grid(data, fs_grid.computational_grid)
  nx, ny, nz = (axis.size - 1 for axis in edges)
  if not 0 <= fs_grid.x_idx < nx:
    raise ValueError(
        f"x_idx {fs_grid.x_idx} is out of bounds for data with Nx={nx}.")
  if (fs_grid.zc.shape != (nz, )
      or fs_grid.phi_2d.shape != (ny, fs_grid.zf.size)
      or fs_grid.phi_tor_list.ndim != 1):
    raise ValueError(
        "Incompatible flux-surface grid: projection and data grid shapes differ."
    )

  _, _, values = _interpolate_component(data, comp)
  vals_zf = PchipInterpolator(fs_grid.zc, values, axis=-1,
                              extrapolate=True)(fs_grid.zf)
  vals_2d = vals_zf[fs_grid.x_idx, :, :]

  flux_surf_data = np.empty((fs_grid.phi_tor_list.size, fs_grid.zf.size))
  for iz in range(fs_grid.zf.size):
    phi_y = fs_grid.phi_2d[:, iz]
    val_y = vals_2d[:, iz]
    box = np.mean(np.diff(phi_y)) * ny
    if not np.isfinite(box) or np.isclose(box, 0.0):
      raise ValueError(
          "Toroidal geometry has a zero or non-finite binormal angular span.")
    phi_ext = np.concatenate([phi_y - box, phi_y, phi_y + box])
    val_ext = np.concatenate([val_y, val_y, val_y])
    order = np.argsort(phi_ext)
    folded = phi_y[0] + np.mod(fs_grid.phi_tor_list - phi_y[0], box)
    flux_surf_data[:, iz] = np.interp(folded, phi_ext[order], val_ext[order])

  return data._result([fs_grid.phi_tor_list, fs_grid.zf],
                      flux_surf_data[..., np.newaxis],
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True)
