"""Evaluate cell-local DG polynomials at physical mesh locations.

Gkeyll basis matrices act on each complete field block. Nodal inputs first
use their exact inverse basis transform. Both paths return independent NumPy
values, preserving each original cell's physical edges.

``interpolate`` samples interior subcell centers. ``local_poly`` includes cell
endpoints and inserts NaN separators so plots preserve discontinuous traces.
"""

from __future__ import annotations

import numpy as np

from postgkyl.gpython import basis as gpython_basis


def num_basis(dim: int, poly_order: int, basis_type: str,
              **basis_kwargs) -> int:
  """Number of DG basis functions, straight from Gkeyll's basis object."""
  return gpython_basis.num_basis(basis_type, dim, poly_order, **basis_kwargs)


def _validate_coefficients(values, grid, block_size, surface_dir=None):
  if values.shape[-1] == 0 or values.shape[-1] % block_size:
    raise ValueError(f"component count {values.shape[-1]} must contain "
                     f"complete field blocks of {block_size}")
  if len(grid) != values.ndim - 1 or any(
      np.ndim(g) != 1 or len(g) != values.shape[d] + (d != surface_dir)
      for d, g in enumerate(grid)):
    raise ValueError(
        "DG interpolation requires cell edges in expansion directions "
        "and face coordinates in the surface direction")


def _expansion_basis(num_dims, poly_order, surface_dir, split):
  """Resolve the tangential basis of a surface field or the full cell basis."""
  if surface_dir is None:
    return num_dims, poly_order, split
  if not 0 <= surface_dir < num_dims:
    raise ValueError("surface direction is out of range")
  return num_dims - 1, poly_order, {"cdim": None, "vdim": None}


def _point_surface(values, grid):
  """A 1-D surface is a 0-D point: Gkeyll stores its value directly."""
  _validate_coefficients(values, grid, 1, surface_dir=0)
  return [np.array(grid[0], copy=True)], np.array(values, copy=True)


def _make_mesh(num_interp: int, edges: np.ndarray) -> np.ndarray:
  """Subdivide each physical cell into ``num_interp`` uniform subcells."""
  fractions = np.arange(num_interp) / num_interp
  subedges = edges[:-1, None] + np.diff(edges)[:, None] * fractions
  return np.r_[subedges.ravel(), edges[-1]]


def _interpolate_on_mesh(c_mat: np.ndarray, q_in: np.ndarray,
                         refinement: tuple[int, ...]) -> np.ndarray:
  """Apply the interpolation matrix on every cell (per-point scatter)."""
  num_cells = np.array(q_in.shape)[:-1]  # drop the coefficient axis
  num_dims = int(len(num_cells))
  ni = np.asarray(refinement)
  q_out = np.zeros(num_cells * ni, np.float64)
  q_in = np.moveaxis(q_in, -1, 0)  # coefficient index first
  for n in range(int(np.prod(ni))):
    temp = np.tensordot(c_mat[n, :], q_in, axes=1)
    start_idx = np.unravel_index(n, ni, order="F")
    idxs = [
        slice(int(start_idx[i]), int(num_cells[i] * ni[i]), int(ni[i]))
        for i in range(num_dims)
    ]
    q_out[tuple(idxs)] = temp
  return q_out


def interpolate(values: np.ndarray,
                grid: list,
                *,
                poly_order: int,
                basis_type: str,
                nodal: bool = False,
                num_interp: int | None = None,
                cdim: int | None = None,
                vdim: int | None = None,
                surface_dir: int | None = None):
  """Interpolate DG coefficients onto a refined cell-edge mesh.

  Args:
    values: ``(cells..., total_comps)`` array of DG coefficients.
    grid: 1-D cell edges, except surface face coordinates on ``surface_dir``.
    poly_order: polynomial order of the basis.
    basis_type: long basis name (``"serendipity"``, ``"tensor"``,
      ``"hybrid"``, or ``"gkhybrid"``).
    nodal: Whether the data are field-blocked node values per cell;
      converted through the exact ``nodal_to_modal`` matrix first.
    num_interp: interpolation points per cell; defaults to ``poly_order + 1``.
    surface_dir: Perpendicular axis of a surface field, if present. This axis
      is not refined or included in the expansion basis.

  Returns:
    ``(grid_out, values_out)`` -- the refined edge grid and a **new**
    ``(refined_cells..., num_fields)`` NumPy value array.
  """
  num_dims = len(grid)
  split = {"cdim": cdim, "vdim": vdim}
  if num_dims == 1 and basis_type == "hybrid":
    basis_type = "serendipity"  # PKPM hybrid degenerates to serendipity in 1D
  if num_interp is None:
    num_interp = poly_order + 1

  if not isinstance(num_interp, (int, np.integer)) or num_interp < 1:
    raise ValueError("num_interp must be a positive integer")
  if num_dims == 1 and surface_dir == 0:
    return _point_surface(values, grid)
  expansion_dim, expansion_order, split = _expansion_basis(
      num_dims, poly_order, surface_dir, split)
  nodes = num_basis(expansion_dim, expansion_order, basis_type, **split)
  _validate_coefficients(values, grid, nodes, surface_dir)
  num_fields = values.shape[-1] // nodes
  c_mat = gpython_basis.interpolation_matrix(basis_type, expansion_dim,
                                             expansion_order, num_interp,
                                             **split)

  n2m = (gpython_basis.nodal_to_modal_matrix(
      basis_type, expansion_dim, expansion_order, **split) if nodal else None)
  refinement = tuple(1 if d == surface_dir else num_interp
                     for d in range(num_dims))
  out = None
  for c in range(num_fields):
    q = values[..., c * nodes:(c + 1) * nodes]
    if n2m is not None:
      q = np.einsum("jk,...k->...j", n2m, q)
    interpolated_c = _interpolate_on_mesh(c_mat, q, refinement)[..., np.newaxis]
    out = interpolated_c if out is None else np.append(
        out, interpolated_c, axis=-1)

  grid_out = [
      np.array(grid[d], copy=True) if d == surface_dir else _make_mesh(
          num_interp, grid[d]) for d in range(num_dims)
  ]
  return grid_out, out


def _cell_edges_to_nodes(edges: np.ndarray, nodes_1d: np.ndarray) -> np.ndarray:
  """Physical coordinates of ``nodes_1d`` (in ``[-1, 1]``) within every cell
  of a 1-D ``edges`` array, flattened cell-major. Each cell is scaled and
  shifted using its own width, including on nonuniform grids."""
  cell_center = 0.5 * (edges[:-1] + edges[1:])
  dx = edges[1:] - edges[:-1]
  return (cell_center[:, np.newaxis] +
          nodes_1d[np.newaxis, :] * dx[:, np.newaxis] / 2).reshape(-1)


def local_poly(values: np.ndarray,
               grid: list,
               *,
               poly_order: int,
               basis_type: str,
               nodal: bool = False,
               npoints: int = 2,
               cdim: int | None = None,
               vdim: int | None = None,
               surface_dir: int | None = None):
  """Evaluate the DG polynomial cell-by-cell onto a discontinuity-preserving
  plotting mesh.

  Unlike :func:`interpolate`, points span the whole reference cell
  ``[-1, 1]`` (``npoints`` of them, endpoints included) and a NaN is
  inserted at every cell interface, so the true inter-cell jump of the DG
  solution is visible when plotted instead of hidden by a spuriously smooth
  curve.

  Args:
    values: ``(cells..., total_comps)`` array of DG coefficients.
    grid: 1-D cell edges, except surface face coordinates on ``surface_dir``.
    poly_order: polynomial order of the basis.
    basis_type: long basis name (``"serendipity"``, ``"tensor"``,
      ``"hybrid"``, or ``"gkhybrid"``).
    nodal: Whether the data use a nodal basis; converted through the exact
      ``nodal_to_modal`` matrix first.
    npoints: evaluation points per cell, from one face to the other.
    surface_dir: Perpendicular axis of a surface field, if present. Its
      coordinates are retained and no polynomial is expanded along it.

  Returns:
    ``(grid_out, values_out)`` -- a NaN-separated edge-grid list and value
    array, one entry longer per cell interface than the plain ``npoints``
    x ``num_cells`` mesh.
  """
  num_dims = len(grid)
  split = {"cdim": cdim, "vdim": vdim}
  if num_dims == 1 and basis_type == "hybrid":
    basis_type = "serendipity"  # PKPM hybrid degenerates to serendipity in 1D

  if not isinstance(npoints, (int, np.integer)) or npoints < 2:
    raise ValueError("npoints must be an integer >= 2")
  if num_dims == 1 and surface_dir == 0:
    return _point_surface(values, grid)
  nodes_1d = np.linspace(-1.0, 1.0, npoints)
  num_nodes = len(nodes_1d)

  expansion_dim, expansion_order, split = _expansion_basis(
      num_dims, poly_order, surface_dir, split)
  nb = num_basis(expansion_dim, expansion_order, basis_type, **split)
  _validate_coefficients(values, grid, nb, surface_dir)
  num_fields = values.shape[-1] // nb
  c_mat = gpython_basis.eval_matrix(
      basis_type, expansion_dim, expansion_order,
      gpython_basis.tensor_points(nodes_1d, expansion_dim), **split)

  n2m = (gpython_basis.nodal_to_modal_matrix(
      basis_type, expansion_dim, expansion_order, **split) if nodal else None)
  refinement = tuple(1 if d == surface_dir else num_nodes
                     for d in range(num_dims))
  out = None
  for c in range(num_fields):
    q = values[..., c * nb:(c + 1) * nb]
    if n2m is not None:
      q = np.einsum("jk,...k->...j", n2m, q)
    field_c = _interpolate_on_mesh(c_mat, q, refinement)[..., np.newaxis]
    out = field_c if out is None else np.append(out, field_c, axis=-1)

  num_cells = np.array(values.shape[:-1])
  grid_out = [
      np.array(grid[d], copy=True)
      if d == surface_dir else _cell_edges_to_nodes(grid[d], nodes_1d)
      for d in range(num_dims)
  ]
  for d in range(num_dims):
    if d == surface_dir:
      continue
    sep = np.arange(num_nodes, num_nodes * num_cells[d], num_nodes)
    out = np.insert(out, sep, np.nan, axis=d)
    grid_out[d] = np.insert(grid_out[d], sep, grid_out[d][sep - 1])

  return grid_out, out
