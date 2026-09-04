"""Discontinuous-Galerkin interpolation -- modal coefficients -> mesh values.

**This is the one-way bridge between the two domains**: DG coefficients in
(read through the container's NumPy view of the native array), plain NumPy
values out. The interpolation matrix is built from Gkeyll's own basis
functions (:mod:`postgkyl.gpython.basis` calls the ``eval`` pointer carried by
``struct gkyl_basis``), then applied per cell with a NumPy ``tensordot`` --
so the result is always a *new, by-value* NumPy array, never a view of C
memory. The vendored sympy matrix tables this replaced lived in
``matrices.py`` (see ``src_bak`` history).

:func:`local_poly` is the same bridge with a different evaluation-point
convention: points span the whole reference cell ``[-1, 1]`` (endpoints
included) instead of interior subcell centers, and a NaN is spliced in at
every cell interface -- so a plot shows the true DG inter-cell discontinuity
instead of the spuriously smooth curve :func:`interpolate` produces. The
hand-derived per-order polynomial tables the old implementation used
(``modalDG/kernels/expand_*d.py``, serendipity only) are superseded by
:func:`postgkyl.gpython.basis.eval_matrix`, which evaluates *any* basis at
arbitrary points through Gkeyll's own compiled basis-eval.
"""

from __future__ import annotations

import numpy as np

from postgkyl.gpython import basis as gpython_basis


def num_basis(dim: int, poly_order: int, basis_type: str) -> int:
  """Number of DG basis functions, straight from Gkeyll's basis object."""
  return gpython_basis.num_basis(basis_type, dim, poly_order)


def _make_mesh(num_interp: int, edges: np.ndarray) -> np.ndarray:
  """Refine a 1-D nodal mesh by ``num_interp`` points per cell (uniform)."""
  nx = edges.shape[0] - 1
  return np.linspace(edges[0], edges[-1], num_interp * nx + 1)


def _interpolate_on_mesh(c_mat: np.ndarray, q_in: np.ndarray,
                         num_interp: int) -> np.ndarray:
  """Apply the interpolation matrix on every cell (per-point scatter)."""
  num_cells = np.array(q_in.shape)[:-1]  # drop the coefficient axis
  num_dims = int(len(num_cells))
  ni = np.array([num_interp] * num_dims)
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
                modal: bool = True,
                num_interp: int | None = None):
  """Interpolate DG coefficients onto a refined uniform mesh.

  Args:
    values: ``(cells..., total_comps)`` array of DG coefficients.
    grid: list of 1-D nodal edge arrays (one per dimension).
    poly_order: polynomial order of the basis.
    basis_type: long basis name (``"serendipity"``, ``"tensor"``,
      ``"hybrid"``, or ``"gkhybrid"``).
    modal: False for nodal-basis data (field-blocked node values per cell);
      converted through the exact ``nodal_to_modal`` matrix first.
    num_interp: interpolation points per cell; defaults to ``poly_order + 1``.

  Returns:
    ``(grid_out, values_out)`` -- the refined edge grid and a **new**
    ``(refined_cells..., num_fields)`` NumPy value array.
  """
  num_dims = len(grid)
  if num_dims == 1 and basis_type == "hybrid":
    basis_type = "serendipity"  # PKPM hybrid degenerates to serendipity in 1D
  if num_interp is None:
    num_interp = poly_order + 1

  nodes = num_basis(num_dims, poly_order, basis_type)
  num_fields = values.shape[-1] // nodes
  c_mat = gpython_basis.interpolation_matrix(basis_type, num_dims, poly_order,
                                             num_interp)

  n2m = (None if modal else gpython_basis.nodal_to_modal_matrix(
      basis_type, num_dims, poly_order))
  out = None
  for c in range(num_fields):
    q = values[..., c * nodes:(c + 1) * nodes]
    if n2m is not None:
      q = np.einsum("jk,...k->...j", n2m, q)
    interpolated_c = _interpolate_on_mesh(c_mat, q, num_interp)[..., np.newaxis]
    out = interpolated_c if out is None else np.append(
        out, interpolated_c, axis=-1)

  grid_out = [_make_mesh(num_interp, grid[d]) for d in range(num_dims)]
  return grid_out, out


def _cell_edges_to_nodes(edges: np.ndarray, nodes_1d: np.ndarray) -> np.ndarray:
  """Physical coordinates of ``nodes_1d`` (in ``[-1, 1]``) within every cell
  of a 1-D ``edges`` array, flattened cell-major. Works for a non-uniform
  grid: each cell is scaled/shifted from its own actual width, not assumed
  uniform across the domain (unlike :func:`_make_mesh`)."""
  cell_center = 0.5 * (edges[:-1] + edges[1:])
  dx = edges[1:] - edges[:-1]
  return (cell_center[:, np.newaxis] +
          nodes_1d[np.newaxis, :] * dx[:, np.newaxis] / 2).reshape(-1)


def local_poly(values: np.ndarray,
               grid: list,
               *,
               poly_order: int,
               basis_type: str,
               modal: bool = True,
               npoints: int = 2):
  """Evaluate the DG polynomial cell-by-cell onto a discontinuity-preserving
  plotting mesh.

  Unlike :func:`interpolate`, points span the whole reference cell
  ``[-1, 1]`` (``npoints`` of them, endpoints included) and a NaN is
  inserted at every cell interface, so the true inter-cell jump of the DG
  solution is visible when plotted instead of hidden by a spuriously smooth
  curve.

  Args:
    values: ``(cells..., total_comps)`` array of DG coefficients.
    grid: list of 1-D nodal edge arrays (one per dimension).
    poly_order: polynomial order of the basis.
    basis_type: long basis name (``"serendipity"``, ``"tensor"``,
      ``"hybrid"``, or ``"gkhybrid"``).
    modal: False for nodal-basis data; converted through the exact
      ``nodal_to_modal`` matrix first.
    npoints: evaluation points per cell, from one face to the other.

  Returns:
    ``(grid_out, values_out)`` -- a NaN-separated edge-grid list and value
    array, one entry longer per cell interface than the plain ``npoints``
    x ``num_cells`` mesh.
  """
  num_dims = len(grid)
  if num_dims == 1 and basis_type == "hybrid":
    basis_type = "serendipity"  # PKPM hybrid degenerates to serendipity in 1D

  nodes_1d = np.linspace(-1.0, 1.0, npoints)
  num_nodes = len(nodes_1d)

  nb = num_basis(num_dims, poly_order, basis_type)
  num_fields = values.shape[-1] // nb
  c_mat = gpython_basis.eval_matrix(
      basis_type, num_dims, poly_order,
      gpython_basis.tensor_points(nodes_1d, num_dims))

  n2m = (None if modal else gpython_basis.nodal_to_modal_matrix(
      basis_type, num_dims, poly_order))
  out = None
  for c in range(num_fields):
    q = values[..., c * nb:(c + 1) * nb]
    if n2m is not None:
      q = np.einsum("jk,...k->...j", n2m, q)
    field_c = _interpolate_on_mesh(c_mat, q, num_nodes)[..., np.newaxis]
    out = field_c if out is None else np.append(out, field_c, axis=-1)

  num_cells = np.array(values.shape[:-1])
  grid_out = [_cell_edges_to_nodes(grid[d], nodes_1d) for d in range(num_dims)]
  for d in range(num_dims):
    sep = np.arange(num_nodes, num_nodes * num_cells[d], num_nodes)
    out = np.insert(out, sep, np.nan, axis=d)
    grid_out[d] = np.insert(grid_out[d], sep, grid_out[d][sep - 1])

  return grid_out, out
