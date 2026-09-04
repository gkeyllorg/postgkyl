"""Grid mapping -- evaluate a coordinate-map DG field at a target's grid points.

See ``MAPPING.md`` for the full design. **The core semantic**: a mapping file
is a DG field whose components hold the coefficients of the physical
coordinates :math:`x_d(z)` of each mapped dimension ``d``; mapping a grid means
evaluating those coefficients at the *target*'s existing grid points -- there is
no resolution parameter and no alignment arithmetic, so the new grid always has
exactly the shape of the one it replaces.

Two functions, one per step of ``MAPPING.md``'s "evaluation algorithm":
:func:`eval_at_points` evaluates ONE coordinate's coefficients at an arbitrary
point set (steps 1-4: cell locate, reference-coordinate conversion, grouped
basis evaluation, reshape); :func:`map_grid` builds the tensor point set for
the target axes and calls it once per mapped dimension.
"""

from __future__ import annotations

import numpy as np

from postgkyl import gpython


def eval_at_points(coeffs: np.ndarray,
                   lower: np.ndarray,
                   upper: np.ndarray,
                   cells: np.ndarray,
                   points: np.ndarray,
                   *,
                   basis_type: str,
                   poly_order: int,
                   modal: bool = True) -> np.ndarray:
  """Evaluate one coordinate's DG coefficients at arbitrary computational points.

  Args:
    coeffs: ``(*cells, num_basis)`` array -- the mapping's per-cell
      coefficients for a single physical coordinate ``x_d(z)`` over its own
      uniform grid. Modal by default; pass ``modal=False`` for a nodal-basis
      mapping file (converted through the exact ``nodal_to_modal`` matrix
      first, same pattern as :func:`postgkyl.dg.interpolate.interpolate`).
    lower: length-``m`` array, the mapping's own domain lower bounds.
    upper: length-``m`` array, the mapping's own domain upper bounds.
    cells: length-``m`` array, the mapping's own cell counts (must match
      ``coeffs.shape[:-1]``).
    points: ``(*shape, m)`` array of evaluation points in computational
      coordinates, within the mapping's bounds.
    basis_type: long basis name (``"serendipity"``, ``"tensor"``,
      ``"hybrid"``, or ``"gkhybrid"``).
    poly_order: polynomial order of the mapping's basis.
    modal: False for nodal-basis mapping coefficients.

  Returns:
    ``(*shape,)`` array -- ``x_d`` evaluated at every point.

  Raises:
    ValueError: ``cells`` does not match ``coeffs.shape[:-1]``, or the last
      axis of ``points`` does not have length ``m``.
  """
  lower = np.asarray(lower, dtype=np.float64)
  upper = np.asarray(upper, dtype=np.float64)
  cells = np.asarray(cells, dtype=np.int64)
  m = lower.shape[0]

  if coeffs.shape[:-1] != tuple(int(c) for c in cells):
    raise ValueError(
        f"eval_at_points: coeffs cell shape {coeffs.shape[:-1]} does not "
        f"match cells {tuple(cells)}")
  points = np.asarray(points, dtype=np.float64)
  if points.shape[-1] != m:
    raise ValueError(
        f"eval_at_points: points last axis has length {points.shape[-1]}, "
        f"expected {m} (len(lower))")

  if not modal:
    n2m = gpython.basis.nodal_to_modal_matrix(basis_type, m, poly_order)
    coeffs = np.einsum("jk,...k->...j", n2m, coeffs)

  shape = points.shape[:-1]
  z = points.reshape(-1, m)
  dz = (upper - lower) / cells

  # Step 1: locate the containing cell (clip fixes the boundary convention).
  idx = np.clip(np.floor((z - lower) / dz).astype(np.int64), 0, cells - 1)
  # Step 2: reference coordinates in [-1, 1]^m.
  centers = lower + (idx + 0.5) * dz
  eta = 2.0 * (z - centers) / dz

  flat_coeffs = coeffs.reshape(-1, coeffs.shape[-1])
  cell_lin = np.ravel_multi_index(tuple(idx[:, d] for d in range(m)),
                                  tuple(int(c) for c in cells))

  # Step 3: group points by containing cell -> one matrix-vector product each.
  out = np.empty(z.shape[0], dtype=np.float64)
  for lin in np.unique(cell_lin):
    sel = cell_lin == lin
    b = gpython.basis.eval_matrix(basis_type, m, poly_order, eta[sel])
    out[sel] = b @ flat_coeffs[lin]

  # Step 4: reshape to the point-set shape.
  return out.reshape(shape)


def map_grid(map_coeffs: np.ndarray, map_ctx: dict,
             target_axes: list[np.ndarray]) -> list[np.ndarray]:
  """Evaluate every mapped dimension's coordinates at the target's grid points.

  Args:
    map_coeffs: ``(*cells, m * num_basis)`` array -- the mapping field's raw
      coefficients (``GDataState.get_values()``); components
      ``d*num_basis:(d+1)*num_basis`` are the coefficients of ``x_d(z)``.
    map_ctx: the mapping dataset's ``ctx`` dict; reads ``lower``, ``upper``,
      ``cells``, ``basis_type``, ``poly_order``, and ``value_form`` (default
      ``"modal"``).
    target_axes: the target's own edge/node arrays for the ``m`` axes being
      deformed, one 1-D array per mapped dimension.

  Returns:
    A list of ``m`` new grid arrays, one per mapped dimension: 1-D when
    ``m == 1``; an ``m``-dimensional nodal array (the full tensor product of
    ``target_axes``, ``indexing="ij"``) for every dimension when ``m > 1``,
    so non-separable (curvilinear) maps are handled.
  """
  lower = map_ctx["lower"]
  upper = map_ctx["upper"]
  cells = map_ctx["cells"]
  basis_type = map_ctx["basis_type"]
  poly_order = map_ctx["poly_order"]
  modal = map_ctx.get("value_form", "modal") == "modal"
  m = len(target_axes)

  if m == 1:
    points = np.asarray(target_axes[0], dtype=np.float64)[:, np.newaxis]
  else:
    points = np.stack(np.meshgrid(*target_axes, indexing="ij"), axis=-1)

  nb = gpython.basis.num_basis(basis_type, m, poly_order)
  return [
      eval_at_points(map_coeffs[..., d * nb:(d + 1) * nb],
                     lower,
                     upper,
                     cells,
                     points,
                     basis_type=basis_type,
                     poly_order=poly_order,
                     modal=modal) for d in range(m)
  ]


def map_grid_separable(map_coeffs: np.ndarray, map_ctx: dict,
                       target_axes: list[np.ndarray]) -> list[np.ndarray]:
  """Evaluate each mapped dimension's coordinates independently.

  Gkeyll's velocity-space coordinate maps (``mapc2p_vel``) are diagonal:
  dimension ``d`` is a **separate** 1-D map ``v_d(z_d)`` over its own axis
  only, not a joint ``m``-dimensional curvilinear map like :func:`map_grid`
  handles for configuration-space maps (``mapc2p``/``mc2nu``). Component
  block ``d*num_basis:(d+1)*num_basis`` (``num_basis`` for a *1-D* basis of
  the given order) holds dimension ``d``'s own coefficients; Gkeyll's writer
  stores them on the full ``m``-dimensional cell grid but broadcasts them
  along every axis other than ``d``, so cell index 0 on every other axis
  already carries the full set of values.

  Args:
    map_coeffs: ``(*cells, m * num_basis)`` array -- ``GDataState.get_values()``,
      ``num_basis`` being the 1-D basis size for ``basis_type``/``poly_order``.
    map_ctx: the mapping dataset's ``ctx`` dict; reads ``lower``, ``upper``,
      ``cells``, ``basis_type``, ``poly_order``, and ``value_form`` (default
      ``"modal"``).
    target_axes: the target's own 1-D edge arrays for the ``m`` axes being
      deformed, one per mapped dimension.

  Returns:
    A list of ``m`` new 1-D grid arrays, one per mapped dimension.
  """
  lower = map_ctx["lower"]
  upper = map_ctx["upper"]
  cells = map_ctx["cells"]
  basis_type = map_ctx["basis_type"]
  poly_order = map_ctx["poly_order"]
  modal = map_ctx.get("value_form", "modal") == "modal"
  m = len(target_axes)

  nb = gpython.basis.num_basis(basis_type, 1, poly_order)
  new_axes = []
  for d in range(m):
    idx = [0] * m
    idx[d] = slice(None)
    coeffs_d = map_coeffs[tuple(idx)][:, d * nb:(d + 1) * nb]
    points = np.asarray(target_axes[d], dtype=np.float64)[:, np.newaxis]
    new_axes.append(
        eval_at_points(coeffs_d,
                       lower[d:d + 1],
                       upper[d:d + 1],
                       cells[d:d + 1],
                       points,
                       basis_type=basis_type,
                       poly_order=poly_order,
                       modal=modal))
  return new_axes
