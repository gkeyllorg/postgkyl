"""Pure NumPy geometry for curvilinear (non-separable, ``.map()``-produced)
grid blocks.

A curvilinear block (see ``postgkyl.operations.map``, ``space="conf"`` with
``m > 1``) stores ``m`` physical-coordinate arrays, each shaped like the
block's own joint ``m``-D nodal (edge) grid -- unlike a separable axis, no
single 1-D coordinate array exists per dimension, so a plain ``np.gradient``
or coordinate-difference has no meaning. These helpers compute the local
geometry needed to differentiate or integrate data on such a block via the
chain rule / change of variables instead: the index-space Jacobian
(``jacobian``), its determinant as the physical cell volume (``cell_volume``,
used by ``integrate``), and the resulting physical-space gradient
(``physical_gradient``, used by ``differentiate``).

The Jacobian is evaluated with unit index spacing (one cell = one index
step) rather than the block's original computational grid spacing, which
``.map()`` does not retain. This is exact for ``cell_volume``/
``physical_gradient``'s purposes: both only ever use the Jacobian in a ratio
(inverted against a same-parametrization gradient, or as a *relative* cell
weight normalized by the sum of the block's cells), so the arbitrary choice
of index units cancels out.
"""

from __future__ import annotations

import numpy as np


def cell_center(nodal: np.ndarray) -> np.ndarray:
  """Average an ``ndim``-D nodal (edge) array over its ``2**ndim`` corners.

  The curvilinear analogue of the ``0.5 * (coord[1:] + coord[:-1])``
  cell-center convention used elsewhere for a single (separable) axis:
  reduces every axis' length by one.
  """
  out = nodal
  for ax in range(out.ndim):
    lo = tuple(
        slice(0, -1) if k == ax else slice(None) for k in range(out.ndim))
    hi = tuple(
        slice(1, None) if k == ax else slice(None) for k in range(out.ndim))
    out = 0.5 * (out[lo] + out[hi])
  return out


def jacobian(block_coords: list) -> np.ndarray:
  """The index-space Jacobian of an ``m``-D curvilinear block.

  Args:
    block_coords: the block's ``m`` physical-coordinate arrays (one per
      mapped dimension, in the block's own local-axis order), each of the
      block's own nodal (edge) shape.

  Returns:
    ``J`` of shape ``cells_shape + (m, m)``, where ``cells_shape`` is
    ``block_coords[0]``'s shape reduced by one per axis (cell-centered) and
    ``J[..., i, j] = d(block_coords[i]) / d(local cell index j)``, evaluated
    by central differences at unit index spacing.
  """
  m = len(block_coords)
  centers = [cell_center(c) for c in block_coords]
  shape = centers[0].shape
  J = np.empty(shape + (m, m))
  for i in range(m):
    for j in range(m):
      J[..., i, j] = np.gradient(centers[i], axis=j, edge_order=2)
  return J


def cell_volume(block_coords: list) -> np.ndarray:
  """Per-cell physical volume (area in 2-D) of an ``m``-D curvilinear block.

  The change-of-variables volume element (the Jacobian determinant): exact
  for a bilinear/trilinear cell, second-order accurate otherwise -- matching
  this codebase's numerical (not exact) differentiate/integrate philosophy.
  Unit index spacing already gives the *physical* cell volume directly
  (not merely a relative one up to some missing scale): ``block_coords``
  holds true physical coordinates against a unit-index abscissa, so a
  central difference of one index step is ``dxi`` times the continuous
  ``d(physical)/d(index)`` derivative, and that same ``dxi`` factor appears
  once per row of the Jacobian -- i.e. ``m`` times in its determinant,
  exactly cancelling the ``m``-fold ``1/dxi`` of the physical volume
  element ``dxi_0 * dxi_1 * ... `` No separate pre-map cell-width metadata
  is needed (or, unlike ``physical_gradient``, would even help: here it
  would double-count the very same factor).

  Shape: the block's ``cells_shape`` (one entry per mapped dimension).
  """
  return np.abs(np.linalg.det(jacobian(block_coords)))


def physical_gradient(block_coords: list, values: np.ndarray,
                      block_axes: tuple) -> np.ndarray:
  """The physical-space gradient of ``values`` along a curvilinear block's
  directions, via the chain rule ``grad_x f = (J^-1)^T grad_xi f``.

  Args:
    block_coords: the block's ``m`` physical-coordinate arrays, each of the
      block's own nodal (edge) shape, in ``block_axes`` order.
    values: the dataset's cell-centered values (any number of axes; the
      block's cells occupy the absolute axes named in ``block_axes``).
    block_axes: the absolute axis of ``values`` differentiated by each of
      ``block_coords``'s local dimensions, in order.

  Returns:
    ``values.shape + (m,)``: the physical derivative along each of the
    block's ``m`` directions (``block_axes`` order) on a new trailing axis;
    every other axis keeps ``values``'s own shape and position.
  """
  m = len(block_coords)
  jinv = np.linalg.inv(jacobian(block_coords))  # cells_shape + (m, m)

  moved = np.moveaxis(values, block_axes, range(m))
  dfdxi = np.stack([np.gradient(moved, axis=j, edge_order=2) for j in range(m)],
                   axis=-1)

  # jinv only varies over the block's own m axes; insert size-1 axes for
  # every other axis of `moved` (now trailing, after the moveaxis above) so
  # it broadcasts against dfdxi positionally.
  n_between = moved.ndim - m
  jinv = jinv.reshape(jinv.shape[:m] + (1, ) * n_between + (m, m))
  out_moved = np.einsum("...ji,...j->...i", jinv, dfdxi)
  return np.moveaxis(out_moved, range(m), block_axes)
