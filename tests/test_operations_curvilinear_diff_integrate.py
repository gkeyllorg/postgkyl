"""``differentiate``/``integrate`` on curvilinear (``.map(space="conf")``)
grids -- both verbs need the block's Jacobian (``numerics.curvilinear``)
instead of treating each axis as separable. Uses exactly linear coordinate
maps (rotation, shear) so the finite-difference machinery reproduces the
analytic answer up to floating-point precision, independent of resolution.
"""

from __future__ import annotations

import numpy as np
import pytest

from postgkyl import gpython, operations
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.numerics import curvilinear

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")
pytestmark = needs_gkeyll


# --------------------------------------------------------------- test helpers
def _project_2d(fn, lower, upper, cells, basis_type, poly_order):
  """Exact per-cell modal coefficients of ``fn(z0, z1)`` (mirrors
  ``tests/test_operations_map.py``'s helper of the same name)."""
  node_eta = gpython.basis.node_coords(basis_type, 2, poly_order)
  n2m = gpython.basis.nodal_to_modal_matrix(basis_type, 2, poly_order)
  dz = [(upper[d] - lower[d]) / cells[d] for d in range(2)]
  c0 = lower[0] + (np.arange(cells[0]) + 0.5) * dz[0]
  c1 = lower[1] + (np.arange(cells[1]) + 0.5) * dz[1]
  centers = np.stack(np.meshgrid(c0, c1, indexing="ij"), axis=-1)
  node_phys = (
      centers[:, :, None, :] +
      0.5 * np.array(dz)[None, None, None, :] * node_eta[None, None, :, :])
  nodal_vals = fn(node_phys[..., 0], node_phys[..., 1])
  return np.einsum("ij,...j->...i", n2m, nodal_vals)


def _synthetic_map(coeffs,
                   lower,
                   upper,
                   cells,
                   *,
                   basis_type="serendipity",
                   poly_order=1):
  d = GDataState()
  d.ctx.update(basis_type=basis_type,
               poly_order=poly_order,
               value_form="modal",
               cells=np.asarray(cells, dtype=np.int64))
  grid = [
      np.linspace(lower[i], upper[i],
                  int(cells[i]) + 1) for i in range(len(cells))
  ]
  d.push(grid, gpython.GkylArray.from_numpy(coeffs))
  return d


def _linear_map_dataset(a00,
                        a01,
                        a10,
                        a11,
                        *,
                        lower=(0.0, 0.0),
                        upper=(4.0, 4.0),
                        cells=(8, 8),
                        poly_order=1):
  """A target dataset whose grid is deformed by the linear map
  ``x = a00*z0 + a01*z1``, ``y = a10*z0 + a11*z1`` -- exact for any
  ``poly_order >= 1`` (an affine function)."""
  m0 = _project_2d(lambda z0, z1: a00 * z0 + a01 * z1, lower, upper, cells,
                   "serendipity", poly_order)
  m1 = _project_2d(lambda z0, z1: a10 * z0 + a11 * z1, lower, upper, cells,
                   "serendipity", poly_order)
  modal = np.concatenate([m0, m1], axis=-1)
  mapping = _synthetic_map(modal, lower, upper, cells, poly_order=poly_order)
  target = GDataState()
  target.push([np.linspace(lower[i], upper[i], cells[i] + 1) for i in range(2)],
              np.zeros(tuple(cells) + (1, )))
  mapped = operations.map(target, mapping, space="conf")

  x = curvilinear.cell_center(mapped.grid[0])
  y = curvilinear.cell_center(mapped.grid[1])
  return mapped, x, y


# --------------------------------------------------------------- differentiate
class TestDifferentiateCurvilinear:

  def test_rotation_gradient_matches_analytic(self):
    """A rigid rotation: physical gradient of f = x**2 + y is exactly
    (2x, 1) regardless of the rotation angle."""
    theta = 0.3
    c, s = np.cos(theta), np.sin(theta)
    mapped, x, y = _linear_map_dataset(c, -s, s, c)
    field = mapped._result(mapped.grid, (x**2 + y)[..., np.newaxis])

    dfdx = operations.differentiate(field, direction=0)
    dfdy = operations.differentiate(field, direction=1)
    np.testing.assert_allclose(dfdx.values[..., 0], 2 * x, atol=1e-8)
    np.testing.assert_allclose(dfdy.values[..., 0], np.ones_like(y), atol=1e-8)

  def test_shear_gradient_of_linear_field_is_exact(self):
    """A non-orthogonal (shear) map: for a field that is itself linear in
    the physical coordinates, the chain-rule reconstruction is exact."""
    mapped, x, y = _linear_map_dataset(1.0, 0.5, 0.0, 1.0)
    a, b = 3.0, -2.0
    field = mapped._result(mapped.grid, (a * x + b * y)[..., np.newaxis])

    grad = operations.differentiate(field)
    np.testing.assert_allclose(grad.values[..., 0], a, atol=1e-8)
    np.testing.assert_allclose(grad.values[..., 1], b, atol=1e-8)

  def test_full_gradient_direction_none_matches_per_direction_calls(self):
    mapped, x, y = _linear_map_dataset(1.0, 0.3, -0.2, 1.0)
    field = mapped._result(mapped.grid, (x**2 + y**2)[..., np.newaxis])

    full = operations.differentiate(field)
    dx = operations.differentiate(field, direction=0)
    dy = operations.differentiate(field, direction=1)
    np.testing.assert_allclose(full.values[..., 0], dx.values[..., 0])
    np.testing.assert_allclose(full.values[..., 1], dy.values[..., 0])


# ------------------------------------------------------------------ integrate
class TestIntegrateCurvilinear:

  def test_rotation_preserves_area(self):
    """Integrating the constant field 1 over a rotated square recovers the
    original (unrotated) domain area -- rotation preserves area."""
    theta = 0.4
    c, s = np.cos(theta), np.sin(theta)
    lower, upper, cells = (0.0, 0.0), (4.0, 2.0), (16, 16)
    mapped, x, _y = _linear_map_dataset(c,
                                        -s,
                                        s,
                                        c,
                                        lower=lower,
                                        upper=upper,
                                        cells=cells)
    field = mapped._result(mapped.grid, np.ones(tuple(cells) + (1, )))

    total = operations.integrate(field, "0,1")
    expected_area = (upper[0] - lower[0]) * (upper[1] - lower[1])
    np.testing.assert_allclose(total, expected_area, rtol=1e-6)

  def test_shear_area_matches_determinant(self):
    """A shear map's area scales by the (constant) Jacobian determinant of
    the linear transform, |a00*a11 - a01*a10|."""
    a00, a01, a10, a11 = 1.0, 0.5, 0.0, 1.0
    lower, upper, cells = (0.0, 0.0), (2.0, 3.0), (10, 12)
    mapped, x, _y = _linear_map_dataset(a00,
                                        a01,
                                        a10,
                                        a11,
                                        lower=lower,
                                        upper=upper,
                                        cells=cells)
    field = mapped._result(mapped.grid, np.ones(tuple(cells) + (1, )))

    total = operations.integrate(field, "0,1")
    det = abs(a00 * a11 - a01 * a10)
    expected_area = det * (upper[0] - lower[0]) * (upper[1] - lower[1])
    np.testing.assert_allclose(total, expected_area, rtol=1e-6)

  def test_partial_block_reduction_raises(self):
    mapped, x, y = _linear_map_dataset(1.0, 0.3, -0.2, 1.0)
    field = mapped._result(mapped.grid, (x + y)[..., np.newaxis])
    with pytest.raises(ValueError, match="curvilinear"):
      operations.integrate(field, "0")
