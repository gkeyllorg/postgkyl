"""Tests for ``postgkyl.dg.map`` -- grid mapping by evaluation at target points.

See ``MAPPING.md`` for the design. Test fixtures build modal (or nodal)
coefficients for the mapping field synthetically with ``gpython.basis`` matrices
(no mapc2p file is required, per the layer instructions), by exactly
projecting a chosen physical-coordinate function onto the basis's own node
points, per cell -- this guarantees the coefficients exactly represent the
chosen function, so the expected result can be computed independently
(directly from the function), never from the code under test.

Run:  PYTHONPATH=src pytest tests/test_dg_map.py -v
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

from postgkyl import gpython, dg  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")
pytestmark = needs_gkeyll


def _project_1d(fn, lower, upper, cells, basis_type, poly_order):
  """Exact per-cell modal coefficients of ``fn(z)`` for a 1-D basis.

  ``fn`` must be exactly representable in the basis on every cell (e.g. any
  polynomial of degree <= poly_order) -- projecting through the node points
  and back through the exact nodal<->modal change of basis reproduces it
  exactly, independent of the mapping code under test.
  """
  nb = gpython.basis.num_basis(basis_type, 1, poly_order)
  node_eta = gpython.basis.node_coords(basis_type, 1, poly_order)[:, 0]
  n2m = gpython.basis.nodal_to_modal_matrix(basis_type, 1, poly_order)
  dz = (upper - lower) / cells
  centers = lower + (np.arange(cells) + 0.5) * dz
  nodal_z = centers[:, None] + 0.5 * dz * node_eta[None, :]  # (cells, nb)
  nodal_vals = fn(nodal_z)
  return nodal_vals @ n2m.T, nodal_vals  # (modal, nodal) both (cells, nb)


def _project_2d(fn, lower, upper, cells, basis_type, poly_order):
  """Exact per-cell modal coefficients of ``fn(z0, z1)`` for a 2-D basis."""
  nb = gpython.basis.num_basis(basis_type, 2, poly_order)
  node_eta = gpython.basis.node_coords(basis_type, 2, poly_order)  # (nb, 2)
  n2m = gpython.basis.nodal_to_modal_matrix(basis_type, 2, poly_order)
  dz = [(upper[d] - lower[d]) / cells[d] for d in range(2)]
  c0 = lower[0] + (np.arange(cells[0]) + 0.5) * dz[0]
  c1 = lower[1] + (np.arange(cells[1]) + 0.5) * dz[1]
  centers = np.stack(np.meshgrid(c0, c1, indexing="ij"), axis=-1)  # (*cells,2)
  # physical coordinates of every node, every cell: (*cells, nb, 2)
  node_phys = (
      centers[:, :, None, :] +
      0.5 * np.array(dz)[None, None, None, :] * node_eta[None, None, :, :])
  nodal_vals = fn(node_phys[..., 0], node_phys[..., 1])  # (*cells, nb)
  modal = np.einsum("ij,...j->...i", n2m, nodal_vals)
  return modal, nodal_vals


# --------------------------------------------------------------------- 1-D
def test_eval_at_points_identity_map_1d_is_exact_to_machine_precision():
  lower, upper, cells = 0.0, 4.0, 4
  modal, _ = _project_1d(lambda z: z, lower, upper, cells, "serendipity", 1)
  targets = np.linspace(lower, upper, 33)  # finer than the mapping's own grid
  got = dg.map.eval_at_points(modal, [lower], [upper], [cells],
                              targets[:, None],
                              basis_type="serendipity",
                              poly_order=1)
  np.testing.assert_allclose(got, targets, atol=1e-12)


def test_map_grid_identity_1d_matches_target_axis():
  lower, upper, cells = -1.0, 3.0, 5
  modal, _ = _project_1d(lambda z: z, lower, upper, cells, "serendipity", 1)
  target_axes = [np.linspace(lower, upper, 17)]
  map_ctx = dict(lower=np.array([lower]),
                 upper=np.array([upper]),
                 cells=np.array([cells]),
                 basis_type="serendipity",
                 poly_order=1,
                 value_form="modal")
  out = dg.map_grid(modal, map_ctx, target_axes)
  assert len(out) == 1
  assert out[0].shape == target_axes[0].shape  # m == 1 stays 1-D
  np.testing.assert_allclose(out[0], target_axes[0], atol=1e-12)


def test_eval_at_points_in_basis_quadratic_is_exact_at_edges():
  """A single cell spanning the whole domain sidesteps cell-boundary
  continuity questions entirely, isolating the eval_matrix/reshape math."""
  lower, upper, cells = -1.0, 3.0, 1
  fn = lambda z: 0.5 * z**2 - z + 1.0  # degree 2, in-basis for p2
  modal, _ = _project_1d(fn, lower, upper, cells, "serendipity", 2)
  targets = np.array([lower, -0.3, 0.7, 2.1, upper])  # includes both edges
  got = dg.map.eval_at_points(modal, [lower], [upper], [cells],
                              targets[:, None],
                              basis_type="serendipity",
                              poly_order=2)
  np.testing.assert_allclose(got, fn(targets), atol=1e-12)


def test_eval_at_points_rejects_cells_mismatch():
  modal, _ = _project_1d(lambda z: z, 0.0, 1.0, 2, "serendipity", 1)
  with pytest.raises(ValueError, match="does not match cells"):
    dg.map.eval_at_points(
        modal,
        [0.0],
        [1.0],
        [3],  # wrong cell count
        np.array([[0.5]]),
        basis_type="serendipity",
        poly_order=1)


def test_eval_at_points_rejects_points_dim_mismatch():
  modal, _ = _project_1d(lambda z: z, 0.0, 1.0, 2, "serendipity", 1)
  with pytest.raises(ValueError, match="expected 1"):
    dg.map.eval_at_points(
        modal,
        [0.0],
        [1.0],
        [2],
        np.array([[0.5, 0.5]]),  # last axis length 2, expected 1
        basis_type="serendipity",
        poly_order=1)


def test_eval_at_points_nodal_basis_path_matches_modal():
  """A nodal-basis mapping file (``nodal=True``) converts through the exact
  nodal<->modal change of basis, then evaluates identically to the modal path."""
  lower, upper, cells = 0.0, 4.0, 4
  modal, nodal = _project_1d(lambda z: z, lower, upper, cells, "serendipity", 1)
  targets = np.linspace(lower, upper, 11)
  got_modal = dg.map.eval_at_points(modal, [lower], [upper], [cells],
                                    targets[:, None],
                                    basis_type="serendipity",
                                    poly_order=1,
                                    nodal=False)
  got_nodal = dg.map.eval_at_points(nodal, [lower], [upper], [cells],
                                    targets[:, None],
                                    basis_type="serendipity",
                                    poly_order=1,
                                    nodal=True)
  np.testing.assert_allclose(got_nodal, got_modal, atol=1e-12)
  np.testing.assert_allclose(got_nodal, targets, atol=1e-12)


def test_map_grid_nodal_basis_map_file():
  lower, upper, cells = 0.0, 2.0, 2
  _, nodal = _project_1d(lambda z: z, lower, upper, cells, "serendipity", 1)
  target_axes = [np.linspace(lower, upper, 9)]
  map_ctx = dict(lower=np.array([lower]),
                 upper=np.array([upper]),
                 cells=np.array([cells]),
                 basis_type="serendipity",
                 poly_order=1,
                 value_form="nodal")
  out = dg.map_grid(nodal, map_ctx, target_axes)
  np.testing.assert_allclose(out[0], target_axes[0], atol=1e-12)


# --------------------------------------------------------------------- 2-D
def test_map_grid_identity_2d_curvilinear_matches_meshgrid():
  """Every physical coordinate is evaluated over all m dims, so the same
  algorithm handles the non-separable (curvilinear) case."""
  lower, upper, cells = [0.0, 0.0], [2.0, 3.0], [2, 3]
  m0, _ = _project_2d(lambda z0, z1: z0, lower, upper, cells, "serendipity", 1)
  m1, _ = _project_2d(lambda z0, z1: z1, lower, upper, cells, "serendipity", 1)
  map_coeffs = np.concatenate([m0, m1], axis=-1)
  target_axes = [
      np.linspace(lower[0], upper[0], 5),
      np.linspace(lower[1], upper[1], 7)
  ]
  map_ctx = dict(lower=np.array(lower),
                 upper=np.array(upper),
                 cells=np.array(cells),
                 basis_type="serendipity",
                 poly_order=1,
                 value_form="modal")
  out = dg.map_grid(map_coeffs, map_ctx, target_axes)

  expected = np.meshgrid(*target_axes, indexing="ij")
  assert len(out) == 2
  for d in range(2):
    assert out[d].shape == (5, 7)  # shape of the axes it replaces
    np.testing.assert_allclose(out[d], expected[d], atol=1e-12)


def test_map_grid_2d_rotation_is_exact_non_separable():
  """A genuine rotation mixes both computational coordinates into each
  physical one -- exercises the non-separable (curvilinear) evaluation."""
  lower, upper, cells = [-1.0, -1.0], [1.0, 1.0], [2, 2]
  theta = 0.4
  cos_t, sin_t = np.cos(theta), np.sin(theta)
  fn0 = lambda z0, z1: cos_t * z0 - sin_t * z1
  fn1 = lambda z0, z1: sin_t * z0 + cos_t * z1
  m0, _ = _project_2d(fn0, lower, upper, cells, "serendipity", 1)
  m1, _ = _project_2d(fn1, lower, upper, cells, "serendipity", 1)
  map_coeffs = np.concatenate([m0, m1], axis=-1)
  target_axes = [
      np.linspace(lower[0], upper[0], 6),
      np.linspace(lower[1], upper[1], 4)
  ]
  map_ctx = dict(lower=np.array(lower),
                 upper=np.array(upper),
                 cells=np.array(cells),
                 basis_type="serendipity",
                 poly_order=1,
                 value_form="modal")
  out = dg.map_grid(map_coeffs, map_ctx, target_axes)

  z0, z1 = np.meshgrid(*target_axes, indexing="ij")
  np.testing.assert_allclose(out[0], fn0(z0, z1), atol=1e-12)
  np.testing.assert_allclose(out[1], fn1(z0, z1), atol=1e-12)
