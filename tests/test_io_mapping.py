"""Tests for ``postgkyl.io.mapping``.

Run:  PYTHONPATH=src pytest tests/test_io_mapping.py -v
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

from postgkyl.io import mapping  # noqa: E402


def test_uniform_grid_has_cells_plus_one_edges():
  grid = mapping.uniform_grid(np.array([0.0, -1.0]), np.array([2.0, 1.0]),
                              np.array([4, 2]))
  assert len(grid) == 2
  np.testing.assert_allclose(grid[0], [0.0, 0.5, 1.0, 1.5, 2.0])
  np.testing.assert_allclose(grid[1], [-1.0, 0.0, 1.0])


def test_adjust_for_ghost_cells_no_op_when_shapes_match():
  lower = np.array([0.0])
  upper = np.array([1.0])
  cells = np.array([4])
  lo, up, c = mapping.adjust_for_ghost_cells(lower, upper, cells, (4, ))
  assert c[0] == 4
  assert lo[0] == pytest.approx(0.0)
  assert up[0] == pytest.approx(1.0)


def test_c2p_grid_splits_packed_node_axis_by_hand():
  """A hand-computed 1-D case: 3 nodes, 2 dims -> each dim gets 1 coeff."""
  # nodes[..., 0] is the x-coefficient, nodes[..., 1] the y-coefficient.
  nodes = np.array([
      [0.0, 10.0],
      [1.0, 11.0],
      [2.0, 12.0],
  ])
  blocks = mapping.c2p_grid(nodes, num_dims=2)
  assert len(blocks) == 2
  np.testing.assert_array_equal(blocks[0], np.array([[0.0], [1.0], [2.0]]))
  np.testing.assert_array_equal(blocks[1], np.array([[10.0], [11.0], [12.0]]))


def test_c2p_grid_with_multiple_coefficients_per_dim():
  """3 dims, 2 modal coefficients per dim -> 6 packed components."""
  nodes = np.arange(2 * 6, dtype=float).reshape(2, 6)
  blocks = mapping.c2p_grid(nodes, num_dims=3)
  assert len(blocks) == 3
  for d in range(3):
    np.testing.assert_array_equal(blocks[d], nodes[:, d * 2:(d + 1) * 2])
