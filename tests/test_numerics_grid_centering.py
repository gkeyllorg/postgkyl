"""Tests for postgkyl.numerics.grid_centering -- nodal_to_cell_centered_grid."""

from __future__ import annotations

import numpy as np
import pytest

from postgkyl.numerics.grid_centering import nodal_to_cell_centered_grid


class TestNodalToCellCenteredGrid:

  def test_1d_nodal_grid_is_centered(self):
    grid = [np.linspace(0.0, 1.0, 5)]  # 4 cells, nodal (5 points)
    out = nodal_to_cell_centered_grid(grid, cells=np.array([4]))
    assert len(out) == 1
    assert out[0].shape == (4, )
    np.testing.assert_allclose(out[0], 0.5 * (grid[0][:-1] + grid[0][1:]))

  def test_1d_already_cell_centered_passthrough(self):
    grid = [np.linspace(0.1, 0.9, 4)]  # already 4 cell centers
    out = nodal_to_cell_centered_grid(grid, cells=np.array([4]))
    np.testing.assert_allclose(out[0], grid[0])

  def test_2d_nodal_grids(self):
    grid = [np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 4)]
    out = nodal_to_cell_centered_grid(grid, cells=np.array([4, 3]))
    assert len(out) == 2
    assert out[0].shape == (4, )
    assert out[1].shape == (3, )

  def test_dimension_mismatch_raises(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    with pytest.raises(ValueError, match="doesn't match"):
      nodal_to_cell_centered_grid(grid, cells=np.array([4, 3]))

  def test_bad_axis_length_raises(self):
    grid = [np.linspace(0.0, 1.0, 6)]  # neither 4 nor 5 points
    with pytest.raises(ValueError, match="terribly wrong"):
      nodal_to_cell_centered_grid(grid, cells=np.array([4]))

  def test_meshgrid_true_returns_ij_indexed_grid(self):
    grid = [np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 4)]
    out = nodal_to_cell_centered_grid(grid,
                                      cells=np.array([4, 3]),
                                      meshgrid=True)
    assert len(out) == 2
    assert out[0].shape == (4, 3)
    assert out[1].shape == (4, 3)

  def test_meshgrid_false_keeps_1d_axes(self):
    grid = [np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 4)]
    out = nodal_to_cell_centered_grid(grid,
                                      cells=np.array([4, 3]),
                                      meshgrid=False)
    assert out[0].ndim == 1
    assert out[1].ndim == 1

  def test_meshgrid_ignored_for_1d(self):
    grid = [np.linspace(0.0, 1.0, 5)]
    out = nodal_to_cell_centered_grid(grid, cells=np.array([4]), meshgrid=True)
    assert len(out) == 1
    assert out[0].ndim == 1

  def test_2d_array_grid_nodal(self):
    """Multi-dimensional (already-meshgridded) coordinate arrays: the
    2-D-shaped-grid branch of the function."""
    x_nodal = np.linspace(0.0, 1.0, 5)
    y_nodal = np.linspace(0.0, 2.0, 4)
    X, Y = np.meshgrid(x_nodal, y_nodal, indexing="ij")
    out = nodal_to_cell_centered_grid([X, Y], cells=np.array([4, 3]))
    assert out[0].shape == (4, 3)
    assert out[1].shape == (4, 3)

  def test_2d_array_grid_already_cell_centered_passthrough(self):
    """Multi-dimensional grid array whose axis already matches ``cells``
    (no averaging needed) -- the ``grid[d].shape[d] == cells[d]``
    passthrough branch for array-shaped (already-meshgridded) grids."""
    x_cc = np.linspace(0.1, 0.9, 4)
    y_cc = np.linspace(0.2, 1.8, 3)
    X, Y = np.meshgrid(x_cc, y_cc, indexing="ij")
    out = nodal_to_cell_centered_grid([X, Y], cells=np.array([4, 3]))
    np.testing.assert_allclose(out[0], X)
    np.testing.assert_allclose(out[1], Y)

  def test_multidim_grid_array_with_single_dimension(self):
    """A multi-dimensional (ndim > 1) coordinate array in a 1-D grid (the
    ``num_dims == 1`` branch of the array-shaped-grid case): averaging
    happens along axis 0 only, leaving the other axis untouched."""
    grid = [
        np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    ]
    out = nodal_to_cell_centered_grid(grid, cells=np.array([4]))
    assert len(out) == 1
    assert out[0].shape == (4, 2)
    np.testing.assert_allclose(out[0],
                               [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

  def test_2d_array_grid_bad_shape_raises(self):
    x_nodal = np.linspace(0.0, 1.0, 6)  # neither 4 nor 5 along axis 0
    y_nodal = np.linspace(0.0, 2.0, 4)
    X, Y = np.meshgrid(x_nodal, y_nodal, indexing="ij")
    with pytest.raises(ValueError, match="terribly wrong"):
      nodal_to_cell_centered_grid([X, Y], cells=np.array([4, 3]))
