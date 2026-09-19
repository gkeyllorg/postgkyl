"""Tests for postgkyl.numerics.mag_sq / rel_change / rotation_matrix."""

from __future__ import annotations

import numpy as np

from postgkyl.numerics.mag_sq import mag_sq
from postgkyl.numerics.rel_change import rel_change
from postgkyl.numerics.rotation_matrix import rotation_matrix

_G1 = [np.array([0.0, 1.0])]

# ---------------------------------------------------------------------------
# mag_sq
# ---------------------------------------------------------------------------


class TestMagSq:

  def test_unit_x_vector(self):
    _, out = mag_sq(_G1, np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(out.flat[0], 1.0)

  def test_3_4_0_vector(self):
    _, out = mag_sq(_G1, np.array([[3.0, 4.0, 0.0]]))
    np.testing.assert_allclose(out.flat[0], 25.0)

  def test_output_has_trailing_dim(self):
    _, out = mag_sq(_G1, np.array([[1.0, 2.0, 3.0]]))
    assert out.ndim == 2
    assert out.shape[-1] == 1

  def test_custom_coords(self):
    _, out = mag_sq(_G1,
                    np.array([[0.0, 0.0, 0.0, 3.0, 4.0, 0.0]]),
                    coords="3:6")
    np.testing.assert_allclose(out.flat[0], 25.0)

  def test_multi_cell(self):
    grid = [np.linspace(0.0, 1.0, 4)]
    values = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    _, out = mag_sq(grid, values)
    np.testing.assert_allclose(out[:, 0], [1.0, 1.0, 2.0])

  def test_grid_returned_unchanged(self):
    grid = [np.linspace(0.0, 1.0, 3)]
    out_grid, _ = mag_sq(grid, np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(out_grid[0], grid[0])


# ---------------------------------------------------------------------------
# rel_change
# ---------------------------------------------------------------------------


class TestRelChange:

  def test_doubled_values(self):
    grid = [np.linspace(0.0, 1.0, 4)]
    v0 = np.array([[1.0], [2.0], [3.0]])
    v1 = np.array([[2.0], [4.0], [6.0]])
    _, out = rel_change(grid, v0, v1)
    np.testing.assert_allclose(out[:, 0], [1.0, 1.0, 1.0])

  def test_no_change_gives_zero(self):
    grid = [np.linspace(0.0, 1.0, 4)]
    v = np.array([[1.0], [2.0], [3.0]])
    _, out = rel_change(grid, v.copy(), v.copy())
    np.testing.assert_allclose(out[:, 0], 0.0, atol=1e-14)

  def test_with_comp_normalizes_by_selected_component(self):
    grid = [np.linspace(0.0, 1.0, 3)]
    v0 = np.array([[2.0, 4.0], [1.0, 2.0]])
    v1 = np.array([[4.0, 8.0], [2.0, 4.0]])
    _, out = rel_change(grid, v0, v1, comp=0)
    np.testing.assert_allclose(out[0, 0], 1.0)
    np.testing.assert_allclose(out[0, 1], 2.0)

  def test_multi_component(self):
    grid = [np.linspace(0.0, 1.0, 3)]
    v0 = np.array([[1.0, 2.0], [1.0, 4.0]])
    v1 = np.array([[2.0, 4.0], [3.0, 8.0]])
    _, out = rel_change(grid, v0, v1)
    np.testing.assert_allclose(out[0, 0], 1.0)
    np.testing.assert_allclose(out[0, 1], 1.0)
    np.testing.assert_allclose(out[1, 0], 2.0)
    np.testing.assert_allclose(out[1, 1], 1.0)


# ---------------------------------------------------------------------------
# rotation_matrix
# ---------------------------------------------------------------------------


class TestRotationMatrix:

  def test_basic_shape(self):
    v = np.array([1.0, 2.0, 3.0])
    R = rotation_matrix(v)
    assert R.shape == (3, 3)

  def test_returns_ndarray(self):
    v = np.array([1.0, 2.0, 3.0])
    assert isinstance(rotation_matrix(v), np.ndarray)

  def test_arbitrary_vector_first_row_is_direction(self):
    v = np.array([3.0, 4.0, 1.0])
    R = rotation_matrix(v)
    k = v / np.abs(v)
    np.testing.assert_allclose(R[0], k, atol=1e-10)

  def test_positive_vector(self):
    v = np.array([1.0, 2.0, 3.0])
    R = rotation_matrix(v)
    np.testing.assert_allclose(R[0], np.array([1.0, 1.0, 1.0]), atol=1e-10)

  def test_returns_non_zero_matrix(self):
    v = np.array([1.0, 2.0, 3.0])
    R = rotation_matrix(v)
    assert R.dtype == float
    assert np.any(R != 0)

  def test_rows_are_mutually_orthogonal(self):
    """Analytic check: rotation_matrix builds an (unnormalized) orthogonal
    frame -- each row should be perpendicular to every other row."""
    v = np.array([2.0, -3.0, 5.0])
    R = rotation_matrix(v)
    np.testing.assert_allclose(R[0] @ R[1], 0.0, atol=1e-10)
    np.testing.assert_allclose(R[0] @ R[2], 0.0, atol=1e-10)
    np.testing.assert_allclose(R[1] @ R[2], 0.0, atol=1e-10)
