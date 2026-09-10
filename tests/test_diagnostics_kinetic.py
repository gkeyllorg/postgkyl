"""Tests for postgkyl.diagnostics.vm.kinetic -- distribution-function frame
transform, folding the array-math analytic tests (formerly
tests_models_frame.py) with the verb-level guard/inplace tests (formerly
part of tests_ops_physics.py)."""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.vm import kinetic
from postgkyl.gdatastate.gdatastate import GDataState

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
F1 = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")


def _make(grid, values, **ctx):
  d = GDataState(ctx=ctx or None)
  d.push(list(grid), values)
  return d


class TestTransformFrameCdim1:

  def _distribution(self, nx=2, nv=3):
    x_edges = np.linspace(0.0, 1.0, nx + 1)
    v_edges = np.linspace(-2.0, 2.0, nv + 1)
    values = np.ones((nx, nv, 1))
    return _make([x_edges, v_edges], values)

  def test_basic_returns_unchanged_values(self):
    f = _make([np.linspace(0.0, 1.0, 4),
               np.linspace(-3.0, 3.0, 5)], np.ones((3, 4, 1)))
    bulk = _make([f.grid[0]], np.ones((3, 1)) * 0.5)
    out = kinetic.transform_frame(f, bulk, cdim=1)
    np.testing.assert_array_equal(out.values, f.values)
    assert len(out.grid) == 2

  def test_zero_velocity_leaves_grid_unshifted(self):
    v_grid = np.linspace(-2.0, 2.0, 4)
    f = _make([np.linspace(0.0, 1.0, 3), v_grid],
              np.random.default_rng(0).random((2, 3, 1)))
    bulk = _make([f.grid[0]], np.zeros((2, 1)))
    out = kinetic.transform_frame(f, bulk, cdim=1)
    np.testing.assert_array_equal(out.values, f.values)
    np.testing.assert_allclose(out.grid[1], np.tile(v_grid, (3, 1)))

  def test_shifts_velocity_grid_by_bulk_velocity(self):
    v_grid = np.linspace(-2.0, 2.0, 4)
    f = _make([np.linspace(0.0, 1.0, 3), v_grid], np.ones((2, 3, 1)))
    bulk = _make([f.grid[0]], np.full((2, 1), 0.5))
    out = kinetic.transform_frame(f, bulk, cdim=1)
    # Interior nodes see the average of the two neighboring cells' shift
    # (both 0.5 here); edge nodes see the single adjacent cell's shift.
    np.testing.assert_allclose(out.grid[1][0], v_grid + 0.5)
    np.testing.assert_allclose(out.grid[1][-1], v_grid + 0.5)

  def test_matches_private_helper(self):
    f = self._distribution()
    bulk = _make([f.grid[0]], np.array([[0.1], [0.2]]))
    out = kinetic.transform_frame(f, bulk, cdim=1)
    grid, values = kinetic._transform_frame(f.grid, f.values, bulk.values, 1)
    for d in range(2):
      np.testing.assert_allclose(out.grid[d], grid[d])
    np.testing.assert_allclose(out.values, values)

  def test_inplace_mutates_distribution(self):
    f = self._distribution()
    bulk = _make([f.grid[0]], np.array([[0.1], [0.2]]))
    out = kinetic.transform_frame(f, bulk, cdim=1, inplace=True)
    assert out is f

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    bulk = _make([np.array([0.0, 1.0])], np.array([[0.1]]))
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      kinetic.transform_frame(d, bulk, cdim=1)


class TestTransformFrameCdim2:

  def test_zero_velocity_leaves_grid_unshifted(self):
    nx, ny, nv = 2, 2, 3
    x_grid = np.linspace(0.0, 1.0, nx + 1)
    y_grid = np.linspace(0.0, 1.0, ny + 1)
    grid_f = [x_grid, y_grid, np.linspace(-2.0, 2.0, nv + 1)]
    values_f = np.ones((nx, ny, nv, 1))
    f = _make(grid_f, values_f)
    bulk = _make([f.grid[0], f.grid[1]], np.zeros((nx, ny, 1)))
    out = kinetic.transform_frame(f, bulk, cdim=2)
    np.testing.assert_array_equal(out.values, values_f)
    assert len(out.grid) == 3
    np.testing.assert_allclose(out.grid[2],
                               np.tile(grid_f[2], (nx + 1, ny + 1, 1)))

  def test_shifts_velocity_grid_by_bulk_velocity(self):
    nx, ny, nv = 2, 2, 3
    v_grid = np.linspace(-2.0, 2.0, nv + 1)
    grid_f = [
        np.linspace(0.0, 1.0, nx + 1),
        np.linspace(0.0, 1.0, ny + 1), v_grid
    ]
    values_f = np.ones((nx, ny, nv, 1))
    f = _make(grid_f, values_f)
    bulk = _make([f.grid[0], f.grid[1]], np.full((nx, ny, 1), 0.5))
    out = kinetic.transform_frame(f, bulk, cdim=2)
    np.testing.assert_array_equal(out.values, values_f)
    # Every corner node sees the same 0.5 shift, since the bulk velocity is
    # uniform.
    np.testing.assert_allclose(out.grid[2][0, 0], v_grid + 0.5)
    np.testing.assert_allclose(out.grid[2][-1, -1], v_grid + 0.5)


class TestTransformFrameCdim3:

  def test_zero_velocity_leaves_grid_unshifted(self):
    nx, ny, nz, nv = 2, 2, 2, 2
    grid_f = [
        np.linspace(0.0, 1.0, nx + 1),
        np.linspace(0.0, 1.0, ny + 1),
        np.linspace(0.0, 1.0, nz + 1),
        np.linspace(-2.0, 2.0, nv + 1)
    ]
    values_f = np.ones((nx, ny, nz, nv, 1))
    f = _make(grid_f, values_f)
    bulk = _make([f.grid[0], f.grid[1], f.grid[2]], np.zeros((nx, ny, nz, 1)))
    out = kinetic.transform_frame(f, bulk, cdim=3)
    np.testing.assert_array_equal(out.values, values_f)
    assert len(out.grid) == 4
    np.testing.assert_allclose(out.grid[3],
                               np.tile(grid_f[3], (nx + 1, ny + 1, nz + 1, 1)))

  def test_shifts_velocity_grid_by_bulk_velocity(self):
    nx, ny, nz, nv = 2, 2, 2, 2
    v_grid = np.linspace(-2.0, 2.0, nv + 1)
    grid_f = [
        np.linspace(0.0, 1.0, nx + 1),
        np.linspace(0.0, 1.0, ny + 1),
        np.linspace(0.0, 1.0, nz + 1), v_grid
    ]
    values_f = np.ones((nx, ny, nz, nv, 1))
    f = _make(grid_f, values_f)
    bulk = _make([f.grid[0], f.grid[1], f.grid[2]], np.full((nx, ny, nz, 1),
                                                            0.5))
    out = kinetic.transform_frame(f, bulk, cdim=3)
    np.testing.assert_array_equal(out.values, values_f)
    np.testing.assert_allclose(out.grid[3][0, 0, 0], v_grid + 0.5)
    np.testing.assert_allclose(out.grid[3][-1, -1, -1], v_grid + 0.5)
