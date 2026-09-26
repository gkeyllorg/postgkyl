"""Tests for postgkyl.diagnostics.mom.rotations -- parrotate/perprotate, folding
the array-math analytic tests (formerly tests_models_rotations.py) with the
verb-level guard/inplace tests (formerly part of tests_ops_physics.py)."""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.mom import rotations
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


class TestParrotate:

  def test_u_parallel_to_v_returns_u(self):
    u = _make([np.linspace(0.0, 1.0, 3)],
              np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    v = _make([np.linspace(0.0, 1.0, 3)],
              np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    out = rotations.parrotate(u, v)
    np.testing.assert_allclose(out.values, u.values, atol=1e-12)

  def test_u_perpendicular_to_v_returns_zero(self):
    u = _make([np.linspace(0.0, 1.0, 3)],
              np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]))
    v = _make([np.linspace(0.0, 1.0, 3)],
              np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    out = rotations.parrotate(u, v)
    np.testing.assert_allclose(out.values, np.zeros_like(u.values), atol=1e-12)

  def test_u_oblique_to_v(self):
    u = _make([np.array([0.0, 1.0])], np.array([[3.0, 4.0, 0.0]]))
    v = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0, 0.0]]))
    out = rotations.parrotate(u, v)
    np.testing.assert_allclose(out.values[0], [3.0, 0.0, 0.0], atol=1e-12)

  def test_custom_rotate_coords(self):
    u = _make([np.array([0.0, 1.0])], np.array([[3.0, 4.0, 0.0]]))
    field = _make([np.array([0.0, 1.0])],
                  np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))
    out = rotations.parrotate(u, field, coords="3:6")
    np.testing.assert_allclose(out.values[0], [3.0, 0.0, 0.0], atol=1e-12)

  def test_grid_passed_through(self):
    grid = [np.linspace(0.0, 1.0, 2)]
    u = _make(grid, np.array([[1.0, 0.0, 0.0]]))
    v = _make(grid, np.array([[1.0, 0.0, 0.0]]))
    out = rotations.parrotate(u, v)
    np.testing.assert_allclose(out.grid[0], grid[0])

  def test_mismatched_components_raises(self):
    u = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0]]))  # only 2 comps
    v = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="three-component"):
      rotations.parrotate(u, v)

  def test_inplace_mutates_array(self):
    u = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0, 0.0]]))
    v = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0, 0.0]]))
    out = rotations.parrotate(u, v, inplace=True)
    assert out is u

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    v = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      rotations.parrotate(d, v)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      rotations.parrotate(v, d)


class TestPerprotate:

  def test_u_parallel_to_v_gives_zero(self):
    u = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0, 0.0]]))
    v = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0, 0.0]]))
    out = rotations.perprotate(u, v)
    np.testing.assert_allclose(out.values, np.zeros_like(u.values), atol=1e-12)

  def test_u_perpendicular_to_v_gives_u(self):
    u = _make([np.array([0.0, 1.0])], np.array([[0.0, 1.0, 0.0]]))
    v = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0, 0.0]]))
    out = rotations.perprotate(u, v)
    np.testing.assert_allclose(out.values, u.values, atol=1e-12)

  def test_perp_plus_par_equals_u(self):
    u = _make([np.array([0.0, 1.0])], np.array([[3.0, 4.0, 0.0]]))
    v = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0, 0.0]]))
    par = rotations.parrotate(u, v)
    perp = rotations.perprotate(u, v)
    np.testing.assert_allclose(par.values + perp.values, u.values, atol=1e-12)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    v = _make([np.array([0.0, 1.0])], np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      rotations.perprotate(d, v)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      rotations.perprotate(v, d)
