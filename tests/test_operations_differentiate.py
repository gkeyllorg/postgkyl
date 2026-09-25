"""Numerical point-value gradients and exact cell-local DG derivatives."""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, operations
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


def _quadratic_1d(n=40):
  edges = np.linspace(0.0, 1.0, n + 1)
  centers = 0.5 * (edges[:-1] + edges[1:])
  y = centers**2  # d/dx = 2x
  return _make([edges], y[:, np.newaxis]), centers


def test_full_gradient_matches_analytic_derivative_1d():
  d, centers = _quadratic_1d()
  out = operations.differentiate(d)
  np.testing.assert_allclose(out.get_values().flatten(),
                             2.0 * centers,
                             atol=1e-2)
  assert out.get_num_comps() == 1  # 1 comp * 1 dim = 1


def test_direction_matches_full_gradient_in_1d():
  d, _ = _quadratic_1d()
  full = operations.differentiate(d)
  by_dir = operations.differentiate(d, direction=0)
  np.testing.assert_allclose(full.get_values(), by_dir.get_values())


def test_grid_unchanged():
  d, _ = _quadratic_1d()
  out = operations.differentiate(d)
  np.testing.assert_allclose(out.get_grid()[0], d.get_grid()[0])


def test_2d_full_gradient_stacks_components():
  e0 = np.linspace(0.0, 1.0, 21)
  e1 = np.linspace(0.0, 1.0, 21)
  c0 = 0.5 * (e0[:-1] + e0[1:])
  c1 = 0.5 * (e1[:-1] + e1[1:])
  X, Y = np.meshgrid(c0, c1, indexing="ij")
  values = (X**2 + Y)[..., np.newaxis]  # d/dx = 2x, d/dy = 1
  d = _make([e0, e1], values)
  out = operations.differentiate(d)
  assert out.get_num_comps() == 2
  np.testing.assert_allclose(out.get_values()[..., 0], 2 * X, atol=1e-2)
  np.testing.assert_allclose(out.get_values()[..., 1],
                             np.ones_like(Y),
                             atol=1e-2)

  single = operations.differentiate(d, direction=1)
  np.testing.assert_allclose(single.get_values()[..., 0],
                             np.ones_like(Y),
                             atol=1e-2)


def test_inplace_and_tag_label():
  d, _ = _quadratic_1d()
  out = operations.differentiate(d, tag="grad", label="dq/dx", inplace=True)
  assert out is d
  assert d.get_tag() == "grad"
  assert d.get_label() == "dq/dx"


def test_mismatched_grid_length_raises():
  # Cell-centered grid (matches value count, not the expected nodal edges)
  # cannot form the required cell-widths -- this is the documented caveat.
  x = np.linspace(0.0, 1.0, 10)
  d = _make([x], (x**2)[:, np.newaxis])
  with pytest.raises(ValueError):
    operations.differentiate(d)


@needs_gkeyll
@pytest.mark.parametrize("poly_order", [1, 2])
def test_modal_derivative_preserves_higher_modes(poly_order):
  data = pg.load(os.path.join(DATA, "generated", f"1d_ms_p{poly_order}.gkyl"))
  before = data.values.copy()
  dx = data.grid[0][1] - data.grid[0][0]
  expected = np.zeros_like(before)
  expected[:, 0] = 2 * np.sqrt(3.) * before[:, 1] / dx
  if poly_order == 2:
    expected[:, 1] = 2 * np.sqrt(15.) * before[:, 2] / dx
  out = operations.differentiate(data, direction=0)
  assert out.backend == "gkyl"
  assert out.ctx["value_form"] == "modal"
  np.testing.assert_allclose(out.values, expected, atol=1e-12)
  np.testing.assert_array_equal(data.values, before)


@needs_gkeyll
def test_modal_full_gradient_stacks_packed_fields_and_supports_inplace():
  data = pg.load(os.path.join(DATA, "generated", "gk_drift_2d_p1.gkyl"))
  expected = np.zeros((*data.num_cells, 2 * data.num_comps))
  expected[..., 0] = 4.  # d(phi)/dx = 2, normalized constant basis = 1/2
  expected[..., data.num_comps] = 8.  # d(phi)/dy = 4
  out = operations.differentiate(data,
                                 inplace=True,
                                 tag="grad",
                                 label="gradient")
  assert out is data
  assert out.tag == "grad"
  assert out.label == "gradient"
  assert out.backend == "gkyl"
  np.testing.assert_allclose(out.values, expected, atol=1e-13)


@needs_gkeyll
@pytest.mark.parametrize("direction", [-1, 1])
def test_modal_invalid_direction_raises(direction):
  data = pg.load(os.path.join(DATA, "generated", "gk_moments_p1.gkyl"))
  with pytest.raises(ValueError, match="out of range"):
    operations.differentiate(data, direction=direction)


@needs_gkeyll
def test_modal_nonuniform_grid_raises():
  data = pg.load(os.path.join(DATA, "generated", "gk_moments_p1.gkyl"))
  data.grid = [np.array([0., 1., 3.])]
  with pytest.raises(ValueError, match="uniform Cartesian cell edges"):
    operations.differentiate(data)


@needs_gkeyll
@pytest.mark.parametrize("representation", ["nodal", "quad"])
def test_native_point_values_require_explicit_conversion(representation):
  data = pg.load(os.path.join(DATA, "generated", "gk_moments_p1.gkyl"))
  data = data.represent(to=representation)
  with pytest.raises(ValueError, match=r"represent\(to='modal'\)"):
    operations.differentiate(data)
