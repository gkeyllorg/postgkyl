"""Tests for the ``differentiate`` verb -- numerical gradient of field data.

Per the layer-03 differentiate-decision note, this is a post-``.interpolate()``
verb: it takes NumPy field values and refuses native modal (gkyl-backed)
data, exactly like ``select``.
"""

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
def test_rejects_modal_data():
  d = pg.load(F1)
  with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
    operations.differentiate(d)
