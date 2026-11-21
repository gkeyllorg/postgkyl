"""Tests for the ``cell_average`` verb.

Coefficients are written in closed form on Gkeyll's orthonormal Legendre
bases: 1D ``1/sqrt(2)``, ``sqrt(3/2) x``, ``sqrt(5/2) (3 x^2 - 1) / 2``;
2D p = 1 ``1/2, sqrt(3) x / 2, sqrt(3) y / 2, 3 x y / 2``.
"""

from __future__ import annotations

import numpy as np
import pytest

import postgkyl as pg
from generate_test_data import write_gkyl_field
from postgkyl import gpython

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
def test_p1_2d_average_is_the_constant_mode(tmp_path):
  # f = a + b x + c y + d x y on each cell averages to a; two fields.
  rng = np.random.default_rng(3)
  a = rng.normal(size=(3, 4, 2))
  coeffs = np.zeros((3, 4, 8))
  for f in range(2):
    coeffs[..., 4 * f] = 2 * a[..., f]
    coeffs[..., 4 * f + 1:4 * f + 4] = rng.normal(size=(3, 4, 3))
  path = tmp_path / "f.gkyl"
  write_gkyl_field(path, [3, 4], [0.0, -1.0], [3.0, 1.0], coeffs, 1,
                   "serendipity")
  out = pg.GData(str(path)).cell_average()
  np.testing.assert_allclose(out.values, a, rtol=1e-15, atol=1e-15)
  np.testing.assert_allclose(out.grid[0], np.linspace(0.0, 3.0, 4))


@needs_gkeyll
def test_p2_average_is_not_the_centre_value(tmp_path):
  # f = a + b x^2 averages to a + b / 3 over [-1, 1]; its centre value is a.
  a, b = 2.0, 1.5
  coeffs = np.array([[(a + b / 3) * np.sqrt(2), 0.0,
                      (2 * b / 3) / np.sqrt(5 / 2)]])
  path = tmp_path / "g.gkyl"
  write_gkyl_field(path, [1], [-1.0], [1.0], coeffs, 2, "serendipity")
  data = pg.GData(str(path))
  np.testing.assert_allclose(data.cell_average().values.ravel(),
                             a + b / 3,
                             rtol=1e-15)
  np.testing.assert_allclose(data.interpolate(num_interp=1).values.ravel(),
                             a,
                             rtol=1e-14)


@needs_gkeyll
def test_rejects_point_values(tmp_path):
  coeffs = np.ones((2, 2, 4))
  path = tmp_path / "h.gkyl"
  write_gkyl_field(path, [2, 2], [0.0, 0.0], [1.0, 1.0], coeffs, 1,
                   "serendipity")
  with pytest.raises(ValueError):
    pg.GData(str(path)).cell_average().cell_average()
