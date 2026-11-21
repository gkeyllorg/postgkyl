"""Analytic tests for ``postgkyl.diagnostics.gk.sheath``.

Face averages are checked against the closed-form face means of the
orthonormal serendipity p = 1 basis and of polynomials projected onto p = 2
and 3D bases; the sheath terms against drifting Maxwellian moments with known
n, u and T, the floating drop against the zero-current flux balance it
solves, and the Mach number against a flow set to a known multiple of c_s.

Run: PYTHONPATH=src pytest tests/test_diagnostics_gk_sheath.py -v
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from postgkyl import dg, gpython
from postgkyl.diagnostics.gk import sheath

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

ME, MI, E = 9.10938215e-31, 3.34358348e-27, 1.602176487e-19  # electron, D


def project(f, basis_type, ndim, poly_order):
  """Coefficients of ``f`` on one reference cell by Gauss-Legendre
  projection (exact for the polynomials used here): the orthonormal basis's
  coefficients are int f b_k over [-1, 1]^ndim."""
  x, w = np.polynomial.legendre.leggauss(poly_order + 2)
  pts = np.array(list(itertools.product(x, repeat=ndim)))
  wts = np.prod(list(itertools.product(w, repeat=ndim)), axis=1)
  b = dg.basis_matrix(basis_type, ndim, poly_order, pts)
  return (wts * f(*pts.T)) @ b


@needs_gkeyll
class TestFaceWeights:

  @pytest.mark.parametrize("side", [-1, 1])
  def test_serendipity_p1_closed_form(self, side):
    # The face y = side mean of 1/2, sqrt(3) x / 2, sqrt(3) y / 2, 3 x y / 2.
    w = sheath.face_weights("serendipity", 2, 1, 1, side)
    np.testing.assert_allclose(w, [0.5, 0.0, side * np.sqrt(3) / 2, 0.0],
                               rtol=0,
                               atol=1e-15)

  @pytest.mark.parametrize("basis_type", ["serendipity", "tensor"])
  def test_p2_polynomial_face_means(self, basis_type):
    # f = 1 + x + 2 y + 3 x^2 + x y^2 + y^2: over x at y = +1 its mean is
    # 1 + 2 + 1 + 1 = 5, at y = -1 it is 1 - 2 + 1 + 1 = 1; over y at
    # x = +1, 1 + 1 + 3 + 1/3 + 1/3 = 17/3, at x = -1, 1 - 1 + 3 - 1/3 + 1/3.
    def f(x, y):
      return 1 + x + 2 * y + 3 * x**2 + x * y**2 + y**2

    c = project(f, basis_type, 2, 2)
    for direction, side, mean in ((1, 1, 5.0), (1, -1, 1.0), (0, 1, 17 / 3),
                                  (0, -1, 3.0)):
      w = sheath.face_weights(basis_type, 2, 2, direction, side)
      assert c @ w == pytest.approx(mean, abs=1e-13)

  def test_3d_face_means(self):
    # f = 2 + x + y z + 4 z on [-1, 1]^3: its mean over the face z = +1 is 6,
    # over x = -1 it is 1 (y z and z average out).
    def f(x, y, z):
      return 2 + x + y * z + 4 * z

    c = project(f, "serendipity", 3, 1)
    assert c @ sheath.face_weights("serendipity", 3, 1, 2, 1) == pytest.approx(
        6.0, abs=1e-13)
    assert c @ sheath.face_weights("serendipity", 3, 1, 0, -1) == pytest.approx(
        1.0, abs=1e-13)


def maxwellian_moments(n, u, temp, mass):
  """Gyrokinetic M0, M1, M2 of a drifting Maxwellian: M2 = int (v_par^2 +
  2 mu B / m) f = n u^2 + n T / m (parallel) + 2 n T / m (perpendicular)."""
  n, u, temp = (np.asarray(v, float) for v in (n, u, temp))
  return n, n * u, n * u**2 + 3 * n * temp / mass


class TestTemperature:

  def test_recovers_a_drifting_maxwellian(self):
    n, u, temp = [1e19, 3e18], [2e4, -5e4], [40 * E, 250 * E]
    np.testing.assert_allclose(sheath.temperature(
        *maxwellian_moments(n, u, temp, MI), MI),
                               temp,
                               rtol=1e-14)


class TestSheathTerms:

  def setup_method(self):
    self.te = np.array([5.0, 20.0, 80.0]) * E
    self.ti = np.array([5.0, 40.0, 30.0]) * E
    self.n = np.array([2e18, 1e19, 4e19])
    self.cs = np.sqrt((self.te + self.ti) / MI)

  def terms(self, vcut_sq, mach, toward):
    elc = maxwellian_moments(self.n, 0.0, self.te, ME)
    ion = maxwellian_moments(self.n, toward * mach * self.cs, self.ti, MI)
    return sheath.sheath_terms(vcut_sq,
                               elc,
                               ion,
                               elc_mass=ME,
                               ion_mass=MI,
                               elementary_charge=E,
                               toward=toward)

  @pytest.mark.parametrize("toward", [-1, 1])
  def test_known_drop_mach_and_temperatures(self, toward):
    drop = np.array([1.0, 2.5, 3.2])
    vcut_sq = 2 * drop * self.te / ME  # m_e v_cut^2 / 2 = drop T_e
    mach = np.array([0.9, 1.0, 1.3])
    got_drop, _, got_mach, te, ti = self.terms(vcut_sq, mach, toward)
    np.testing.assert_allclose(got_drop, drop, rtol=1e-14)
    np.testing.assert_allclose(got_mach, mach, rtol=1e-12)
    np.testing.assert_allclose(te, [5.0, 20.0, 80.0], rtol=1e-12)
    np.testing.assert_allclose(ti, [5.0, 40.0, 30.0], rtol=1e-12)

  def test_floating_drop_balances_the_currents(self):
    # Zero current at the wall: the electron flux n v_e / 4 exp(-e dphi / T_e)
    # (v_e = sqrt(8 T_e / (pi m_e)), the mean speed) equals the ion's Bohm
    # flux n c_s.
    _, floating, _, _, _ = self.terms(np.zeros(3), np.ones(3), 1)
    v_e = np.sqrt(8 * self.te / (np.pi * ME))
    np.testing.assert_allclose(self.n * v_e / 4 * np.exp(-floating),
                               self.n * self.cs,
                               rtol=1e-13)
    # Deuterium, T_i = T_e: the familiar ~2.8.
    assert floating[0] == pytest.approx(2.84, abs=0.01)

  def test_empty_rows_are_not_finite_and_silent(self):
    elc = maxwellian_moments(self.n, 0.0, self.te, ME)
    ion = maxwellian_moments(self.n, 0.0, self.ti, MI)
    elc = tuple(np.where([True, False, True], v, 0.0) for v in elc)
    with np.errstate(all="raise"):
      drop, floating, mach, te, _ = sheath.sheath_terms(np.ones(3),
                                                        elc,
                                                        ion,
                                                        elc_mass=ME,
                                                        ion_mass=MI,
                                                        elementary_charge=E,
                                                        toward=1)
    np.testing.assert_array_equal(np.isfinite(drop), [True, False, True])
    assert not np.isfinite(te[1])
