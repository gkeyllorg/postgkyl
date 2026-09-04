"""Tests for postgkyl.diagnostics.mom.plasma -- plasma-parameter GData verbs
(magB, vt, vA, omegaC, omegaP, d, lambdaD, rho, beta), porting the analytic
array-math assertions of tests_models_plasma_params.py onto the new
GData-facing wrappers -- these functions never had a verb layer before this
restructure, so there is no old ops-level dispatch to preserve."""

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.constants as const

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.mom import plasma as pp
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


_G1 = [np.array([0.0, 1.0])]

# EM field: [Ex, Ey, Ez, Bx, By, Bz]  Bx=3, By=4, Bz=0 -> |B|=5
_FIELD_VALS = np.array([[0.0, 0.0, 0.0, 3.0, 4.0, 0.0]])
_MAGB = 5.0

# 5-moment species: rho=2, vx=0.5, vy=0, vz=0, p=0.6
_GAMMA = 5.0 / 3.0
_RHO = 2.0
_VX = 0.5
_P = 0.6
_E = _P / (_GAMMA - 1) + 0.5 * _RHO * _VX**2
_MOM5 = np.array([[_RHO, _RHO * _VX, 0.0, 0.0, _E]])


def _field():
  return _make(_G1, _FIELD_VALS)


def _species():
  return _make(_G1, _MOM5)


class TestMagB:

  def test_magnitude(self):
    out = pp.magB(_field())
    np.testing.assert_allclose(out.values.flat[0], _MAGB, rtol=1e-10)

  def test_inplace_mutates_field(self):
    field = _field()
    out = pp.magB(field, inplace=True)
    assert out is field

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.magB(d)


class TestVt:

  def test_no_sqrt2_defaults_false(self):
    out = pp.vt(_species())
    T = _P / _RHO
    np.testing.assert_allclose(out.values.flat[0], np.sqrt(2.0 * T), rtol=1e-10)

  def test_no_sqrt2(self):
    out = pp.vt(_species(), no_sqrt2=True)
    T = _P / _RHO
    np.testing.assert_allclose(out.values.flat[0], np.sqrt(T), rtol=1e-10)

  def test_mass_scales_result(self):
    out = pp.vt(_species(), mass=2.0, no_sqrt2=True)
    T = _P / _RHO
    np.testing.assert_allclose(out.values.flat[0], np.sqrt(T / 2.0), rtol=1e-10)

  def test_mhd_uses_mhd_temperature(self):
    bx, by, bz = 1.0, 0.0, 0.0
    mag_p = 0.5 * (bx**2 + by**2 + bz**2)
    e_mhd = 0.5 * _RHO * _VX**2 + _P / (_GAMMA - 1) + mag_p
    mhd_vals = np.array([[_RHO, _RHO * _VX, 0.0, 0.0, e_mhd, bx, by, bz]])
    d = _make(_G1, mhd_vals)
    out = pp.vt(d, gas_gamma=_GAMMA, mhd=True, no_sqrt2=True)
    np.testing.assert_allclose(out.values.flat[0],
                               np.sqrt(_P / _RHO),
                               rtol=1e-10)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.vt(d)


class TestVA:

  def test_alfven_speed(self):
    out = pp.vA(_species(), _field())
    expected = _MAGB / np.sqrt(_RHO)
    np.testing.assert_allclose(out.values.flat[0], expected, rtol=1e-10)

  def test_mu0_scales_result(self):
    out = pp.vA(_species(), _field(), mu_0=2.0)
    expected = _MAGB / np.sqrt(2.0 * _RHO)
    np.testing.assert_allclose(out.values.flat[0], expected, rtol=1e-10)

  def test_result_carries_species_grid(self):
    species, field = _species(), _field()
    out = pp.vA(species, field, inplace=True)
    assert out is species

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    field = _field()
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.vA(d, field)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.vA(_species(), d)


class TestOmegaC:

  def test_cyclotron_frequency(self):
    out = pp.omegaC(_field(), mass=1.0, charge=1.0)
    np.testing.assert_allclose(out.values.flat[0], _MAGB, rtol=1e-10)

  def test_uses_absolute_charge(self):
    oC_pos = pp.omegaC(_field(), mass=1.0, charge=1.0)
    oC_neg = pp.omegaC(_field(), mass=1.0, charge=-1.0)
    np.testing.assert_allclose(oC_pos.values.flat[0],
                               oC_neg.values.flat[0],
                               rtol=1e-10)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.omegaC(d)


class TestOmegaP:

  def test_plasma_frequency(self):
    out = pp.omegaP(_species(), mass=1.0, charge=1.0, epsilon_0=1.0)
    expected = np.sqrt(_RHO)
    np.testing.assert_allclose(out.values.flat[0], expected, rtol=1e-10)

  def test_hydrogen_matches_nrl_formulary(self):
    # NRL Plasma Formulary: f_pi[Hz] = 2.1e2 * Z * sqrt(n[cm^-3] / mu) for a
    # singly-charged ion of mass number mu; compare our SI computation
    # (mass density rho = n * m_p, as fluid moment data stores it) against
    # this textbook approximation to its own (2-digit) precision.
    n = 1.0e20  # m^-3
    rho_vals = np.array([[n * const.m_p]])
    d = _make(_G1, rho_vals)
    out = pp.omegaP(d,
                    mass=const.m_p,
                    charge=const.e,
                    epsilon_0=const.epsilon_0)
    expected_exact = np.sqrt(n * const.e**2 / (const.epsilon_0 * const.m_p))
    np.testing.assert_allclose(out.values.flat[0], expected_exact, rtol=1e-9)

    n_cm3 = n * 1e-6
    omega_nrl = 2 * np.pi * 2.1e2 * np.sqrt(n_cm3)
    np.testing.assert_allclose(out.values.flat[0], omega_nrl, rtol=5e-3)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.omegaP(d)


class TestD:

  def test_skin_depth(self):
    dd = pp.d(_species(), mass=1.0, charge=1.0, epsilon_0=1.0, mu_0=1.0)
    omegaP = pp.omegaP(_species(), mass=1.0, charge=1.0, epsilon_0=1.0)
    expected = 1.0 / omegaP.values.flat[0]
    np.testing.assert_allclose(dd.values.flat[0], expected, rtol=1e-10)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    modal = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.d(modal)


class TestLambdaD:

  def test_debye_length(self):
    out = pp.lambdaD(_species(),
                     mass=1.0,
                     charge=1.0,
                     epsilon_0=1.0,
                     mu_0=1.0,
                     no_sqrt2=False)
    vt_out = pp.vt(_species(), no_sqrt2=False)
    omegaP_out = pp.omegaP(_species(), mass=1.0, charge=1.0, epsilon_0=1.0)
    expected = vt_out.values.flat[0] / omegaP_out.values.flat[0] / np.sqrt(2.0)
    np.testing.assert_allclose(out.values.flat[0], expected, rtol=1e-10)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.lambdaD(d)


class TestRho:

  def test_larmor_radius(self):
    out = pp.rho(_species(), _field(), mass=1.0, charge=1.0, no_sqrt2=False)
    vt_out = pp.vt(_species(), no_sqrt2=False)
    omegaC_out = pp.omegaC(_field(), mass=1.0, charge=1.0)
    expected = vt_out.values.flat[0] / omegaC_out.values.flat[0]
    np.testing.assert_allclose(out.values.flat[0], expected, rtol=1e-10)

  def test_no_sqrt2_matches_default_after_normalization(self):
    rho_default = pp.rho(_species(), _field(), mass=1.0, charge=1.0)
    rho_no_sqrt2 = pp.rho(_species(),
                          _field(),
                          mass=1.0,
                          charge=1.0,
                          no_sqrt2=True)
    np.testing.assert_allclose(rho_no_sqrt2.values.flat[0] /
                               rho_default.values.flat[0],
                               1.0,
                               rtol=1e-8)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    field = _field()
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.rho(d, field)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.rho(_species(), d)


class TestBeta:

  def test_plasma_beta(self):
    out = pp.beta(_species(), _field(), mu_0=1.0, no_sqrt2=False)
    vt_out = pp.vt(_species(), no_sqrt2=False)
    vA_out = pp.vA(_species(), _field(), mu_0=1.0)
    expected = vt_out.values.flat[0]**2 / vA_out.values.flat[0]**2
    np.testing.assert_allclose(out.values.flat[0], expected, rtol=1e-10)

  def test_no_sqrt2_matches_default(self):
    # The "* 2.0" correction for no_sqrt2=True exactly compensates for the
    # missing sqrt(2) factor squared in v_th**2, so both conventions give
    # the same beta.
    beta_default = pp.beta(_species(), _field(), mu_0=1.0)
    beta_no_sqrt2 = pp.beta(_species(),
                            _field(),
                            mu_0=1.0,
                            no_sqrt2=True)
    np.testing.assert_allclose(beta_no_sqrt2.values.flat[0],
                               beta_default.values.flat[0],
                               rtol=1e-10)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    field = _field()
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.beta(d, field)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      pp.beta(_species(), d)
