"""Tests for postgkyl.diagnostics.mom.ten_moment -- 10-moment pressure tensor,
field-aligned pressure diagnostics (p_par, p_perp, agyrotropy), folding the
array-math analytic tests (formerly tests_models_ten_moment.py) with the
verb-level guard/inplace/VARIABLES tests (formerly part of
tests_ops_moments.py / tests_ops_physics.py)."""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.mom import ten_moment as tm
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


_G1D = [np.array([0.0, 1.0])]

_RHO = 1.0
_VX, _VY, _VZ = 0.5, 0.25, 0.1
_P_T = 0.4
_MOM10 = np.array([[
    _RHO, _RHO * _VX, _RHO * _VY, _RHO * _VZ, _P_T + _RHO * _VX**2,
    _RHO * _VX * _VY, _RHO * _VX * _VZ, _P_T + _RHO * _VY**2, _RHO * _VY * _VZ,
    _P_T + _RHO * _VZ**2
]])


def _diagonal_pressure(pxx, pyy, pzz):
  return _make(_G1D, np.array([[pxx, 0.0, 0.0, pyy, 0.0, pzz]]))


def _b(bx, by, bz):
  return _make(_G1D, np.array([[bx, by, bz]]))


class TestPressureTensorComponents:

  def test_pxx(self):
    d = _make(_G1D, _MOM10)
    out = tm.pxx(d)
    np.testing.assert_allclose(out.values[0, 0], _P_T, rtol=1e-10)

  def test_pxy_pxz_pyz_zero_for_diagonal_flow(self):
    d = _make(_G1D, _MOM10)
    np.testing.assert_allclose(tm.pxy(d).values[0, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(tm.pxz(d).values[0, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(tm.pyz(d).values[0, 0], 0.0, atol=1e-14)

  def test_pyy(self):
    d = _make(_G1D, _MOM10)
    out = tm.pyy(d)
    np.testing.assert_allclose(out.values[0, 0], _P_T, rtol=1e-10)

  def test_pzz(self):
    d = _make(_G1D, _MOM10)
    out = tm.pzz(d)
    np.testing.assert_allclose(out.values[0, 0], _P_T, rtol=1e-10)

  def test_pressure_tensor_shape_and_diagonal(self):
    d = _make(_G1D, _MOM10)
    out = tm.pressure_tensor(d)
    assert out.values.shape[-1] == 6
    np.testing.assert_allclose(out.values[0, 0], _P_T, rtol=1e-10)
    np.testing.assert_allclose(out.values[0, 3], _P_T, rtol=1e-10)
    np.testing.assert_allclose(out.values[0, 5], _P_T, rtol=1e-10)
    np.testing.assert_allclose(out.values[0, [1, 2, 4]], 0.0, atol=1e-14)

  @needs_gkeyll
  def test_pxx_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      tm.pxx(d)


class TestPPar:

  def test_b_along_x_pxx_is_p_par(self):
    p = _diagonal_pressure(1.0, 0.5, 0.5)
    b = _b(1.0, 0.0, 0.0)
    out = tm.p_par(p, b)
    np.testing.assert_allclose(out.values.flat[0], 1.0, rtol=1e-12)

  def test_b_along_y_pyy_is_p_par(self):
    p = _diagonal_pressure(0.5, 2.0, 0.5)
    b = _b(0.0, 1.0, 0.0)
    out = tm.p_par(p, b)
    np.testing.assert_allclose(out.values.flat[0], 2.0, rtol=1e-12)

  def test_b_along_z_pzz_is_p_par(self):
    p = _diagonal_pressure(0.5, 0.5, 3.0)
    b = _b(0.0, 0.0, 1.0)
    out = tm.p_par(p, b)
    np.testing.assert_allclose(out.values.flat[0], 3.0, rtol=1e-12)

  def test_isotropic_pressure_p_par_equals_p(self):
    p = _diagonal_pressure(2.0, 2.0, 2.0)
    b = _b(1.0, 1.0, 0.0)
    out = tm.p_par(p, b)
    np.testing.assert_allclose(out.values.flat[0], 2.0, rtol=1e-10)

  def test_b_diagonal_gives_average(self):
    p = _diagonal_pressure(1.0, 2.0, 0.0)
    b = _b(1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0)
    out = tm.p_par(p, b)
    np.testing.assert_allclose(out.values.flat[0], 1.5, rtol=1e-12)

  def test_inplace_mutates_ptensor(self):
    p = _diagonal_pressure(1.0, 0.5, 0.5)
    b = _b(1.0, 0.0, 0.0)
    out = tm.p_par(p, b, inplace=True)
    assert out is p

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    b = _b(1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      tm.p_par(d, b)


class TestPPerp:

  def test_b_along_x_perp_is_average_of_pyy_pzz(self):
    p = _diagonal_pressure(1.0, 0.6, 0.4)
    b = _b(1.0, 0.0, 0.0)
    out = tm.p_perp(p, b)
    np.testing.assert_allclose(out.values.flat[0], 0.5, rtol=1e-12)

  def test_isotropic_pressure_perp_equals_par(self):
    p = _diagonal_pressure(1.5, 1.5, 1.5)
    b = _b(1.0, 0.0, 0.0)
    par_out = tm.p_par(p, b)
    perp_out = tm.p_perp(p, b)
    np.testing.assert_allclose(perp_out.values.flat[0],
                               par_out.values.flat[0],
                               rtol=1e-10)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    b = _b(1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      tm.p_perp(d, b)


class TestAgyro:

  @pytest.mark.parametrize("measure", ["frobenius", "swisdak"])
  def test_isotropic_tensor_is_gyrotropic(self, measure):
    p = _diagonal_pressure(2.0, 2.0, 2.0)
    b = _b(0.0, 0.0, 1.0)
    out = tm.agyro(p, b, measure=measure)
    np.testing.assert_allclose(out.values, 0.0, atol=1e-10)

  def test_swisdak_case_insensitive(self):
    p = _make(_G1D, np.array([[2.0, 0.5, 0.0, 1.0, 0.0, 1.0]]))
    b = _b(1.0, 0.0, 0.0)
    out1 = tm.agyro(p, b, measure="swisdak")
    out2 = tm.agyro(p, b, measure="Swisdak")
    np.testing.assert_allclose(out1.values, out2.values)

  def test_frobenius_case_insensitive(self):
    p = _make(_G1D, np.array([[2.0, 0.5, 0.0, 1.0, 0.0, 1.0]]))
    b = _b(1.0, 0.0, 0.0)
    out1 = tm.agyro(p, b, measure="frobenius")
    out2 = tm.agyro(p, b, measure="Frobenius")
    np.testing.assert_allclose(out1.values, out2.values)

  def test_invalid_measure_raises(self):
    p = _diagonal_pressure(1.0, 1.0, 1.0)
    b = _b(1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="swisdak.*frobenius"):
      tm.agyro(p, b, measure="invalid")

  def test_agyrotropic_swisdak_nonzero(self):
    p = _make(_G1D, np.array([[2.0, 0.5, 0.0, 1.0, 0.0, 1.0]]))
    b = _b(1.0, 0.0, 0.0)
    out = tm.agyro(p, b, measure="swisdak")
    assert out.values.flat[0] > 0.0

  def test_agyrotropic_frobenius_nonzero(self):
    p = _make(_G1D, np.array([[2.0, 0.5, 0.0, 1.0, 0.0, 1.0]]))
    b = _b(1.0, 0.0, 0.0)
    out = tm.agyro(p, b, measure="frobenius")
    assert out.values.flat[0] > 0.0

  def test_default_measure_is_frobenius(self):
    p = _make(_G1D, np.array([[2.0, 0.5, 0.0, 1.0, 0.0, 1.0]]))
    b = _b(1.0, 0.0, 0.0)
    default_out = tm.agyro(p, b)
    explicit_out = tm.agyro(p, b, measure="frobenius")
    np.testing.assert_allclose(default_out.values, explicit_out.values)

  def test_inplace_mutates_ptensor(self):
    p = _diagonal_pressure(2.0, 2.0, 2.0)
    b = _b(0.0, 0.0, 1.0)
    out = tm.agyro(p, b, inplace=True)
    assert out is p

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    b = _b(0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      tm.agyro(d, b)


class TestMomAgyro:

  def _species_and_field(self):
    species = _make(
        _G1D, np.array([[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0]]))
    field = _make(_G1D, np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
    return species, field

  def test_isotropic_species_is_gyrotropic(self):
    species, field = self._species_and_field()
    out = tm.mom_agyro(species, field)
    np.testing.assert_allclose(out.values, 0.0, atol=1e-12)

  def test_matches_private_helper(self):
    species, field = self._species_and_field()
    out = tm.mom_agyro(species, field, measure="swisdak")
    _, expected = tm._get_gkyl_10m_agyro(species.grid,
                                         species.values,
                                         field.grid,
                                         field.values,
                                         measure="swisdak")
    np.testing.assert_allclose(out.values, expected)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    field = _make(_G1D, np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      tm.mom_agyro(d, field)


class TestGkyl10mPrivateWrappers:
  """The ``_get_gkyl_10m_p_par``/``_get_gkyl_10m_p_perp`` helpers have no
  public GData wrapper (the target layout table for this module lists no
  'mom_p_par'/'mom_p_perp' verb, unlike ``mom_agyro``) -- ported directly
  against the private array-level functions, matching the old
  ``models``-level tests exactly."""

  @staticmethod
  def _species_and_field():
    rho, vx = 1.0, 0.5
    Pxx = 2.0 + rho * vx**2
    Pxy = 0.3
    mom10 = np.array([[rho, rho * vx, 0.0, 0.0, Pxx, Pxy, 0.0, 1.0, 0.0, 1.0]])
    field_vals = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    g = [np.array([0.0, 1.0])]
    return g, mom10, g, field_vals

  def test_p_par_wrapper(self):
    sg, sv, fg, fv = self._species_and_field()
    _, p_par = tm._get_gkyl_10m_p_par(sg, sv, fg, fv)
    np.testing.assert_allclose(p_par.flat[0], 2.0, rtol=1e-10)

  def test_p_perp_wrapper(self):
    sg, sv, fg, fv = self._species_and_field()
    _, p_perp = tm._get_gkyl_10m_p_perp(sg, sv, fg, fv)
    np.testing.assert_allclose(p_perp.flat[0], 1.0, rtol=1e-10)


class TestFiveMomentSetFixedAtTenMoments:

  def _tenmoment_state(self):
    vals = np.array([[1.0, 2.0, 0.0, 0.0, 6.0, 0.0, 0.0, 3.0, 0.0, 3.0]])
    return _make([np.array([0.0, 1.0])], vals)

  def test_density_reused_from_five_moment(self):
    from postgkyl.diagnostics.mom import five_moment as fm
    assert tm.density is fm.density
    assert tm.xvel is fm.xvel
    assert tm.vel is fm.vel

  def test_pressure_uses_num_moms_10(self):
    d = self._tenmoment_state()
    out = tm.pressure(d)
    from postgkyl.diagnostics.mom.five_moment import _get_p
    _, expected = _get_p(d.grid, d.values, gas_gamma=5.0 / 3, num_moms=10)
    np.testing.assert_allclose(out.values, expected)

  def test_ke_uses_num_moms_10(self):
    d = self._tenmoment_state()
    out = tm.ke(d)
    from postgkyl.diagnostics.mom.five_moment import _get_ke
    _, expected = _get_ke(d.grid, d.values, gas_gamma=5.0 / 3, num_moms=10)
    np.testing.assert_allclose(out.values, expected)

  def test_temp_uses_num_moms_10(self):
    d = self._tenmoment_state()
    out = tm.temp(d)
    from postgkyl.diagnostics.mom.five_moment import _get_temp
    _, expected = _get_temp(d.grid, d.values, gas_gamma=5.0 / 3, num_moms=10)
    np.testing.assert_allclose(out.values, expected)

  def test_sound_uses_num_moms_10(self):
    d = self._tenmoment_state()
    out = tm.sound(d)
    from postgkyl.diagnostics.mom.five_moment import _get_sound
    _, expected = _get_sound(d.grid, d.values, gas_gamma=5.0 / 3, num_moms=10)
    np.testing.assert_allclose(out.values, expected)

  def test_mach_uses_num_moms_10(self):
    d = self._tenmoment_state()
    out = tm.mach(d)
    from postgkyl.diagnostics.mom.five_moment import _get_mach
    _, expected = _get_mach(d.grid, d.values, gas_gamma=5.0 / 3, num_moms=10)
    np.testing.assert_allclose(out.values, expected)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      tm.density(d)

  @needs_gkeyll
  @pytest.mark.parametrize("fn_name", ["ke", "temp", "sound", "mach"])
  def test_all_scalar_quantities_reject_modal_data(self, fn_name):
    d = pg.load(F1)
    fn = getattr(tm, fn_name)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      fn(d)


class TestVariables:

  def test_variables_table_has_exactly_the_old_tenmoment_vocabulary(self):
    assert set(tm.VARIABLES) == {
        "density", "xvel", "yvel", "zvel", "vel", "pressure", "ke", "temp",
        "sound", "mach", "pressureTensor", "pxx", "pxy", "pxz", "pyy", "pyz",
        "pzz"
    }

  def test_variables_table_maps_to_public_functions(self):
    assert tm.VARIABLES["pressureTensor"] is tm.pressure_tensor
    assert tm.VARIABLES["pxx"] is tm.pxx
    assert tm.VARIABLES["density"] is tm.density
