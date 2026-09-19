"""Tests for postgkyl.diagnostics.mom.five_moment -- the 5-/10-moment primitive
variable family (density, velocity, pressure, temperature, sound, Mach),
folding the array-math analytic tests (formerly tests_models_five_moment.py)
with the verb-level guard/inplace/tag/label/VARIABLES tests (formerly part of
tests_ops_moments.py)."""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.mom import five_moment as fm
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

# 5-moment Euler fluid: [rho, rho*vx, rho*vy, rho*vz, E]
_RHO = 1.0
_VX, _VY, _VZ = 0.5, 0.25, 0.1
_P_THERMAL = 0.6
_GAMMA = 5.0 / 3.0
_E_5 = _P_THERMAL / (_GAMMA - 1) + 0.5 * _RHO * (_VX**2 + _VY**2 + _VZ**2)
_MOM5 = np.array([[_RHO, _RHO * _VX, _RHO * _VY, _RHO * _VZ, _E_5]])

# 10-moment fluid: [rho, mx, my, mz, Pxx, Pxy, Pxz, Pyy, Pyz, Pzz]
_P_T = 0.4
_Pxx = _P_T + _RHO * _VX**2
_Pxy = 0.0 + _RHO * _VX * _VY
_Pxz = 0.0 + _RHO * _VX * _VZ
_Pyy = _P_T + _RHO * _VY**2
_Pyz = 0.0 + _RHO * _VY * _VZ
_Pzz = _P_T + _RHO * _VZ**2
_MOM10 = np.array([[
    _RHO, _RHO * _VX, _RHO * _VY, _RHO * _VZ, _Pxx, _Pxy, _Pxz, _Pyy, _Pyz, _Pzz
]])


class TestDensity:

  def test_value(self):
    d = _make(_G1D, _MOM5)
    out = fm.density(d)
    np.testing.assert_allclose(out.values[0, 0], _RHO)

  def test_output_shape_has_trailing_dim(self):
    d = _make(_G1D, _MOM5)
    out = fm.density(d)
    assert out.values.ndim == _MOM5.ndim
    assert out.values.shape[-1] == 1

  def test_multi_cell(self):
    grid = [np.linspace(0.0, 1.0, 4)]
    values = np.hstack([np.array([[1.0], [2.0], [3.0]]), np.zeros((3, 4))])
    d = _make(grid, values)
    out = fm.density(d)
    np.testing.assert_allclose(out.values[:, 0], [1.0, 2.0, 3.0])

  def test_inplace_mutates(self):
    d = _make(_G1D, _MOM5)
    out = fm.density(d, inplace=True)
    assert out is d

  def test_tag_and_label(self):
    d = _make(_G1D, _MOM5)
    out = fm.density(d, tag="rho", label="lbl")
    assert out.get_tag() == "rho"
    assert out.get_label() == "lbl"

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      fm.density(d)


class TestVelocityComponents:

  def test_xvel(self):
    d = _make(_G1D, _MOM5)
    out = fm.xvel(d)
    np.testing.assert_allclose(out.values[0, 0], _VX)

  def test_yvel(self):
    d = _make(_G1D, _MOM5)
    out = fm.yvel(d)
    np.testing.assert_allclose(out.values[0, 0], _VY)

  def test_zvel(self):
    d = _make(_G1D, _MOM5)
    out = fm.zvel(d)
    np.testing.assert_allclose(out.values[0, 0], _VZ)

  def test_vel_three_components(self):
    d = _make(_G1D, _MOM5)
    out = fm.vel(d)
    assert out.values.shape[-1] == 3
    np.testing.assert_allclose(out.values[0, 0], _VX)
    np.testing.assert_allclose(out.values[0, 1], _VY)
    np.testing.assert_allclose(out.values[0, 2], _VZ)

  def test_fabricated_maxwellian_recovers_bulk_velocity(self):
    # density=1, momentum=(2, 0, 0), energy=10: analytic case from the
    # legacy TestMomentFluent euler() fixture -- vx should recover 2.0.
    d = _make([np.array([0.0, 1.0])], np.array([[1.0, 2.0, 0.0, 0.0, 10.0]]))
    rho_out = fm.density(d)
    vx_out = fm.xvel(d)
    np.testing.assert_allclose(rho_out.values.flat[0], 1.0)
    np.testing.assert_allclose(vx_out.values.flat[0], 2.0)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      fm.xvel(d)


class TestPressureScalar:

  def test_5mom_auto_detect(self):
    d = _make(_G1D, _MOM5)
    out = fm.pressure(d)
    np.testing.assert_allclose(out.values[0, 0], _P_THERMAL, rtol=1e-10)

  def test_5mom_explicit(self):
    d = _make(_G1D, _MOM5)
    out = fm.pressure(d, num_moms=5)
    np.testing.assert_allclose(out.values[0, 0], _P_THERMAL, rtol=1e-10)

  def test_10mom_auto_detect(self):
    d = _make(_G1D, _MOM10)
    out = fm.pressure(d)
    np.testing.assert_allclose(out.values[0, 0], _P_T, rtol=1e-10)

  def test_10mom_explicit(self):
    d = _make(_G1D, _MOM10)
    out = fm.pressure(d, num_moms=10)
    np.testing.assert_allclose(out.values[0, 0], _P_T, rtol=1e-10)

  def test_wrong_num_comps_raises(self):
    d = _make(_G1D, np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError, match="num_moms"):
      fm.pressure(d)

  def test_multi_cell(self):
    grid = [np.linspace(0.0, 1.0, 3)]
    values = np.concatenate([_MOM5, _MOM5 * 2.0], axis=0)
    d = _make(grid, values)
    out = fm.pressure(d, num_moms=5)
    np.testing.assert_allclose(out.values[0, 0], _P_THERMAL, rtol=1e-9)
    np.testing.assert_allclose(out.values[1, 0], 2.0 * _P_THERMAL, rtol=1e-9)

  def test_gas_gamma_is_forwarded(self):
    d = _make(_G1D, _MOM5)
    out = fm.pressure(d, gas_gamma=1.4)
    _, expected = fm._get_p(d.grid, d.values, gas_gamma=1.4, num_moms=5)
    np.testing.assert_allclose(out.values, expected)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      fm.pressure(d)


class TestKineticEnergy:

  def test_5mom(self):
    d = _make(_G1D, _MOM5)
    out = fm.ke(d)
    expected = 0.5 * _RHO * (_VX**2 + _VY**2 + _VZ**2)
    np.testing.assert_allclose(out.values[0, 0], expected, rtol=1e-10)

  def test_10mom(self):
    d = _make(_G1D, _MOM10)
    out = fm.ke(d, num_moms=10)
    expected = 0.5 * _RHO * (_VX**2 + _VY**2 + _VZ**2)
    np.testing.assert_allclose(out.values[0, 0], expected, rtol=1e-10)

  def test_wrong_num_comps_raises(self):
    d = _make(_G1D, np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError):
      fm.ke(d)

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      fm.ke(d)


class TestTempSoundMach:

  def test_temp_5mom(self):
    d = _make(_G1D, _MOM5)
    out = fm.temp(d)
    np.testing.assert_allclose(out.values[0, 0], _P_THERMAL / _RHO, rtol=1e-10)

  def test_temp_10mom(self):
    d = _make(_G1D, _MOM10)
    out = fm.temp(d, num_moms=10)
    np.testing.assert_allclose(out.values[0, 0], _P_T / _RHO, rtol=1e-10)

  def test_sound_speed(self):
    d = _make(_G1D, _MOM5)
    out = fm.sound(d)
    expected = np.sqrt(_GAMMA * _P_THERMAL / _RHO)
    np.testing.assert_allclose(out.values[0, 0], expected, rtol=1e-10)

  def test_mach(self):
    d = _make(_G1D, _MOM5)
    out = fm.mach(d)
    v = np.sqrt(_VX**2 + _VY**2 + _VZ**2)
    cs = np.sqrt(_GAMMA * _P_THERMAL / _RHO)
    np.testing.assert_allclose(out.values[0, 0], v / cs, rtol=1e-10)

  def test_grid_is_passed_through_unchanged(self):
    d = _make(_G1D, _MOM5)
    out = fm.mach(d)
    np.testing.assert_allclose(out.grid[0], _G1D[0])

  @needs_gkeyll
  def test_temp_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      fm.temp(d)

  @needs_gkeyll
  def test_sound_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      fm.sound(d)

  @needs_gkeyll
  def test_mach_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      fm.mach(d)


class TestVelocityVerb:

  def test_divides_momentum_by_density(self):
    density = _make([np.array([0.0, 1.0, 2.0])], np.array([[1.0], [2.0]]))
    momentum = _make([np.array([0.0, 1.0, 2.0])],
                     np.array([[3.0, 6.0], [4.0, 8.0]]))
    out = fm.velocity(density, momentum)
    np.testing.assert_allclose(out.values, [[3.0, 6.0], [2.0, 4.0]])

  def test_inplace_mutates_density(self):
    density = _make([np.array([0.0, 1.0])], np.array([[1.0]]))
    momentum = _make([np.array([0.0, 1.0])], np.array([[2.0]]))
    out = fm.velocity(density, momentum, inplace=True)
    assert out is density

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    field = _make([np.array([0.0, 1.0])], np.array([[1.0]]))
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      fm.velocity(d, field)


class TestVariables:

  @pytest.mark.parametrize("name", [
      "density", "xvel", "yvel", "zvel", "vel", "pressure", "ke", "temp",
      "sound", "mach"
  ])
  def test_variables_table_matches_public_function(self, name):
    assert fm.VARIABLES[name] is getattr(fm, name)

  def test_variables_table_has_exactly_the_old_euler_vocabulary(self):
    assert set(fm.VARIABLES) == {
        "density", "xvel", "yvel", "zvel", "vel", "pressure", "ke", "temp",
        "sound", "mach"
    }
