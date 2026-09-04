"""Tests for postgkyl.diagnostics.mom.mhd -- MHD B-field, pressure, temperature,
sound speed, Mach number, folding the array-math analytic tests (formerly
tests_models_mhd.py) with the verb-level guard/VARIABLES tests (formerly
part of tests_ops_moments.py)."""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.mom import mhd
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
_VX = 0.5
_P_THERMAL = 0.6
_GAMMA = 5.0 / 3.0
_BX, _BY, _BZ = 1.0, 0.0, 0.0
_MAG_P = 0.5 * (_BX**2 + _BY**2 + _BZ**2)
_E_MHD = 0.5 * _RHO * _VX**2 + _P_THERMAL / (_GAMMA - 1) + _MAG_P
_MHD8 = np.array([[_RHO, _RHO * _VX, 0.0, 0.0, _E_MHD, _BX, _BY, _BZ]])


class TestFieldExtraction:

  def test_bx(self):
    d = _make(_G1D, _MHD8)
    np.testing.assert_allclose(mhd.bx(d).values[0, 0], _BX)

  def test_by(self):
    d = _make(_G1D, _MHD8)
    np.testing.assert_allclose(mhd.by(d).values[0, 0], _BY)

  def test_bz(self):
    d = _make(_G1D, _MHD8)
    np.testing.assert_allclose(mhd.bz(d).values[0, 0], _BZ)

  def test_bi_shape_and_values(self):
    d = _make(_G1D, _MHD8)
    out = mhd.bi(d)
    assert out.values.shape[-1] == 3
    np.testing.assert_allclose(out.values[0], [_BX, _BY, _BZ])

  def test_mag_pressure(self):
    d = _make(_G1D, _MHD8)
    out = mhd.mag_pressure(d)
    np.testing.assert_allclose(out.values[0, 0], _MAG_P)

  @needs_gkeyll
  def test_bx_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      mhd.bx(d)


class TestThermo:

  def test_pressure(self):
    d = _make(_G1D, _MHD8)
    out = mhd.pressure(d)
    np.testing.assert_allclose(out.values[0, 0], _P_THERMAL, rtol=1e-10)

  def test_temp(self):
    d = _make(_G1D, _MHD8)
    out = mhd.temp(d)
    np.testing.assert_allclose(out.values[0, 0], _P_THERMAL / _RHO, rtol=1e-10)

  def test_sound(self):
    d = _make(_G1D, _MHD8)
    out = mhd.sound(d)
    expected = np.sqrt(_GAMMA * _P_THERMAL / _RHO)
    np.testing.assert_allclose(out.values[0, 0], expected, rtol=1e-10)

  def test_mach(self):
    d = _make(_G1D, _MHD8)
    out = mhd.mach(d)
    cs = np.sqrt(_GAMMA * _P_THERMAL / _RHO)
    np.testing.assert_allclose(out.values[0, 0], _VX / cs, rtol=1e-10)

  def test_mag_p_zero_field_gives_pure_gas_pressure(self):
    e = _P_THERMAL / (_GAMMA - 1) + 0.5 * _RHO * _VX**2
    values = np.array([[_RHO, _RHO * _VX, 0.0, 0.0, e, 0.0, 0.0, 0.0]])
    d = _make(_G1D, values)
    out = mhd.pressure(d)
    np.testing.assert_allclose(out.values[0, 0], _P_THERMAL, rtol=1e-10)

  def test_mu_0_is_forwarded_to_mag_pressure(self):
    d = _make(_G1D, _MHD8)
    out = mhd.mag_pressure(d, mu_0=2.0)
    np.testing.assert_allclose(out.values[0, 0], _MAG_P / 2.0)

  @needs_gkeyll
  def test_pressure_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      mhd.pressure(d)


class TestFiveMomentSetReused:

  def test_density_xvel_reused_from_five_moment(self):
    from postgkyl.diagnostics.mom import five_moment as fm
    assert mhd.density is fm.density
    assert mhd.xvel is fm.xvel
    assert mhd.yvel is fm.yvel
    assert mhd.zvel is fm.zvel
    assert mhd.vel is fm.vel


class TestVariables:

  def test_variables_table_has_exactly_the_old_mhd_vocabulary(self):
    assert set(mhd.VARIABLES) == {
        "density", "xvel", "yvel", "zvel", "vel", "Bx", "By", "Bz", "Bi",
        "magpressure", "pressure", "temp", "sound", "mach"
    }

  def test_variables_table_maps_to_public_functions(self):
    assert mhd.VARIABLES["Bx"] is mhd.bx
    assert mhd.VARIABLES["Bi"] is mhd.bi
    assert mhd.VARIABLES["magpressure"] is mhd.mag_pressure
    assert mhd.VARIABLES["density"] is mhd.density
