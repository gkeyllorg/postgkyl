"""Tests for postgkyl.diagnostics.moments.multispecies -- energy decomposition and
current accumulation, folding the array-math analytic tests (formerly
tests_models_energetics.py) with the verb-level guard/inplace tests
(formerly part of tests_ops_physics.py)."""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.moments import multispecies as ms
from postgkyl.gdatastate.gdatastate import GDataState

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
    reason="no compiled Gkeyll (libg0core.so) found")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
F1 = os.path.join(DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")

_G1D = [np.array([0.0, 1.0])]
_GAMMA = 5.0 / 3.0


def _make(grid, values, **ctx):
  d = GDataState(ctx=ctx or None)
  d.push(list(grid), values)
  return d
# end


def _make_5mom(rho, vx, p):
  E = p / (_GAMMA - 1) + 0.5 * rho * vx**2
  return _make(_G1D, np.array([[rho, rho * vx, 0.0, 0.0, E]]))
# end


class TestEnergetics:
  def test_components_and_total(self):
    elc = _make_5mom(rho=1.0, vx=1.0, p=0.3)
    ion = _make_5mom(rho=1.0, vx=0.5, p=0.6)
    field = _make(_G1D, np.array([[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]))  # Ex=1, Bx=2

    out = ms.energetics(elc, ion, field)

    assert out.values.shape[-1] == 7
    pre_expected = 0.3
    kee_expected = 0.5 * 1.0 * 1.0**2
    pri_expected = 0.6
    kei_expected = 0.5 * 1.0 * 0.5**2
    esq_expected = 1.0**2 / 2.0
    bsq_expected = 2.0**2 / 2.0
    np.testing.assert_allclose(out.values[0, 0], pre_expected, rtol=1e-10)
    np.testing.assert_allclose(out.values[0, 1], kee_expected, rtol=1e-10)
    np.testing.assert_allclose(out.values[0, 2], pri_expected, rtol=1e-10)
    np.testing.assert_allclose(out.values[0, 3], kei_expected, rtol=1e-10)
    np.testing.assert_allclose(out.values[0, 4], esq_expected, rtol=1e-10)
    np.testing.assert_allclose(out.values[0, 5], bsq_expected, rtol=1e-10)
    total = (pre_expected + kee_expected + pri_expected + kei_expected
        + esq_expected + bsq_expected)
    np.testing.assert_allclose(out.values[0, 6], total, rtol=1e-10)
  # end

  def test_result_carries_field_grid(self):
    elc = _make_5mom(rho=1.0, vx=0.0, p=1.0)
    ion = _make_5mom(rho=1.0, vx=0.0, p=1.0)
    field = _make(_G1D, np.zeros((1, 6)))
    out = ms.energetics(elc, ion, field, inplace=True)
    assert out is field
  # end

  def test_component_layout(self):
    elc = _make_5mom(rho=1.0, vx=2.0, p=16.0 / 3.0)
    ion = _make_5mom(rho=1.0, vx=2.0, p=16.0 / 3.0)
    field = _make(_G1D, np.array([[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]]))
    out = ms.energetics(elc, ion, field)
    comps = out.values[0]
    np.testing.assert_allclose(comps[0], 16.0 / 3.0)  # electron thermal
    np.testing.assert_allclose(comps[1], 2.0)          # electron kinetic
    np.testing.assert_allclose(comps[2], 16.0 / 3.0)  # ion thermal
    np.testing.assert_allclose(comps[3], 2.0)          # ion kinetic
    np.testing.assert_allclose(comps[4], 0.5)          # electric
    np.testing.assert_allclose(comps[5], 2.0)          # magnetic
    np.testing.assert_allclose(comps[6], comps[:6].sum())  # total
  # end

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    elc = _make_5mom(rho=1.0, vx=0.0, p=1.0)
    field = _make(_G1D, np.zeros((1, 6)))
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      ms.energetics(d, elc, field)
    # end
  # end
# end


class TestAccumulateCurrent:
  def _species(self):
    return _make(_G1D, np.array([[1.0, 2.0, -3.0]]))
  # end

  def test_default_negates(self):
    d = self._species()
    out = ms.accumulate_current(d)
    np.testing.assert_allclose(out.values, -d.values)
  # end

  def test_qbym_scales_by_charge_over_mass(self):
    d = self._species()
    out = ms.accumulate_current(d, qbym=True, charge=2.0, mass=4.0)
    np.testing.assert_allclose(out.values, 0.5 * d.values)
  # end

  def test_qbym_negative_charge(self):
    d = self._species()
    out = ms.accumulate_current(d, qbym=True, charge=-1.0, mass=2.0)
    np.testing.assert_allclose(out.values, -0.5 * d.values)
  # end

  def test_qbym_without_mass_raises(self):
    d = self._species()
    with pytest.raises(ValueError, match="qbym"):
      ms.accumulate_current(d, qbym=True, charge=2.0)  # mass missing
    # end
  # end

  def test_qbym_without_charge_raises(self):
    d = self._species()
    with pytest.raises(ValueError, match="qbym"):
      ms.accumulate_current(d, qbym=True, mass=4.0)  # charge missing
    # end
  # end

  def test_inplace_mutates(self):
    d = self._species()
    out = ms.accumulate_current(d, inplace=True)
    assert out is d
  # end

  def test_grid_passed_through(self):
    d = self._species()
    out = ms.accumulate_current(d)
    np.testing.assert_allclose(out.grid[0], _G1D[0])
  # end

  @needs_gkeyll
  def test_rejects_modal_data(self):
    d = pg.load(F1)
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      ms.accumulate_current(d)
    # end
  # end
# end


class TestAccumulateCurrentPrivateHelperFallback:
  """``_accumulate_current`` (the moved array-level ``models.energetics``
  function) still silently falls back to the ``qbym=False`` formula when
  ``mass``/``charge`` are missing -- the public verb now refuses that
  combination before ever calling through (see ``accumulate_current``'s own
  qbym guard above), so this behavior is only reachable by calling the
  private helper directly, exactly as the pre-restructure
  ``tests_models_energetics.py`` did against ``models.accumulate_current``."""

  def test_qbym_without_mass_falls_back_to_negation(self):
    values = np.array([[1.0, 2.0, 3.0]])
    _, out = ms._accumulate_current(_G1D, values, qbym=True, charge=-1.0,
        mass=None)
    np.testing.assert_allclose(out, -values)
  # end

  def test_qbym_without_charge_falls_back_to_negation(self):
    values = np.array([[1.0, 2.0, 3.0]])
    _, out = ms._accumulate_current(_G1D, values, qbym=True, charge=None,
        mass=1.0)
    np.testing.assert_allclose(out, -values)
  # end
# end
