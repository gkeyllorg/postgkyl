"""Tests for the radial transport diagnostic ``pg.gk.transport``.

Three levels are checked:

- ``transport_coefficients`` and ``gyro_bohm_normalize``, plain algebra on
  radial profiles, against manufactured profiles;
- ``transport`` end to end on a synthetic 3x simulation whose fluxes and
  profiles are known in closed form;
- the generated ``gk_transport`` command.

The synthetic simulation has profiles linear in x and uniform in (y, z),
``n = N0 - GN x``, ``T = T0 - GT x``, ``M2 = M20 + GM2 x``, a potential
``phi = s(frame) (a y + e z)`` and uniform geometry, so that
``v_E^x = s (b_y e - b_z a)/(J B)`` is uniform. Every weak product and
average is then exact in p1, and so are the expected values::

  Gamma = n v_E,  Q = (m/2) M2 v_E,  q = Q - conv T Gamma,
  D = Gamma/(g^xx GN),  chi = q/(n g^xx GT),

at the radial interpolation points, ``s`` averaged over the frames.
"""

from __future__ import annotations

import importlib
import os

from click.testing import CliRunner
import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.cli.app import cli
from postgkyl.diagnostics.gk import quantity as qmod
from postgkyl.diagnostics.gk.registry import gk_quant_registry

# The package re-exports the ``transport`` function under the module's name.
gkt = importlib.import_module("postgkyl.diagnostics.gk.transport")

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
                                  reason="requires Gkeyll")

# Exact operations on O(1)-relative data: only double round-off remains.
RTOL = 1e-10


class TestTransportCoefficients:
  x = np.linspace(0.0, 1.0, 9)
  n0, gn, t0, gt, gxx = 2.0e19, 1.5e19, 3.0e-17, 2.0e-17, 1.7

  def _profiles(self):
    n = self.n0 - self.gn * self.x
    T = self.t0 - self.gt * self.x
    gamma = 4.0e19 * (1.0 + self.x)
    return gamma, 9.0 * gamma * T, n, T

  def test_manufactured_profiles(self):
    gamma, Q, n, T = self._profiles()
    ones = np.ones_like(n)
    res = gkt.transport_coefficients(gamma,
                                     Q,
                                     n,
                                     T,
                                     self.gxx,
                                     -self.gn * ones,
                                     -self.gt * ones,
                                     conv=1.5)
    q = Q - 1.5 * T * gamma
    np.testing.assert_allclose(res["q"], q, rtol=1e-14)
    np.testing.assert_allclose(res["D"],
                               gamma / (self.gxx * self.gn),
                               rtol=1e-14)
    np.testing.assert_allclose(res["chi"],
                               q / (n * self.gxx * self.gt),
                               rtol=1e-14)

  def test_conv_zero_keeps_the_energy_flux(self):
    gamma, Q, n, T = self._profiles()
    ones = np.ones_like(n)
    res = gkt.transport_coefficients(gamma,
                                     Q,
                                     n,
                                     T,
                                     self.gxx,
                                     -ones,
                                     -ones,
                                     conv=0.0)
    np.testing.assert_array_equal(res["q"], Q)

  def test_flat_gradients_are_masked(self):
    gamma, Q, n, T = self._profiles()
    dn_dx = -self.gn * np.cos(np.pi * self.x)  # Vanishes at x = 0.5.
    res = gkt.transport_coefficients(gamma, Q, n, T, self.gxx, dn_dx,
                                     np.zeros_like(n))
    assert np.isnan(res["D"][4])
    assert np.all(np.isfinite(np.delete(res["D"], 4)))
    assert np.all(np.isnan(res["chi"])), "a flat T profile has no chi"


class TestGyroBohm:
  x = np.linspace(0.0, 1.0, 5)
  m_i, q_i, m_e = 3.343e-27, 1.602e-19, 9.109e-31
  gxx = 1.7

  def _window(self, n, T, dn_dx, dT_dx, mass, charge, bmag=2.0):
    res = {
        "frames": [1, 2],
        "n": n,
        "T": T,
        "dn_dx": dn_dx,
        "dT_dx": dT_dx,
        "gxx": self.gxx * np.ones_like(n),
        "bmag": bmag * np.ones_like(n),
        "mass": mass,
        "charge": charge
    }
    res.update(
        gkt.transport_coefficients(3.0e19 * np.ones_like(n),
                                   2.0e3 * np.ones_like(n), n, T, res["gxx"],
                                   dn_dx, dT_dx))
    return res

  def _ion(self, **kw):
    n, T = 2.0e19 * (1.0 - 0.3 * self.x), 3.0e-17 * (1.0 - 0.5 * self.x)
    return self._window(n, T, -0.6e19 * np.ones_like(n),
                        -1.5e-17 * np.ones_like(n), self.m_i, self.q_i, **kw)

  def _elc(self):
    n, T = 2.0e19 * (1.0 - 0.3 * self.x), 5.0e-17 * (1.0 - 0.2 * self.x)
    return self._window(n, T, -0.6e19 * np.ones_like(n),
                        -1.0e-17 * np.ones_like(n), self.m_e, -self.q_i)

  def _expected(self, res, Te, bmag=2.0):
    c_s = np.sqrt(Te / self.m_i)
    D0 = (c_s * self.m_i / (self.q_i * bmag))**2 * c_s
    L_n = -res["n"] / (np.sqrt(self.gxx) * res["dn_dx"])
    L_T = -res["T"] / (np.sqrt(self.gxx) * res["dT_dx"])
    return res["D"] * L_n / D0, res["chi"] * L_T / D0

  def test_adiabatic_electrons_use_ti_over_te(self):
    results = {"ion": [self._ion()]}
    gkt.gyro_bohm_normalize(results, ti_over_te=2.0)
    res = results["ion"][0]
    D_gB, chi_gB = self._expected(res, res["T"] / 2.0)
    np.testing.assert_allclose(res["D_gB"], D_gB, rtol=1e-12)
    np.testing.assert_allclose(res["chi_gB"], chi_gB, rtol=1e-12)

  def test_kinetic_electrons_use_their_temperature_and_the_ion_mass(self):
    results = {"elc": [self._elc()], "ion": [self._ion()]}
    gkt.gyro_bohm_normalize(results, ti_over_te=7.0)  # Unused: elc listed.
    Te = results["elc"][0]["T"]
    for species in ("elc", "ion"):
      res = results[species][0]
      D_gB, chi_gB = self._expected(res, Te)
      np.testing.assert_allclose(res["D_gB"], D_gB, rtol=1e-12)
      np.testing.assert_allclose(res["chi_gB"], chi_gB, rtol=1e-12)

  def test_reference_values_replace_the_profiles(self):
    results = {"ion": [self._ion(bmag=5.0)]}
    gkt.gyro_bohm_normalize(results, te_ref=4.0e-17, bmag_ref=2.0)
    res = results["ion"][0]
    D_gB, chi_gB = self._expected(res, 4.0e-17, bmag=2.0)
    np.testing.assert_allclose(res["D_gB"], D_gB, rtol=1e-12)
    np.testing.assert_allclose(res["chi_gB"], chi_gB, rtol=1e-12)

  def test_errors(self):
    with pytest.raises(ValueError, match="ion species"):
      gkt.gyro_bohm_normalize({"elc": [self._elc()]})
    res = self._ion()
    res["bmag"] = None
    with pytest.raises(ValueError, match="geo_int_bmag"):
      gkt.gyro_bohm_normalize({"ion": [res]})
    res = self._ion()
    res["mass"] = None
    with pytest.raises(ValueError, match="mass option"):
      gkt.gyro_bohm_normalize({"ion": [res]})


# ----------------------------------------------- synthetic 3x simulation
_NAME, _SPECIES, _FRAMES = "gksynth", "ion", (3, 4, 5)
_CELLS, _LENGTHS = (6, 4, 3), (0.8, 1.3, 2.1)
_PSI0 = 2.0**-1.5
_MASS, _CHARGE = 3.343e-27, 1.0
_N0, _GN = 3.0e19, 1.2e19
_T0, _GT = 4.0e-17, 1.5e-17
_M20, _GM2 = 5.0e10, 2.0e10
_PHI_Y, _PHI_Z = 250.0, -40.0
_B_Y, _B_Z = 0.3, 0.95
_JACOBGEO, _JACOBTOT_INV, _GXX, _BMAG = 1.6, 0.45, 2.2, 1.9
_VE_X = (_B_Y * _PHI_Z - _B_Z * _PHI_Y) * _JACOBTOT_INV


def _scale(frame):
  """The potential grows with the frame so the time average is not one
  frame's value."""
  return 1.0 + 0.1 * (frame - _FRAMES[0])


def _grid():
  return [np.linspace(0.0, L, n + 1) for L, n in zip(_LENGTHS, _CELLS)]


def _linear(comps, scale=1.0):
  """Native p1 field whose components are ``offset + sum_d slope_d x_d``."""
  centers = np.meshgrid(*[(g[:-1] + g[1:]) / 2 for g in _grid()], indexing="ij")
  values = np.zeros((*_CELLS, 8 * len(comps)))
  for comp, (offset, slopes) in enumerate(comps):
    values[...,
           8 * comp] = scale * (offset +
                                sum(s * c
                                    for s, c in zip(slopes, centers))) / _PSI0
    for d, slope in enumerate(slopes):
      dx = _LENGTHS[d] / _CELLS[d]
      values[..., 8 * comp + 1 + d] = (scale * slope * dx /
                                       (2.0 * np.sqrt(3.0) * _PSI0))
  data = pg.GData(
      ctx={
          "basis_type": "serendipity",
          "poly_order": 1,
          "value_form": "modal",
          "cells": np.array(_CELLS),
          "mass": _MASS,
          "charge": _CHARGE
      })
  data.push(_grid(), gpython.GkylArray.from_numpy(values))
  return data


def _const(*avgs):
  return _linear([(a, (0.0, 0.0, 0.0)) for a in avgs])


_FACTORIES = {
    "geo_int_jacobgeo":
    lambda f: _const(_JACOBGEO),
    "geo_int_jacobtot_inv":
    lambda f: _const(_JACOBTOT_INV),
    "geo_int_b_i":
    lambda f: _const(0.0, _B_Y, _B_Z),
    "geo_int_gij":
    lambda f: _const(_GXX, 0.0, 0.0, 1.0, 0.0, 1.0),
    "geo_int_bmag":
    lambda f: _const(_BMAG),
    "field":
    lambda f: _linear([(0.0, (0.0, _PHI_Y, _PHI_Z))], _scale(f)),
    "M0":
    lambda f: _linear([(_N0, (-_GN, 0.0, 0.0))]),
    "M2":
    lambda f: _linear([(_M20, (_GM2, 0.0, 0.0))]),
    "MaxwellianMoments":
    lambda f: _linear([(_N0, (-_GN, 0.0, 0.0)), (0.0, (0.0, 0.0, 0.0)),
                       (_T0 / _MASS, (-_GT / _MASS, 0.0, 0.0))]),
}


def _load(file_name, *args, **kwargs):
  base = os.path.basename(str(file_name))[len(_NAME) + 1:-len(".gkyl")]
  if base.startswith("geo_"):
    return _FACTORIES[base](None)
  stem, _, frame = base.rpartition("_")
  stem = stem.removeprefix(f"{_SPECIES}_")
  return _FACTORIES[stem](int(frame))


@pytest.fixture
def synthetic_sim(tmp_path, monkeypatch):
  """Marker files for the synthetic simulation; their data are served by
  :func:`_load` in place of the reader."""
  for stem in _FACTORIES:
    if stem.startswith("geo_"):
      (tmp_path / f"{_NAME}-{stem}.gkyl").touch()
    else:
      prefix = f"{_NAME}-" + ("" if stem == "field" else f"{_SPECIES}_")
      for f in _FRAMES:
        (tmp_path / f"{prefix}{stem}_{f}.gkyl").touch()
  monkeypatch.setattr(qmod, "GData", _load)
  monkeypatch.setattr(gkt, "GData", _load)
  return str(tmp_path)


def _expected(x, conv=1.5, scale=None):
  scale = np.mean([_scale(f) for f in _FRAMES]) if scale is None else scale
  n = _N0 - _GN * x
  T = _T0 - _GT * x
  gamma = n * _VE_X * scale
  Q = 0.5 * _MASS * (_M20 + _GM2 * x) * _VE_X * scale
  q = Q - conv * T * gamma
  return {
      "n": n,
      "T": T,
      "gamma": gamma,
      "Q": Q,
      "q": q,
      "gxx": _GXX * np.ones_like(x),
      "D": gamma / (_GXX * _GN),
      "chi": q / (n * _GXX * _GT)
  }


def _centers(edges):
  return (edges[:-1] + edges[1:]) / 2


def test_electrostatic_run_falls_back_to_the_ExB_flux(synthetic_sim):
  """Without apar output the total fluxes use their electrostatic source
  combination, although some electromagnetic sources (M1 from the
  Maxwellian moments) exist."""
  for qname in ("part_flux", "energy_flux"):
    combo_idx, frames = gk_quant_registry.get(qname).get_avail_source(
        synthetic_sim + "/", _NAME, _SPECIES, ":")
    assert combo_idx == 1, qname
    assert frames == list(_FRAMES)


@needs_gkeyll
class TestTransport:

  def test_time_averaged_profiles(self, synthetic_sim):
    outputs = list(gkt.TRANSPORT_OUTPUTS)
    outputs = [o for o in outputs if o not in gkt.GYRO_BOHM_OUTPUTS]
    datasets = pg.gk.transport(_NAME,
                               _SPECIES,
                               path=synthetic_sim,
                               outputs=outputs)
    assert [d.tag for d in datasets] == [f"transport_{o}" for o in outputs]
    x = _centers(datasets[0].grid[0])
    for output, data in zip(outputs, datasets):
      np.testing.assert_allclose(data.values[..., 0],
                                 _expected(x)[output],
                                 rtol=RTOL)
    assert datasets[0].ctx["frame"] == _FRAMES[-1]

  def test_frame_selection_and_per_frame(self, synthetic_sim):
    datasets = pg.gk.transport(_NAME,
                               _SPECIES,
                               "4:6",
                               path=synthetic_sim,
                               outputs=["gamma"],
                               per_frame=True)
    assert [d.ctx["frame"] for d in datasets] == [4, 5]
    x = _centers(datasets[1].grid[0])
    np.testing.assert_allclose(datasets[1].values[..., 0],
                               _expected(x, scale=_scale(5))["gamma"],
                               rtol=RTOL)
    assert datasets[1].label.endswith(" f5")

  def test_uniform_fluxes_have_no_turbulent_part(self, synthetic_sim):
    (gamma, ) = pg.gk.transport(_NAME,
                                _SPECIES,
                                path=synthetic_sim,
                                outputs=["gamma"],
                                fluct="yz")
    np.testing.assert_allclose(gamma.values, 0.0, atol=1e-10 * _N0 * abs(_VE_X))

  def test_gyro_Bohm_with_adiabatic_electrons(self, synthetic_sim):
    """``T_e = T_i/ti_over_te``, B from geo_int_bmag, ``m_i``/``q_i`` from
    the files."""
    D_gB, chi_gB = pg.gk.transport(_NAME,
                                   _SPECIES,
                                   path=synthetic_sim,
                                   outputs=["D_gB", "chi_gB"],
                                   ti_over_te=2.0)
    x = _centers(D_gB.grid[0])
    exp = _expected(x)
    c_s = np.sqrt(0.5 * exp["T"] / _MASS)
    D0 = (c_s * _MASS / (_CHARGE * _BMAG))**2 * c_s
    L_n = exp["n"] / (np.sqrt(_GXX) * _GN)
    L_T = exp["T"] / (np.sqrt(_GXX) * _GT)
    np.testing.assert_allclose(D_gB.values[..., 0],
                               exp["D"] * L_n / D0,
                               rtol=RTOL)
    np.testing.assert_allclose(chi_gB.values[..., 0],
                               exp["chi"] * L_T / D0,
                               rtol=RTOL)

  def test_unknown_output_is_rejected(self, synthetic_sim):
    with pytest.raises(ValueError, match="unknown output"):
      pg.gk.transport(_NAME, _SPECIES, path=synthetic_sim, outputs=["nope"])

  def test_cli_command(self, synthetic_sim):
    """The generated command runs the same function; the values are
    covered above."""
    result = CliRunner().invoke(cli, [
        "gk_transport", "--name", _NAME, "--species", _SPECIES, "--path",
        synthetic_sim, "--outputs", "D", "--outputs", "chi", "info"
    ])
    assert result.exit_code == 0, result.output
    assert "(transport_D#" in result.output
    assert "(transport_chi#" in result.output
