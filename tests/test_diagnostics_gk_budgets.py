"""Tests for ``postgkyl.diagnostics.gk.budgets``.

The interval integrals are checked against antiderivatives: the trapezoid
rule is exact for a linear rate, overestimates ``t^2`` over ``[a, b]`` by
``(b - a) h^2 / 6`` (``h`` the sample spacing), and a constant rate's
cumulative integral is linear, so interpolating it at an edge between two
samples is exact. The run-level commands read the synthetic run of
``gk_tokamak_fixture`` with rates linear in time, whose budgets integrate in
closed form.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

import gk_tokamak_fixture as fx
from postgkyl import gpython
from postgkyl.diagnostics.gk import budgets

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


def test_device_factor():
  assert budgets.device_factor(False) == pytest.approx(2 * np.pi, rel=1e-15)
  assert budgets.device_factor(True) == pytest.approx(4 * np.pi, rel=1e-15)


def test_interval_integrals_exact_for_linear_rate():
  t = np.linspace(0.0, 1.0, 11)
  edges = np.array([0.0, 0.3, 0.7, 1.0])  # on samples
  antiderivative = lambda s: 3 * s + s**2  # of 3 + 2 t  # noqa: E731
  expected = np.diff(antiderivative(edges))
  got = budgets.interval_integrals(t, 3 + 2 * t, edges)
  np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0)


def test_interval_integrals_trapezoid_error_of_a_quadratic():
  h = 0.05
  t = np.arange(0.0, 1.0 + h / 2, h)
  edges = np.array([0.0, 0.25, 1.0])
  exact = np.diff(edges**3 / 3)
  got = budgets.interval_integrals(t, t**2, edges)
  np.testing.assert_allclose(got - exact,
                             np.diff(edges) * h**2 / 6,
                             rtol=1e-10,
                             atol=1e-15)


def test_interval_integrals_constant_rate_between_samples():
  t = np.array([0.0, 0.4, 0.5, 1.3, 2.0])  # nonuniform
  edges = np.array([0.1, 0.45, 1.9])  # between samples
  got = budgets.interval_integrals(t, np.full(t.size, 2.5), edges)
  np.testing.assert_allclose(got, 2.5 * np.diff(edges), rtol=1e-14)


def test_imbalance_values_and_nan_rules():
  src = np.array([2.0, 3.0, 0.0, 1.0, 0.0])
  out = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
  gain = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
  got = budgets.imbalance(src, out, gain)
  # (2-1-0)/2; closes; a whole outflow unaccounted; nothing recorded; zeros
  np.testing.assert_array_equal(got[:3], [0.5, 0.0, -1.0])
  assert np.isnan(got[3]) and np.isnan(got[4])


def test_interval_budget_skips_samples_holding_nothing():
  t = np.linspace(0.0, 1.0, 11)
  source = 2 + t
  outflow = 1 + 0.5 * t
  residual = np.full(t.size, 0.1)
  for a in (source, outflow, residual):
    a[:2] = 0.0  # the first two samples hold nothing yet
  got = budgets.interval_budget(t, source, outflow, residual,
                                np.array([0.0, 0.5, 1.0]), "particle")
  np.testing.assert_allclose(got.edges, [0.2, 0.5, 1.0], rtol=1e-15)
  S = lambda a, b: 2 * (b - a) + (b**2 - a**2) / 2  # noqa: E731
  O = lambda a, b: (b - a) + (b**2 - a**2) / 4  # noqa: E731
  e = got.edges
  src = np.array([S(e[0], e[1]), S(e[1], e[2])])
  out = np.array([O(e[0], e[1]), O(e[1], e[2])])
  res = 0.1 * np.diff(e)
  np.testing.assert_allclose(got.source, src, rtol=1e-13)
  np.testing.assert_allclose(got.outflow, out, rtol=1e-13)
  np.testing.assert_allclose(got.residual, res, rtol=1e-13)
  np.testing.assert_allclose(got.gain, src - out - res, rtol=1e-13)
  np.testing.assert_allclose(got.imbalance, res / src, rtol=1e-12)
  np.testing.assert_allclose(got.midpoints, [0.35, 0.75], rtol=1e-15)
  assert "particle budget over 2 intervals" in str(got)


def test_interval_budget_that_closes_has_no_imbalance():
  t = np.linspace(0.0, 2.0, 9)
  got = budgets.interval_budget(t, 1 + t**2, 0.5 * t, np.zeros(t.size),
                                np.array([0.0, 1.0, 2.0]))
  np.testing.assert_array_equal(got.imbalance, [0.0, 0.0])


def test_interval_budget_with_fewer_than_two_edges_is_empty():
  t = np.linspace(0.0, 1.0, 5)
  ones = np.ones(t.size)
  got = budgets.interval_budget(t, ones, ones, 0 * ones, np.array([5.0]))
  assert got.edges.size == 0 and got.imbalance.size == 0
  assert "fewer than two" in str(got)


# --------------------------------------------------------- run-level reads
def _hamiltonian(m0, h):
  """``(n, 3)`` HamiltonianMoments samples: M0, M1 = 0, H."""
  return np.stack([m0, np.zeros_like(m0), h], axis=-1)


def _plate_rates():
  t = fx.SAMPLES
  return {
      "b2-elc_bflux_yupper_integrated_HamiltonianMoments":
      _hamiltonian(1 + t, 10 * t),
      "b2-ion_bflux_yupper_integrated_HamiltonianMoments":
      _hamiltonian(2 * t, 5 + t**2),
  }


@needs_gkeyll
def test_target_fluxes(tmp_path):
  path = fx.write_run(tmp_path / "run", _plate_rates())
  t = fx.SAMPLES[1:]
  got = budgets.target_fluxes(fx.NAME, ["elc", "ion"], path=str(path))
  assert [p.name for p in got.plates] == ["lower inner target"]
  plate = got.plates[0]
  assert plate.blocks == (2, ) and plate.face == "yupper"
  np.testing.assert_array_equal(got.time, t)
  np.testing.assert_allclose(plate.particles, 1 + 3 * t, rtol=1e-15)
  np.testing.assert_allclose(plate.power, 5 + 10 * t + t**2, rtol=1e-15)
  ring = budgets.target_fluxes(fx.NAME, ["elc", "ion"],
                               path=str(path),
                               full_torus=True)
  np.testing.assert_allclose(ring.plates[0].power,
                             2 * np.pi * (5 + 10 * t + t**2),
                             rtol=1e-15)
  assert "lower inner target (blocks 2, yupper)" in str(got)


@needs_gkeyll
def test_target_fluxes_leave_out_a_plate_without_files(tmp_path):
  path = fx.write_run(tmp_path / "run")
  got = budgets.target_fluxes(fx.NAME, ["elc"], path=str(path))
  assert got.plates == () and got.time.size == 0


def _particle_rates():
  """S = 2 + t (block 0), outflow 1 + t / 2 (block 2's plate and block 1's
  psi-upper face), gain 0.9 + t / 2 (blocks 0 and 1): residual 0.1."""
  t = fx.SAMPLES
  z = np.zeros(t.size)
  return {
      "b0-elc_source_integrated_HamiltonianMoments": _hamiltonian(2 + t, z),
      "b2-elc_bflux_yupper_integrated_HamiltonianMoments":
      _hamiltonian(1 + z, z),
      "b1-elc_bflux_xupper_integrated_HamiltonianMoments":
      _hamiltonian(0.5 * t, z),
      "b0-elc_fdot_integrated_HamiltonianMoments": _hamiltonian(0.9 + z, z),
      "b1-elc_fdot_integrated_HamiltonianMoments": _hamiltonian(0.5 * t, z),
      "b2-elc_fdot_integrated_HamiltonianMoments": _hamiltonian(z, z),
  }


@needs_gkeyll
def test_frame_times(tmp_path):
  path = fx.write_run(tmp_path / "run")
  frames, times = budgets.frame_times(fx.NAME, "elc", path=str(path))
  np.testing.assert_array_equal(frames, [0, 1, 2])
  np.testing.assert_array_equal(times, fx.FRAME_TIMES)


@needs_gkeyll
def test_balance_intervals(tmp_path):
  path = fx.write_run(tmp_path / "run", _particle_rates())
  got = budgets.balance_intervals(fx.NAME, ["elc"], path=str(path))
  # balance() skips t = 0: the first interval starts at the next sample.
  np.testing.assert_allclose(got.edges, [0.1, 0.5, 1.0], rtol=1e-15)
  e = got.edges
  S = lambda a, b: 2 * (b - a) + (b**2 - a**2) / 2  # noqa: E731
  O = lambda a, b: (b - a) + (b**2 - a**2) / 4  # noqa: E731
  src = np.array([S(e[0], e[1]), S(e[1], e[2])])
  out = np.array([O(e[0], e[1]), O(e[1], e[2])])
  res = 0.1 * np.diff(e)
  np.testing.assert_allclose(got.source, src, rtol=1e-13)
  np.testing.assert_allclose(got.outflow, out, rtol=1e-13)
  np.testing.assert_allclose(got.residual, res, rtol=1e-12)
  np.testing.assert_allclose(got.imbalance, res / src, rtol=1e-11)
  whole = budgets.balance_intervals(fx.NAME, ["elc"],
                                    path=str(path),
                                    whole_device=True)
  # A half domain: 2 pi, times 2 for the mirror image.
  np.testing.assert_allclose(whole.source, 4 * np.pi * src, rtol=1e-13)
  np.testing.assert_allclose(whole.imbalance, got.imbalance, rtol=1e-15)
  assert whole.whole_device
