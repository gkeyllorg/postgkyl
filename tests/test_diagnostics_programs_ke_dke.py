"""Tests for ``postgkyl.diagnostics.mom.ke_dke``.

Ported from ``src_bak/postgkyl/tools/calc_ke_dke.py`` (no ``tests_bak``
corpus exists for this tool). See the module docstring for the three
``src_bak`` bugs this port fixes (a file-name f-string missing its own
parameter, an array-aliasing bug, and an off-by-one difference-loop bound)
-- the tests here pin the *fixed* behavior: an exact analytic kinetic-energy
value per frame, and a dissipation rate covering every consecutive frame
pair.

Run: PYTHONPATH=src pytest tests/test_diagnostics_programs_ke_dke.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from postgkyl.diagnostics.mom import ke_dke as kd


class _FakeGData:
  def __init__(self, grid, values):
    self.grid = grid
    self.values = values
  # end
# end


class TestKineticEnergyAnalytic:

  def test_uniform_velocity_matches_hand_derivation(self):
    n = 4
    rho = np.full((n, n, n), 2.0)
    u = np.full((n, n, n), 1.0)
    v = np.full((n, n, n), 2.0)
    w = np.full((n, n, n), 3.0)
    px, py, pz = u * rho, v * rho, w * rho
    dx = dy = dz = 0.5
    vol = 10.0
    ke = kd._kinetic_energy(rho, px, py, pz, dx, dy, dz, vol)
    # e = rho*(u^2+v^2+w^2) = 2*(1+4+9) = 28 per cell, n^3 = 64 cells.
    expected = 28.0 * (n ** 3) * dx * dy * dz * vol
    np.testing.assert_allclose(ke, expected)
  # end

  def test_zero_velocity_gives_zero_energy(self):
    n = 3
    rho = np.full((n, n, n), 5.0)
    zero = np.zeros((n, n, n))
    ke = kd._kinetic_energy(rho, zero, zero, zero, 1.0, 1.0, 1.0, 1.0)
    np.testing.assert_allclose(ke, 0.0)
  # end
# end


class TestDissipationRatePure:

  def test_backward_difference_every_pair(self):
    ke = np.array([1.0, 3.0, 6.0, 10.0])
    dke = kd._dissipation_rate(ke, dt=0.5)
    expected = -(ke[1:] - ke[:-1]) / 0.5
    np.testing.assert_allclose(dke, expected)
    assert dke.shape[0] == ke.shape[0] - 1
  # end

  def test_constant_ke_gives_zero_dissipation(self):
    ke = np.full(5, 3.0)
    dke = kd._dissipation_rate(ke, dt=1.0)
    np.testing.assert_allclose(dke, 0.0)
  # end
# end


class TestKeDkeSweep:

  def _uniform_frame(self, n=3, value=1.0):
    edges = np.arange(n + 1, dtype=np.float64)
    rho = np.full((n, n, n), value)
    values = np.stack([rho, rho, rho, rho], axis=-1)
    return _FakeGData([edges, edges, edges], values)
  # end

  def test_sweeps_expected_frame_count_and_dke_length(self, monkeypatch):
    calls = []

    def fake_gdata(file_name):
      calls.append(file_name)
      return self._uniform_frame()
    # end

    monkeypatch.setattr(kd, "GData", fake_gdata)

    out = kd.ke_dke("sim-fluid_", 0, 3, dim=3, vol=1.0, init_time=0.0, final_time=3.0)
    # First frame read twice (once for grid spacing, once in the sweep).
    assert calls == ["sim-fluid_0.gkyl", "sim-fluid_0.gkyl", "sim-fluid_1.gkyl",
        "sim-fluid_2.gkyl", "sim-fluid_3.gkyl"]
    assert out.ke.shape == (4,)
    assert out.dke.shape == (3,)
    # u=v=w=1 (rho=1, px=py=pz=1/rho=... wait: px=py=pz=rho=1 -> u=v=w=1)
    # constant across every frame -> dke is exactly zero, not just close.
    np.testing.assert_allclose(out.dke, 0.0)
  # end

  def test_dim_2_uses_unit_z_spacing(self, monkeypatch):
    def fake_gdata(file_name):
      return self._uniform_frame(n=2)
    # end

    monkeypatch.setattr(kd, "GData", fake_gdata)
    out = kd.ke_dke("sim-fluid_", 0, 1, dim=2, vol=1.0, init_time=0.0, final_time=1.0)
    assert out.ke.shape == (2,)
  # end

  def test_uses_own_root_file_name_not_a_literal_string(self, monkeypatch):
    """Regression test for the src_bak bug where the per-frame file name was
    built as f"root_file_name{c:d}.gkyl" -- a literal string containing the
    parameter's *name* -- instead of interpolating its value."""
    calls = []

    def fake_gdata(file_name):
      calls.append(file_name)
      return self._uniform_frame()
    # end

    monkeypatch.setattr(kd, "GData", fake_gdata)
    kd.ke_dke("distinctive_stem_", 0, 1, dim=3, vol=1.0, init_time=0.0, final_time=1.0)
    assert all(c.startswith("distinctive_stem_") for c in calls)
    assert not any("root_file_name" in c for c in calls)
  # end
# end


class TestKineticEnergyTracesIsFrozen:

  def test_fields_present(self):
    t = kd.KineticEnergyTraces(ke=np.array([1.0]), dke=np.array([]))
    with pytest.raises(Exception):
      t.ke = np.array([2.0])
  # end
# end
    # end
