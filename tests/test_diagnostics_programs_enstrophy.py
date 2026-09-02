"""Tests for ``postgkyl.diagnostics.moments.enstrophy``.

Ported from ``src_bak/postgkyl/tools/calc_enstrophy.py`` (no ``tests_bak``
corpus exists for this tool). The pure per-frame math (``_enstrophy_terms``)
is checked against an analytic velocity field where the curl and the
velocity-gradient tensor are hand-computable exactly (linear-in-coordinate
components, so ``np.gradient(..., edge_order=2)`` on a uniform grid
reproduces the analytic derivative exactly); the frame-sweep wiring
(``enstrophy``) is exercised against a synthetic multi-frame file family
stubbed through ``postgkyl.diagnostics.moments.enstrophy.GData`` -- the repo ships
no multi-frame 3-D five-moment ``.gkyl`` fixture family for this tool, so no
real-fixture path is attempted (see this layer's report).

Run: PYTHONPATH=src pytest tests/test_diagnostics_programs_enstrophy.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from postgkyl.diagnostics.moments import enstrophy as ens


class _FakeGData:
  def __init__(self, grid, values):
    self.grid = grid
    self.values = values
  # end
# end


class TestEnstrophyTermsAnalytic:
  """u = x, v = y, w = -2z (irrotational, incompressible): curl is exactly
  zero everywhere, and the velocity-gradient tensor's diagonal is constant
  (1, 1, -2) everywhere, so both integrals are exactly computable by hand."""

  def _field(self, n=4, rho0=2.0):
    dx = dy = dz = 1.0
    coords = np.arange(n, dtype=np.float64)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    rho = np.full((n, n, n), rho0)
    u, v, w = x, y, -2.0 * z
    px, py, pz = u * rho, v * rho, w * rho
    return rho, px, py, pz, dx, dy, dz
  # end

  def test_curl_is_zero_for_irrotational_field(self):
    rho, px, py, pz, dx, dy, dz = self._field()
    enstrophy_val, _ = ens._enstrophy_terms(rho, px, py, pz, dx, dy, dz)
    np.testing.assert_allclose(enstrophy_val, 0.0, atol=1e-10)
  # end

  def test_incompressible_term_matches_hand_derivation(self):
    n = 4
    rho, px, py, pz, dx, dy, dz = self._field(n=n, rho0=2.0)
    _, incompressible = ens._enstrophy_terms(rho, px, py, pz, dx, dy, dz)
    # diag(grad) = (1, 1, -2) everywhere -> trace(M^T (*) M) = 1^2+1^2+(-2)^2 = 6.
    # incom_mag = 6 * rho = 12, summed only over the (n-1)^3 sub-cube the
    # nested loop's `range(n - 1)` bound reaches (a quirk preserved verbatim
    # from src_bak -- see the module docstring), times dx*dy*dz = 1.
    expected = 6.0 * 2.0 * (n - 1) ** 3
    np.testing.assert_allclose(incompressible, expected)
  # end

  def test_zero_velocity_gives_zero_both_terms(self):
    n = 3
    rho = np.full((n, n, n), 1.0)
    zero = np.zeros((n, n, n))
    enstrophy_val, incompressible = ens._enstrophy_terms(
        rho, zero, zero, zero, 1.0, 1.0, 1.0)
    np.testing.assert_allclose(enstrophy_val, 0.0)
    np.testing.assert_allclose(incompressible, 0.0)
  # end
# end


class TestEnstrophySweep:
  """Frame-sweep wiring: ``enstrophy()`` reads ``stem{frame}.ext`` for each
  frame in ``[init_frame, final_frame]`` and stacks the per-frame results."""

  def test_sweeps_expected_frame_range(self, monkeypatch):
    n = 3
    dx = dy = dz = 1.0
    coords = np.arange(n, dtype=np.float64)
    edges = np.arange(n + 1, dtype=np.float64)
    rho = np.full((n, n, n), 1.0)

    calls = []

    def fake_gdata(file_name):
      calls.append(file_name)
      values = np.stack([rho, rho, rho, rho], axis=-1)  # rho, px=py=pz=rho
      return _FakeGData([edges, edges, edges], values)
    # end

    monkeypatch.setattr(ens, "GData", fake_gdata)

    out = ens.enstrophy("sim-fluid_", 2, 4, extension="dat")
    # The first frame is read twice: once up front for the grid spacing,
    # then again inside the sweep loop.
    assert calls == ["sim-fluid_2.dat", "sim-fluid_2.dat", "sim-fluid_3.dat",
        "sim-fluid_4.dat"]
    assert out.enstrophy.shape == (3,)
    assert out.incompressible_enstrophy.shape == (3,)
    # u = v = w = px/rho = 1 (constant) -> zero curl and zero gradient.
    np.testing.assert_allclose(out.enstrophy, 0.0)
    np.testing.assert_allclose(out.incompressible_enstrophy, 0.0)
  # end

  def test_single_frame_range(self, monkeypatch):
    n = 3
    edges = np.arange(n + 1, dtype=np.float64)
    rho = np.full((n, n, n), 1.0)

    def fake_gdata(file_name):
      values = np.stack([rho, rho, rho, rho], axis=-1)
      return _FakeGData([edges, edges, edges], values)
    # end

    monkeypatch.setattr(ens, "GData", fake_gdata)
    out = ens.enstrophy("sim-fluid_", 0, 0)
    assert out.enstrophy.shape == (1,)
  # end
# end


class TestEnstrophyTracesIsFrozen:

  def test_fields_present(self):
    t = ens.EnstrophyTraces(enstrophy=np.array([1.0]),
        incompressible_enstrophy=np.array([2.0]))
    with pytest.raises(Exception):
      t.enstrophy = np.array([3.0])
  # end
# end
    # end
