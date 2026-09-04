"""Tests for ``postgkyl.diagnostics.gk.particle_balance``.

See ``test_diagnostics_programs_energy_balance.py`` for the shared testing
strategy (no ``tests_bak`` corpus exists for this app; the repo ships no
multi-file gyrokinetic particle-balance fixture set, so the full figure path
is exercised against synthetic per-file datasets stubbed through
``utils.GData``).

Run: PYTHONPATH=src pytest tests/test_diagnostics_programs_particle_balance.py -v
"""

from __future__ import annotations

import importlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from postgkyl.diagnostics.gk import utils as gk_utils

pb = importlib.import_module(
    "postgkyl.diagnostics.gk.particle_balance")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")


class _FakeGData:
  def __init__(self, grid, values, ctx=None):
    self._grid = grid
    self._values = values
    self.ctx = ctx or {}
  # end

  def get_grid(self):
    return self._grid
  # end

  def get_values(self):
    return self._values
  # end
# end


class _StubFiles:
  def __init__(self, tmp_path, monkeypatch):
    self._registry: dict[str, _FakeGData] = {}
    monkeypatch.setattr(gk_utils, "GData", self._dispatch)
  # end

  def _dispatch(self, file_name):
    return self._registry[file_name]
  # end

  def add(self, file_name: str, time, values) -> None:
    open(file_name, "w").close()
    self._registry[file_name] = _FakeGData([np.asarray(time)], np.asarray(values))
  # end
# end


@pytest.fixture
def stub(tmp_path, monkeypatch):
  return _StubFiles(tmp_path, monkeypatch)
# end


def _build_sim(stub, tmp_path, name="sim", species="ion", *, with_src=True,
    with_bflux=True, n=5):
  path = str(tmp_path) + "/"
  # A dynvector's grid is exactly one time stamp per recorded sample, not
  # N+1 cell edges like a field file (see io/gkyl_reader.py's _read_t2_v1).
  time = np.linspace(0.0, 1.0, n)

  # 2 components (M0, M1): a single-component array would collapse to 1-D
  # under np.squeeze, breaking the `v[:, _DENSITY_MOMENT]` indexing every
  # integrated-moments file family needs.
  fdot_vals = np.zeros((n, 2))
  fdot_vals[:, 0] = np.linspace(1.0, 2.0, n)
  stub.add(f"{path}{name}-{species}_fdot_integrated_moms.gkyl", time, fdot_vals)

  if with_src:
    src_vals = np.zeros((n, 2))
    src_vals[:, 0] = 0.1
    stub.add(f"{path}{name}-{species}_source_integrated_moms.gkyl", time, src_vals)
  # end
  if with_bflux:
    bflux_vals = np.zeros((n, 2))
    bflux_vals[:, 0] = 0.05
    stub.add(f"{path}{name}-{species}_bflux_xlower_integrated_HamiltonianMoments.gkyl",
        time, bflux_vals)
  # end
  return path
# end


class TestParticleBalanceErrorPure:

  def test_formula(self):
    fdot = np.array([2.0, 3.0])
    src = np.array([1.0, 1.0])
    bflux = np.array([0.5, 0.5])
    err = pb.particle_balance_error(fdot, src, bflux)
    np.testing.assert_allclose(err, src - bflux - fdot)
  # end
# end


class TestAccumulatePure:

  def test_first_use_copies_not_aliases(self):
    a = np.array([1.0, 2.0])
    out = pb._accumulate(None, a)
    out[0] = 99.0
    assert a[0] == 1.0
  # end

  def test_accumulates_sum(self):
    out = pb._accumulate(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(out, [4.0, 6.0])
  # end
# end


class TestResolvePure:

  def test_no_override_uses_default(self):
    assert pb._resolve("/p/", None, "default.gkyl", 0) == "default.gkyl"
  # end

  def test_override_substitutes_block(self):
    assert pb._resolve("/p/", "custom_*.gkyl", "unused", 3) == "/p/custom_3.gkyl"
  # end
# end


class TestGkParticleBalanceSynthetic:

  def test_full_path_with_src_and_bflux(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    fig, traces = pb.particle_balance("sim", "ion", path=path)
    try:
      assert traces.src is not None
      assert traces.bflux_tot is not None
      assert traces.mom_err is not None
      assert traces.time.shape[0] == 5
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_missing_source_and_bflux(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, with_src=False, with_bflux=False)
    fig, traces = pb.particle_balance("sim", "ion", path=path)
    try:
      assert traces.src is None
      assert traces.bflux_tot is None
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_relative_error_branch(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    f_vals = np.zeros((n, 2))
    f_vals[:, 0] = 10.0
    stub.add(f"{path}sim-ion_integrated_moms.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    dt_vals = np.full((n - 1, 1), 0.2)
    stub.add(f"{path}sim-dt.gkyl", dt_time, dt_vals)

    fig, traces = pb.particle_balance("sim", "ion", path=path, relative_error=True)
    try:
      assert traces.mom_err is None
      assert traces.mom_err_norm is not None
      assert traces.mom_err_norm.shape[0] == n - 1
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_relative_error_absy_saveas_and_show(self, stub, tmp_path):
    """Covers ``absy`` wrapping the relative-error ylabel in ``||`` together
    with ``saveas``/``show`` (default ``ylabel_string`` is non-empty on this
    branch, unlike the absolute-error branch's ``""`` default)."""
    path = _build_sim(stub, tmp_path)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    f_vals = np.zeros((n, 2))
    f_vals[:, 0] = 10.0
    stub.add(f"{path}sim-ion_integrated_moms.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    dt_vals = np.full((n - 1, 1), 0.2)
    stub.add(f"{path}sim-dt.gkyl", dt_time, dt_vals)

    out_path = str(tmp_path / "out.png")
    fig, traces = pb.particle_balance(
        "sim", "ion", path=path, relative_error=True, absy=True, show=True,
        saveas=out_path)
    try:
      assert traces.mom_err_norm is not None
      assert os.path.exists(out_path)
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_missing_required_fdot_file_raises(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    with pytest.raises(FileNotFoundError, match="fdot_integrated_moms"):
      pb.particle_balance("sim", "ion", path=path)
  # end
    # end

  def test_bflux_override_and_absy_logy(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, with_bflux=False)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    override_vals = np.zeros((n, 2))
    override_vals[:, 0] = -0.05
    override_name = f"{path}custom_bflux.gkyl"
    stub.add(override_name, time, override_vals)

    fig, traces = pb.particle_balance(
        "sim", "ion", path=path, bflux_files={"xlower": "custom_bflux.gkyl"},
        absy=True, logy=True)
    try:
      assert traces.bflux_tot is not None
      np.testing.assert_allclose(traces.bflux_tot, -0.05)
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_multiblock_sums_over_blocks(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    n = 4
    time = np.linspace(0.0, 1.0, n)
    for block in (0, 1):
      fdot_vals = np.zeros((n, 2))
      fdot_vals[:, 0] = 1.0
      stub.add(f"{path}sim_b{block}-ion_fdot_integrated_moms.gkyl", time, fdot_vals)
    # end
    fig, traces = pb.particle_balance("sim", "ion", path=path, multib="0,1")
    try:
      # Two blocks, each contributing fdot=1.0, sum to 2.0 everywhere.
      np.testing.assert_allclose(traces.fdot, 2.0)
    # end
    finally:
      plt.close(fig)
  # end
# end
    # end


class TestGkParticleBalanceRealFixtures:

  def test_real_fixture_particle_balance(self):
    required = ("_fdot_integrated_moms.gkyl",)
    if not any(
        any(f.endswith(suffix) for f in os.listdir(DATA)) for suffix in required):
      pytest.skip(
          "tests/test_data ships no gyrokinetic particle-balance file family "
          "(needs e.g. '<name>-<species>_fdot_integrated_moms.gkyl'); see "
          "TestGkParticleBalanceSynthetic for full-path coverage against "
          "stubbed data instead.")
    # end
    pytest.fail("fixture files appeared -- wire up a real-data assertion here")
  # end
# end
