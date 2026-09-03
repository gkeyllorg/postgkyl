"""Tests for ``postgkyl.diagnostics.gyrokinetics.energy_balance``.

Ported/extended from ``src_bak/postgkyl/apps/gk_energy_balance.py`` (no
``tests_bak`` corpus exists for this app -- it was never covered upstream).
The repo does not ship a multi-file gyrokinetic energy-balance fixture set
(``<name>-field_energy_dot.gkyl``, ``..._fdot_integrated_moms.gkyl``, ...),
so the full figure path is exercised against synthetic per-file datasets
stubbed through ``utils.GData`` (the same technique
``tests/test_diagnostics_gk_load.py`` uses for the quantity registry),
rather than skipped outright -- this gives real coverage of the block/
species accumulation loop and both (absolute- and relative-error) plotting
branches. The pure residual formula and accumulation helper are unit-tested
directly with no I/O at all.

Run: PYTHONPATH=src pytest tests/test_diagnostics_programs_energy_balance.py -v
"""

from __future__ import annotations

import importlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from postgkyl.diagnostics.gyrokinetics import utils as gk_utils

eb = importlib.import_module(
    "postgkyl.diagnostics.gyrokinetics.energy_balance")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")


class _FakeGData:
  """Stands in for ``postgkyl.gdata.GData`` -- just enough surface for
  ``utils.read_gfile``/``read_gfile_if_present`` (``get_grid``/``get_values``/
  ``ctx``)."""

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
  """Registers ``(grid, values)`` for a set of file names and monkeypatches
  ``utils.GData`` to serve them, touching each file on disk so the
  existence checks in ``read_gfile_if_present`` pass."""

  def __init__(self, tmp_path, monkeypatch):
    self._tmp_path = tmp_path
    self._registry: dict[str, _FakeGData] = {}
    monkeypatch.setattr(gk_utils, "GData", self._dispatch)
  # end

  def _dispatch(self, file_name):
    return self._registry[file_name]
  # end

  def add(self, file_name: str, time_edges: np.ndarray, values: np.ndarray) -> None:
    open(file_name, "w").close()
    self._registry[file_name] = _FakeGData([np.asarray(time_edges)], np.asarray(values))
  # end
# end


@pytest.fixture
def stub(tmp_path, monkeypatch):
  return _StubFiles(tmp_path, monkeypatch)
# end


def _build_sim(stub, tmp_path, name="sim", species=("ion",), *, with_src=True,
    with_bflux=True, with_apar=False, n=5):
  """Populate a minimal single-block energy-balance file set."""
  path = str(tmp_path) + "/"
  # A dynvector's grid is exactly one time stamp per recorded sample (see
  # ``io/gkyl_reader.py``'s ``_read_t2_v1``), not N+1 cell edges like a
  # field file -- the fake GData below mimics that real convention.
  time = np.linspace(0.0, 1.0, n)

  for sp in species:
    fdot_vals = np.zeros((n, 3))
    fdot_vals[:, 2] = np.linspace(1.0, 2.0, n)
    stub.add(f"{path}{name}-{sp}_fdot_integrated_moms.gkyl", time, fdot_vals)

    if with_src:
      src_vals = np.zeros((n, 3))
      src_vals[:, 2] = 0.1
      stub.add(f"{path}{name}-{sp}_source_integrated_moms.gkyl", time, src_vals)
    # end
    if with_bflux:
      bflux_vals = np.zeros((n, 3))
      bflux_vals[:, 2] = 0.05
      stub.add(f"{path}{name}-{sp}_bflux_xlower_integrated_HamiltonianMoments.gkyl",
          time, bflux_vals)
    # end
  # end

  field_dot_vals = np.zeros((n, 1))
  field_dot_vals[:, 0] = 0.2
  stub.add(f"{path}{name}-field_energy_dot.gkyl", time, field_dot_vals)

  if with_apar:
    apar_dot_vals = np.zeros((n, 1))
    apar_dot_vals[:, 0] = 0.15
    stub.add(f"{path}{name}-apar_energy_dot.gkyl", time, apar_dot_vals)
  # end

  return path
# end


class TestEnergyBalanceErrorPure:
  """The residual formula -- pure array arithmetic, no I/O."""

  def test_no_apar(self):
    fdot = np.array([2.0, 3.0])
    src = np.array([1.0, 1.0])
    bflux = np.array([0.5, 0.5])
    field_dot = np.array([1.0, 1.0])
    err = eb.energy_balance_error(fdot, src, bflux, field_dot)
    np.testing.assert_allclose(err, src - bflux - (fdot - field_dot))
  # end

  def test_with_apar(self):
    fdot = np.array([2.0])
    src = np.array([1.0])
    bflux = np.array([0.5])
    field_dot = np.array([1.0])
    apar_dot = np.array([0.25])
    err = eb.energy_balance_error(fdot, src, bflux, field_dot, apar_dot)
    np.testing.assert_allclose(err, src - bflux - (fdot - field_dot - apar_dot))
  # end
# end


class TestAccumulatePure:

  def test_first_use_copies_not_aliases(self):
    a = np.array([1.0, 2.0])
    out = eb._accumulate(None, a)
    out[0] = 99.0
    assert a[0] == 1.0
  # end

  def test_accumulates_sum(self):
    out = eb._accumulate(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(out, [4.0, 6.0])
  # end
# end


class TestResolvePure:

  def test_no_override_uses_default(self):
    assert eb._resolve("/p/", None, "default.gkyl", 0) == "default.gkyl"
  # end

  def test_override_substitutes_block_then_species(self):
    out = eb._resolve("/p/", "custom_*_*.gkyl", "unused", 3, "ion")
    assert out == "/p/custom_3_ion.gkyl"
  # end
# end


class TestGkEnergyBalanceSynthetic:
  """Full figure-path coverage against stubbed per-file datasets."""

  def test_full_path_with_src_and_bflux(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      assert traces.src is not None
      assert traces.bflux_tot is not None
      assert traces.mom_err is not None
      assert traces.time.shape[0] == 5
      # src[0] is zeroed before computing the residual.
      assert traces.mom_err.shape == (5,)
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_missing_source_and_bflux(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, with_src=False, with_bflux=False)
    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      assert traces.src is None
      assert traces.bflux_tot is None
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_electromagnetic_branch(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, with_apar=True)
    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      assert traces.apar_dot is not None
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_multi_species_sums(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, species=("ion", "elc"))
    fig, traces = eb.energy_balance("sim", ["ion", "elc"], path=path)
    try:
      # Two identical species contributions sum to double a single one.
      single_dir = tmp_path / "single"
      single_dir.mkdir()
      single_path = _build_sim(stub, single_dir, species=("ion",))
      _, single_traces = eb.energy_balance("sim", ["ion"], path=single_path)
      np.testing.assert_allclose(traces.fdot, 2 * single_traces.fdot)
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_relative_error_branch(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    field_vals = np.full((n, 1), 3.0)
    stub.add(f"{path}sim-field_energy.gkyl", time, field_vals)
    f_vals = np.zeros((n, 3))
    f_vals[:, 2] = 10.0
    stub.add(f"{path}sim-ion_integrated_moms.gkyl", time, f_vals)
    # dt.gkyl records the timestep *between* frames, so it naturally has one
    # fewer entry than the per-frame traces -- matching src_bak, which slices
    # every per-frame trace with [1:] but never slices dt itself.
    dt_time = np.linspace(0.0, 1.0, n - 1)
    dt_vals = np.full((n - 1, 1), 0.2)
    stub.add(f"{path}sim-dt.gkyl", dt_time, dt_vals)

    fig, traces = eb.energy_balance("sim", ["ion"], path=path, relative_error=True)
    try:
      assert traces.mom_err is None
      assert traces.mom_err_norm is not None
      # One point is dropped (t=0) relative to the absolute-error path.
      assert traces.mom_err_norm.shape[0] == n - 1
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_relative_error_electromagnetic_absy_and_saveas(self, stub, tmp_path):
    """Covers the apar branch inside the relative-error path together with
    ``absy``/``saveas``/``show``."""
    path = _build_sim(stub, tmp_path, with_apar=True)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    stub.add(f"{path}sim-field_energy.gkyl", time, np.full((n, 1), 3.0))
    stub.add(f"{path}sim-apar_energy.gkyl", time, np.full((n, 1), 1.0))
    f_vals = np.zeros((n, 3))
    f_vals[:, 2] = 10.0
    stub.add(f"{path}sim-ion_integrated_moms.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    stub.add(f"{path}sim-dt.gkyl", dt_time, np.full((n - 1, 1), 0.2))

    out_path = str(tmp_path / "out.png")
    fig, traces = eb.energy_balance(
        "sim", ["ion"], path=path, relative_error=True, absy=True, logy=True,
        saveas=out_path)
    try:
      assert traces.mom_err_norm is not None
      assert os.path.exists(out_path)
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_relative_error_apar_dot_present_without_apar_energy(self, stub, tmp_path):
    """Regression for C1: a run can ship ``apar_energy_dot.gkyl`` (read in
    the unrelated, earlier per-block loop that sets ``has_apar_dot``)
    without shipping ``apar_energy.gkyl`` (read inside the relative-error
    branch's own loop, which sets ``has_apar``). The relative-error branch
    must gate every apar-dependent line -- the ``[1:]`` slicing, the
    ``energy_balance_error`` call, and the ``denom`` computation -- on
    ``has_apar``, not ``has_apar_dot``; gating on the wrong flag leaves
    ``apar`` as ``None`` (never accumulated, since ``has_apar`` is False)
    while still trying to slice it, raising
    ``TypeError: 'NoneType' object is not subscriptable``."""
    path = _build_sim(stub, tmp_path, with_apar=True)  # stages apar_energy_dot.gkyl only.
    n = 5
    time = np.linspace(0.0, 1.0, n)
    # No "sim-apar_energy.gkyl" staged -- has_apar stays False.
    stub.add(f"{path}sim-field_energy.gkyl", time, np.full((n, 1), 3.0))
    f_vals = np.zeros((n, 3))
    f_vals[:, 2] = 10.0
    stub.add(f"{path}sim-ion_integrated_moms.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    stub.add(f"{path}sim-dt.gkyl", dt_time, np.full((n - 1, 1), 0.2))

    fig, traces = eb.energy_balance("sim", ["ion"], path=path, relative_error=True)
    try:
      # No TypeError, and the electromagnetic term is correctly excluded
      # (has_apar-gated) -- matches the electrostatic relative-error formula.
      assert traces.mom_err_norm is not None
      assert traces.mom_err_norm.shape[0] == n - 1
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_missing_required_field_dot_file_raises(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    with pytest.raises(FileNotFoundError, match="field_energy_dot"):
      eb.energy_balance("sim", ["ion"], path=path)
  # end
    # end

  def test_missing_required_fdot_file_raises(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    n = 5
    time = np.linspace(0.0, 1.0, n)
    stub.add(f"{path}sim-field_energy_dot.gkyl", time, np.zeros((n, 1)))
    with pytest.raises(FileNotFoundError, match="fdot_integrated_moms"):
      eb.energy_balance("sim", ["ion"], path=path)
  # end
    # end

  def test_bflux_override_and_absy_logy(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, with_bflux=False)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    override_vals = np.zeros((n, 3))
    override_vals[:, 2] = -0.05
    override_name = f"{path}custom_bflux.gkyl"
    stub.add(override_name, time, override_vals)

    fig, traces = eb.energy_balance(
        "sim", ["ion"], path=path, bflux_files={"xlower": "custom_bflux.gkyl"},
        absy=True, logy=True, show=True)
    try:
      assert traces.bflux_tot is not None
      np.testing.assert_allclose(traces.bflux_tot, -0.05)
    # end
    finally:
      plt.close(fig)
  # end
# end
    # end


class TestGkEnergyBalanceRealFixtures:
  """Real end-to-end run against ``tests/test_data`` -- skipped loudly since
  the repo does not ship a gyrokinetic energy-balance file family (only
  single-frame distribution/geometry fixtures for a different diagnostic are
  staged there)."""

  def test_real_fixture_energy_balance(self):
    required = ("field_energy_dot.gkyl", "_fdot_integrated_moms.gkyl")
    if not any(
        any(f.endswith(suffix) for f in os.listdir(DATA)) for suffix in required):
      pytest.skip(
          "tests/test_data ships no gyrokinetic energy-balance file family "
          "(needs e.g. '<name>-field_energy_dot.gkyl', "
          "'<name>-<species>_fdot_integrated_moms.gkyl'); see "
          "TestGkEnergyBalanceSynthetic for full-path coverage against "
          "stubbed data instead.")
    # end
    pytest.fail("fixture files appeared -- wire up a real-data assertion here")
  # end
# end
