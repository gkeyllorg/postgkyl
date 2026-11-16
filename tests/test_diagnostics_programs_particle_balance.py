"""Tests for ``postgkyl.diagnostics.gk.particle_balance``.

See ``test_diagnostics_programs_energy_balance.py`` for the shared testing
strategy. Stubbed datasets exercise branches, and generated dynvector files
exercise the real reader and figure path together.

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

pb = importlib.import_module("postgkyl.diagnostics.gk.particle_balance")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")


class _FakeGData:

  def __init__(self, grid, values, ctx=None):
    self._grid = grid
    self._values = values
    self.ctx = ctx or {}

  def get_grid(self):
    return self._grid

  def get_values(self):
    return self._values


class _StubFiles:

  def __init__(self, tmp_path, monkeypatch):
    self._registry: dict[str, _FakeGData] = {}
    monkeypatch.setattr(gk_utils, "GData", self._dispatch)

  def _dispatch(self, file_name):
    return self._registry[file_name]

  def add(self, file_name: str, time, values, ctx: dict | None = None) -> None:
    open(file_name, "w").close()
    self._registry[file_name] = _FakeGData([np.asarray(time)],
                                           np.asarray(values), ctx)

  def add_balance(self, file_name: str, time: np.ndarray,
                  columns: dict[str, np.ndarray]) -> None:
    """Stage a Gkeyll balance file: columns named by ``Columns`` metadata."""
    self.add(file_name, time, np.stack(list(columns.values()), axis=-1),
             {"Columns": ", ".join(columns)})


@pytest.fixture
def stub(tmp_path, monkeypatch):
  return _StubFiles(tmp_path, monkeypatch)


def _build_sim(stub,
               tmp_path,
               name="sim",
               species="ion",
               *,
               with_src=True,
               with_bflux=True,
               n=5):
  path = str(tmp_path) + "/"
  # A dynvector's grid is exactly one time stamp per recorded sample, not
  # N+1 cell edges like a field file (see io/gkyl_reader.py's _read_t2_v1).
  time = np.linspace(0.0, 1.0, n)

  # Gkeyll's default integrated moments, M0M1M2parM2perp.
  fdot_vals = np.zeros((n, 4))
  fdot_vals[:, 0] = np.linspace(1.0, 2.0, n)
  stub.add(f"{path}{name}-{species}_fdot_integrated_M0M1M2parM2perp.gkyl", time,
           fdot_vals)

  if with_src:
    src_vals = np.zeros((n, 4))
    src_vals[:, 0] = 0.1
    stub.add(f"{path}{name}-{species}_source_integrated_M0M1M2parM2perp.gkyl",
             time, src_vals)
  if with_bflux:
    bflux_vals = np.zeros((n, 4))
    bflux_vals[:, 0] = 0.05
    stub.add(
        f"{path}{name}-{species}_bflux_xlower_integrated_HamiltonianMoments.gkyl",
        time, bflux_vals)
  return path


class TestParticleBalanceErrorPure:

  def test_formula(self):
    fdot = np.array([2.0, 3.0])
    src = np.array([1.0, 1.0])
    bflux = np.array([0.5, 0.5])
    err = pb.particle_balance_error(fdot, src, bflux)
    np.testing.assert_allclose(err, src - bflux - fdot)


class TestAccumulatePure:

  def test_first_use_copies_not_aliases(self):
    a = np.array([1.0, 2.0])
    out = pb._accumulate(None, a)
    out[0] = 99.0
    assert a[0] == 1.0

  def test_accumulates_sum(self):
    out = pb._accumulate(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(out, [4.0, 6.0])


class TestResolvePure:

  def test_no_override_uses_default(self):
    assert pb._resolve("/p/", None, "default.gkyl", 0) == "default.gkyl"

  def test_override_substitutes_block(self):
    assert pb._resolve("/p/", "custom_*.gkyl", "unused",
                       3) == "/p/custom_3.gkyl"


class TestGkParticleBalanceSynthetic:

  def test_full_path_with_src_and_bflux(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    fig, traces = pb.particle_balance("sim", "ion", path=path)
    try:
      assert traces.src is not None
      assert traces.bflux_tot is not None
      assert traces.mom_err is not None
      assert traces.time.shape[0] == 4  # t=0 is skipped.
    finally:
      plt.close(fig)

  def test_only_t0_raises(self, stub, tmp_path):
    """No step was taken: no rate is defined."""
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-ion_fdot_integrated_M0M1M2parM2perp.gkyl", [0.0],
             np.zeros((1, 4)))
    with pytest.raises(ValueError, match="No time step"):
      pb.particle_balance("sim", "ion", path=path)

  def test_missing_source_and_bflux(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, with_src=False, with_bflux=False)
    fig, traces = pb.particle_balance("sim", "ion", path=path)
    try:
      assert traces.src is None
      assert traces.bflux_tot is None
    finally:
      plt.close(fig)

  def test_relative_error_branch(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    f_vals = np.zeros((n, 4))
    f_vals[:, 0] = 10.0
    stub.add(f"{path}sim-ion_integrated_M0M1M2parM2perp.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    dt_vals = np.full((n - 1, 1), 0.2)
    stub.add(f"{path}sim-dt.gkyl", dt_time, dt_vals)

    fig, traces = pb.particle_balance("sim",
                                      "ion",
                                      path=path,
                                      relative_error=True)
    try:
      assert traces.mom_err is None
      assert traces.mom_err_norm is not None
      assert traces.mom_err_norm.shape[0] == n - 1
    finally:
      plt.close(fig)

  def test_relative_error_absy_saveas_and_show(self, stub, tmp_path):
    """Covers ``absy`` wrapping the relative-error ylabel in ``||`` together
    with ``saveas``/``show`` (default ``ylabel_string`` is non-empty on this
    branch, unlike the absolute-error branch's ``""`` default)."""
    path = _build_sim(stub, tmp_path)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    f_vals = np.zeros((n, 4))
    f_vals[:, 0] = 10.0
    stub.add(f"{path}sim-ion_integrated_M0M1M2parM2perp.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    dt_vals = np.full((n - 1, 1), 0.2)
    stub.add(f"{path}sim-dt.gkyl", dt_time, dt_vals)

    out_path = str(tmp_path / "out.png")
    fig, traces = pb.particle_balance("sim",
                                      "ion",
                                      path=path,
                                      relative_error=True,
                                      absy=True,
                                      show=True,
                                      saveas=out_path)
    try:
      assert traces.mom_err_norm is not None
      assert os.path.exists(out_path)
    finally:
      plt.close(fig)

  def test_missing_required_fdot_file_raises(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    with pytest.raises(FileNotFoundError, match="fdot_integrated_M0.gkyl"):
      pb.particle_balance("sim", "ion", path=path)

  def test_bflux_override_and_absy_logy(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, with_bflux=False)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    override_vals = np.zeros((n, 4))
    override_vals[:, 0] = -0.05
    override_name = f"{path}custom_bflux.gkyl"
    stub.add(override_name, time, override_vals)

    fig, traces = pb.particle_balance(
        "sim",
        "ion",
        path=path,
        bflux_files={"xlower": "custom_bflux.gkyl"},
        absy=True,
        logy=True)
    try:
      assert traces.bflux_tot is not None
      np.testing.assert_allclose(traces.bflux_tot, -0.05)
    finally:
      plt.close(fig)

  def test_multiblock_sums_over_blocks(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    n = 4
    time = np.linspace(0.0, 1.0, n)
    for block in (0, 1):
      fdot_vals = np.zeros((n, 4))
      fdot_vals[:, 0] = 1.0
      stub.add(f"{path}sim_b{block}-ion_fdot_integrated_M0M1M2parM2perp.gkyl",
               time, fdot_vals)
    fig, traces = pb.particle_balance("sim", "ion", path=path, multib="0,1")
    try:
      # Two blocks, each contributing fdot=1.0, sum to 2.0 everywhere.
      np.testing.assert_allclose(traces.fdot, 2.0)
    finally:
      plt.close(fig)


class TestGkParticleBalanceMomentFileNames:
  """Gkeyll's ``<species>_<kind>integrated_<moment>.gkyl`` naming."""

  def test_single_component_m0_files(self, stub, tmp_path):
    """Gkeyll adds ``M0`` when no requested moment has the density."""
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add(f"{path}sim-ion_fdot_integrated_M0.gkyl", time, (1.0 + time)[:,
                                                                          None])
    stub.add(f"{path}sim-ion_bflux_xlower_integrated_M0.gkyl", time,
             np.full((5, 1), 0.05))
    stub.add(f"{path}sim-ion_integrated_M0.gkyl", time, np.full((5, 1), 10.0))
    stub.add(f"{path}sim-dt.gkyl", time[1:], np.full((4, 1), 0.25))

    for relative in (False, True):
      fig, traces = pb.particle_balance("sim",
                                        "ion",
                                        path=path,
                                        relative_error=relative)
      try:
        mom_err = -0.05 - (1.0 + time[1:])
        if relative:
          np.testing.assert_allclose(traces.mom_err_norm, mom_err * 0.025)
        else:
          np.testing.assert_allclose(traces.mom_err, mom_err)
      finally:
        plt.close(fig)

  def test_bgk_source_adds_to_source(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    bgk_vals = np.zeros((5, 3))
    bgk_vals[:, 0] = 0.02
    stub.add(f"{path}sim-ion_source_bgk_integrated_M0M1M2.gkyl", time, bgk_vals)

    fig, traces = pb.particle_balance("sim", "ion", path=path)
    try:
      np.testing.assert_allclose(traces.src, 0.12)
    finally:
      plt.close(fig)

  def test_reads_older_integrated_moms_names(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    fdot_vals = np.zeros((5, 4))
    fdot_vals[:, 0] = 1.0 + time
    stub.add(f"{path}sim-ion_fdot_integrated_moms.gkyl", time, fdot_vals)

    fig, traces = pb.particle_balance("sim", "ion", path=path)
    try:
      np.testing.assert_array_equal(traces.fdot, 1.0 + time[1:])
    finally:
      plt.close(fig)


def _particle_balance_columns(time: np.ndarray) -> dict[str, np.ndarray]:
  """Balance-file columns in a non-Gkeyll order (lookup must be by name):
  ``mom_dot=1+t``, ``src=0.1``, ``bflux=0.05``, number 10 and dt 0.25;
  every rate is zero at t=0."""
  mom_dot = 1.0 + time
  src = np.full_like(time, 0.1)
  bflux = np.full_like(time, 0.05)
  mom_err = src - bflux - mom_dot
  columns = {
      "mom_dot": mom_dot,
      "bflux_tot_mom": bflux,
      "src_mom": src,
      "mom_err": mom_err,
      "mom_err_norm": mom_err * 0.25 / 10.0,
  }
  for values in columns.values():
    values[0] = 0.0
  return columns


class TestGkParticleBalanceFromBalanceFiles:
  """Fallback to Gkeyll's ``<name>-<species>_balance_particle.gkyl``."""

  def test_reads_columns_by_name(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _particle_balance_columns(time)
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time, columns)

    for relative in (False, True):
      fig, traces = pb.particle_balance("sim",
                                        "ion",
                                        path=path,
                                        relative_error=relative)
      try:
        np.testing.assert_array_equal(traces.time, time[1:])
        np.testing.assert_array_equal(traces.fdot, columns["mom_dot"][1:])
        np.testing.assert_array_equal(traces.src, columns["src_mom"][1:])
        np.testing.assert_array_equal(traces.bflux_tot,
                                      columns["bflux_tot_mom"][1:])
        if relative:
          np.testing.assert_array_equal(traces.mom_err_norm,
                                        columns["mom_err_norm"][1:])
          assert traces.mom_err is None
        else:
          np.testing.assert_array_equal(traces.mom_err, columns["mom_err"][1:])
          assert traces.mom_err_norm is None
      finally:
        plt.close(fig)

  def test_species_sum_is_not_a_species_balance(self, stub, tmp_path):
    """``<name>-balance_particle.gkyl`` sums over species: no fallback."""
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-balance_particle.gkyl", time,
                     _particle_balance_columns(time))
    with pytest.raises(FileNotFoundError, match="ion_fdot_integrated_M0.gkyl"):
      pb.particle_balance("sim", "ion", path=path)

  def test_moment_files_take_precedence(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time,
                     _particle_balance_columns(time))

    fig, traces = pb.particle_balance("sim", "ion", path=path)
    try:
      np.testing.assert_array_equal(traces.fdot, np.linspace(1.0, 2.0, 5)[1:])
    finally:
      plt.close(fig)

  def test_path_override_selects_moment_files(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time,
                     _particle_balance_columns(time))
    with pytest.raises(FileNotFoundError, match="custom_fdot"):
      pb.particle_balance("sim", "ion", path=path, fdot_file="custom_fdot.gkyl")

  def test_multiblock_sums_blocks(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    block0 = _particle_balance_columns(time)
    block1 = {key: 3.0 * values for key, values in block0.items()}
    stub.add_balance(f"{path}sim_b0-ion_balance_particle.gkyl", time, block0)
    stub.add_balance(f"{path}sim_b1-ion_balance_particle.gkyl", time, block1)

    fig, traces = pb.particle_balance("sim", "ion", path=path, multib="0,1")
    try:
      np.testing.assert_allclose(traces.mom_err,
                                 4.0 * block0["mom_err"][1:],
                                 rtol=0,
                                 atol=1e-14)
    finally:
      plt.close(fig)
    with pytest.raises(ValueError, match="several blocks"):
      pb.particle_balance("sim",
                          "ion",
                          path=path,
                          multib="0,1",
                          relative_error=True)

  def test_no_species_reads_species_sum(self, stub, tmp_path):
    """Without species, ``<name>-balance_particle.gkyl`` is read even if the
    species files are present."""
    path = _build_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    columns = _particle_balance_columns(time)
    stub.add_balance(f"{path}sim-balance_particle.gkyl", time, columns)
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time, {
        key: 2.0 * v
        for key, v in columns.items()
    })

    for relative in (False, True):
      fig, traces = pb.particle_balance("sim",
                                        path=path,
                                        relative_error=relative)
      try:
        if relative:
          np.testing.assert_array_equal(traces.mom_err_norm,
                                        columns["mom_err_norm"][1:])
        else:
          np.testing.assert_array_equal(traces.fdot, columns["mom_dot"][1:])
          np.testing.assert_array_equal(traces.mom_err, columns["mom_err"][1:])
      finally:
        plt.close(fig)

  def test_no_species_without_balance_file_raises(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    with pytest.raises(FileNotFoundError, match="sim-balance_particle"):
      pb.particle_balance("sim", path=path)

  def test_no_species_rejects_path_overrides(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-balance_particle.gkyl", time,
                     _particle_balance_columns(time))
    with pytest.raises(ValueError, match="need a species"):
      pb.particle_balance("sim", path=path, fdot_file="custom_fdot.gkyl")


class TestGkParticleBalanceRealFixtures:

  def test_real_fixture_particle_balance(self):
    path = os.path.join(DATA, "generated", "gk_programs")
    time = np.linspace(0.0, 1.0, 5)[1:]  # t=0 is skipped.
    fdot = 1.0 + time
    mom_err = 0.1 - 0.05 - fdot
    for relative in (False, True):
      fig, traces = pb.particle_balance("sim",
                                        "ion",
                                        path=path,
                                        relative_error=relative)
      try:
        np.testing.assert_array_equal(traces.time, time)
        np.testing.assert_allclose(traces.fdot, fdot, rtol=0, atol=1e-14)
        np.testing.assert_allclose(traces.bflux_tot, 0.05, rtol=0, atol=1e-14)
        if relative:
          np.testing.assert_allclose(traces.mom_err_norm,
                                     mom_err * 0.25 / 10.0,
                                     rtol=0,
                                     atol=1e-14)
          assert traces.mom_err is None
        else:
          np.testing.assert_allclose(traces.mom_err,
                                     mom_err,
                                     rtol=0,
                                     atol=1e-14)
      finally:
        plt.close(fig)

  @pytest.mark.parametrize("species", ["ion", None])
  def test_real_fixture_balance_file(self, species):
    """Gkeyll's balance files without the moment files (``bal`` fixture), of
    the ion and summed over its single species."""
    path = os.path.join(DATA, "generated", "gk_programs")
    time = np.linspace(0.0, 1.0, 5)[1:]  # t=0 is skipped.
    fdot = 1.0 + time
    mom_err = 0.1 - 0.05 - fdot
    for relative in (False, True):
      fig, traces = pb.particle_balance("bal",
                                        species,
                                        path=path,
                                        relative_error=relative)
      try:
        np.testing.assert_array_equal(traces.time, time)
        for actual, expected in ((traces.fdot, fdot), (traces.src, 0.1),
                                 (traces.bflux_tot, 0.05)):
          np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        if relative:
          np.testing.assert_allclose(traces.mom_err_norm,
                                     mom_err * 0.25 / 10.0,
                                     rtol=0,
                                     atol=1e-14)
          assert traces.mom_err is None
        else:
          np.testing.assert_allclose(traces.mom_err,
                                     mom_err,
                                     rtol=0,
                                     atol=1e-14)
      finally:
        plt.close(fig)
