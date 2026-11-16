"""Tests for ``postgkyl.diagnostics.gk.energy_balance``.

Ported/extended from ``src_bak/postgkyl/apps/gk_energy_balance.py`` (no
``tests_bak`` corpus exists for this app -- it was never covered upstream).
The full figure path is exercised against synthetic per-file datasets
stubbed through ``utils.GData`` (the same technique
``tests/test_diagnostics_gk_load.py`` uses for the quantity registry),
rather than skipped outright -- this gives real coverage of the block/
species accumulation loop and both (absolute- and relative-error) plotting
branches. The pure residual formula and accumulation helper are unit-tested
directly with no I/O at all. Generated dynvector files also exercise the real
reader and figure path together.

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

from postgkyl.diagnostics.gk import utils as gk_utils

eb = importlib.import_module("postgkyl.diagnostics.gk.energy_balance")

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

  def get_grid(self):
    return self._grid

  def get_values(self):
    return self._values


class _StubFiles:
  """Registers ``(grid, values)`` for a set of file names and monkeypatches
  ``utils.GData`` to serve them, touching each file on disk so the
  existence checks in ``read_gfile_if_present`` pass."""

  def __init__(self, tmp_path, monkeypatch):
    self._tmp_path = tmp_path
    self._registry: dict[str, _FakeGData] = {}
    monkeypatch.setattr(gk_utils, "GData", self._dispatch)

  def _dispatch(self, file_name):
    return self._registry[file_name]

  def add(self,
          file_name: str,
          time_edges: np.ndarray,
          values: np.ndarray,
          ctx: dict | None = None) -> None:
    open(file_name, "w").close()
    self._registry[file_name] = _FakeGData([np.asarray(time_edges)],
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
               species=("ion", ),
               *,
               with_src=True,
               with_bflux=True,
               with_apar=False,
               n=5):
  """Populate a minimal single-block energy-balance file set."""
  path = str(tmp_path) + "/"
  # A dynvector's grid is exactly one time stamp per recorded sample (see
  # ``io/gkyl_reader.py``'s ``_read_t2_v1``), not N+1 cell edges like a
  # field file -- the fake GData below mimics that real convention.
  time = np.linspace(0.0, 1.0, n)

  for sp in species:
    fdot_vals = np.zeros((n, 3))
    fdot_vals[:, 2] = np.linspace(1.0, 2.0, n)
    stub.add(f"{path}{name}-{sp}_fdot_integrated_HamiltonianMoments.gkyl", time,
             fdot_vals)

    if with_src:
      src_vals = np.zeros((n, 3))
      src_vals[:, 2] = 0.1
      stub.add(f"{path}{name}-{sp}_source_integrated_HamiltonianMoments.gkyl",
               time, src_vals)
    if with_bflux:
      bflux_vals = np.zeros((n, 3))
      bflux_vals[:, 2] = 0.05
      stub.add(
          f"{path}{name}-{sp}_bflux_xlower_integrated_HamiltonianMoments.gkyl",
          time, bflux_vals)

  field_dot_vals = np.zeros((n, 1))
  field_dot_vals[:, 0] = 0.2
  stub.add(f"{path}{name}-field_energy_dot.gkyl", time, field_dot_vals)

  if with_apar:
    apar_dot_vals = np.zeros((n, 1))
    apar_dot_vals[:, 0] = 0.15
    stub.add(f"{path}{name}-apar_energy_dot.gkyl", time, apar_dot_vals)

  return path


class TestEnergyBalanceErrorPure:
  """The residual formula -- pure array arithmetic, no I/O."""

  def test_no_apar(self):
    fdot = np.array([2.0, 3.0])
    src = np.array([1.0, 1.0])
    bflux = np.array([0.5, 0.5])
    field_dot = np.array([1.0, 1.0])
    err = eb.energy_balance_error(fdot, src, bflux, field_dot)
    np.testing.assert_allclose(err, src - bflux - (fdot - field_dot))

  def test_with_apar(self):
    fdot = np.array([2.0])
    src = np.array([1.0])
    bflux = np.array([0.5])
    field_dot = np.array([1.0])
    apar_dot = np.array([0.25])
    err = eb.energy_balance_error(fdot, src, bflux, field_dot, apar_dot)
    np.testing.assert_allclose(err, src - bflux - (fdot - field_dot - apar_dot))


class TestAccumulatePure:

  def test_first_use_copies_not_aliases(self):
    a = np.array([1.0, 2.0])
    out = eb._accumulate(None, a)
    out[0] = 99.0
    assert a[0] == 1.0

  def test_accumulates_sum(self):
    out = eb._accumulate(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(out, [4.0, 6.0])


class TestResolvePure:

  def test_no_override_uses_default(self):
    assert eb._resolve("/p/", None, "default.gkyl", 0) == "default.gkyl"

  def test_override_substitutes_block_then_species(self):
    out = eb._resolve("/p/", "custom_*_*.gkyl", "unused", 3, "ion")
    assert out == "/p/custom_3_ion.gkyl"


class TestGkEnergyBalanceSynthetic:
  """Full figure-path coverage against stubbed per-file datasets."""

  def test_full_path_with_src_and_bflux(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      assert traces.src is not None
      assert traces.bflux_tot is not None
      assert traces.mom_err is not None
      # Rates are only defined after the first step: t=0 is skipped.
      assert traces.time.shape[0] == 4
      assert traces.mom_err.shape == (4, )
    finally:
      plt.close(fig)

  def test_only_t0_raises(self, stub, tmp_path):
    """No step was taken: no rate is defined."""
    path = _build_sim(stub, tmp_path, with_src=False, with_bflux=False, n=1)
    with pytest.raises(ValueError, match="No time step"):
      eb.energy_balance("sim", ["ion"], path=path)

  def test_missing_source_and_bflux(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, with_src=False, with_bflux=False)
    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      assert traces.src is None
      assert traces.bflux_tot is None
    finally:
      plt.close(fig)

  def test_electromagnetic_branch(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, with_apar=True)
    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      assert traces.apar_dot is not None
    finally:
      plt.close(fig)

  def test_multi_species_sums(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, species=("ion", "elc"))
    fig, traces = eb.energy_balance("sim", ["ion", "elc"], path=path)
    try:
      # Two identical species contributions sum to double a single one.
      single_dir = tmp_path / "single"
      single_dir.mkdir()
      single_path = _build_sim(stub, single_dir, species=("ion", ))
      _, single_traces = eb.energy_balance("sim", ["ion"], path=single_path)
      np.testing.assert_allclose(traces.fdot, 2 * single_traces.fdot)
    finally:
      plt.close(fig)

  def test_relative_error_branch(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    field_vals = np.full((n, 1), 3.0)
    stub.add(f"{path}sim-field_energy.gkyl", time, field_vals)
    f_vals = np.zeros((n, 3))
    f_vals[:, 2] = 10.0
    stub.add(f"{path}sim-ion_integrated_HamiltonianMoments.gkyl", time, f_vals)
    # dt.gkyl records the timestep *between* frames, so it naturally has one
    # fewer entry than the per-frame traces -- matching src_bak, which slices
    # every per-frame trace with [1:] but never slices dt itself.
    dt_time = np.linspace(0.0, 1.0, n - 1)
    dt_vals = np.full((n - 1, 1), 0.2)
    stub.add(f"{path}sim-dt.gkyl", dt_time, dt_vals)

    fig, traces = eb.energy_balance("sim", ["ion"],
                                    path=path,
                                    relative_error=True)
    try:
      assert traces.mom_err is None
      assert traces.mom_err_norm is not None
      # One point is dropped (t=0) relative to the absolute-error path.
      assert traces.mom_err_norm.shape[0] == n - 1
    finally:
      plt.close(fig)

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
    stub.add(f"{path}sim-ion_integrated_HamiltonianMoments.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    stub.add(f"{path}sim-dt.gkyl", dt_time, np.full((n - 1, 1), 0.2))

    out_path = str(tmp_path / "out.png")
    fig, traces = eb.energy_balance("sim", ["ion"],
                                    path=path,
                                    relative_error=True,
                                    absy=True,
                                    logy=True,
                                    saveas=out_path)
    try:
      assert traces.mom_err_norm is not None
      assert os.path.exists(out_path)
    finally:
      plt.close(fig)

  def test_relative_error_apar_dot_present_without_apar_energy(
      self, stub, tmp_path):
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
    path = _build_sim(stub, tmp_path,
                      with_apar=True)  # stages apar_energy_dot.gkyl only.
    n = 5
    time = np.linspace(0.0, 1.0, n)
    # No "sim-apar_energy.gkyl" staged -- has_apar stays False.
    stub.add(f"{path}sim-field_energy.gkyl", time, np.full((n, 1), 3.0))
    f_vals = np.zeros((n, 3))
    f_vals[:, 2] = 10.0
    stub.add(f"{path}sim-ion_integrated_HamiltonianMoments.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    stub.add(f"{path}sim-dt.gkyl", dt_time, np.full((n - 1, 1), 0.2))

    fig, traces = eb.energy_balance("sim", ["ion"],
                                    path=path,
                                    relative_error=True)
    try:
      # No TypeError, and the electromagnetic term is correctly excluded
      # (has_apar-gated) -- matches the electrostatic relative-error formula.
      assert traces.mom_err_norm is not None
      assert traces.mom_err_norm.shape[0] == n - 1
    finally:
      plt.close(fig)

  def test_missing_required_field_dot_file_raises(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    with pytest.raises(FileNotFoundError, match="field_energy_dot"):
      eb.energy_balance("sim", ["ion"], path=path)

  def test_missing_required_fdot_file_raises(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    n = 5
    time = np.linspace(0.0, 1.0, n)
    stub.add(f"{path}sim-field_energy_dot.gkyl", time, np.zeros((n, 1)))
    with pytest.raises(FileNotFoundError,
                       match="fdot_integrated_HamiltonianMoments"):
      eb.energy_balance("sim", ["ion"], path=path)

  def test_bflux_override_and_absy_logy(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path, with_bflux=False)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    override_vals = np.zeros((n, 3))
    override_vals[:, 2] = -0.05
    override_name = f"{path}custom_bflux.gkyl"
    stub.add(override_name, time, override_vals)

    fig, traces = eb.energy_balance("sim", ["ion"],
                                    path=path,
                                    bflux_files={"xlower": "custom_bflux.gkyl"},
                                    absy=True,
                                    logy=True,
                                    show=True)
    try:
      assert traces.bflux_tot is not None
      np.testing.assert_allclose(traces.bflux_tot, -0.05)
    finally:
      plt.close(fig)


class TestGkEnergyBalanceMomentFileNames:
  """Gkeyll's ``<species>_<kind>integrated_<moment>.gkyl`` naming."""

  def test_bgk_source_adds_to_source(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    bgk_vals = np.zeros((5, 3))
    bgk_vals[:, 2] = 0.02
    stub.add(f"{path}sim-ion_source_bgk_integrated_HamiltonianMoments.gkyl",
             time, bgk_vals)

    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      np.testing.assert_allclose(traces.src, 0.12)
    finally:
      plt.close(fig)

  def test_source_of_one_species_is_kept(self, stub, tmp_path):
    """The source of the first species survives the second having none."""
    path = _build_sim(stub, tmp_path, species=("ion", "elc"), with_src=False)
    time = np.linspace(0.0, 1.0, 5)
    bgk_vals = np.zeros((5, 3))
    bgk_vals[:, 2] = 0.02
    stub.add(f"{path}sim-ion_source_bgk_integrated_HamiltonianMoments.gkyl",
             time, bgk_vals)

    fig, traces = eb.energy_balance("sim", ["ion", "elc"], path=path)
    try:
      np.testing.assert_allclose(traces.src, 0.02)
    finally:
      plt.close(fig)

  def test_reads_older_integrated_moms_names(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    fdot_vals = np.zeros((5, 3))
    fdot_vals[:, 2] = 1.0 + time
    stub.add(f"{path}sim-ion_fdot_integrated_moms.gkyl", time, fdot_vals)
    stub.add(f"{path}sim-ion_source_integrated_moms.gkyl", time,
             np.full((5, 3), 0.1))
    stub.add(f"{path}sim-field_energy_dot.gkyl", time, np.zeros((5, 1)))

    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      np.testing.assert_array_equal(traces.fdot, 1.0 + time[1:])
      np.testing.assert_allclose(traces.src, 0.1)
    finally:
      plt.close(fig)


def _energy_balance_columns(time: np.ndarray,
                            bg: float = 0.05) -> dict[str, np.ndarray]:
  """Balance-file columns in a non-Gkeyll order (lookup must be by name):
  ``mom_dot=2+t``, ``src=0.3``, ``bflux=0.1``, ``field_dot=0.4``,
  ``bg_dot=bg``, energy 18 and dt 0.25; every rate is zero at t=0."""
  mom_dot = 2.0 + time
  src = np.full_like(time, 0.3)
  bflux = np.full_like(time, 0.1)
  field_dot = np.full_like(time, 0.4)
  bg_dot = np.full_like(time, bg)
  mom_err = src - bflux - (mom_dot - field_dot + bg_dot)
  columns = {
      "bg_dot": bg_dot,
      "field_dot": field_dot,
      "mom_dot": mom_dot,
      "src_mom": src,
      "bflux_tot_mom": bflux,
      "mom_err_norm": mom_err * 0.25 / 18.0,
      "mom_err": mom_err,
  }
  for values in columns.values():
    values[0] = 0.0
  return columns


def _legend_texts(fig) -> list[str]:
  return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]


class TestGkEnergyBalanceFromBalanceFiles:
  """Fallback to Gkeyll's ``<name>-balance_energy.gkyl``."""

  def test_reads_columns_by_name(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _energy_balance_columns(time)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = eb.energy_balance("sim", ["ion", "elc"], path=path)
    try:
      # Rates are only defined after the first step: t=0 is skipped.
      np.testing.assert_array_equal(traces.time, time[1:])
      for trace, column in ((traces.fdot, "mom_dot"), (traces.src, "src_mom"),
                            (traces.bflux_tot,
                             "bflux_tot_mom"), (traces.field_dot, "field_dot"),
                            (traces.bg_dot, "bg_dot"), (traces.mom_err,
                                                        "mom_err")):
        np.testing.assert_array_equal(trace, columns[column][1:])
      assert traces.apar_dot is None
      assert traces.mom_err_norm is None
      assert any(r"\rho_{bg}" in text for text in _legend_texts(fig))
    finally:
      plt.close(fig)

  def test_relative_error(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _energy_balance_columns(time)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = eb.energy_balance("sim", ["ion"],
                                    path=path,
                                    relative_error=True)
    try:
      np.testing.assert_array_equal(traces.time, time[1:])
      np.testing.assert_array_equal(traces.fdot, columns["mom_dot"][1:])
      np.testing.assert_array_equal(traces.mom_err_norm,
                                    columns["mom_err_norm"][1:])
      assert traces.mom_err is None
    finally:
      plt.close(fig)

  def test_zero_terms_are_absent(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _energy_balance_columns(time, bg=0.0)
    columns["src_mom"][:] = 0.0
    columns["bflux_tot_mom"][:] = 0.0
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      assert traces.src is None
      assert traces.bflux_tot is None
      assert traces.bg_dot is None
      # Field, fdot and residual curves only.
      assert len(_legend_texts(fig)) == 3
    finally:
      plt.close(fig)

  def test_moment_files_take_precedence(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time,
                     _energy_balance_columns(time))

    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      np.testing.assert_array_equal(traces.fdot, np.linspace(1.0, 2.0, 5)[1:])
      assert traces.bg_dot is None
    finally:
      plt.close(fig)

  def test_missing_field_energy_dot_falls_back(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add(f"{path}sim-ion_fdot_integrated_HamiltonianMoments.gkyl", time,
             np.ones((5, 3)))
    columns = _energy_balance_columns(time)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = eb.energy_balance("sim", ["ion"], path=path)
    try:
      np.testing.assert_array_equal(traces.fdot, columns["mom_dot"][1:])
    finally:
      plt.close(fig)

  def test_path_override_selects_moment_files(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add(f"{path}sim-field_energy_dot.gkyl", time, np.zeros((5, 1)))
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time,
                     _energy_balance_columns(time))
    with pytest.raises(FileNotFoundError, match="custom_fdot"):
      eb.energy_balance("sim", ["ion"], path=path, fdot_file="custom_fdot.gkyl")

  def test_multiblock_sums_blocks(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    block0 = _energy_balance_columns(time)
    block1 = {key: 3.0 * values for key, values in block0.items()}
    stub.add_balance(f"{path}sim_b0-balance_energy.gkyl", time, block0)
    stub.add_balance(f"{path}sim_b1-balance_energy.gkyl", time, block1)

    fig, traces = eb.energy_balance("sim", ["ion"], path=path, multib="-1")
    try:
      np.testing.assert_allclose(traces.fdot,
                                 4.0 * block0["mom_dot"][1:],
                                 rtol=0,
                                 atol=1e-14)
      np.testing.assert_allclose(traces.mom_err,
                                 4.0 * block0["mom_err"][1:],
                                 rtol=0,
                                 atol=1e-14)
    finally:
      plt.close(fig)

  def test_multiblock_relative_error_raises(self, stub, tmp_path):
    """Each block normalizes by its own energy: the sum is not defined."""
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    for block in (0, 1):
      stub.add_balance(f"{path}sim_b{block}-balance_energy.gkyl", time,
                       _energy_balance_columns(time))
    with pytest.raises(ValueError, match="several blocks"):
      eb.energy_balance("sim", ["ion"],
                        path=path,
                        multib="-1",
                        relative_error=True)

  def test_no_species_reads_balance_file(self, stub, tmp_path):
    """Without species the moment files cannot be read, even if present."""
    path = _build_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    columns = _energy_balance_columns(time)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = eb.energy_balance("sim", path=path)
    try:
      np.testing.assert_array_equal(traces.fdot, columns["mom_dot"][1:])
      np.testing.assert_array_equal(traces.mom_err, columns["mom_err"][1:])
    finally:
      plt.close(fig)

  def test_no_species_without_balance_file_raises(self, stub, tmp_path):
    path = _build_sim(stub, tmp_path)
    with pytest.raises(FileNotFoundError, match="sim-balance_energy"):
      eb.energy_balance("sim", path=path)

  def test_no_species_rejects_path_overrides(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time,
                     _energy_balance_columns(time))
    with pytest.raises(ValueError, match="need the species"):
      eb.energy_balance("sim", path=path, fdot_file="custom_fdot.gkyl")


class TestReadBalanceFiles:

  def test_single_sample_keeps_time_axis(self, stub, tmp_path):
    """A run stopped before its first step has one (t=0) row, which
    ``read_gfile`` squeezes to 0-D time and 1-D values."""
    name = str(tmp_path / "one-balance_particle.gkyl")
    stub.add(name, np.array([0.0]), np.zeros((1, 2)),
             {"Columns": "mom_err, mom_err_norm"})
    time, columns = gk_utils.read_balance_files([name])
    np.testing.assert_array_equal(time, [0.0])
    np.testing.assert_array_equal(columns["mom_err"], [0.0])

  def test_missing_columns_metadata_raises(self, stub, tmp_path):
    name = str(tmp_path / "bare-balance_particle.gkyl")
    stub.add(name, np.linspace(0.0, 1.0, 3), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="Columns"):
      gk_utils.read_balance_files([name])


class TestGkEnergyBalanceRealFixtures:
  """Real end-to-end run against generated dynvector files."""

  def test_real_fixture_energy_balance(self):
    path = os.path.join(DATA, "generated", "gk_programs")
    time = np.linspace(0.0, 1.0, 5)[1:]  # t=0 is skipped.
    fdot = 2.0 + time
    mom_err = 0.3 - 0.1 - (fdot - 0.4)
    for relative in (False, True):
      fig, traces = eb.energy_balance(
          "sim", ["ion"],
          path=path,
          relative_error=relative,
          field_dot_file="sim-field_energy_dot.gkyl")
      try:
        np.testing.assert_array_equal(traces.time, time)
        np.testing.assert_allclose(traces.fdot, fdot, rtol=0, atol=1e-14)
        np.testing.assert_allclose(traces.field_dot, 0.4, rtol=0, atol=1e-14)
        if relative:
          np.testing.assert_allclose(traces.mom_err_norm,
                                     mom_err * 0.25 / 18.0,
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

  def test_real_fixture_balance_file(self):
    """Gkeyll's balance file without the moment files (``bal`` fixture)."""
    path = os.path.join(DATA, "generated", "gk_programs")
    time = np.linspace(0.0, 1.0, 5)[1:]  # t=0 is skipped.
    fdot = 2.0 + time
    mom_err = 0.3 - 0.1 - (fdot - 0.4 + 0.05)
    for relative in (False, True):
      fig, traces = eb.energy_balance("bal", ["ion"],
                                      path=path,
                                      relative_error=relative)
      try:
        np.testing.assert_array_equal(traces.time, time)
        for actual, expected in ((traces.fdot, fdot), (traces.src, 0.3),
                                 (traces.bflux_tot, 0.1), (traces.field_dot,
                                                           0.4), (traces.bg_dot,
                                                                  0.05)):
          np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        if relative:
          np.testing.assert_allclose(traces.mom_err_norm,
                                     mom_err * 0.25 / 18.0,
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
