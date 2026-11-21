"""Tests for ``postgkyl.diagnostics.gk.balance``.

Ported/extended from ``src_bak/postgkyl/apps/gk_energy_balance.py`` and
``gk_particle_balance.py`` (no ``tests_bak`` corpus exists for these apps --
they were never covered upstream). The full figure path is exercised against
synthetic per-file datasets stubbed through ``utils.GData`` (the same
technique ``tests/test_diagnostics_gk_load.py`` uses for the quantity
registry), rather than skipped outright -- this gives real coverage of the
block/species accumulation loop and both (absolute- and relative-error)
plotting branches. The pure residual formula and accumulation helper are
unit-tested directly with no I/O at all. Generated dynvector files also
exercise the real reader and figure path together.

Run: PYTHONPATH=src pytest tests/test_diagnostics_programs_balance.py -v
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

bal = importlib.import_module("postgkyl.diagnostics.gk.balance")

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


def _legend_texts(fig) -> list[str]:
  return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]


def _build_energy_sim(stub,
                      tmp_path,
                      name="sim",
                      species=("ion", ),
                      *,
                      with_src=True,
                      with_bflux=True,
                      with_apar=False,
                      n=5):
  """Populate a minimal single-block energy-balance file set: H fdot=1..2,
  source 0.1, bflux 0.05, field_dot 0.2 and, if requested, apar_dot 0.15."""
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


def _build_particle_sim(stub,
                        tmp_path,
                        name="sim",
                        species=("ion", ),
                        *,
                        with_src=True,
                        with_bflux=True,
                        n=5):
  """Populate a minimal single-block particle-balance file set: M0
  fdot=1..2, source 0.1 and bflux 0.05 per species."""
  path = str(tmp_path) + "/"
  time = np.linspace(0.0, 1.0, n)

  for sp in species:
    # Gkeyll's default integrated moments, M0M1M2parM2perp.
    fdot_vals = np.zeros((n, 4))
    fdot_vals[:, 0] = np.linspace(1.0, 2.0, n)
    stub.add(f"{path}{name}-{sp}_fdot_integrated_M0M1M2parM2perp.gkyl", time,
             fdot_vals)

    if with_src:
      src_vals = np.zeros((n, 4))
      src_vals[:, 0] = 0.1
      stub.add(f"{path}{name}-{sp}_source_integrated_M0M1M2parM2perp.gkyl",
               time, src_vals)
    if with_bflux:
      bflux_vals = np.zeros((n, 4))
      bflux_vals[:, 0] = 0.05
      stub.add(
          f"{path}{name}-{sp}_bflux_xlower_integrated_HamiltonianMoments.gkyl",
          time, bflux_vals)
  return path


class TestBalanceErrorPure:
  """The residual formula -- pure array arithmetic, no I/O."""

  def test_particle(self):
    fdot = np.array([2.0, 3.0])
    src = np.array([1.0, 1.0])
    bflux = np.array([0.5, 0.5])
    err = bal.balance_error(fdot, src, bflux)
    np.testing.assert_allclose(err, src - bflux - fdot)

  def test_energy_no_apar(self):
    fdot = np.array([2.0, 3.0])
    src = np.array([1.0, 1.0])
    bflux = np.array([0.5, 0.5])
    field_dot = np.array([1.0, 1.0])
    err = bal.balance_error(fdot, src, bflux, field_dot)
    np.testing.assert_allclose(err, src - bflux - (fdot - field_dot))

  def test_energy_with_apar(self):
    fdot = np.array([2.0])
    src = np.array([1.0])
    bflux = np.array([0.5])
    field_dot = np.array([1.0])
    apar_dot = np.array([0.25])
    err = bal.balance_error(fdot, src, bflux, field_dot, apar_dot)
    np.testing.assert_allclose(err, src - bflux - (fdot - field_dot - apar_dot))


class TestAccumulatePure:

  def test_first_use_copies_not_aliases(self):
    a = np.array([1.0, 2.0])
    out = bal._accumulate(None, a)
    out[0] = 99.0
    assert a[0] == 1.0

  def test_accumulates_sum(self):
    out = bal._accumulate(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(out, [4.0, 6.0])


class TestResolvePure:

  def test_no_override_uses_default(self):
    assert bal._resolve("/p/", None, "default.gkyl", 0) == "default.gkyl"

  def test_override_substitutes_block(self):
    assert bal._resolve("/p/", "custom_*.gkyl", "unused",
                        3) == "/p/custom_3.gkyl"

  def test_override_substitutes_block_then_species(self):
    out = bal._resolve("/p/", "custom_*_*.gkyl", "unused", 3, "ion")
    assert out == "/p/custom_3_ion.gkyl"


class TestOptions:

  def test_field_overrides_only_for_energy(self, stub, tmp_path):
    path = _build_particle_sim(stub, tmp_path)
    with pytest.raises(ValueError, match="only apply to the energy"):
      bal.balance("sim", ["ion"],
                  path=path,
                  field_dot_file="sim-field_energy_dot.gkyl")


class TestEnergySynthetic:
  """Full figure-path coverage against stubbed per-file datasets."""

  def test_full_path_with_src_and_bflux(self, stub, tmp_path):
    path = _build_energy_sim(stub, tmp_path)
    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      assert traces["src"] is not None
      assert traces["bflux_tot"] is not None
      assert traces["mom_err"] is not None
      # Rates are only defined after the first step: t=0 is skipped.
      assert traces["time"].shape[0] == 4
      assert traces["mom_err"].shape == (4, )
      assert fig.axes[0].get_title() == "Energy balance"
    finally:
      plt.close(fig)

  def test_only_t0_raises(self, stub, tmp_path):
    """No step was taken: no rate is defined."""
    path = _build_energy_sim(stub,
                             tmp_path,
                             with_src=False,
                             with_bflux=False,
                             n=1)
    with pytest.raises(ValueError, match="No time step"):
      bal.balance("sim", ["ion"], quantity="energy", path=path)

  def test_missing_source_and_bflux(self, stub, tmp_path):
    path = _build_energy_sim(stub, tmp_path, with_src=False, with_bflux=False)
    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      assert traces["src"] is None
      assert traces["bflux_tot"] is None
    finally:
      plt.close(fig)

  def test_electromagnetic_branch(self, stub, tmp_path):
    path = _build_energy_sim(stub, tmp_path, with_apar=True)
    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      assert traces["apar_dot"] is not None
    finally:
      plt.close(fig)

  def test_plotted_terms_sum_to_the_residual(self, stub, tmp_path):
    """The legend writes the residual as the sum of the plotted terms: with
    S=0.1, bflux=0.05, field_dot=0.2, apar_dot=0.15 and fdot=1+t, the curves
    S, -bflux, +field_dot, +apar_dot, -fdot add up to the residual."""
    path = _build_energy_sim(stub, tmp_path, with_apar=True)
    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      lines = fig.axes[0].get_lines()[1:]  # Skip the zero line.
      np.testing.assert_allclose(sum(line.get_ydata() for line in lines[:-1]),
                                 lines[-1].get_ydata(),
                                 rtol=0,
                                 atol=1e-14)
      np.testing.assert_allclose(lines[-1].get_ydata(),
                                 0.4 - (1.0 + traces["time"]),
                                 rtol=0,
                                 atol=1e-14)
      assert r"$+\dot{\phi}$$+\dot{A}_{\parallel}$" in _legend_texts(fig)[-1]
    finally:
      plt.close(fig)

  def test_multi_species_sums(self, stub, tmp_path):
    path = _build_energy_sim(stub, tmp_path, species=("ion", "elc"))
    fig, traces = bal.balance("sim", ["ion", "elc"],
                              quantity="energy",
                              path=path)
    try:
      # Two identical species contributions sum to double a single one.
      single_dir = tmp_path / "single"
      single_dir.mkdir()
      single_path = _build_energy_sim(stub, single_dir, species=("ion", ))
      _, single_traces = bal.balance("sim", ["ion"],
                                     quantity="energy",
                                     path=single_path)
      np.testing.assert_allclose(traces["fdot"], 2 * single_traces["fdot"])
    finally:
      plt.close(fig)

  def test_relative_error_branch(self, stub, tmp_path):
    path = _build_energy_sim(stub, tmp_path)
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

    fig, traces = bal.balance("sim", ["ion"],
                              quantity="energy",
                              path=path,
                              relative_error=True)
    try:
      assert traces["mom_err"] is None
      # One point is dropped (t=0) relative to the absolute-error path.
      fdot = np.linspace(1.0, 2.0, n)[1:]
      np.testing.assert_allclose(traces["mom_err_norm"],
                                 (0.1 - 0.05 - (fdot - 0.2)) * 0.2 / 7.0,
                                 rtol=0,
                                 atol=1e-15)
      assert fig.axes[0].get_title() == (
          "Relative error in energy conservation")
    finally:
      plt.close(fig)

  def test_relative_error_electromagnetic_absy_and_saveas(self, stub, tmp_path):
    """Covers the apar branch inside the relative-error path together with
    ``absy``/``saveas``/``show``: the energy is 10 - 3 - 1."""
    path = _build_energy_sim(stub, tmp_path, with_apar=True)
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
    fig, traces = bal.balance("sim", ["ion"],
                              quantity="energy",
                              path=path,
                              relative_error=True,
                              absy=True,
                              logy=True,
                              saveas=out_path)
    try:
      fdot = np.linspace(1.0, 2.0, n)[1:]
      np.testing.assert_allclose(traces["mom_err_norm"],
                                 (0.1 - 0.05 - (fdot - 0.2 - 0.15)) * 0.2 / 6.0,
                                 rtol=0,
                                 atol=1e-15)
      assert os.path.exists(out_path)
    finally:
      plt.close(fig)

  def test_relative_error_apar_dot_present_without_apar_energy(
      self, stub, tmp_path):
    """A run can ship ``apar_energy_dot.gkyl`` without ``apar_energy.gkyl``:
    the residual keeps the apar term, which does not depend on the
    normalization, and the energy (10 - 3) has no apar term."""
    path = _build_energy_sim(
        stub, tmp_path, with_apar=True)  # stages apar_energy_dot.gkyl only.
    n = 5
    time = np.linspace(0.0, 1.0, n)
    stub.add(f"{path}sim-field_energy.gkyl", time, np.full((n, 1), 3.0))
    f_vals = np.zeros((n, 3))
    f_vals[:, 2] = 10.0
    stub.add(f"{path}sim-ion_integrated_HamiltonianMoments.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    stub.add(f"{path}sim-dt.gkyl", dt_time, np.full((n - 1, 1), 0.2))

    fig, traces = bal.balance("sim", ["ion"],
                              quantity="energy",
                              path=path,
                              relative_error=True)
    try:
      fdot = np.linspace(1.0, 2.0, n)[1:]
      np.testing.assert_allclose(traces["mom_err_norm"],
                                 (0.1 - 0.05 - (fdot - 0.2 - 0.15)) * 0.2 / 7.0,
                                 rtol=0,
                                 atol=1e-15)
    finally:
      plt.close(fig)

  def test_relative_error_without_field_energy_raises(self, stub, tmp_path):
    path = _build_energy_sim(stub, tmp_path)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    f_vals = np.zeros((n, 3))
    f_vals[:, 2] = 10.0
    stub.add(f"{path}sim-ion_integrated_HamiltonianMoments.gkyl", time, f_vals)
    stub.add(f"{path}sim-dt.gkyl", time[1:], np.full((n - 1, 1), 0.2))
    with pytest.raises(FileNotFoundError, match="sim-field_energy.gkyl"):
      bal.balance("sim", ["ion"],
                  quantity="energy",
                  path=path,
                  relative_error=True)

  def test_missing_required_field_dot_file_raises(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    with pytest.raises(FileNotFoundError, match="field_energy_dot"):
      bal.balance("sim", ["ion"], quantity="energy", path=path)

  def test_missing_required_fdot_file_raises(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    n = 5
    time = np.linspace(0.0, 1.0, n)
    stub.add(f"{path}sim-field_energy_dot.gkyl", time, np.zeros((n, 1)))
    with pytest.raises(FileNotFoundError,
                       match="fdot_integrated_HamiltonianMoments"):
      bal.balance("sim", ["ion"], quantity="energy", path=path)

  def test_bflux_override_and_absy_logy(self, stub, tmp_path):
    path = _build_energy_sim(stub, tmp_path, with_bflux=False)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    override_vals = np.zeros((n, 3))
    override_vals[:, 2] = -0.05
    override_name = f"{path}custom_bflux.gkyl"
    stub.add(override_name, time, override_vals)

    fig, traces = bal.balance("sim", ["ion"],
                              quantity="energy",
                              path=path,
                              bflux_files={"xlower": "custom_bflux.gkyl"},
                              absy=True,
                              logy=True,
                              show=True)
    try:
      assert traces["bflux_tot"] is not None
      np.testing.assert_allclose(traces["bflux_tot"], -0.05)
    finally:
      plt.close(fig)


class TestParticleSynthetic:

  def test_full_path_with_src_and_bflux(self, stub, tmp_path):
    path = _build_particle_sim(stub, tmp_path)
    fig, traces = bal.balance("sim", ["ion"], quantity="particle", path=path)
    try:
      assert traces["src"] is not None
      assert traces["bflux_tot"] is not None
      assert traces["mom_err"] is not None
      assert traces["field_dot"] is None
      assert traces["time"].shape[0] == 4  # t=0 is skipped.
      assert fig.axes[0].get_title() == "Particle balance"
      assert _legend_texts(fig)[-1].startswith(r"$E_{\dot{\mathcal{N}}}=$")
    finally:
      plt.close(fig)

  def test_particle_is_the_default(self, stub, tmp_path):
    path = _build_particle_sim(stub, tmp_path)
    fig, traces = bal.balance("sim", ["ion"], path=path)
    try:
      np.testing.assert_allclose(traces["mom_err"],
                                 0.1 - 0.05 - np.linspace(1.0, 2.0, 5)[1:])
    finally:
      plt.close(fig)

  def test_only_t0_raises(self, stub, tmp_path):
    """No step was taken: no rate is defined."""
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-ion_fdot_integrated_M0M1M2parM2perp.gkyl", [0.0],
             np.zeros((1, 4)))
    with pytest.raises(ValueError, match="No time step"):
      bal.balance("sim", ["ion"], quantity="particle", path=path)

  def test_missing_source_and_bflux(self, stub, tmp_path):
    path = _build_particle_sim(stub, tmp_path, with_src=False, with_bflux=False)
    fig, traces = bal.balance("sim", ["ion"], quantity="particle", path=path)
    try:
      assert traces["src"] is None
      assert traces["bflux_tot"] is None
    finally:
      plt.close(fig)

  def test_multi_species_sums(self, stub, tmp_path):
    """Two species of fdot 1..2, source 0.1 and bflux 0.05 each."""
    path = _build_particle_sim(stub, tmp_path, species=("ion", "elc"))
    fig, traces = bal.balance("sim", ["ion", "elc"],
                              quantity="particle",
                              path=path)
    try:
      fdot = np.linspace(1.0, 2.0, 5)[1:]
      np.testing.assert_allclose(traces["fdot"], 2.0 * fdot)
      np.testing.assert_allclose(traces["mom_err"], 2.0 * (0.1 - 0.05 - fdot))
    finally:
      plt.close(fig)

  def test_relative_error_branch(self, stub, tmp_path):
    path = _build_particle_sim(stub, tmp_path)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    f_vals = np.zeros((n, 4))
    f_vals[:, 0] = 10.0
    stub.add(f"{path}sim-ion_integrated_M0M1M2parM2perp.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    dt_vals = np.full((n - 1, 1), 0.2)
    stub.add(f"{path}sim-dt.gkyl", dt_time, dt_vals)

    fig, traces = bal.balance("sim", ["ion"],
                              quantity="particle",
                              path=path,
                              relative_error=True)
    try:
      assert traces["mom_err"] is None
      fdot = np.linspace(1.0, 2.0, n)[1:]
      np.testing.assert_allclose(traces["mom_err_norm"],
                                 (0.1 - 0.05 - fdot) * 0.2 / 10.0,
                                 rtol=0,
                                 atol=1e-15)
    finally:
      plt.close(fig)

  def test_relative_error_absy_saveas_and_show(self, stub, tmp_path):
    """Covers ``absy`` wrapping the relative-error ylabel in ``||`` together
    with ``saveas``/``show`` (default ``ylabel_string`` is non-empty on this
    branch, unlike the absolute-error branch's ``""`` default)."""
    path = _build_particle_sim(stub, tmp_path)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    f_vals = np.zeros((n, 4))
    f_vals[:, 0] = 10.0
    stub.add(f"{path}sim-ion_integrated_M0M1M2parM2perp.gkyl", time, f_vals)
    dt_time = np.linspace(0.0, 1.0, n - 1)
    dt_vals = np.full((n - 1, 1), 0.2)
    stub.add(f"{path}sim-dt.gkyl", dt_time, dt_vals)

    out_path = str(tmp_path / "out.png")
    fig, traces = bal.balance("sim", ["ion"],
                              quantity="particle",
                              path=path,
                              relative_error=True,
                              absy=True,
                              show=True,
                              saveas=out_path)
    try:
      assert traces["mom_err_norm"] is not None
      assert os.path.exists(out_path)
      assert fig.axes[0].get_ylabel() == (
          r"|$E_{\dot{\mathcal{N}}}~\Delta t/\mathcal{N}$|")
    finally:
      plt.close(fig)

  def test_missing_required_fdot_file_raises(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    with pytest.raises(FileNotFoundError, match="fdot_integrated_M0.gkyl"):
      bal.balance("sim", ["ion"], quantity="particle", path=path)

  def test_bflux_override_and_absy_logy(self, stub, tmp_path):
    path = _build_particle_sim(stub, tmp_path, with_bflux=False)
    n = 5
    time = np.linspace(0.0, 1.0, n)
    override_vals = np.zeros((n, 4))
    override_vals[:, 0] = -0.05
    override_name = f"{path}custom_bflux.gkyl"
    stub.add(override_name, time, override_vals)

    fig, traces = bal.balance("sim", ["ion"],
                              quantity="particle",
                              path=path,
                              bflux_files={"xlower": "custom_bflux.gkyl"},
                              absy=True,
                              logy=True)
    try:
      assert traces["bflux_tot"] is not None
      np.testing.assert_allclose(traces["bflux_tot"], -0.05)
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
    fig, traces = bal.balance("sim", ["ion"],
                              quantity="particle",
                              path=path,
                              multib="0,1")
    try:
      # Two blocks, each contributing fdot=1.0, sum to 2.0 everywhere.
      np.testing.assert_allclose(traces["fdot"], 2.0)
    finally:
      plt.close(fig)


class TestEnergyMomentFileNames:
  """Gkeyll's ``<species>_<kind>integrated_<moment>.gkyl`` naming."""

  def test_bgk_source_is_its_own_term(self, stub, tmp_path):
    """S=0.1 and S_BGK=0.02 stay separate; the residual has both:
    0.1 + 0.02 - 0.05 - ((1+t) - 0.2)."""
    path = _build_energy_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    bgk_vals = np.zeros((5, 3))
    bgk_vals[:, 2] = 0.02
    stub.add(f"{path}sim-ion_source_bgk_integrated_HamiltonianMoments.gkyl",
             time, bgk_vals)

    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      np.testing.assert_allclose(traces["src"], 0.1, rtol=0, atol=1e-15)
      np.testing.assert_allclose(traces["src_bgk"], 0.02, rtol=0, atol=1e-15)
      np.testing.assert_allclose(traces["mom_err"],
                                 0.27 - (1.0 + time[1:]),
                                 rtol=0,
                                 atol=1e-14)
      # The residual's formula adds the BGK source.
      assert r"$+\mathcal{S}_{\mathrm{BGK}}$" in _legend_texts(fig)[-1]
    finally:
      plt.close(fig)

  def test_bgk_source_of_one_species_is_kept(self, stub, tmp_path):
    """The BGK source of the first species survives the second having none,
    and no source is not a zero source."""
    path = _build_energy_sim(stub,
                             tmp_path,
                             species=("ion", "elc"),
                             with_src=False)
    time = np.linspace(0.0, 1.0, 5)
    bgk_vals = np.zeros((5, 3))
    bgk_vals[:, 2] = 0.02
    stub.add(f"{path}sim-ion_source_bgk_integrated_HamiltonianMoments.gkyl",
             time, bgk_vals)

    fig, traces = bal.balance("sim", ["ion", "elc"],
                              quantity="energy",
                              path=path)
    try:
      assert traces["src"] is None
      np.testing.assert_allclose(traces["src_bgk"], 0.02, rtol=0, atol=1e-15)
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

    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      np.testing.assert_array_equal(traces["fdot"], 1.0 + time[1:])
      np.testing.assert_allclose(traces["src"], 0.1)
    finally:
      plt.close(fig)


class TestParticleMomentFileNames:
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
      fig, traces = bal.balance("sim", ["ion"],
                                quantity="particle",
                                path=path,
                                relative_error=relative)
      try:
        mom_err = -0.05 - (1.0 + time[1:])
        if relative:
          np.testing.assert_allclose(traces["mom_err_norm"], mom_err * 0.025)
        else:
          np.testing.assert_allclose(traces["mom_err"], mom_err)
      finally:
        plt.close(fig)

  def test_bgk_source_is_its_own_term(self, stub, tmp_path):
    """S=0.1 and S_BGK=0.02 stay separate; the residual has both:
    0.1 + 0.02 - 0.05 - (1+t)."""
    path = _build_particle_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    bgk_vals = np.zeros((5, 3))
    bgk_vals[:, 0] = 0.02
    stub.add(f"{path}sim-ion_source_bgk_integrated_M0M1M2.gkyl", time, bgk_vals)

    fig, traces = bal.balance("sim", ["ion"], quantity="particle", path=path)
    try:
      np.testing.assert_allclose(traces["src"], 0.1, rtol=0, atol=1e-15)
      np.testing.assert_allclose(traces["src_bgk"], 0.02, rtol=0, atol=1e-15)
      np.testing.assert_allclose(traces["mom_err"],
                                 0.07 - (1.0 + time[1:]),
                                 rtol=0,
                                 atol=1e-14)
    finally:
      plt.close(fig)

  def test_bgk_source_without_source(self, stub, tmp_path):
    path = _build_particle_sim(stub, tmp_path, with_src=False)
    time = np.linspace(0.0, 1.0, 5)
    bgk_vals = np.zeros((5, 3))
    bgk_vals[:, 0] = 0.02
    stub.add(f"{path}sim-ion_source_bgk_integrated_M0M1M2.gkyl", time, bgk_vals)

    fig, traces = bal.balance("sim", ["ion"], quantity="particle", path=path)
    try:
      assert traces["src"] is None
      np.testing.assert_allclose(traces["src_bgk"], 0.02, rtol=0, atol=1e-15)
    finally:
      plt.close(fig)

  def test_reads_older_integrated_moms_names(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    fdot_vals = np.zeros((5, 4))
    fdot_vals[:, 0] = 1.0 + time
    stub.add(f"{path}sim-ion_fdot_integrated_moms.gkyl", time, fdot_vals)

    fig, traces = bal.balance("sim", ["ion"], quantity="particle", path=path)
    try:
      np.testing.assert_array_equal(traces["fdot"], 1.0 + time[1:])
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


class TestEnergyFromBalanceFiles:
  """Fallback to Gkeyll's ``<name>-balance_energy.gkyl``."""

  def test_reads_columns_by_name(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _energy_balance_columns(time)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = bal.balance("sim", ["ion", "elc"],
                              quantity="energy",
                              path=path)
    try:
      # Rates are only defined after the first step: t=0 is skipped.
      np.testing.assert_array_equal(traces["time"], time[1:])
      for trace, column in ((traces["fdot"], "mom_dot"),
                            (traces["src"], "src_mom"), (traces["bflux_tot"],
                                                         "bflux_tot_mom"),
                            (traces["field_dot"], "field_dot"),
                            (traces["bg_dot"], "bg_dot"), (traces["mom_err"],
                                                           "mom_err")):
        np.testing.assert_array_equal(trace, columns[column][1:])
      assert traces["apar_dot"] is None
      assert traces["mom_err_norm"] is None
      # An older file: no src_bgk_mom column.
      assert traces["src_bgk"] is None
      assert any(r"\rho_{bg}" in text for text in _legend_texts(fig))
    finally:
      plt.close(fig)

  def test_reads_bgk_source_column(self, stub, tmp_path):
    """``src_bgk_mom=0.04`` is its own term and enters the residual's
    formula; a zero column is absent."""
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _energy_balance_columns(time)
    src_bgk = np.full_like(time, 0.04)
    src_bgk[0] = 0.0
    columns["src_bgk_mom"] = src_bgk
    columns["mom_err"] = columns["mom_err"] + src_bgk
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      np.testing.assert_array_equal(traces["src"], columns["src_mom"][1:])
      np.testing.assert_array_equal(traces["src_bgk"], src_bgk[1:])
      np.testing.assert_array_equal(traces["mom_err"], columns["mom_err"][1:])
      legend = _legend_texts(fig)
      assert r"$\mathcal{S}_{\mathrm{BGK}}$" in legend
      assert legend[-1].startswith(r"$E_{\dot{\mathcal{E}}}=$$\mathcal{S}$"
                                   r"$+\mathcal{S}_{\mathrm{BGK}}$$-")
    finally:
      plt.close(fig)

    columns["src_bgk_mom"] = np.zeros_like(time)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)
    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      assert traces["src_bgk"] is None
    finally:
      plt.close(fig)

  def test_relative_error(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _energy_balance_columns(time)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = bal.balance("sim", ["ion"],
                              quantity="energy",
                              path=path,
                              relative_error=True)
    try:
      np.testing.assert_array_equal(traces["time"], time[1:])
      np.testing.assert_array_equal(traces["fdot"], columns["mom_dot"][1:])
      np.testing.assert_array_equal(traces["mom_err_norm"],
                                    columns["mom_err_norm"][1:])
      assert traces["mom_err"] is None
    finally:
      plt.close(fig)

  def test_zero_terms_are_absent(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _energy_balance_columns(time, bg=0.0)
    columns["src_mom"][:] = 0.0
    columns["bflux_tot_mom"][:] = 0.0
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      assert traces["src"] is None
      assert traces["bflux_tot"] is None
      assert traces["bg_dot"] is None
      # Field, fdot and residual curves only.
      assert len(_legend_texts(fig)) == 3
    finally:
      plt.close(fig)

  def test_moment_files_take_precedence(self, stub, tmp_path):
    path = _build_energy_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time,
                     _energy_balance_columns(time))

    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      np.testing.assert_array_equal(traces["fdot"],
                                    np.linspace(1.0, 2.0, 5)[1:])
      assert traces["bg_dot"] is None
    finally:
      plt.close(fig)

  def test_missing_field_energy_dot_falls_back(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add(f"{path}sim-ion_fdot_integrated_HamiltonianMoments.gkyl", time,
             np.ones((5, 3)))
    columns = _energy_balance_columns(time)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = bal.balance("sim", ["ion"], quantity="energy", path=path)
    try:
      np.testing.assert_array_equal(traces["fdot"], columns["mom_dot"][1:])
    finally:
      plt.close(fig)

  def test_species_balance_files_are_not_read(self, stub, tmp_path):
    """The per-species energy balance files lack the field terms."""
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-ion_balance_energy.gkyl", time,
                     _particle_balance_columns(time))
    with pytest.raises(FileNotFoundError, match="field_energy_dot"):
      bal.balance("sim", ["ion"], quantity="energy", path=path)

  def test_path_override_selects_moment_files(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add(f"{path}sim-field_energy_dot.gkyl", time, np.zeros((5, 1)))
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time,
                     _energy_balance_columns(time))
    with pytest.raises(FileNotFoundError, match="custom_fdot"):
      bal.balance("sim", ["ion"],
                  quantity="energy",
                  path=path,
                  fdot_file="custom_fdot.gkyl")

  def test_multiblock_sums_blocks(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    block0 = _energy_balance_columns(time)
    block1 = {key: 3.0 * values for key, values in block0.items()}
    stub.add_balance(f"{path}sim_b0-balance_energy.gkyl", time, block0)
    stub.add_balance(f"{path}sim_b1-balance_energy.gkyl", time, block1)

    fig, traces = bal.balance("sim", ["ion"],
                              quantity="energy",
                              path=path,
                              multib="-1")
    try:
      np.testing.assert_allclose(traces["fdot"],
                                 4.0 * block0["mom_dot"][1:],
                                 rtol=0,
                                 atol=1e-14)
      np.testing.assert_allclose(traces["mom_err"],
                                 4.0 * block0["mom_err"][1:],
                                 rtol=0,
                                 atol=1e-14)
    finally:
      plt.close(fig)

  def test_multiblock_relative_error_per_block(self, stub, tmp_path):
    """Each block normalizes by its own energy: one curve per block, with
    block 1's errors 3 times block 0's."""
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    block0 = _energy_balance_columns(time)
    stub.add_balance(f"{path}sim_b0-balance_energy.gkyl", time, block0)
    stub.add_balance(f"{path}sim_b1-balance_energy.gkyl", time, {
        key: 3.0 * values
        for key, values in block0.items()
    })

    fig, traces = bal.balance("sim", ["ion"],
                              quantity="energy",
                              path=path,
                              multib="-1",
                              relative_error=True)
    try:
      assert traces["mom_err"] is None
      assert traces["mom_err_norm"] is None
      assert list(traces["mom_err_norm_parts"]) == ["block 0", "block 1"]
      np.testing.assert_array_equal(traces["mom_err_norm_parts"]["block 0"],
                                    block0["mom_err_norm"][1:])
      np.testing.assert_array_equal(traces["mom_err_norm_parts"]["block 1"],
                                    3.0 * block0["mom_err_norm"][1:])
      assert _legend_texts(fig) == ["block 0", "block 1"]
    finally:
      plt.close(fig)

  def test_no_species_reads_balance_file(self, stub, tmp_path):
    """Without species the moment files cannot be read, even if present."""
    path = _build_energy_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    columns = _energy_balance_columns(time)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time, columns)

    fig, traces = bal.balance("sim", quantity="energy", path=path)
    try:
      np.testing.assert_array_equal(traces["fdot"], columns["mom_dot"][1:])
      np.testing.assert_array_equal(traces["mom_err"], columns["mom_err"][1:])
    finally:
      plt.close(fig)

  def test_no_species_without_balance_file_raises(self, stub, tmp_path):
    path = _build_energy_sim(stub, tmp_path)
    with pytest.raises(FileNotFoundError, match="sim-balance_energy"):
      bal.balance("sim", quantity="energy", path=path)

  def test_no_species_rejects_path_overrides(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-balance_energy.gkyl", time,
                     _energy_balance_columns(time))
    with pytest.raises(ValueError, match="need the species"):
      bal.balance("sim",
                  quantity="energy",
                  path=path,
                  fdot_file="custom_fdot.gkyl")


class TestParticleFromBalanceFiles:
  """Fallback to Gkeyll's ``<name>-<species>_balance_particle.gkyl``."""

  def test_reads_columns_by_name(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _particle_balance_columns(time)
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time, columns)

    for relative in (False, True):
      fig, traces = bal.balance("sim", ["ion"],
                                quantity="particle",
                                path=path,
                                relative_error=relative)
      try:
        np.testing.assert_array_equal(traces["time"], time[1:])
        np.testing.assert_array_equal(traces["fdot"], columns["mom_dot"][1:])
        np.testing.assert_array_equal(traces["src"], columns["src_mom"][1:])
        np.testing.assert_array_equal(traces["bflux_tot"],
                                      columns["bflux_tot_mom"][1:])
        # An older file: no src_bgk_mom column.
        assert traces["src_bgk"] is None
        assert traces["field_dot"] is None
        if relative:
          np.testing.assert_array_equal(traces["mom_err_norm"],
                                        columns["mom_err_norm"][1:])
          assert traces["mom_err"] is None
        else:
          np.testing.assert_array_equal(traces["mom_err"],
                                        columns["mom_err"][1:])
          assert traces["mom_err_norm"] is None
      finally:
        plt.close(fig)

  def test_reads_bgk_source_column(self, stub, tmp_path):
    """``src_bgk_mom=0.02`` is its own term; a zero column is absent."""
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _particle_balance_columns(time)
    src_bgk = np.full_like(time, 0.02)
    src_bgk[0] = 0.0
    columns["src_bgk_mom"] = src_bgk
    columns["mom_err"] = columns["mom_err"] + src_bgk
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time, columns)

    fig, traces = bal.balance("sim", ["ion"], quantity="particle", path=path)
    try:
      np.testing.assert_array_equal(traces["src"], columns["src_mom"][1:])
      np.testing.assert_array_equal(traces["src_bgk"], src_bgk[1:])
      np.testing.assert_array_equal(traces["mom_err"], columns["mom_err"][1:])
    finally:
      plt.close(fig)

    columns["src_bgk_mom"] = np.zeros_like(time)
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time, columns)
    fig, traces = bal.balance("sim", ["ion"], quantity="particle", path=path)
    try:
      assert traces["src_bgk"] is None
    finally:
      plt.close(fig)

  def test_sums_species_files(self, stub, tmp_path):
    """The elc file is twice the ion one: the sum is three times it, and
    each file normalizes by its own particle number."""
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    columns = _particle_balance_columns(time)
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time, columns)
    stub.add_balance(f"{path}sim-elc_balance_particle.gkyl", time, {
        key: 2.0 * v
        for key, v in columns.items()
    })

    fig, traces = bal.balance("sim", ["ion", "elc"],
                              quantity="particle",
                              path=path)
    try:
      np.testing.assert_allclose(traces["mom_err"],
                                 3.0 * columns["mom_err"][1:],
                                 rtol=0,
                                 atol=1e-14)
    finally:
      plt.close(fig)

    fig, traces = bal.balance("sim", ["ion", "elc"],
                              quantity="particle",
                              path=path,
                              relative_error=True)
    try:
      assert traces["mom_err_norm"] is None
      assert list(traces["mom_err_norm_parts"]) == ["ion", "elc"]
      np.testing.assert_array_equal(traces["mom_err_norm_parts"]["ion"],
                                    columns["mom_err_norm"][1:])
      np.testing.assert_array_equal(traces["mom_err_norm_parts"]["elc"],
                                    2.0 * columns["mom_err_norm"][1:])
    finally:
      plt.close(fig)

  def test_species_sum_is_not_a_species_balance(self, stub, tmp_path):
    """``<name>-balance_particle.gkyl`` sums over species: no fallback."""
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-balance_particle.gkyl", time,
                     _particle_balance_columns(time))
    with pytest.raises(FileNotFoundError, match="ion_fdot_integrated_M0.gkyl"):
      bal.balance("sim", ["ion"], quantity="particle", path=path)

  def test_moment_files_take_precedence(self, stub, tmp_path):
    path = _build_particle_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time,
                     _particle_balance_columns(time))

    fig, traces = bal.balance("sim", ["ion"], quantity="particle", path=path)
    try:
      np.testing.assert_array_equal(traces["fdot"],
                                    np.linspace(1.0, 2.0, 5)[1:])
    finally:
      plt.close(fig)

  def test_path_override_selects_moment_files(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time,
                     _particle_balance_columns(time))
    with pytest.raises(FileNotFoundError, match="custom_fdot"):
      bal.balance("sim", ["ion"],
                  quantity="particle",
                  path=path,
                  fdot_file="custom_fdot.gkyl")

  def test_multiblock_sums_blocks(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    block0 = _particle_balance_columns(time)
    block1 = {key: 3.0 * values for key, values in block0.items()}
    stub.add_balance(f"{path}sim_b0-ion_balance_particle.gkyl", time, block0)
    stub.add_balance(f"{path}sim_b1-ion_balance_particle.gkyl", time, block1)

    fig, traces = bal.balance("sim", ["ion"],
                              quantity="particle",
                              path=path,
                              multib="0,1")
    try:
      np.testing.assert_allclose(traces["mom_err"],
                                 4.0 * block0["mom_err"][1:],
                                 rtol=0,
                                 atol=1e-14)
    finally:
      plt.close(fig)

    fig, traces = bal.balance("sim", ["ion"],
                              quantity="particle",
                              path=path,
                              multib="0,1",
                              relative_error=True)
    try:
      assert list(
          traces["mom_err_norm_parts"]) == ["block 0 ion", "block 1 ion"]
      np.testing.assert_array_equal(traces["mom_err_norm_parts"]["block 1 ion"],
                                    block1["mom_err_norm"][1:])
    finally:
      plt.close(fig)

  def test_twelve_parts_change_line_style_after_ten_colors(
      self, stub, tmp_path):
    """Matplotlib cycles 10 colors: blocks 10 and 11 are dashed."""
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    for block in range(12):
      stub.add_balance(f"{path}sim_b{block}-balance_particle.gkyl", time,
                       _particle_balance_columns(time))

    fig, traces = bal.balance("sim",
                              quantity="particle",
                              path=path,
                              multib="-1",
                              relative_error=True)
    try:
      assert len(traces["mom_err_norm_parts"]) == 12
      # The first line is the zero reference.
      styles = [line.get_linestyle() for line in fig.axes[0].get_lines()[1:]]
      assert styles == ["-"] * 10 + ["--"] * 2
    finally:
      plt.close(fig)

  def test_no_species_reads_species_sum(self, stub, tmp_path):
    """Without species, ``<name>-balance_particle.gkyl`` is read even if the
    species files are present."""
    path = _build_particle_sim(stub, tmp_path)
    time = np.linspace(0.0, 1.0, 5)
    columns = _particle_balance_columns(time)
    stub.add_balance(f"{path}sim-balance_particle.gkyl", time, columns)
    stub.add_balance(f"{path}sim-ion_balance_particle.gkyl", time, {
        key: 2.0 * v
        for key, v in columns.items()
    })

    for relative in (False, True):
      fig, traces = bal.balance("sim",
                                quantity="particle",
                                path=path,
                                relative_error=relative)
      try:
        if relative:
          np.testing.assert_array_equal(traces["mom_err_norm"],
                                        columns["mom_err_norm"][1:])
        else:
          np.testing.assert_array_equal(traces["fdot"], columns["mom_dot"][1:])
          np.testing.assert_array_equal(traces["mom_err"],
                                        columns["mom_err"][1:])
      finally:
        plt.close(fig)

  def test_no_species_without_balance_file_raises(self, stub, tmp_path):
    path = _build_particle_sim(stub, tmp_path)
    with pytest.raises(FileNotFoundError, match="sim-balance_particle"):
      bal.balance("sim", quantity="particle", path=path)

  def test_no_species_rejects_path_overrides(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    time = np.linspace(0.0, 1.0, 5)
    stub.add_balance(f"{path}sim-balance_particle.gkyl", time,
                     _particle_balance_columns(time))
    with pytest.raises(ValueError, match="need the species"):
      bal.balance("sim",
                  quantity="particle",
                  path=path,
                  fdot_file="custom_fdot.gkyl")


class TestReadBalanceFiles:

  def test_single_sample_keeps_time_axis(self, stub, tmp_path):
    """A run stopped before its first step has one (t=0) row, which
    ``read_gfile`` squeezes to 0-D time and 1-D values."""
    name = str(tmp_path / "one-balance_particle.gkyl")
    stub.add(name, np.array([0.0]), np.zeros((1, 2)),
             {"Columns": "mom_err, mom_err_norm"})
    time, columns = bal._read_balance_files([name])
    np.testing.assert_array_equal(time, [0.0])
    np.testing.assert_array_equal(columns["mom_err"], [0.0])

  def test_missing_columns_metadata_raises(self, stub, tmp_path):
    name = str(tmp_path / "bare-balance_particle.gkyl")
    stub.add(name, np.linspace(0.0, 1.0, 3), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="Columns"):
      bal._read_balance_files([name])


class TestRealFixtures:
  """Real end-to-end runs against generated dynvector files."""

  def test_energy_balance(self):
    path = os.path.join(DATA, "generated", "gk_programs")
    time = np.linspace(0.0, 1.0, 5)[1:]  # t=0 is skipped.
    fdot = 2.0 + time
    mom_err = 0.3 - 0.1 - (fdot - 0.4)
    for relative in (False, True):
      fig, traces = bal.balance("sim", ["ion"],
                                quantity="energy",
                                path=path,
                                relative_error=relative,
                                field_dot_file="sim-field_energy_dot.gkyl")
      try:
        np.testing.assert_array_equal(traces["time"], time)
        np.testing.assert_allclose(traces["fdot"], fdot, rtol=0, atol=1e-14)
        np.testing.assert_allclose(traces["field_dot"], 0.4, rtol=0, atol=1e-14)
        if relative:
          np.testing.assert_allclose(traces["mom_err_norm"],
                                     mom_err * 0.25 / 18.0,
                                     rtol=0,
                                     atol=1e-14)
          assert traces["mom_err"] is None
        else:
          np.testing.assert_allclose(traces["mom_err"],
                                     mom_err,
                                     rtol=0,
                                     atol=1e-14)
      finally:
        plt.close(fig)

  def test_particle_balance(self):
    path = os.path.join(DATA, "generated", "gk_programs")
    time = np.linspace(0.0, 1.0, 5)[1:]  # t=0 is skipped.
    fdot = 1.0 + time
    mom_err = 0.1 - 0.05 - fdot
    for relative in (False, True):
      fig, traces = bal.balance("sim", ["ion"],
                                quantity="particle",
                                path=path,
                                relative_error=relative)
      try:
        np.testing.assert_array_equal(traces["time"], time)
        np.testing.assert_allclose(traces["fdot"], fdot, rtol=0, atol=1e-14)
        np.testing.assert_allclose(traces["bflux_tot"],
                                   0.05,
                                   rtol=0,
                                   atol=1e-14)
        if relative:
          np.testing.assert_allclose(traces["mom_err_norm"],
                                     mom_err * 0.25 / 10.0,
                                     rtol=0,
                                     atol=1e-14)
          assert traces["mom_err"] is None
        else:
          np.testing.assert_allclose(traces["mom_err"],
                                     mom_err,
                                     rtol=0,
                                     atol=1e-14)
      finally:
        plt.close(fig)

  def test_energy_balance_file(self):
    """Gkeyll's balance file without the moment files (``bal`` fixture)."""
    path = os.path.join(DATA, "generated", "gk_programs")
    time = np.linspace(0.0, 1.0, 5)[1:]  # t=0 is skipped.
    fdot = 2.0 + time
    mom_err = 0.3 + 0.04 - 0.1 - (fdot - 0.4 + 0.05)
    for relative in (False, True):
      fig, traces = bal.balance("bal", ["ion"],
                                quantity="energy",
                                path=path,
                                relative_error=relative)
      try:
        np.testing.assert_array_equal(traces["time"], time)
        for actual, expected in ((traces["fdot"], fdot), (traces["src"], 0.3),
                                 (traces["src_bgk"],
                                  0.04), (traces["bflux_tot"], 0.1),
                                 (traces["field_dot"], 0.4), (traces["bg_dot"],
                                                              0.05)):
          np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        if relative:
          np.testing.assert_allclose(traces["mom_err_norm"],
                                     mom_err * 0.25 / 18.0,
                                     rtol=0,
                                     atol=1e-14)
          assert traces["mom_err"] is None
        else:
          np.testing.assert_allclose(traces["mom_err"],
                                     mom_err,
                                     rtol=0,
                                     atol=1e-14)
      finally:
        plt.close(fig)

  @pytest.mark.parametrize("species", [["ion"], None])
  def test_particle_balance_file(self, species):
    """Gkeyll's balance files without the moment files (``bal`` fixture), of
    the ion and summed over its single species."""
    path = os.path.join(DATA, "generated", "gk_programs")
    time = np.linspace(0.0, 1.0, 5)[1:]  # t=0 is skipped.
    fdot = 1.0 + time
    mom_err = 0.1 + 0.02 - 0.05 - fdot
    for relative in (False, True):
      fig, traces = bal.balance("bal",
                                species,
                                quantity="particle",
                                path=path,
                                relative_error=relative)
      try:
        np.testing.assert_array_equal(traces["time"], time)
        for actual, expected in ((traces["fdot"], fdot), (traces["src"], 0.1),
                                 (traces["src_bgk"],
                                  0.02), (traces["bflux_tot"], 0.05)):
          np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        if relative:
          np.testing.assert_allclose(traces["mom_err_norm"],
                                     mom_err * 0.25 / 10.0,
                                     rtol=0,
                                     atol=1e-14)
          assert traces["mom_err"] is None
        else:
          np.testing.assert_allclose(traces["mom_err"],
                                     mom_err,
                                     rtol=0,
                                     atol=1e-14)
      finally:
        plt.close(fig)


@pytest.mark.parametrize("source_moments, columns, energy", [
    ("HamiltonianMoments", (0.0, 0.0, 4.0), 4.0),
    ("M0M1M2", (0.0, 0.0, 4.0), 0.5 * 2.0 * 4.0),
    ("M0M1M2parM2perp", (0.0, 0.0, 3.0, 1.0), 0.5 * 2.0 * (3.0 + 1.0)),
])
def test_energy_source_of_an_older_moments_file(stub, tmp_path, source_moments,
                                                columns, energy):
  # ``<species>_source_integrated_moms`` does not name its moments: for
  # M0M1M2 kinds the energy is m/2 M2 (mass 2 here), for the Hamiltonian its
  # third component.
  path = _build_energy_sim(stub, tmp_path, with_src=False)
  time = np.linspace(0.0, 1.0, 5)
  stub.add(f"{path}sim-ion_source_integrated_moms.gkyl", time,
           np.tile(columns, (5, 1)), {"mass": 2.0})
  fig, traces = bal.balance("sim", ["ion"],
                            quantity="energy",
                            path=path,
                            source_moments=source_moments)
  plt.close(fig)
  np.testing.assert_allclose(traces["src"], energy, rtol=1e-15)


def test_source_moments_leaves_named_and_particle_files_alone(stub, tmp_path):
  # A file named for its moments says what it holds; the particle balance
  # reads M0 whatever the kind.
  path = _build_energy_sim(stub, tmp_path)
  fig, traces = bal.balance("sim", ["ion"],
                            quantity="energy",
                            path=path,
                            source_moments="M0M1M2")
  plt.close(fig)
  np.testing.assert_allclose(traces["src"], 0.1, rtol=1e-15)
