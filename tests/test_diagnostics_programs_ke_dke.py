"""Energy conservation references for actual 2D/3D fluid frame files."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.mom import ke_dke as kd

GEN = Path(__file__).parent / "test_data" / "generated"
needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("timing, slopes", [
    ("uniform", [-15.0, -25.0]),
    ("irregular", [-15.0, -25.0 / 3.0]),
    ("untimed", [-15.0, -25.0]),
])
def test_real_frames_use_physical_energy_and_timestamps(ndim, timing, slopes):
  # rho=1, u=(1,0,0), (2,0,0), (3,0,0) on a unit domain. Their
  # energies are independently 1/2, 2, 9/2. Frame timestamps take precedence
  # over legacy endpoints; only untimed files need those endpoints.
  fallback_end = 0.2 if timing == "untimed" else 99.0
  out = kd.ke_dke(str(GEN / f"moments_{ndim}d_{timing}_"),
                  0,
                  2,
                  dim=ndim,
                  vol=1.0,
                  init_time=0.0,
                  final_time=fallback_end)
  np.testing.assert_allclose(out.ke, [0.5, 2.0, 4.5], rtol=2e-14, atol=2e-14)
  np.testing.assert_allclose(out.dke, slopes, rtol=2e-14, atol=2e-13)


@needs_gkeyll
@pytest.mark.parametrize("ndim, energy", [(2, 4.0 / 3.0), (3, 13.0 / 6.0)])
def test_nonconstant_momentum_has_exact_integrated_energy(ndim, energy):
  # rho=2 and u=(y+z,-x,2*x-y), with z=0 in 2D. Integrating
  # 1/2*rho*|u|² on the unit domain gives 4/3 (2D) and 13/6 (3D).
  out = kd.ke_dke(str(GEN / f"moments_{ndim}d_affine_"),
                  0,
                  0,
                  dim=ndim,
                  vol=2.5,
                  init_time=0.0,
                  final_time=0.0)
  np.testing.assert_allclose(out.ke, [2.5 * energy], rtol=2e-14, atol=2e-14)
  assert out.dke.shape == (0, )


@pytest.mark.parametrize("ndim", [2, 3])
def test_point_cell_values_use_true_dimensions_and_all_velocity_components(
    ndim, monkeypatch):
  grid = [np.linspace(0, 1, n + 1) for n in [2, 3, 4][:ndim]]
  cells = tuple(len(edges) - 1 for edges in grid)
  loaded = []

  def load(name):
    loaded.append(name)
    frame = int(Path(name).stem.rsplit("_", 1)[1])
    # rho=2, u=(a, 2a, -a); KE=6a² on a unit domain.
    a = frame - 3
    values = np.broadcast_to([2, 2 * a, 4 * a, -2 * a], (*cells, 4)).copy()
    data = pg.GData(ctx={"time": (frame - 4) * 0.1})
    data.push(grid, values)
    return data

  monkeypatch.setattr(kd, "GData", load)
  out = kd.ke_dke("physical-fluid_",
                  4,
                  6,
                  dim=ndim,
                  vol=1,
                  init_time=0,
                  final_time=100,
                  extension="dat")
  assert loaded == [f"physical-fluid_{i}.dat" for i in (4, 5, 6)]
  np.testing.assert_allclose(out.ke, [6, 24, 54], rtol=0, atol=2e-13)
  np.testing.assert_allclose(out.dke, [-180, -300], rtol=0, atol=3e-12)


@pytest.mark.parametrize("times", [[0, 0, 1], [0, -1, 1], [0, np.nan, 1]])
def test_invalid_timestamp_intervals_are_rejected(times):
  with pytest.raises(ValueError, match="timestamps"):
    kd._dissipation_rate(np.array([0.5, 2, 4.5]), times)


@needs_gkeyll
def test_dimension_mismatch_is_explicit():
  with pytest.raises(ValueError, match="has 2 dimensions, expected 3"):
    kd.ke_dke(str(GEN / "moments_2d_uniform_"),
              0,
              0,
              dim=3,
              vol=1,
              init_time=0,
              final_time=0)


@needs_gkeyll
def test_mixed_missing_and_present_timestamps_are_rejected(monkeypatch):

  def load(name):
    frame = int(Path(name).stem.rsplit("_", 1)[1])
    timing = "untimed" if frame == 1 else "uniform"
    return pg.load(str(GEN / f"moments_2d_{timing}_{frame}.gkyl"))

  monkeypatch.setattr(kd, "GData", load)
  with pytest.raises(ValueError, match="every frame or none"):
    kd.ke_dke("fluid_", 0, 2, dim=2, vol=1, init_time=0, final_time=1)


def test_traces_are_frozen():
  traces = kd.KineticEnergyTraces(ke=np.array([1.0]), dke=np.array([]))
  with pytest.raises(FrozenInstanceError):
    traces.ke = np.array([2.0])


@pytest.mark.parametrize("dim, first, last, message", [
    (1, 0, 1, "requires dim=2 or dim=3"),
    (2, 2, 1, "final_frame must be at least init_frame"),
])
def test_invalid_frame_requests_fail_before_loading(dim, first, last, message):
  with pytest.raises(ValueError, match=message):
    kd.ke_dke("unused_", first, last, dim, 1., 0., 1.)
