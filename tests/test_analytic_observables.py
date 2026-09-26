"""Physical observables with analytic answers from existing generated data."""

from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.mom import five_moment as fm

GEN = Path(__file__).parent / "test_data" / "generated"
needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
@pytest.mark.parametrize("frame", [0, 3, 11])
@pytest.mark.parametrize("psd", [False, True])
def test_travelling_wave_fourier_amplitude_phase_and_frequencies(frame, psd):
  # These p0 files store center samples, not the exact continuous cosine's
  # cell averages. One interpolation point per cell recovers those samples.
  data = pg.load(GEN /
                 f"travelling_wave_{frame:03d}.gkyl").interpolate(num_interp=1)
  spectrum = data.fft(psd=psd)
  n = 64
  phase = np.pi / n - 2 * np.pi * frame / 16
  expected = np.zeros((n, 1), dtype=complex)
  expected[0, 0] = n
  expected[1, 0] = 0.3 * n * np.exp(1j * phase)
  expected[-1, 0] = 0.3 * n * np.exp(-1j * phase)
  # DFT bins in cycles/unit length on [0,2*pi], in standard FFT order.
  frequencies = np.r_[np.arange(n // 2), np.arange(-n // 2, 0)] / (2 * np.pi)
  if psd:
    expected = np.abs(expected[:n // 2])**2
    frequencies = frequencies[:n // 2]
  np.testing.assert_allclose(spectrum.grid[0],
                             frequencies,
                             rtol=2e-14,
                             atol=2e-14)
  np.testing.assert_allclose(spectrum.values, expected, rtol=2e-13, atol=3e-12)
  if not psd:
    # Mean square = 1 + 0.6^2/2. All bins, not just the peak, enter Parseval.
    assert np.sum(np.abs(spectrum.values)**2) / n**2 == pytest.approx(1.18,
                                                                      rel=2e-13)


@needs_gkeyll
@pytest.mark.parametrize("frame", [0, 5])
def test_wave_surface_fourier_modes_include_both_axis_phases(frame):
  data = pg.load(GEN /
                 f"wave_surface_{frame:03d}.gkyl").interpolate(num_interp=1)
  spectrum = data.fft()
  n = 32
  time = 2 * np.pi * frame / 12
  expected = np.zeros((n, n, 1), dtype=complex)
  expected[0, 0, 0] = n * n
  # cos(x-t)*cos(y) has four complex Fourier modes of amplitude 0.6/4.
  for kx in (-1, 1):
    for ky in (-1, 1):
      expected[kx, ky,
               0] = 0.15 * n * n * np.exp(1j *
                                          ((kx + ky) * np.pi / n - kx * time))
  frequencies = np.r_[np.arange(n // 2), np.arange(-n // 2, 0)] / (2 * np.pi)
  for axis in spectrum.grid:
    np.testing.assert_allclose(axis, frequencies, rtol=2e-14, atol=2e-14)
  np.testing.assert_allclose(spectrum.values, expected, rtol=2e-13, atol=3e-11)
  assert np.sum(np.abs(spectrum.values)**2) / n**4 == pytest.approx(1.09,
                                                                    rel=2e-13)


@pytest.mark.parametrize("method", ["fit", "growth"])
def test_exponential_energy_recovers_amplitude_growth_rate(method):
  data = pg.load(GEN / "exponential_energy.gkyl")
  # E(t)=1e-6 exp(2*gamma*t): the amplitude growth rate is gamma=0.2,
  # not the energy exponent 0.4. Start away from the analytic parameters.
  result = (data.fit("exp2", guess=(5e-7, 0.15))
            if method == "fit" else data.growth(guess=(5e-7, 0.15), min_n=20))
  np.testing.assert_allclose(result.ctx["fit_params"], [[1e-6, 0.2]],
                             rtol=2e-7,
                             atol=1e-14)
  times = np.linspace(0., 10., 101)
  np.testing.assert_allclose(result.values, (1e-6 * np.exp(0.4 * times))[:,
                                                                         None],
                             rtol=2e-7,
                             atol=1e-14)


@needs_gkeyll
@pytest.mark.parametrize(
    "quantity", ["density", "vel", "pressure", "ke", "temp", "sound", "mach"])
def test_shock_tube_primitives_on_both_sides_of_discontinuity(quantity):
  data = pg.load(GEN / "shock_tube_1d_p0.gkyl").interpolate(num_interp=1)
  rho = np.r_[np.ones(50), np.full(50, 0.125)]
  pressure = np.r_[np.ones(50), np.full(50, 0.1)]
  expected = {
      "density": rho[:, None],
      "vel": np.zeros((100, 3)),
      "pressure": pressure[:, None],
      "ke": np.zeros((100, 1)),
      "temp": (pressure / rho)[:, None],
      "sound": np.sqrt((5 / 3) * pressure / rho)[:, None],
      "mach": np.zeros((100, 1)),
  }[quantity]
  result = getattr(fm, quantity)(data)
  np.testing.assert_allclose(result.values, expected, rtol=3e-14, atol=3e-14)


@needs_gkeyll
def test_shock_tube_integrated_mass_momentum_and_energy():
  # Native p0 integration is unsupported. Sampling once per cell recovers
  # the exact cell averages of this piecewise-constant state.
  data = pg.load(GEN / "shock_tube_1d_p0.gkyl").interpolate(num_interp=1)
  # Each state occupies half the unit interval: M=(1+1/8)/2,
  # E=(1+1/10)/(2*(5/3-1)); all three momenta vanish.
  np.testing.assert_allclose(data.integrate(), [0.5625, 0., 0., 0., 0.825],
                             rtol=3e-14,
                             atol=3e-14)
