"""Tests for postgkyl.numerics.filters -- fft_filtering and butter_filtering."""

from __future__ import annotations

import numpy as np
import pytest

from postgkyl.numerics.filters import fft_filtering, butter_filtering


class TestFftFiltering:

  def test_removes_high_frequency_component(self):
    N = 256
    dt = 1.0 / N
    t = np.linspace(0.0, 1.0 - dt, N)
    signal = np.sin(2 * 2 * np.pi * t) + 0.5 * np.sin(50 * 2 * np.pi * t)
    filtered = fft_filtering(signal, dt=dt, cutoff=10.0)
    high_freq_power_before = 0.5
    high_freq_power_after = np.std(
        np.real(filtered) - np.sin(2 * 2 * np.pi * t))
    assert high_freq_power_after < 0.1 * high_freq_power_before

  def test_preserves_dc_component(self):
    N = 128
    dt = 1.0 / N
    signal = np.ones(N) * 3.0
    filtered = fft_filtering(signal, dt=dt, cutoff=1.0)
    np.testing.assert_allclose(np.real(filtered), 3.0, atol=1e-10)

  def test_output_same_length(self):
    N = 64
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(N)
    filtered = fft_filtering(signal, dt=0.01, cutoff=10.0)
    assert len(filtered) == N

  def test_cutoff_zero_removes_all(self):
    N = 64
    signal = np.sin(2 * np.pi * np.linspace(0, 1, N))
    filtered = fft_filtering(signal, dt=1.0 / N, cutoff=0.0)
    np.testing.assert_allclose(np.abs(filtered).max(), 0.0, atol=1e-10)

  def test_cutoff_is_keyword_only(self):
    with pytest.raises(TypeError):
      fft_filtering(np.ones(8), 1.0, 5.0)  # type: ignore[misc]

  def test_cutoff_required(self):
    with pytest.raises(TypeError):
      fft_filtering(np.ones(8))  # type: ignore[call-arg]


class TestButterFiltering:

  def test_removes_high_frequency(self):
    N = 512
    dt = 1.0 / N
    t = np.linspace(0.0, 1.0, N)
    low = np.sin(2 * 2 * np.pi * t)
    high = 0.5 * np.sin(100 * 2 * np.pi * t)
    filtered = butter_filtering(low + high, dt=dt, cutoff=10.0)
    skip = N // 5
    std_filtered = np.std(filtered[skip:])
    std_original = np.std((low + high)[skip:])
    assert std_filtered < std_original

  def test_output_same_length(self):
    N = 64
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(N)
    filtered = butter_filtering(signal, dt=0.01, cutoff=5.0)
    assert len(filtered) == N

  def test_preserves_low_frequency(self):
    N = 512
    dt = 1.0 / N
    t = np.linspace(0.0, 1.0, N)
    freq = 1.0
    signal = np.sin(2 * np.pi * freq * t)
    filtered = butter_filtering(signal, dt=dt, cutoff=100.0)
    skip = N // 5
    np.testing.assert_allclose(np.max(np.abs(filtered[skip:])), 1.0, atol=0.05)

  def test_cutoff_required(self):
    with pytest.raises(TypeError):
      butter_filtering(np.ones(8))  # type: ignore[call-arg]
