"""Analytic checks of periodic poloidal reconstruction."""
import numpy as np
import pytest

from postgkyl.numerics.rz import fft_poloidal_project


@pytest.mark.parametrize("ny", [7, 8])
@pytest.mark.parametrize("angle", [0.0, 0.37, np.pi / 2])
def test_fourier_mode_reconstructs_at_physical_angle(ny, angle):
  zc = np.linspace(-0.75, 0.75, 4)
  samples = np.cos(2 * np.pi * np.arange(ny) / ny)
  values = np.broadcast_to(samples[None, :, None], (2, ny, 4))
  zf = np.linspace(-1.0, 1.0, 9)
  reconstructed = fft_poloidal_project(values, zc, 2 * np.pi, np.zeros(2),
                                       np.zeros((2, 9)), zf, angle)
  np.testing.assert_allclose(reconstructed, np.cos(angle), atol=1e-14)
