"""Fourier reconstruction of a periodic field-aligned poloidal slice.

The binormal direction is periodic; parallel endpoints obey twist-and-shift.
All coordinate and boundary information is supplied as numerical arrays.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator


def fft_poloidal_project(values: np.ndarray, zc: np.ndarray, box: float,
                         wind: np.ndarray, phi0_zf: np.ndarray, zf: np.ndarray,
                         phi_tor: float) -> np.ndarray:
  """FFT twist-and-shift reconstruction at one physical toroidal angle."""
  nx, ny, nz = values.shape
  fk = np.fft.rfft(values, axis=1, norm="forward")
  mode_count = fk.shape[1]

  dz = zc[1] - zc[0]
  z_extended = np.concatenate(([zc[0] - dz / 2], zc, [zc[-1] + dz / 2]))
  fk_extended = np.zeros((nx, mode_count, nz + 2), dtype=complex)
  fk_extended[:, :, 1:-1] = fk
  phase_shift = (2.0 * np.pi / box) * wind
  for mode in range(mode_count):
    phase = np.exp(-1j * mode * phase_shift)
    fk_extended[:, mode, -1] = 0.5 * (fk[:, mode, -1] + phase * fk[:, mode, 0])
    fk_extended[:, mode,
                0] = 0.5 * (fk[:, mode, 0] + np.conj(phase) * fk[:, mode, -1])

  fk_zf = (PchipInterpolator(z_extended, fk_extended.real, axis=2)(zf) +
           1j * PchipInterpolator(z_extended, fk_extended.imag, axis=2)(zf))

  fraction = (phi_tor - phi0_zf) / box
  out = np.zeros((nx, len(zf)))
  for mode in range(mode_count):
    weight = 1.0 if (mode == 0 or
                     (ny % 2 == 0 and mode == mode_count - 1)) else 2.0
    out += weight * np.real(
        fk_zf[:, mode, :] * np.exp(-1j * 2.0 * np.pi * mode * fraction))
  return out
