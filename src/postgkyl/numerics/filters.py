"""Low-pass filtering: FFT brick-wall and Butterworth.

The legacy ``tools/filters.py`` fell back to an interactive matplotlib
click-to-pick cutoff frequency when ``cutoff`` was omitted. That picker is
an effect at the edge (it pops up a figure and blocks on a GUI event) and
does not belong in a pure-array leaf module; it has not been ported here.
If anyone still wants that convenience, it belongs in ``render``/``cli``,
built on top of :func:`fft_filtering`. Consequently ``cutoff`` is a
required argument here rather than optional.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


def fft_filtering(data: np.ndarray,
                  dt: float = 1.0,
                  *,
                  cutoff: float) -> np.ndarray:
  """Low-pass filter ``data`` by zeroing FFT bins above ``cutoff``.

  Args:
    data: 1-D signal.
    dt: Sample spacing.
    cutoff: High-frequency cutoff; bins with ``|freq| > cutoff`` are zeroed.

  Returns:
    The (complex) inverse FFT of the filtered spectrum.
  """
  N = len(data)
  freq = np.fft.fftfreq(N, dt)
  FT = np.fft.fft(data)

  FT[freq > cutoff] = 0
  FT[freq < -cutoff] = 0

  return np.fft.ifft(FT)


def _butter_lowpass(cutoff: float, fs: float, order: int = 5):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype="low", analog=False)
  return b, a


def _butter_lowpass_filter(data: np.ndarray,
                           cutoff: float,
                           fs: float,
                           order: int = 5):
  b, a = _butter_lowpass(cutoff, fs, order=order)
  return lfilter(b, a, data)


def butter_filtering(data: np.ndarray,
                     dt: float = 1.0,
                     *,
                     cutoff: float) -> np.ndarray:
  """Low-pass filter ``data`` with a 6th-order Butterworth filter.

  Args:
    data: 1-D signal.
    dt: Sample spacing.
    cutoff: High-frequency cutoff.

  Returns:
    The filtered signal (same length as ``data``).
  """
  order = 6
  fs = 1 / dt  # sample rate
  return _butter_lowpass_filter(data, cutoff, fs, order)
