"""FFT / PSD of gridded data, plus polar (shell) isotropic binning.

Merges the legacy ``tools/fft.py``, ``tools/init_polar.py``, and
``tools/polar_isotropic.py`` into one module: :func:`fft` is the entry
point (with ``psd``/``iso`` flags), :func:`init_polar` and
:func:`polar_isotropic` are the isotropic-binning helpers it calls for
``iso=True`` and are also useful standalone.
"""

from __future__ import annotations

import numpy as np
import scipy.fft


def fft(grid: list[np.ndarray],
        values: np.ndarray,
        *,
        psd: bool = False,
        iso: bool = False) -> tuple[list[np.ndarray], np.ndarray]:
  """FFT (or power spectral density, optionally isotropic) of gridded data.

  Args:
    grid: Nodal coordinate arrays, one per spatial dimension. Axes of
      length <= 2 are treated as dummy dimensions and squeezed out first.
    values: Data array; the last axis is components.
    psd: If ``True``, return the (one-sided) power spectral density
      instead of the complex FFT.
    iso: If ``True`` (requires ``psd`` and exactly 3 real spatial
      dimensions), additionally shell-average the PSD over polar
      (isotropic) ``k``-bins and return a 1-D isotropic spectrum.

  Returns:
    ``(freq, ft_values)``: ``freq`` is a list of 1-D frequency arrays (one
    per surviving spatial axis, or a single polar-``k`` axis if ``iso``),
    and ``ft_values`` is the (P)FT array.

  Raises:
    ValueError: If ``psd`` is requested for data that is not 1-D, 2-D, or
      3-D.
  """
  grid = list(grid)
  values = values

  # Remove dummy dimensions
  num_dims = len(grid)
  idx = [d for d in range(num_dims) if len(grid[d]) <= 2]
  if idx:
    for i in idx[::-1]:
      grid.pop(i)
    values = np.squeeze(values, tuple(idx))
    num_dims = len(grid)
  num_comps = values.shape[-1]

  if num_dims == 1:
    N = len(grid[0])
    dx = grid[0][1] - grid[0][0]
    freq = [scipy.fft.fftfreq(N, dx)]
    ft_values = np.zeros(values.shape, "complex")
    for comp in np.arange(num_comps):
      ft_values[..., comp] = scipy.fft.fft(values[..., comp])

    if psd:
      freq[0] = freq[0][:N // 2]
      ft_values = np.abs(ft_values[:N // 2, :])**2
    return freq, ft_values

  if num_dims > 3:
    # src_bak raised this same message, but only from deep inside the
    # ``psd`` branch -- unreachable in practice, since the fixed-size
    # ``N = np.zeros(3)`` below always raises a confusing IndexError first
    # for num_dims > 3, psd or not. Raise it up front instead.
    raise ValueError("Only 1D, 2D, and 3D data are currently supported.")

  N = np.zeros(3, dtype=int)
  dx = np.zeros(3)
  freq = []
  for i in range(num_dims):
    N[i] = len(grid[i])
    dx[i] = grid[i][1] - grid[i][0]
    freq.append(scipy.fft.fftfreq(N[i], dx[i]))
  ft_values = np.zeros(values.shape, "complex")
  for comp in np.arange(num_comps):
    ft_values[..., comp] = scipy.fft.fftn(values[..., comp])
  if not psd:
    return freq, ft_values

  for i in range(num_dims):
    freq[i] = freq[i][:N[i] // 2]
  if num_dims == 2:
    ft_values = np.abs(ft_values[:N[0] // 2, :N[1] // 2, :])**2
    if iso:
      freq.append(0)  # dummy third index, only meaningful to init_polar below
  else:  # num_dims == 3 (num_dims > 3 already raised above)
    ft_values = np.abs(ft_values[:N[0] // 2, :N[1] // 2, :N[2] // 2, :])**2

  if not iso:
    return freq, ft_values

  nkpolar = int(np.sqrt(np.sum(N[:]**2)))
  nkx = N[0] // 2
  nky = N[1] // 2
  nkz = N[2] // 2
  kx, ky, kz = freq[0], freq[1], freq[2]
  akp, nbin, polar_index, _ = init_polar(nkx, nky, nkz, kx, ky, kz, nkpolar)
  fft_iso = np.zeros((nkpolar, num_comps))
  for comp in np.arange(num_comps):
    fft_iso[:, comp] = polar_isotropic(nkpolar, nkx, nky, nkz, polar_index,
                                       nbin, ft_values[..., comp], kx, ky, kz)
  return [akp], fft_iso


def init_polar(nkx, nky, nkz, kx, ky, kz, nkpolar):
  """Build a polar (k-perpendicular) binning of a Cartesian wavenumber grid.

  Constructs uniformly spaced polar bins in ``k = sqrt(kx**2 + ky**2 [+ kz**2])``
  and assigns each Cartesian wavenumber cell to a bin, for later isotropic
  (shell) averaging of spectra. Works for 2D grids (set ``nkz`` and ``kz`` to
  ``0``) and 3D grids.

  Args:
    nkx: Number of grid points along the ``kx`` axis.
    nky: Number of grid points along the ``ky`` axis.
    nkz: Number of grid points along the ``kz`` axis; use ``0`` for 2D data.
    kx: 1D array of ``kx`` wavenumbers; ``kx[1]`` sets the spacing ``dkx``.
    ky: 1D array of ``ky`` wavenumbers; ``ky[1]`` sets the spacing ``dky``.
    kz: 1D array of ``kz`` wavenumbers; ``kz[1]`` sets the spacing ``dkz``.
      Use ``0`` for 2D data.
    nkpolar: Number of polar (radial ``k_perp``) bins to create. If ``0``,
      no binning is performed and empty outputs are returned.

  Returns:
    ``(akp, nbin, polar_index, akplim)`` where ``akp`` is the array of
    polar bin centers (the ``k_perp`` grid), ``nbin`` is the count of
    Cartesian cells assigned to each bin, ``polar_index`` is an integer
    array (shape matching the Cartesian grid) giving the bin index of each
    cell, and ``akplim`` is the array of polar bin edges.
  """
  # if 2D, nkz and kz = 0

  if nkpolar == 0:
    akp = []
    nbin = 0
    polar_index = []
    akplim = []
  elif nkz == 0:
    nbin = np.zeros(nkpolar)  # Number of kx,ky in each polar bins
    polar_index = np.zeros((nkx, nky),
                           dtype=int)  # Polar index to simplify binning
    if nkx == 1 and nky == 1:
      # NB: src_bak wrote this as ``nkx == 1 & nky == 1``. ``&`` binds
      # tighter than ``==``, so that parsed as
      # ``nkx == (1 & nky) and (1 & nky) == 1`` -- true or false by the
      # *parity* of nky, not by whether nkx/nky actually equal 1. Fixed to
      # the evidently intended ``and``, proven by the parity-sensitive
      # test in test_numerics_fft.py.
      dkp = 0
    elif nkx == 1:
      dkp = ky[1]
    elif nky == 1:
      dkp = kx[1]
    else:
      dkp = max(kx[1], ky[1])
    akp = (np.linspace(1, nkpolar, nkpolar)) * dkp  # Kperp grid
    akplim = dkp / 2 + (np.linspace(0, nkpolar,
                                    nkpolar + 1)) * dkp  # Bin limits
    # Re-written to avoid loops. Necessary for large grids.
    [kxg, kyg] = np.meshgrid(
        ky, kx)  # Deal with meshgrid weirdness (so do not have to transpose)
    kp = np.sqrt(kxg**2 + kyg**2)
    pn = np.where(kp >= akplim[nkpolar])
    polar_index[pn[0], pn[1]] = nkpolar - 1
    nbin[nkpolar - 1] = nbin[nkpolar - 1] + len(pn[0])
    for ik in range(0, nkpolar):
      pn = np.where((kp < akplim[ik + 1]) & (kp >= akplim[ik]))
      polar_index[pn[0], pn[1]] = ik
      nbin[ik] = nbin[ik] + len(pn[0])
  else:
    # 3D data
    nbin = np.zeros(nkpolar)
    polar_index = np.zeros((nkx, nky, nkz), dtype=int)
    if nkx == 1 and nky == 1 and nkz == 1:
      # NB: same ``&``-vs-``==``-precedence bug as the 2D branch above,
      # fixed the same way.
      dkp = 0
    elif nkx == 1:
      dkp = max(ky[1], kz[1])
    elif nky == 1:
      dkp = max(kx[1], kz[1])
    elif nkz == 1:
      dkp = max(kx[1], ky[1])
    else:
      dkp = max(kx[1], ky[1], kz[1])
    akp = (np.linspace(1, nkpolar, nkpolar)) * dkp  # kperp grid
    akplim = dkp / 2 + (np.linspace(0, nkpolar,
                                    nkpolar + 1)) * dkp  # bin limits
    # Re-written to avoid loops
    [kxg, kyg, kzg] = np.meshgrid(ky, kx, kz)
    kp = np.sqrt(kxg**2 + kyg**2 + kzg**2)
    pn = np.where(kp >= akplim[nkpolar])
    polar_index[pn[0], pn[1], pn[2]] = nkpolar - 1
    nbin[nkpolar - 1] = nbin[nkpolar - 1] + len(pn[0])
    for ik in range(0, nkpolar):
      pn = np.where((kp < akplim[ik + 1]) & (kp >= akplim[ik]))
      polar_index[pn[0], pn[1], pn[2]] = ik
      nbin[ik] = nbin[ik] + len(pn[0])

  return akp, nbin, polar_index, akplim


def polar_isotropic(nkpolar, nkx, nky, nkz, polar_index, nbin, fft_matrix, kx,
                    ky, kz):
  """Average a spectrum over polar (k-perpendicular) shells.

  Accumulates the values of ``fft_matrix`` into the polar bins defined by
  ``polar_index`` (as produced by :func:`init_polar`) and divides by the
  number of cells per bin to obtain the isotropic (shell-averaged)
  spectrum. Works for 2D grids (set ``nkz`` and ``kz`` to ``0``) and 3D
  grids.

  Args:
    nkpolar: Number of polar (radial ``k_perp``) bins.
    nkx: Number of grid points along the ``kx`` axis.
    nky: Number of grid points along the ``ky`` axis.
    nkz: Number of grid points along the ``kz`` axis; use ``0`` for 2D data.
    polar_index: Integer array mapping each Cartesian wavenumber cell to
      its polar bin, as returned by :func:`init_polar`.
    nbin: Number of Cartesian cells in each polar bin, used as the
      averaging denominator.
    fft_matrix: Spectral quantity (e.g. spectral power) defined on the
      Cartesian wavenumber grid to be averaged over shells.
    kx: 1D array of ``kx`` wavenumbers (accepted for interface consistency).
    ky: 1D array of ``ky`` wavenumbers (accepted for interface consistency).
    kz: 1D array of ``kz`` wavenumbers (accepted for interface consistency).

  Returns:
    The shell-averaged (isotropic) spectrum, one value per polar bin
    (shape ``(nkpolar,)``).
  """
  # if 2D, then nkz = kz = 0

  fft_isok = np.zeros(nkpolar)
  if nkz == 0:
    for i in range(nkx):
      for j in range(nky):
        fft_isok[polar_index[i,
                             j]] = fft_isok[polar_index[i, j]] + fft_matrix[i,
                                                                            j]
  else:
    for i in range(nkx):
      for j in range(nky):
        for k in range(nkz):
          fft_isok[polar_index[
              i, j, k]] = fft_isok[polar_index[i, j, k]] + fft_matrix[i, j, k]

  fft_isok = fft_isok / nbin[:]
  return fft_isok
