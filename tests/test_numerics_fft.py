"""Tests for postgkyl.numerics.fft -- fft/psd/iso and the polar helpers."""

from __future__ import annotations

import numpy as np
import pytest

from postgkyl.numerics.fft import fft, init_polar, polar_isotropic


class TestFft1D:

  def test_returns_freq_and_ft_values(self):
    N = 32
    grid = [np.linspace(0.0, 1.0, N + 1)]
    x_cc = 0.5 * (grid[0][:-1] + grid[0][1:])
    values = np.sin(2 * np.pi * x_cc)[:, np.newaxis]
    freq, ft = fft(grid, values)
    assert len(freq) == 1
    assert ft.shape[0] == N

  def test_analytic_fft_of_pure_sine(self):
    """A pure sine of frequency f0, sampled on a grid whose length matches
    the values (``fft`` reads ``N``/``dx`` straight off ``len(grid[0])``,
    so the grid array must already be the N-length sample-location axis,
    not an N+1 nodal/edge array), has FFT power at exactly bins +-f0 and
    zero elsewhere: an analytic, hand-computable reference."""
    N = 64
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    grid = [x]
    f0 = 4
    values = np.sin(2 * np.pi * f0 * x)[:, np.newaxis]
    freq, ft = fft(grid, values)
    power = np.abs(ft[:, 0])
    i_pos = np.argmin(np.abs(freq[0] - f0))
    i_neg = np.argmin(np.abs(freq[0] + f0))
    np.testing.assert_allclose(freq[0][i_pos], f0, atol=1e-9)
    np.testing.assert_allclose(freq[0][i_neg], -f0, atol=1e-9)
    total = np.sum(power**2)
    peak = power[i_pos]**2 + power[i_neg]**2
    assert peak / total > 0.999

  def test_dc_component_for_constant(self):
    N = 16
    grid = [np.linspace(0.0, 1.0, N + 1)]
    values = np.ones((N, 1))
    freq, ft = fft(grid, values)
    np.testing.assert_allclose(np.abs(ft[0, 0]), float(N))

  def test_psd_halves_spectrum(self):
    N = 32
    grid = [np.linspace(0.0, 1.0, N + 1)]
    x_cc = 0.5 * (grid[0][:-1] + grid[0][1:])
    values = np.sin(2 * np.pi * x_cc)[:, np.newaxis]
    freq, ft = fft(grid, values, psd=True)
    assert ft.shape[0] == N // 2
    assert ft.shape[0] == len(freq[0])

  def test_multiple_components(self):
    N = 16
    grid = [np.linspace(0.0, 1.0, N + 1)]
    values = np.column_stack([np.ones(N), np.zeros(N)])
    freq, ft = fft(grid, values)
    assert ft.shape[-1] == 2

  def test_dummy_dimension_squeezed(self):
    N = 16
    grid = [np.linspace(0.0, 1.0, N + 1), np.array([0.0, 1.0])]
    values = np.ones((N, 1, 1))
    freq, ft = fft(grid, values)
    assert len(freq) == 1


class TestFft2D:

  def test_2d_fft_returns_correct_shape(self):
    Nx, Ny = 16, 8
    grid = [np.linspace(0.0, 1.0, Nx + 1), np.linspace(0.0, 1.0, Ny + 1)]
    values = np.ones((Nx, Ny, 1))
    freq, ft = fft(grid, values)
    assert ft.shape == (Nx, Ny, 1)
    assert len(freq) == 2

  def test_2d_psd(self):
    Nx, Ny = 16, 8
    grid = [np.linspace(0.0, 1.0, Nx + 1), np.linspace(0.0, 1.0, Ny + 1)]
    values = np.ones((Nx, Ny, 1))
    freq, ft = fft(grid, values, psd=True)
    assert ft.shape[0] == Nx // 2
    assert ft.shape[1] == Ny // 2

  def test_2d_psd_shape(self):
    Nx, Ny = 8, 8
    grid = [np.linspace(0.0, 1.0, Nx + 1), np.linspace(0.0, 1.0, Ny + 1)]
    values = np.ones((Nx, Ny, 1))
    freq, ft = fft(grid, values, psd=True)
    assert ft.shape == (Nx // 2, Ny // 2, 1)


class TestFft3D:

  def test_3d_fft_runs(self):
    Nx, Ny, Nz = 8, 8, 8
    grid = [
        np.linspace(0.0, 1.0, Nx + 1),
        np.linspace(0.0, 1.0, Ny + 1),
        np.linspace(0.0, 1.0, Nz + 1)
    ]
    values = np.ones((Nx, Ny, Nz, 1))
    freq, ft = fft(grid, values)
    assert ft.shape == (Nx, Ny, Nz, 1)

  def test_3d_psd_halves_dims(self):
    Nx, Ny, Nz = 8, 8, 8
    grid = [
        np.linspace(0.0, 1.0, Nx + 1),
        np.linspace(0.0, 1.0, Ny + 1),
        np.linspace(0.0, 1.0, Nz + 1)
    ]
    values = np.ones((Nx, Ny, Nz, 1))
    freq, ft = fft(grid, values, psd=True)
    assert ft.shape == (Nx // 2, Ny // 2, Nz // 2, 1)

  def test_3d_psd_no_iso(self):
    Nx, Ny, Nz = 4, 4, 4
    grid = [
        np.linspace(0.0, 1.0, Nx + 1),
        np.linspace(0.0, 1.0, Ny + 1),
        np.linspace(0.0, 1.0, Nz + 1)
    ]
    rng = np.random.default_rng(0)
    values = rng.random((Nx, Ny, Nz, 1))
    freq, ft = fft(grid, values, psd=True, iso=False)
    assert ft.shape == (Nx // 2, Ny // 2, Nz // 2, 1)

  @pytest.mark.filterwarnings(
      "ignore:invalid value encountered in divide:RuntimeWarning")
  def test_3d_multi_comp(self):
    Nx, Ny, Nz = 4, 4, 4
    grid = [
        np.linspace(0.0, 1.0, Nx + 1),
        np.linspace(0.0, 1.0, Ny + 1),
        np.linspace(0.0, 1.0, Nz + 1)
    ]
    rng = np.random.default_rng(1)
    values = rng.random((Nx, Ny, Nz, 3))
    freq, ft = fft(grid, values, psd=True, iso=True)
    assert ft.shape[-1] == 3


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide:RuntimeWarning")
class TestFftIsotropic:

  def test_fft_3d_psd_iso(self):
    Nx, Ny, Nz = 4, 4, 4
    grid = [
        np.linspace(0.0, 1.0, Nx + 1),
        np.linspace(0.0, 1.0, Ny + 1),
        np.linspace(0.0, 1.0, Nz + 1)
    ]
    rng = np.random.default_rng(2)
    values = rng.random((Nx, Ny, Nz, 1))
    freq, ft = fft(grid, values, psd=True, iso=True)
    assert isinstance(freq, list)
    assert len(freq) == 1
    assert ft.ndim == 2

  def test_fft_3d_psd_iso_positive(self):
    Nx, Ny, Nz = 4, 4, 4
    grid = [
        np.linspace(0.0, 1.0, Nx + 1),
        np.linspace(0.0, 1.0, Ny + 1),
        np.linspace(0.0, 1.0, Nz + 1)
    ]
    values = np.ones((Nx, Ny, Nz, 1))
    freq, ft = fft(grid, values, psd=True, iso=True)
    finite_vals = ft[np.isfinite(ft)]
    assert np.all(finite_vals >= 0)

  def test_iso_preserves_total_power_end_to_end(self):
    """Physically meaningful invariant that line coverage alone cannot see:
    shell-averaging redistributes power onto k-shells but must not lose or
    gain any of it. Reconstruct the Cartesian PSD independently
    (iso=False) and check that weighting each isotropic bin by its shell's
    cell count (``nbin``) reconstructs the same total."""
    Nx, Ny, Nz = 8, 8, 8
    grid = [
        np.linspace(0.0, 1.0, Nx + 1),
        np.linspace(0.0, 1.0, Ny + 1),
        np.linspace(0.0, 1.0, Nz + 1)
    ]
    rng = np.random.default_rng(4)
    values = rng.random((Nx, Ny, Nz, 1))

    freq_cart, ft_cartesian = fft(grid, values, psd=True, iso=False)
    kx, ky, kz = freq_cart[0], freq_cart[1], freq_cart[2]
    nkx, nky, nkz = len(kx), len(ky), len(kz)
    # fft() derives nkpolar from the *nodal* grid lengths (Nx+1 here), not
    # from the cell counts -- match that exactly to reproduce its binning.
    N = np.array([len(grid[0]), len(grid[1]), len(grid[2])])
    nkpolar = int(np.sqrt(np.sum(N**2)))
    _, nbin, polar_index, _ = init_polar(nkx, nky, nkz, kx, ky, kz, nkpolar)
    expected_iso = polar_isotropic(nkpolar, nkx, nky, nkz, polar_index, nbin,
                                   ft_cartesian[..., 0], kx, ky, kz)

    _, ft_iso = fft(grid, values, psd=True, iso=True)

    # fft(iso=True) must agree with a direct call to the same binning helpers.
    np.testing.assert_allclose(ft_iso[:, 0], expected_iso)

    mask = nbin > 0
    total_from_shells = np.sum(ft_iso[mask, 0] * nbin[mask])
    np.testing.assert_allclose(total_from_shells,
                               np.sum(ft_cartesian[..., 0]),
                               rtol=1e-10)

  def test_iso_on_2d_data_treats_z_as_degenerate(self):
    """iso doesn't check num_dims itself -- for 2D data the (dummy, unset)
    third wavenumber axis is left at ``nkz=0``, so ``init_polar`` takes
    its 2D branch and produces a 1D isotropic spectrum, same as 3D."""
    Nx, Ny = 8, 8
    grid = [np.linspace(0.0, 1.0, Nx + 1), np.linspace(0.0, 1.0, Ny + 1)]
    values = np.ones((Nx, Ny, 1))
    freq, ft = fft(grid, values, psd=True, iso=True)
    assert isinstance(freq, list) and len(freq) == 1
    assert ft.ndim == 2


class TestFftPsdOnlySupported1D2D3D:

  def test_4d_raises_a_clean_value_error(self):
    """src_bak's ``Only 1D, 2D, and 3D`` guard lived deep inside the psd
    branch, behind a fixed-size ``N = np.zeros(3)`` that always raised a
    confusing IndexError first for num_dims > 3 (psd or not) -- an
    unreachable check. Fixed to raise the clean ValueError up front; every
    working (<=3D) input is unaffected."""
    grid = [np.linspace(0.0, 1.0, 3)] * 4
    values = np.ones((2, 2, 2, 2, 1))
    with pytest.raises(ValueError, match="1D, 2D, and 3D"):
      fft(grid, values)
    with pytest.raises(ValueError, match="1D, 2D, and 3D"):
      fft(grid, values, psd=True)


# ---------------------------------------------------------------------------
# init_polar
# ---------------------------------------------------------------------------


class TestInitPolar:

  def test_nkpolar_zero_returns_empty(self):
    akp, nbin, polar_index, akplim = init_polar(4, 4, 0, [], [], [], 0)
    assert akp == []
    assert nbin == 0
    assert polar_index == []
    assert akplim == []

  def test_2d_case_basic(self):
    N = 8
    kx = np.fft.fftfreq(N, 1.0 / N)[:N // 2]
    ky = np.fft.fftfreq(N, 1.0 / N)[:N // 2]
    nkpolar = 5
    akp, nbin, polar_index, akplim = init_polar(len(kx), len(ky), 0, kx, ky, [],
                                                nkpolar)
    assert len(akp) == nkpolar
    assert len(nbin) == nkpolar
    assert polar_index.shape == (len(kx), len(ky))
    assert len(akplim) == nkpolar + 1
    assert np.sum(nbin) > 0

  def test_2d_case_nkx1(self):
    kx = np.array([0.0])
    ky = np.array([0.0, 1.0, 2.0])
    akp, nbin, polar_index, akplim = init_polar(1, 3, 0, kx, ky, [], 3)
    assert len(akp) == 3

  def test_2d_case_nky1(self):
    kx = np.array([0.0, 1.0, 2.0])
    ky = np.array([0.0])
    akp, nbin, polar_index, akplim = init_polar(3, 1, 0, kx, ky, [], 3)
    assert len(akp) == 3

  def test_2d_case_nkx1_and_nky1_uses_zero_spacing(self):
    """The ``nkx == 1 and nky == 1`` branch (dkp = 0): a single-cell grid
    in both directions. Also proves the fixed ``and`` (src_bak's ``&``
    precedence bug would have made the parity of nky, not the actual
    nkx/nky == 1 check, decide this branch)."""
    kx = np.array([0.0])
    ky = np.array([0.0])
    akp, nbin, polar_index, akplim = init_polar(1, 1, 0, kx, ky, [], 2)
    np.testing.assert_allclose(akp, [0.0, 0.0])

  def test_3d_case_basic(self):
    N = 4
    kx = np.fft.fftfreq(N)[:N // 2]
    ky = np.fft.fftfreq(N)[:N // 2]
    kz = np.fft.fftfreq(N)[:N // 2]
    nkpolar = 4
    akp, nbin, polar_index, akplim = init_polar(len(kx), len(ky), len(kz), kx,
                                                ky, kz, nkpolar)
    assert len(akp) == nkpolar
    assert polar_index.shape == (len(kx), len(ky), len(kz))
    assert np.sum(nbin) > 0

  def test_3d_case_nkx1(self):
    kx = np.array([0.0])
    ky = np.array([0.0, 1.0])
    kz = np.array([0.0, 1.0])
    akp, nbin, polar_index, akplim = init_polar(1, 2, 2, kx, ky, kz, 2)
    assert len(akp) == 2

  def test_3d_case_nky1(self):
    kx = np.array([0.0, 1.0])
    ky = np.array([0.0])
    kz = np.array([0.0, 1.0])
    akp, nbin, polar_index, akplim = init_polar(2, 1, 2, kx, ky, kz, 2)
    assert len(akp) == 2

  def test_3d_case_nkz1(self):
    kx = np.array([0.0, 1.0])
    ky = np.array([0.0, 1.0])
    kz = np.array([0.0])
    akp, nbin, polar_index, akplim = init_polar(2, 2, 1, kx, ky, kz, 2)
    assert len(akp) == 2

  def test_3d_case_all_singleton_uses_zero_spacing(self):
    """The ``nkx == 1 and nky == 1 and nkz == 1`` branch (dkp = 0), and a
    parity-sensitive proof of the fixed ``and`` (an even count anywhere
    would have flipped src_bak's ``&``-precedence-bugged condition)."""
    kx = ky = kz = np.array([0.0])
    akp, nbin, polar_index, akplim = init_polar(1, 1, 1, kx, ky, kz, 2)
    np.testing.assert_allclose(akp, [0.0, 0.0])


# ---------------------------------------------------------------------------
# polar_isotropic
# ---------------------------------------------------------------------------


class TestPolarIsotropic:

  def test_2d_case(self):
    N = 8
    kx = np.fft.fftfreq(N)[:N // 2]
    ky = np.fft.fftfreq(N)[:N // 2]
    nkpolar = 3
    akp, nbin, polar_index, _ = init_polar(len(kx), len(ky), 0, kx, ky, [],
                                           nkpolar)
    fft_matrix = np.ones((len(kx), len(ky)))
    result = polar_isotropic(nkpolar, len(kx), len(ky), 0, polar_index, nbin,
                             fft_matrix, kx, ky, [])
    assert result.shape == (nkpolar, )
    assert np.any(nbin > 0)

  @pytest.mark.filterwarnings(
      "ignore:invalid value encountered in divide:RuntimeWarning")
  def test_3d_case(self):
    N = 4
    kx = np.fft.fftfreq(N)[:N // 2]
    ky = np.fft.fftfreq(N)[:N // 2]
    kz = np.fft.fftfreq(N)[:N // 2]
    nkpolar = 3
    akp, nbin, polar_index, _ = init_polar(len(kx), len(ky), len(kz), kx, ky,
                                           kz, nkpolar)
    fft_matrix = np.ones((len(kx), len(ky), len(kz)))
    result = polar_isotropic(nkpolar, len(kx), len(ky), len(kz), polar_index,
                             nbin, fft_matrix, kx, ky, kz)
    assert result.shape == (nkpolar, )
    assert np.any(nbin > 0)
