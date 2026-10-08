"""Analytic tests for the field, flux, and transport GK quantities.

Fields are native p1 serendipity data built from closed-form coefficients.
The orthonormal basis on the reference cell has the constant mode
``psi0 = 2**(-ndim/2)`` and the mode linear in direction ``d`` equal to
``sqrt(3) psi0 xi_d``, ordered ``1, x, y, z, ...``. A globally linear field
``offset + sum_d slope_d x_d`` is then exact (the cell-average mode carries
the value at the cell centre, the linear modes the slopes), and so are its
derivatives and its weak products with uniform geometry. Weak inverses are
exact for constant fields. Expected values therefore hold to round-off,
except where a test states otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import constants

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.gk import quantities as ff

pytestmark = pytest.mark.skipif(not gpython.available(),
                                reason="requires Gkeyll")

ROUND_OFF = dict(rtol=1e-12, atol=0.0)

_CELLS = (2, 3, 4)
_LENGTHS = (0.7, 1.3, 2.1)  # Unequal, so a swapped cell width shows up.
_PSI0 = 2.0**-1.5
_MASS, _CHARGE = 3.3e-27, 1.6e-19

# The metric component holding each (k, l) entry, stated here rather than
# imported so the tests pin the 11, 12, 13, 22, 23, 33 storage order.
_G = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}
_EUCLIDEAN = (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
_SKEW = (2.0, 0.3, -0.5, 3.0, 0.7, 1.5)  # Positive definite.


def _grid(cells=_CELLS, lengths=_LENGTHS):
  return [np.linspace(0.0, L, n + 1) for L, n in zip(lengths, cells)]


def _native(values, cells=_CELLS, lengths=_LENGTHS, **ctx) -> pg.GData:
  data = pg.GData(
      ctx={
          "basis_type": "serendipity",
          "poly_order": 1,
          "value_form": "modal",
          "cells": np.array(cells),
          "mass": _MASS,
          "charge": _CHARGE,
          **ctx
      })
  data.push(_grid(cells, lengths), gpython.GkylArray.from_numpy(values))
  return data


def _linear(comps, cells=_CELLS, lengths=_LENGTHS, **ctx) -> pg.GData:
  """Components ``offset + sum_d slope_d x_d``, given as (offset, slopes)."""
  ndim = len(cells)
  psi0 = 2.0**(-ndim / 2.0)
  nb = 2**ndim
  centers = np.meshgrid(*[(g[:-1] + g[1:]) / 2 for g in _grid(cells, lengths)],
                        indexing="ij")
  values = np.zeros((*cells, nb * len(comps)))
  for comp, (offset, slopes) in enumerate(comps):
    values[...,
           comp * nb] = (offset + sum(s * c
                                      for s, c in zip(slopes, centers))) / psi0
    for d, slope in enumerate(slopes):
      dx = lengths[d] / cells[d]
      values[..., comp * nb + 1 + d] = slope * dx / (2.0 * np.sqrt(3.0) * psi0)
  return _native(values, cells, lengths, **ctx)


def _const(*values, **ctx) -> pg.GData:
  return _linear([(v, (0.0, 0.0, 0.0)) for v in values], **ctx)


def _avg(data, comp=0, ndim=3):
  """Cell averages of physical component ``comp``."""
  return data.values[..., comp * 2**ndim] * 2.0**(-ndim / 2.0)


def _metric(metric):
  return np.array([[metric[_G[(min(i, j), max(i, j))]] for j in range(3)]
                   for i in range(3)])


def _b_hat(metric):
  """``b_i = g_i3/sqrt(g_33)``: b along e_3, as Gkeyll builds it."""
  return tuple(_metric(metric)[:, 2] / np.sqrt(metric[_G[(2, 2)]]))


def test_linear_fields_have_the_slopes_they_claim():
  """Pin the basis convention every closed form below relies on."""
  slopes = (2.5, -1.75, 0.5)
  apar = _linear([(0.3, slopes)])
  for d, slope in enumerate(slopes):
    deriv = apar.differentiate(direction=d)
    np.testing.assert_allclose(_avg(deriv), slope, **ROUND_OFF)
    np.testing.assert_allclose(deriv.values[..., 1:], 0.0, atol=1e-12)


# --------------------------------------------------------------- moments
class TestSecondMomentsFromMaxwellians:
  _N, _U, _VT2 = 2.3e19, 4.1e4, 2.9e9

  def test_from_Maxwellian_moments(self):
    maxwellian = _const(self._N, self._U, self._VT2)
    np.testing.assert_allclose(_avg(ff.fetch_M2par_from_Max([maxwellian])),
                               self._N * (self._U**2 + self._VT2), **ROUND_OFF)
    np.testing.assert_allclose(_avg(ff.fetch_M2perp_from_Max([maxwellian])),
                               2.0 * self._N * self._VT2, **ROUND_OFF)

  def test_from_BiMaxwellian_moments_reads_each_temperature(self):
    """Distinct parallel and perpendicular temperatures expose a swap."""
    bimax = _const(self._N, self._U, 0.7 * self._VT2, 1.9 * self._VT2)
    np.testing.assert_allclose(_avg(ff.fetch_M2par_from_BiMax([bimax])),
                               self._N * (self._U**2 + 0.7 * self._VT2),
                               **ROUND_OFF)
    np.testing.assert_allclose(_avg(ff.fetch_M2perp_from_BiMax([bimax])),
                               2.0 * self._N * 1.9 * self._VT2, **ROUND_OFF)


# ------------------------------------------------- multi-species quantities
_E = constants.elementary_charge
_M_E, _M_I = constants.electron_mass, 3.343e-27
_N_E, _T_E = 2.0e19, 1.6e-17
_N_I, _T_I = 2.0e19, 1.1e-17


def _species(n, T, mass, charge, *extra):
  ctx = dict(mass=mass, charge=charge)
  return [_const(n, **ctx), _const(T, **ctx)
          ] + [_const(v, **ctx) for v in extra]


class TestMachNumber:
  _U_E, _U_I = -3.2e4, 1.7e4

  def test_ion_Mach_number_with_the_hot_ion_sound_speed(self):
    mach = ff.fetch_mach_hot_i([
        _species(_N_I, _T_I, _M_I, _E, self._U_I),
        _species(_N_E, _T_E, _M_E, -_E, self._U_E)
    ],
                               species=["ion", "elc"])
    c_s = np.sqrt((_N_E * _T_E + 3.0 * _N_I * _T_I) / (_N_I * _M_I))
    np.testing.assert_allclose(_avg(mach), self._U_I / c_s, rtol=1e-10)

  def test_listing_the_electrons_first_gives_their_Mach_number(self):
    mach = ff.fetch_mach_cold_i([
        _species(_N_E, _T_E, _M_E, -_E, self._U_E),
        _species(_N_I, _T_I, _M_I, _E, self._U_I)
    ],
                                species=["elc", "ion"])
    np.testing.assert_allclose(_avg(mach),
                               self._U_E / np.sqrt(_T_E / _M_I),
                               rtol=1e-10)

  def test_ions_only_use_adiabatic_electrons(self):
    mach = ff.fetch_mach_cold_i([_species(_N_I, _T_I, _M_I, _E, self._U_I)],
                                species=["ion"],
                                ti_over_te=2.0)
    np.testing.assert_allclose(_avg(mach),
                               self._U_I / np.sqrt(0.5 * _T_I / _M_I),
                               rtol=1e-10)


class TestCollisionFrequency:
  """``nu_sr = norm_nu n_r (v_ts^2 + v_tr^2)^(-3/2)``, ``s`` and ``r`` the
  requested species in order."""

  _REF = dict(den_ref=[1.0e19, 1.2e19], temp_ref=[1.6e-17, 1.3e-17])

  def _nu(self, s, r, **extra):
    return ff.fetch_collision_freq([_species(*s), _species(*r)],
                                   species=["s", "r"],
                                   **{
                                       **self._REF, "bmag_ref": 2.3,
                                       **extra
                                   })

  def test_asymmetry_is_the_mass_and_density_ratio(self):
    """The symmetrized Coulomb logarithm and the thermal-speed sum are the
    same for (e, i) and (i, e), so
    ``nu_ei/nu_ie = (m_i/m_e)(n_i/n_e)`` from the mass prefactors alone."""
    elc, ion = (_N_E, _T_E, _M_E, -_E), (0.5 * _N_I, _T_I, _M_I, _E)
    nu_ei = _avg(self._nu(elc, ion))
    nu_ie = _avg(
        self._nu(ion,
                 elc,
                 den_ref=[1.2e19, 1.0e19],
                 temp_ref=[1.3e-17, 1.6e-17]))
    np.testing.assert_allclose(nu_ei / nu_ie,
                               (_M_I / _M_E) * (0.5 * _N_I / _N_E),
                               rtol=1e-6)

  def test_local_dependence_and_nu_frac(self):
    """At fixed reference values, ``nu`` scales as ``n_r`` and
    ``T^(-3/2)`` and linearly in ``nu_frac``."""
    elc, ion = (_N_E, _T_E, _M_E, -_E), (_N_I, _T_I, _M_I, _E)
    base = _avg(self._nu(elc, ion))
    changed = _avg(self._nu(elc, (3.0 * _N_I, 4.0 * _T_I, _M_I, _E)))
    vt2_sum = _T_E / _M_E + _T_I / _M_I
    vt2_sum_changed = _T_E / _M_E + 4.0 * _T_I / _M_I
    np.testing.assert_allclose(changed / base,
                               3.0 * (vt2_sum / vt2_sum_changed)**1.5,
                               rtol=1e-10)
    np.testing.assert_allclose(_avg(self._nu(elc, ion, nu_frac=0.3)),
                               0.3 * base,
                               rtol=1e-12)

  def test_magnitude_is_a_physical_electron_ion_frequency(self):
    """For n = 2e19 m^-3 and T_e = 100 eV the electron-ion frequency is of
    order 1e6 1/s (NRL formulary: 2.9e-6 n[cm^-3] lnL T[eV]^-3/2), which a
    unit slip would miss by many orders of magnitude."""
    nu = _avg(self._nu((_N_E, _T_E, _M_E, -_E), (_N_I, _T_I, _M_I, _E)))
    nrl = 2.91e-6 * (_N_I * 1e-6) * 15.0 * (_T_E / _E)**-1.5
    assert np.all((nu > 0.1 * nrl) & (nu < 10.0 * nrl))

  def test_needs_two_species_and_reference_values(self):
    elc, ion = (_N_E, _T_E, _M_E, -_E), (_N_I, _T_I, _M_I, _E)
    with pytest.raises(ValueError, match="expected two species"):
      ff.fetch_collision_freq([_species(*elc)], **self._REF, bmag_ref=1.0)
    with pytest.raises(KeyError, match="den_ref"):
      ff.fetch_collision_freq([_species(*elc), _species(*ion)])


# --------------------------------------------------------- gradient lengths
class TestInverseGradientLength:
  _SLOPES = (-2.2, 0.7, 1.3)

  @pytest.mark.parametrize("fetch", [ff.fetch_inv_L_n, ff.fetch_inv_L_T])
  @pytest.mark.parametrize("direction", [0, 1, 2])
  def test_minus_the_gradient_over_the_field(self, fetch, direction):
    """``1/L * X = -dX/dx^k`` exactly, the derivative of a linear field
    being constant; the slopes along the other directions must not enter."""
    field = _linear([(50.0, self._SLOPES)])
    out = fetch([field], dir=direction) * field
    np.testing.assert_allclose(_avg(out), -self._SLOPES[direction], **ROUND_OFF)

  def test_direction_must_be_requested(self):
    with pytest.raises(ValueError, match="direction="):
      ff.fetch_inv_L_T([_linear([(50.0, self._SLOPES)])])

  def test_reduced_runs_carry_only_their_directions(self):
    """A 1x run holds z alone, so 1/L along z uses its only dimension; x is
    refused. A 2x run holds x and z, so y is refused."""
    field_1x = _linear([(1.0, (0.5, ))], cells=(3, ), lengths=(1.0, ))
    out = ff.fetch_inv_L_n([field_1x], dir=2) * field_1x
    np.testing.assert_allclose(_avg(out, ndim=1), -0.5, **ROUND_OFF)
    with pytest.raises(ValueError, match="1x simulation has no x direction"):
      ff.fetch_inv_L_n([field_1x], dir=0)
    field_2x = _linear([(1.0, (0.5, 0.2))], cells=(3, 5), lengths=(0.6, 1.9))
    with pytest.raises(ValueError, match="2x simulation has no y direction"):
      ff.fetch_inv_L_n([field_2x], dir=1)


# --------------------------------------------------- magnetic perturbations
def _dB_sources(slopes, metric=_EUCLIDEAN, b_hat=None, jacob=1.0):
  """``[apar, 1/J, b_i, g_ij]`` with ``apar = sum_d slope_d x_d``."""
  return [
      _linear([(0.0, slopes)]),
      _const(1.0 / jacob),
      _const(*(_b_hat(metric) if b_hat is None else b_hat)),
      _const(*metric)
  ]


class TestMagneticPerturbation:

  def test_uniform_field_along_z(self):
    """Cartesian, b = z, J = 1: ``apar = a x + c y`` gives
    ``dB = (c, -a, 0)``, pinning sign and cyclic wiring at once."""
    a, c = 2.5, -1.75
    srcs = _dB_sources((a, c, 0.0))
    for comp, expected in enumerate((c, -a, 0.0)):
      np.testing.assert_allclose(_avg(
          ff.fetch_dB_perp_contra(srcs[:3], dir=comp)),
                                 expected,
                                 atol=1e-12)
      np.testing.assert_allclose(_avg(ff.fetch_dB_perp_cov(srcs, dir=comp)),
                                 expected,
                                 atol=1e-12)

  def test_gradient_along_b_does_not_contribute(self):
    flat, tilted = _dB_sources((2.5, -1.75, 0.0)), _dB_sources(
        (2.5, -1.75, 9.0))
    for comp in range(3):
      np.testing.assert_allclose(_avg(ff.fetch_dB_perp_cov(flat, dir=comp)),
                                 _avg(ff.fetch_dB_perp_cov(tilted, dir=comp)),
                                 atol=1e-12)

  def test_metric_lowers_the_index(self):
    """b = z and J = 1 hold ``dB^k = (c, -a, 0)`` with a skewed metric, so
    only the lowering ``dB_i = g_ij dB^j`` produces the answer."""
    a, c = 2.5, -1.75
    srcs = _dB_sources((a, c, 0.0), metric=_SKEW, b_hat=(0.0, 0.0, 1.0))
    expected = _metric(_SKEW) @ np.array([c, -a, 0.0])
    for comp in range(3):
      np.testing.assert_allclose(_avg(ff.fetch_dB_perp_cov(srcs, dir=comp)),
                                 expected[comp], **ROUND_OFF)

  def test_general_geometry_magnitude(self):
    """A skewed uniform b gives ``dB^k = (grad(apar) x b)/J`` and
    ``|dB| = sqrt(dB . g . dB)``; the third covariant component vanishes
    because dB is perpendicular to b, which lies along e_3."""
    slopes, jacob = (2.5, -1.75, 0.9), 1.7
    srcs = _dB_sources(slopes, metric=_SKEW, jacob=jacob)
    dB = np.cross(slopes, _b_hat(_SKEW)) / jacob
    np.testing.assert_allclose(_avg(ff.fetch_dB_perp_mag(srcs)),
                               np.sqrt(dB @ _metric(_SKEW) @ dB),
                               rtol=1e-10)
    np.testing.assert_allclose(_avg(ff.fetch_dB_perp_cov(srcs, dir=2)),
                               0.0,
                               atol=1e-12)

  def test_twisting_b_contributes_for_a_uniform_apar(self):
    """``dB = curl(apar b)``: with ``grad(apar) = 0`` and ``b_1 = s y`` the
    whole perturbation is ``dB_3 = -apar s/J``."""
    apar, s, jacob = 0.35, 1.9, 1.7
    b_i = _linear([(0.0, (0.0, s, 0.0)), (0.0, (0.0, 0.0, 0.0)),
                   (1.0, (0.0, 0.0, 0.0))])
    srcs = [_const(apar), _const(1.0 / jacob), b_i, _const(*_EUCLIDEAN)]
    for comp, expected in enumerate((0.0, 0.0, -apar * s / jacob)):
      np.testing.assert_allclose(_avg(ff.fetch_dB_perp_cov(srcs, dir=comp)),
                                 expected,
                                 atol=1e-12)


class TestTotalMagneticField:
  _B, _JACOB = 1.9, 1.7

  def _srcs(self, slopes):
    apar, jinv, b_i, g_ij = _dB_sources(slopes, metric=_SKEW, jacob=self._JACOB)
    return [apar, _const(self._B), jinv, b_i, g_ij]

  def test_equilibrium_representations_describe_the_same_field(self):
    """``B_i = B b_i`` and ``B^i = (B/sqrt(g_33)) delta^i_3`` share
    nothing, so ``B_i B^i = B^2`` checks both and the g_33 component."""
    _apar, bmag, _jinv, b_i, g_ij = self._srcs((0.0, 0.0, 0.0))
    cov = [ff.fetch_B_equilibrium_cov([bmag, b_i], dir=i) for i in range(3)]
    contra = [
        ff.fetch_B_equilibrium_contra([bmag, g_ij], dir=i) for i in range(3)
    ]
    for i in (0, 1):
      np.testing.assert_array_equal(contra[i].values, 0.0)
    total = sum(_avg(cv) * _avg(ct) for cv, ct in zip(cov, contra))
    np.testing.assert_allclose(np.sqrt(total), self._B, **ROUND_OFF)

  def test_total_is_equilibrium_plus_perturbation(self):
    srcs = self._srcs((2.5, -1.75, 0.9))
    apar, bmag, jinv, b_i, g_ij = srcs
    for i in range(3):
      np.testing.assert_allclose(
          _avg(ff.fetch_B_tot_cov(srcs, dir=i)),
          _avg(ff.fetch_B_equilibrium_cov([bmag, b_i], dir=i)) +
          _avg(ff.fetch_dB_perp_cov([apar, jinv, b_i, g_ij], dir=i)),
          **ROUND_OFF)
      np.testing.assert_allclose(
          _avg(ff.fetch_B_tot_contra(srcs, dir=i)),
          _avg(ff.fetch_B_equilibrium_contra([bmag, g_ij], dir=i)) +
          _avg(ff.fetch_dB_perp_contra([apar, jinv, b_i], dir=i)),
          rtol=1e-12,
          atol=1e-14)

  def test_direction_must_be_requested(self):
    with pytest.raises(ValueError, match="direction="):
      ff.fetch_B_tot_cov(self._srcs((1.0, 1.0, 0.0)))


# ---------------------------------------------------------- electric field
class TestElectricField:
  _SLOPES = (1.3e3, -0.7e3, 2.1e3)

  def _phi(self):
    return _linear([(5.0, self._SLOPES)])

  def test_covariant_contravariant_and_magnitude(self):
    E = -np.array(self._SLOPES)
    srcs = [self._phi(), _const(*_SKEW)]
    for comp in range(3):
      np.testing.assert_allclose(_avg(ff.fetch_E_field_cov(srcs[:1], dir=comp)),
                                 E[comp], **ROUND_OFF)
      np.testing.assert_allclose(_avg(ff.fetch_E_field_contra(srcs, dir=comp)),
                                 (_metric(_SKEW) @ E)[comp], **ROUND_OFF)
    np.testing.assert_allclose(_avg(ff.fetch_E_field_mag(srcs)),
                               np.sqrt(E @ _metric(_SKEW) @ E),
                               rtol=1e-10)

  def test_2x_runs_hold_x_and_z(self):
    """A 2x run has no y: ``E_y = 0`` and z is its second dimension."""
    phi = _linear([(0.0, (4.0e2, -9.0e2))], cells=(3, 5), lengths=(0.6, 1.9))
    avg = lambda comp: _avg(ff.fetch_E_field_cov([phi], dir=comp), ndim=2)
    np.testing.assert_allclose(avg(0), -4.0e2, **ROUND_OFF)
    np.testing.assert_array_equal(
        ff.fetch_E_field_cov([phi], dir=1).values, 0.0)
    np.testing.assert_allclose(avg(2), 9.0e2, **ROUND_OFF)


# ------------------------------------------------------- cross-field fluxes
# phi = g x + a y + e z and b_i = (0, b_y, b_z) give the uniform
#   v_E^x = (b_y dphi/dz - b_z dphi/dy)/(J B) = (b_y e - b_z a) jacobtot_inv,
#   v_E^y = (b_z dphi/dx - b_x dphi/dz)/(J B) = b_z g jacobtot_inv.
_GX, _A, _EZ = 2.3, 3.7, -1.9
_BY, _BZ = 0.4, 0.9
_JTOT_INV, _JGEO, _BMAG = 0.37, 1.6, 2.3
_VE_X = (_BY * _EZ - _BZ * _A) * _JTOT_INV
_VE_Y = _BZ * _GX * _JTOT_INV
_DENS, _UPAR, _VT2 = 2.0e19, 1.0e4, 3.0e9


def _es_sources(moment, phi=None):
  """``[moment, phi, J, 1/(J B), b_i]`` of the ExB fluxes."""
  phi = phi if phi is not None else _linear([(0.0, (_GX, _A, _EZ))])
  return [moment, phi, _const(_JGEO), _const(_JTOT_INV), _const(0.0, _BY, _BZ)]


def _em_sources(moment, apar_slopes):
  """``[moment, apar, B, J, 1/J, b_i]`` of the flutter fluxes, b along z."""
  return [
      moment,
      _linear([(0.0, apar_slopes)]),
      _const(_BMAG),
      _const(_JGEO),
      _const(1.0 / _JGEO),
      _const(0.0, 0.0, 1.0)
  ]


def _random(seed) -> pg.GData:
  return _native(np.random.default_rng(seed).standard_normal((*_CELLS, 8)))


class TestCrossFieldFluxes:

  @pytest.mark.parametrize("direction, vel", [(0, _VE_X), (1, _VE_Y)])
  def test_ExB_fluxes(self, direction, vel):
    np.testing.assert_allclose(
        _avg(
            ff.fetch_flux_particle_ExB(_es_sources(_const(_DENS)),
                                       dir=direction)), _DENS * vel,
        **ROUND_OFF)
    np.testing.assert_allclose(
        _avg(
            ff.fetch_flux_energy_ExB(_es_sources(_const(_DENS * _VT2)),
                                     dir=direction)),
        0.5 * _MASS * _DENS * _VT2 * vel, **ROUND_OFF)

  @pytest.mark.parametrize("direction", [0, 1])
  def test_flutter_fluxes(self, direction):
    """With b along z and ``apar = s_x x + s_y y``, ``dB^x = s_y/J`` and
    ``dB^y = -s_x/J``."""
    slopes = (0.13, 0.21, 0.0)
    dB = (slopes[1], -slopes[0])[direction] / _JGEO
    expected = _DENS * _UPAR * dB / _BMAG
    srcs = _em_sources(_const(_DENS * _UPAR), slopes)
    np.testing.assert_allclose(
        _avg(ff.fetch_flux_particle_dB(srcs, dir=direction)), expected,
        **ROUND_OFF)
    np.testing.assert_allclose(
        _avg(ff.fetch_flux_energy_dB(srcs, dir=direction)),
        0.5 * _MASS * expected, **ROUND_OFF)

  def test_direction_is_required_and_cross_field(self):
    """No direction is assumed, and z is refused: it would miss the
    parallel streaming flux."""
    srcs = _es_sources(_const(_DENS))
    with pytest.raises(ValueError, match="direction="):
      ff.fetch_flux_particle_ExB(srcs)
    with pytest.raises(ValueError, match="parallel streaming"):
      ff.fetch_flux_particle_ExB(srcs, dir=2)

  def test_total_flux_is_ExB_plus_flutter(self):
    m0, m1 = _random(1), _random(2)
    es = _es_sources(m0)
    em = _em_sources(m1, (0.3, -0.2, 0.1))
    phi, jgeo, jtot_inv, b_i = es[1:]
    apar, bmag, _jgeo, jgeo_inv, _b = em[1:]
    total = ff.fetch_flux_particle_em(
        [m0, m1, apar, phi, bmag, jgeo, jgeo_inv, jtot_inv, b_i], dir=0)
    expected = (ff.fetch_flux_particle_ExB([m0, phi, jgeo, jtot_inv, b_i],
                                           dir=0).values +
                ff.fetch_flux_particle_dB([m1, apar, bmag, jgeo, jgeo_inv, b_i],
                                          dir=0).values)
    np.testing.assert_allclose(total.values, expected, rtol=1e-12, atol=1e-14)

  def test_uniform_fields_have_no_turbulent_flux(self):
    gamma = ff.fetch_flux_particle_ExB(_es_sources(_const(_DENS)),
                                       dir=0,
                                       fluct="yz")
    np.testing.assert_allclose(gamma.values,
                               0.0,
                               atol=1e-12 * _DENS * abs(_VE_X))

  @pytest.mark.parametrize("fluct, dims", [("y", [1]), ("yz", [1, 2])])
  def test_mean_flux_splits_into_mean_and_turbulent_parts(self, fluct, dims):
    """``<n v> = <n><v> + <dn dv>``, ``<.>`` the J-weighted average that
    defines the fluctuations, for arbitrary n and phi."""
    m0, phi = _random(3), _random(4)
    srcs = _es_sources(m0, phi=phi)
    jgeo = srcs[2]
    total = ff.fetch_flux_particle_ExB(srcs, dir=0)
    turb = ff.fetch_flux_particle_ExB(srcs, dir=0, fluct=fluct)
    v_x = ff._b_cross_grad_div_b_component(phi, srcs[3], srcs[4], 0)
    mean_n = m0 - m0.fluctuation(dims, weight=jgeo)
    mean_v = v_x - v_x.fluctuation(dims, weight=jgeo)

    lhs = total.average(dims, weight=jgeo).values
    rhs = ((mean_n * mean_v).average(dims, weight=jgeo).values +
           turb.average(dims, weight=jgeo).values)
    np.testing.assert_allclose(lhs,
                               rhs,
                               rtol=1e-10,
                               atol=1e-12 * np.abs(lhs).max())
    assert not np.allclose(total.values, turb.values)

  def test_fluxes_need_3x_and_a_known_fluct(self):
    flat = _linear([(_DENS, (0.0, ))], cells=(3, ), lengths=(1.0, ))
    with pytest.raises(ValueError, match="need 3x"):
      ff.fetch_flux_particle_ExB([flat] * 5, dir=0)
    with pytest.raises(ValueError, match="fluct"):
      ff.fetch_flux_particle_ExB(_es_sources(_const(_DENS)), dir=0, fluct="x")


# --------------------------------------------------- transport coefficients
_GXX, _GYY, _N0, _GN, _T0, _GT = 2.2, 0.6, 3.0e19, 1.2e19, 4.0e-17, 1.5e-17
_GAMMA, _Q = 4.0e19, 2.0e3


def _gij():
  return _const(_GXX, 0.0, 0.0, _GYY, 0.0, 1.0)


class TestTransportCoefficients:
  """Uniform fluxes over linear profiles keep every denominator constant,
  so the diffusivities are exact."""

  @pytest.mark.parametrize("direction, g_kk", [(0, _GXX), (1, _GYY)])
  def test_particle_diffusivity(self, direction, g_kk):
    """``D^k = -Gamma^k/(g^kk dn/dx^k)``, the density falling along x^k
    only, so the metric entry and slope of the other direction must not
    enter."""
    slopes = [0.0, 0.0, 0.0]
    slopes[direction] = -_GN
    m0 = _linear([(_N0, tuple(slopes))])
    particle_D = ff.fetch_particle_D(
        [m0, _const(_GAMMA), _gij()], dir=direction)
    np.testing.assert_allclose(_avg(particle_D), _GAMMA / (g_kk * _GN),
                               **ROUND_OFF)

  def test_diffusivities_are_cross_field(self):
    m0 = _linear([(_N0, (0.0, 0.0, -_GN))])
    with pytest.raises(ValueError, match="direction="):
      ff.fetch_particle_D([m0, _const(_GAMMA), _gij()])
    with pytest.raises(ValueError, match="parallel streaming"):
      ff.fetch_particle_D([m0, _const(_GAMMA), _gij()], dir=2)

  @pytest.mark.parametrize("conv", [None, 0.0, 2.5])
  def test_heat_diffusivity_removes_the_convective_flux(self, conv):
    """``chi = -(Q - conv T Gamma)/(n g^xx dT/dx)``, here with a uniform T
    slope and density; ``conv`` defaults to 3/2."""
    temp = _linear([(_T0, (-_GT, 0.0, 0.0))])
    heat_chi = ff.fetch_heat_chi(
        [_const(_N0), temp,
         _const(_GAMMA), _const(_Q),
         _gij()],
        dir=0,
        conv=conv)
    conv_coeff = 1.5 if conv is None else conv
    x_centers = (_grid()[0][:-1] + _grid()[0][1:]) / 2
    heat_flux = _Q - conv_coeff * (_T0 - _GT * x_centers) * _GAMMA
    np.testing.assert_allclose(_avg(heat_chi),
                               (heat_flux / (_N0 * _GXX * _GT))[:, None, None] *
                               np.ones(_CELLS),
                               rtol=1e-12)

  def test_gyro_Bohm_normalization_with_reference_values(self):
    """With constant ``te_ref``/``bmag_ref``, ``rho_s^2 c_s =
    te_ref^(3/2) m_i^(1/2)/(q_i^2 bmag_ref^2)`` and
    ``D/D_gB = Gamma n/(g^xx^(3/2) (dn/dx)^2 rho_s^2 c_s)``, linear in x."""
    te_ref, bmag_ref = 3.0e-17, 2.5
    m0 = _linear([(_N0, (-_GN, 0.0, 0.0))])
    sources = [m0, _const(_T0), _const(_GAMMA), _gij(), _const(_BMAG)]
    out = ff.fetch_particle_D_gB([sources],
                                 species=["ion"],
                                 dir=0,
                                 te_ref=te_ref,
                                 bmag_ref=bmag_ref)
    rho_s2_c_s = te_ref**1.5 * np.sqrt(_MASS) / (_CHARGE**2 * bmag_ref**2)
    x = (_grid()[0][:-1] + _grid()[0][1:]) / 2
    expected = _GAMMA * (_N0 - _GN * x) / (_GXX**1.5 * _GN**2 * rho_s2_c_s)
    np.testing.assert_allclose(_avg(out),
                               expected[:, None, None] * np.ones(_CELLS),
                               rtol=1e-12)

  def test_gyro_Bohm_uses_the_electron_temperature_and_field_profiles(self):
    """Kinetic electrons: ``T_e`` and ``B`` come from the electron
    temperature and the bmag source; constants here keep the check exact."""
    m0 = _linear([(_N0, (-_GN, 0.0, 0.0))])
    ion = [m0, _const(_T0), _const(_GAMMA), _gij(), _const(_BMAG)]
    elc = [
        _const(_N0, charge=-_E, mass=_M_E),
        _const(2.0 * _T0, charge=-_E, mass=_M_E),
        _const(_GAMMA),
        _gij(),
        _const(_BMAG)
    ]
    out = ff.fetch_particle_D_gB([ion, elc], species=["ion", "elc"], dir=0)
    reference = ff.fetch_particle_D_gB([ion],
                                       species=["ion"],
                                       dir=0,
                                       te_ref=2.0 * _T0,
                                       bmag_ref=_BMAG)
    # Modes that vanish analytically hold round-off; scale atol to the field.
    np.testing.assert_allclose(out.values,
                               reference.values,
                               rtol=1e-12,
                               atol=1e-14 * np.abs(reference.values).max())
