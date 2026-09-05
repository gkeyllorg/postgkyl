"""Postgkyl module for testing the gk_quantities fetch functions numerically.

``test_gk_load_quantity`` drives every registered quantity end to end, but it
only asserts that a dataset comes out the other side: an algebra error in a
fetch function would pass it silently. This module closes that gap by checking
the fetch functions against a case whose moments are known in closed form, a
*shifted Maxwellian* with density n, parallel drift u and temperature T (mass m,
two perpendicular velocity dimensions)::

  M0     = n                     M2par  = n*(u^2 + T/m)
  M1     = n*u                   M2perp = n*(2T/m)
                                 M3par  = n*(u^3 + 3*u*T/m)
                                 M3perp = n*u*(2T/m)

The sharpest check available is that a Maxwellian carries *no* heat flux in the
fluid frame, so ``qpar_fluid``/``qperp_fluid`` must cancel to round-off. Because
"is zero" is also what a badly broken function returns, each vanishing check is
paired with a perturbed case that must come out nonzero and equal to a known
value.

The DG fields here are constant within each cell, which makes every weak DG
operation (multiply, invert) exact, so the expected values are matched to
near-machine precision rather than to a loose tolerance.
"""
import os

import numpy as np
import pytest

import postgkyl.utils.gk_quantities.fetch_funcs as ff
import postgkyl.utils.gkeyll_const as gkc
from postgkyl.data import GData

# Synthetic DG dataset parameters: 1D, p1 serendipity (num_basis = 2).
_POLY_ORDER = 1
_BASIS_TYPE = "serendipity"
_NUM_BASIS = 2
_NUM_CELLS = 4
_NUM_DIMS = 1

# Value of the 0th (constant) modal basis function: the cell average of a DG
# field is its 0th coefficient times _PSI0.
_PSI0 = 2.0**(-0.5*_NUM_DIMS)

# Shifted-Maxwellian parameters. Deliberately not round numbers, and of
# realistic magnitude, so that a wrong formula cannot coincidentally agree.
_MASS = 3.343e-27  # Deuterium.
_DENS = 2.7e19
_UPAR = 1.3e4
_TEMP = 9.5e-18
_VT_SQ = _TEMP/_MASS  # Thermal speed squared, T/m.

# Probe whether the gkylsoft DG-operator library is available; fetch functions
# that use weak multiply/invert are skipped if it is not.
try:
  from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops
  GkeyllDGops()
  _DGOPS_AVAILABLE = True
except Exception:  # noqa: BLE001 - any failure means the lib is unusable here
  _DGOPS_AVAILABLE = False

_needs_dgops = pytest.mark.skipif(
  not _DGOPS_AVAILABLE, reason="requires the gkylsoft DG library")


def _const_gdata(comp_avgs, mass: float = _MASS, charge: float = 1.0) -> GData:
  """Return a DG field that is constant in space, with the given cell averages.

  ``comp_avgs`` is a scalar for a single-component field, or a sequence of one
  cell average per physical component. Only the 0th modal coefficient of each
  component is nonzero, which makes the weak DG operations exact.
  """
  avgs = np.atleast_1d(np.asarray(comp_avgs, dtype=float))
  values = np.zeros((_NUM_CELLS, _NUM_BASIS*avgs.size))
  for comp, avg in enumerate(avgs):
    values[:, comp*_NUM_BASIS] = avg/_PSI0

  gdata = GData(ctx={"poly_order": _POLY_ORDER, "basis_type": _BASIS_TYPE,
                     "mass": mass, "charge": charge})
  gdata.push([np.linspace(0.0, 1.0, _NUM_CELLS + 1)], values)
  return gdata


def _cell_avg(gdata: GData, comp: int = 0) -> np.ndarray:
  """Cell averages of the comp-th physical component of a DG field."""
  return gdata.get_values()[:, comp*_NUM_BASIS]*_PSI0


# Moments of the shifted Maxwellian, as single-component DG fields.
def _m0():
  return _const_gdata(_DENS)

def _m1():
  return _const_gdata(_DENS*_UPAR)

def _m2par():
  return _const_gdata(_DENS*(_UPAR**2 + _VT_SQ))

def _m2perp():
  return _const_gdata(_DENS*2.0*_VT_SQ)

def _m3par():
  return _const_gdata(_DENS*(_UPAR**3 + 3.0*_UPAR*_VT_SQ))

def _m3perp():
  return _const_gdata(_DENS*_UPAR*2.0*_VT_SQ)

def _temp():
  return _const_gdata(_TEMP)


def _strip_ctx(gdata: GData, *keys) -> GData:
  """Drop attributes from a GData's context, as if the file did not carry them."""
  for key in keys:
    gdata.ctx.pop(key, None)
  return gdata


class TestGetCtxVal:
  """Resolution of species attributes: '--extra' first, then the file context."""

  def test_extra_overrides_the_context(self):
    """An explicit --extra must win over the attribute stored in the file."""
    gdata = _const_gdata(1.0, mass=_MASS)
    assert ff._get_ctx_val(gdata, "mass", mass=999.0) == 999.0

  def test_extra_array_overrides_the_context_per_species(self):
    """The override must hold for per-species arrays too, entry by entry."""
    gdata = _const_gdata(1.0, mass=_MASS)
    for species_idx, expected in enumerate([1.0, 2.0, 3.0]):
      got = ff._get_ctx_val(gdata, "mass", mass=[1.0, 2.0, 3.0], species_idx=species_idx)
      assert got == expected

  def test_context_is_used_when_extra_does_not_carry_the_key(self):
    """Without an --extra the file's own attribute is still what is used."""
    gdata = _const_gdata(1.0, mass=_MASS)
    assert ff._get_ctx_val(gdata, "mass") == _MASS
    assert ff._get_ctx_val(gdata, "mass", charge=999.0) == _MASS

  def test_scalar_extra_is_used_when_the_context_lacks_it(self):
    gdata = _strip_ctx(_const_gdata(1.0), "mass")
    assert ff._get_ctx_val(gdata, "mass", mass=7.0) == 7.0

  def test_none_in_context_falls_back_to_extra(self):
    gdata = _const_gdata(1.0)
    gdata.ctx["mass"] = None
    assert ff._get_ctx_val(gdata, "mass", mass=7.0) == 7.0

  def test_a_scalar_extra_applies_to_every_species(self):
    gdata = _strip_ctx(_const_gdata(1.0), "mass")
    for species_idx in range(3):
      assert ff._get_ctx_val(gdata, "mass", mass=7.0, species_idx=species_idx) == 7.0

  def test_per_species_array_is_picked_by_species_index(self):
    """'--extra mass=1,2,3' must give each species its own value."""
    gdata = _strip_ctx(_const_gdata(1.0), "mass")
    for species_idx, expected in enumerate([1.0, 2.0, 3.0]):
      got = ff._get_ctx_val(gdata, "mass", mass=[1.0, 2.0, 3.0], species_idx=species_idx)
      assert got == expected

  def test_array_without_a_species_index_is_an_error(self):
    """An array is meaningless where there is no species to index with."""
    gdata = _strip_ctx(_const_gdata(1.0), "mass")
    with pytest.raises(KeyError, match="not computed per species"):
      ff._get_ctx_val(gdata, "mass", mass=[1.0, 2.0])

  def test_too_short_an_array_is_an_error(self):
    """Fewer values than species must be reported, not silently wrap around."""
    gdata = _strip_ctx(_const_gdata(1.0), "mass")
    with pytest.raises(ValueError, match="only 2 values"):
      ff._get_ctx_val(gdata, "mass", mass=[1.0, 2.0], species_idx=2, species="ion2")

  def test_missing_everywhere_is_an_error(self):
    gdata = _strip_ctx(_const_gdata(1.0), "mass")
    with pytest.raises(KeyError, match="mass"):
      ff._get_ctx_val(gdata, "mass")


class TestMoments:
  """Primitive quantities recovered from the Maxwellian's raw moments."""

  @_needs_dgops
  def test_upar_from_M0_M1(self):
    """upar = M1/M0 must return the drift speed the moments were built with."""
    upar = ff.fetch_s1c0_div_s0c0([_m0(), _m1()])
    assert np.allclose(_cell_avg(upar), _UPAR, rtol=1e-12)

  @_needs_dgops
  def test_Tpar_from_M0_M1_M2par(self):
    """Tpar = m*(M2par - upar*M1)/M0 must return T for a Maxwellian."""
    Tpar = ff.fetch_Tpar_from_M0_M1_M2par([_m0(), _m1(), _m2par()])
    assert np.allclose(_cell_avg(Tpar), _TEMP, rtol=1e-10)

  @_needs_dgops
  def test_Tperp_from_M0_M2perp(self):
    """Tperp = m*M2perp/(2*M0) must return T for a Maxwellian."""
    Tperp = ff.fetch_Tperp_from_M0_M2perp([_m0(), _m2perp()])
    assert np.allclose(_cell_avg(Tperp), _TEMP, rtol=1e-12)

  def test_temp_from_Tpar_Tperp(self):
    """An isotropic (Tpar = Tperp = T) split must average back to T."""
    temp = ff.fetch_temp_from_Tpar_Tperp([_temp(), _temp()])
    assert np.allclose(_cell_avg(temp), _TEMP, rtol=1e-14)

  def test_M2_is_M2par_plus_M2perp(self):
    M2 = ff.fetch_s0c0_add_s1c0([_m2par(), _m2perp()])
    expected = _DENS*(_UPAR**2 + _VT_SQ) + _DENS*2.0*_VT_SQ
    assert np.allclose(_cell_avg(M2), expected, rtol=1e-12)

  def test_M3_is_M3par_plus_M3perp(self):
    M3 = ff.fetch_s0c0_add_s1c0([_m3par(), _m3perp()])
    expected = _DENS*(_UPAR**3 + 3.0*_UPAR*_VT_SQ) + _DENS*_UPAR*2.0*_VT_SQ
    assert np.allclose(_cell_avg(M3), expected, rtol=1e-12)

  def test_add_selects_the_requested_components(self):
    """fetch_s0c2_add_s0c3 must add components 2 and 3, not whole arrays."""
    gdata = _const_gdata([2.0, 3.0, 4.0, 5.0])
    out = ff.fetch_s0c2_add_s0c3([gdata])
    assert out.get_values().shape[-1] == _NUM_BASIS, "output must be single-component"
    assert np.allclose(_cell_avg(out), 4.0 + 5.0, rtol=1e-14)

  @_needs_dgops
  def test_press_p(self):
    """p = n*T."""
    press = ff.fetch_press_p([_m0(), _temp()])
    assert np.allclose(_cell_avg(press), _DENS*_TEMP, rtol=1e-12)


class TestMaxwellianMomentSources:
  """Quantities read out of the packed Maxwellian/BiMaxwellian moment files.

  Those files store [n, upar, T/m] and [n, upar, Tpar/m, Tperp/m], so these
  tests pin down both the component indexing and the mass normalization.
  """

  def test_Tpar_from_BiMax(self):
    bimax = _const_gdata([_DENS, _UPAR, _VT_SQ, _VT_SQ])
    Tpar = ff.fetch_Tpar_from_BiMax([bimax])
    assert np.allclose(_cell_avg(Tpar), _TEMP, rtol=1e-12)

  def test_Tperp_from_BiMax(self):
    bimax = _const_gdata([_DENS, _UPAR, _VT_SQ, _VT_SQ])
    Tperp = ff.fetch_Tperp_from_BiMax([bimax])
    assert np.allclose(_cell_avg(Tperp), _TEMP, rtol=1e-12)

  def test_temp_from_Max(self):
    maxmom = _const_gdata([_DENS, _UPAR, _VT_SQ])
    temp = ff.fetch_temp_from_Max([maxmom])
    assert np.allclose(_cell_avg(temp), _TEMP, rtol=1e-12)

  @_needs_dgops
  def test_press_from_Max(self):
    maxmom = _const_gdata([_DENS, _UPAR, _VT_SQ])
    press = ff.fetch_press_from_Max([maxmom])
    assert np.allclose(_cell_avg(press), _DENS*_TEMP, rtol=1e-12)

  @_needs_dgops
  def test_press_from_BiMax(self):
    bimax = _const_gdata([_DENS, _UPAR, _VT_SQ, _VT_SQ])
    press = ff.fetch_press_from_BiMax([bimax])
    assert np.allclose(_cell_avg(press), _DENS*_TEMP, rtol=1e-12)


class TestHeatFluxes:
  """Lab-frame energy fluxes and fluid-frame heat fluxes."""

  def test_qpar_lab_frame(self):
    """qpar = (m/2)*M3par."""
    qpar = ff.fetch_qpar([_m3par()])
    expected = 0.5*_MASS*_DENS*(_UPAR**3 + 3.0*_UPAR*_VT_SQ)
    assert np.allclose(_cell_avg(qpar), expected, rtol=1e-12)

  def test_qperp_lab_frame(self):
    """qperp = (m/2)*M3perp, which is n*u*T for a Maxwellian."""
    qperp = ff.fetch_qperp([_m3perp()])
    assert np.allclose(_cell_avg(qperp), _DENS*_UPAR*_TEMP, rtol=1e-12)

  @_needs_dgops
  def test_qpar_fluid_vanishes_for_a_maxwellian(self):
    """A Maxwellian carries no parallel heat flux in the fluid frame.

    The three terms of (m/2)*[M3par - 3*u*M2par + 2*u^2*M1] cancel exactly, so
    the residual is compared against the size of an individual term rather than
    against an absolute zero.
    """
    qpar_fluid = ff.fetch_qpar_fluid([_m0(), _m1(), _m2par(), _m3par()])
    term_scale = 0.5*_MASS*_DENS*abs(_UPAR)**3
    assert np.all(np.abs(_cell_avg(qpar_fluid))/term_scale < 1e-10)

  @_needs_dgops
  def test_qperp_fluid_vanishes_for_a_maxwellian(self):
    qperp_fluid = ff.fetch_qperp_fluid([_m0(), _m1(), _m2perp(), _m3perp()])
    term_scale = 0.5*_MASS*_DENS*abs(_UPAR)*2.0*_VT_SQ
    assert np.all(np.abs(_cell_avg(qperp_fluid))/term_scale < 1e-10)

  @_needs_dgops
  def test_qpar_fluid_tracks_a_skewed_distribution(self):
    """Guard against the vanishing tests passing for a function that is just 0.

    Skewing M3par away from its Maxwellian value by dM3 is a pure heat-flux
    perturbation, so the fluid-frame flux must become exactly (m/2)*dM3.
    """
    delta_m3 = 0.05*_DENS*_UPAR**3
    m3par_skewed = _const_gdata(_DENS*(_UPAR**3 + 3.0*_UPAR*_VT_SQ) + delta_m3)

    qpar_fluid = ff.fetch_qpar_fluid([_m0(), _m1(), _m2par(), m3par_skewed])
    assert np.allclose(_cell_avg(qpar_fluid), 0.5*_MASS*delta_m3, rtol=1e-8)

  @_needs_dgops
  def test_qperp_fluid_tracks_a_skewed_distribution(self):
    delta_m3 = 0.05*_DENS*_UPAR*2.0*_VT_SQ
    m3perp_skewed = _const_gdata(_DENS*_UPAR*2.0*_VT_SQ + delta_m3)

    qperp_fluid = ff.fetch_qperp_fluid([_m0(), _m1(), _m2perp(), m3perp_skewed])
    assert np.allclose(_cell_avg(qperp_fluid), 0.5*_MASS*delta_m3, rtol=1e-8)


@_needs_dgops
class TestThermalSpeed:
  """vth = sqrt(T/m), m being the requested species' own mass."""

  def test_vt(self):
    vt = ff.fetch_vt([_temp()])
    assert np.allclose(_cell_avg(vt), np.sqrt(_VT_SQ), rtol=1e-12)

  def test_vt_uses_the_species_mass_from_ctx(self):
    """A different species mass must give a different thermal speed."""
    mass = 100.0*_MASS
    vt = ff.fetch_vt([_const_gdata(_TEMP, mass=mass)])
    assert np.allclose(_cell_avg(vt), np.sqrt(_TEMP/mass), rtol=1e-12)


# --- Sound speed: a two-ion-species plasma -----------------------------------
#
# A deuterium species (Z=1) and a doubly-charged impurity (Z=2), with the
# electron density set by quasineutrality. Every value is distinct so that a
# formula which mixes up a species, a charge state or a mass cannot accidentally
# agree.
_E_CHARGE = gkc.GKYL_ELEMENTARY_CHARGE

_N_I1, _T_I1, _M_I1, _Z_I1 = 2.7e19, 6.1e-18, 3.343e-27, 1.0
_N_I2, _T_I2, _M_I2, _Z_I2 = 4.0e18, 4.3e-18, 2.007e-26, 2.0
_N_E = _N_I1*_Z_I1 + _N_I2*_Z_I2  # Quasineutrality.
_T_E = 9.5e-18
_M_E = gkc.GKYL_ELECTRON_MASS


def _species_srcs(dens: float, temp: float, mass: float, charge: float) -> list:
  """The [M0, temp] source pair for one species, as fetch_c_s receives it."""
  return [_const_gdata(dens, mass=mass, charge=charge),
          _const_gdata(temp, mass=mass, charge=charge)]


def _elc_srcs():
  return _species_srcs(_N_E, _T_E, _M_E, -_E_CHARGE)

def _ion1_srcs():
  return _species_srcs(_N_I1, _T_I1, _M_I1, _Z_I1*_E_CHARGE)

def _ion2_srcs():
  return _species_srcs(_N_I2, _T_I2, _M_I2, _Z_I2*_E_CHARGE)


@_needs_dgops
class TestSoundSpeed:
  """The multi-species sound speeds, dispatched by '--extra kind='."""

  def test_ion_acoustic_single_ion_species(self):
    """With one Z=1 ion species the formula collapses to sqrt(Te/mi)."""
    c_s = ff.fetch_c_s([_elc_srcs(), _ion1_srcs()],
                       species=["elc", "ion1"], kind="ion_acoustic")
    assert np.allclose(_cell_avg(c_s), np.sqrt(_T_E/_M_I1), rtol=1e-10)

  def test_ion_acoustic_two_ion_species(self):
    """c_s = sqrt(Te*sum(n_j*Z_j^2/m_j)/sum(n_j*Z_j))."""
    c_s = ff.fetch_c_s([_elc_srcs(), _ion1_srcs(), _ion2_srcs()],
                       species=["elc", "ion1", "ion2"], kind="ion_acoustic")

    numer = _N_I1*_Z_I1**2/_M_I1 + _N_I2*_Z_I2**2/_M_I2
    denom = _N_I1*_Z_I1 + _N_I2*_Z_I2
    assert np.allclose(_cell_avg(c_s), np.sqrt(_T_E*numer/denom), rtol=1e-10)

  def test_thermo_single_ion_species(self):
    """With one Z=1 ion species: sqrt((gamma_e*Te + gamma_i*Ti)/mi)."""
    c_s = ff.fetch_c_s([_elc_srcs(), _ion1_srcs()],
                       species=["elc", "ion1"], kind="thermo")

    # n_e = n_i1 here only if quasineutrality holds for a single species, so
    # use the general formula rather than the reduced one.
    numer = 1.0*_N_E*_T_E + 3.0*_N_I1*_T_I1
    denom = _N_I1*_M_I1
    assert np.allclose(_cell_avg(c_s), np.sqrt(numer/denom), rtol=1e-10)

  def test_thermo_two_ion_species(self):
    """c_s = sqrt((gamma_e*n_e*Te + sum(gamma_j*n_j*Tj))/sum(n_j*m_j))."""
    c_s = ff.fetch_c_s([_elc_srcs(), _ion1_srcs(), _ion2_srcs()],
                       species=["elc", "ion1", "ion2"], kind="thermo")

    numer = 1.0*_N_E*_T_E + 3.0*(_N_I1*_T_I1 + _N_I2*_T_I2)
    denom = _N_I1*_M_I1 + _N_I2*_M_I2
    assert np.allclose(_cell_avg(c_s), np.sqrt(numer/denom), rtol=1e-10)

  def test_thermo_defaults_are_gamma_e_1_gamma_i_3(self):
    """The documented defaults must be what an un-flagged call actually uses."""
    default = ff.fetch_c_s([_elc_srcs(), _ion1_srcs()],
                           species=["elc", "ion1"], kind="thermo")
    explicit = ff.fetch_c_s([_elc_srcs(), _ion1_srcs()],
                            species=["elc", "ion1"], kind="thermo",
                            gamma_e=1.0, gamma_i=3.0)
    assert np.allclose(_cell_avg(default), _cell_avg(explicit), rtol=1e-12)

  def test_thermo_honours_the_gamma_overrides(self):
    c_s = ff.fetch_c_s([_elc_srcs(), _ion1_srcs()],
                       species=["elc", "ion1"], kind="thermo",
                       gamma_e=5.0/3.0, gamma_i=5.0/3.0)

    numer = (5.0/3.0)*(_N_E*_T_E + _N_I1*_T_I1)
    assert np.allclose(_cell_avg(c_s), np.sqrt(numer/(_N_I1*_M_I1)), rtol=1e-10)

  def test_default_kind_is_thermo(self):
    default = ff.fetch_c_s([_elc_srcs(), _ion1_srcs()], species=["elc", "ion1"])
    explicit = ff.fetch_c_s([_elc_srcs(), _ion1_srcs()],
                            species=["elc", "ion1"], kind="thermo")
    assert np.allclose(_cell_avg(default), _cell_avg(explicit), rtol=1e-12)

  def test_species_order_does_not_matter(self):
    """Species are identified by charge sign, so the order is irrelevant."""
    forward = ff.fetch_c_s([_elc_srcs(), _ion1_srcs(), _ion2_srcs()],
                           species=["elc", "ion1", "ion2"], kind="ion_acoustic")
    shuffled = ff.fetch_c_s([_ion2_srcs(), _elc_srcs(), _ion1_srcs()],
                            species=["ion2", "elc", "ion1"], kind="ion_acoustic")
    assert np.allclose(_cell_avg(forward), _cell_avg(shuffled), rtol=1e-12)

  def test_electrons_are_found_by_charge_not_by_name(self):
    """A species called anything must still be treated as the electrons."""
    named = ff.fetch_c_s([_elc_srcs(), _ion1_srcs()],
                         species=["elc", "ion1"], kind="ion_acoustic")
    odd = ff.fetch_c_s([_species_srcs(_N_E, _T_E, _M_E, -_E_CHARGE), _ion1_srcs()],
                       species=["negatron", "deuterium"], kind="ion_acoustic")
    assert np.allclose(_cell_avg(named), _cell_avg(odd), rtol=1e-12)

  def test_no_electron_species_is_an_error(self):
    with pytest.raises(ValueError, match="exactly one negatively charged"):
      ff.fetch_c_s([_ion1_srcs(), _ion2_srcs()], species=["ion1", "ion2"])

  def test_two_electron_species_is_an_error(self):
    with pytest.raises(ValueError, match="exactly one negatively charged"):
      ff.fetch_c_s([_elc_srcs(), _elc_srcs(), _ion1_srcs()],
                   species=["elc1", "elc2", "ion1"])

  def test_no_ion_species_is_an_error(self):
    with pytest.raises(ValueError, match="no positively charged"):
      ff.fetch_c_s([_elc_srcs()], species=["elc"])

  def test_unknown_kind_is_an_error(self):
    with pytest.raises(ValueError, match="unknown kind"):
      ff.fetch_c_s([_elc_srcs(), _ion1_srcs()], species=["elc", "ion1"], kind="bogus")

  def test_missing_charge_attribute_is_an_error(self):
    """Charge missing from both the file and --extra must be reported."""
    srcs = _strip_ctx(_ion1_srcs()[0], "charge"), _ion1_srcs()[1]
    with pytest.raises(KeyError, match="charge"):
      ff.fetch_c_s([_elc_srcs(), list(srcs)], species=["elc", "ion1"])

  def test_attributes_can_come_from_per_species_extra_arrays(self):
    """Species attributes absent from the files can be given per species.

    This is the '--extra mass=..,..,charge=..,..' path: each species must pick
    its own entry, in the order of '--species'.
    """
    def bare(dens, temp):
      return [_strip_ctx(_const_gdata(dens), "mass", "charge"),
              _strip_ctx(_const_gdata(temp), "mass", "charge")]

    c_s = ff.fetch_c_s([bare(_N_E, _T_E), bare(_N_I1, _T_I1), bare(_N_I2, _T_I2)],
                       species=["elc", "ion1", "ion2"], kind="ion_acoustic",
                       mass=[_M_E, _M_I1, _M_I2],
                       charge=[-_E_CHARGE, _Z_I1*_E_CHARGE, _Z_I2*_E_CHARGE])

    numer = _N_I1*_Z_I1**2/_M_I1 + _N_I2*_Z_I2**2/_M_I2
    denom = _N_I1*_Z_I1 + _N_I2*_Z_I2
    assert np.allclose(_cell_avg(c_s), np.sqrt(_T_E*numer/denom), rtol=1e-10)

  def test_extra_arrays_must_cover_every_species(self):
    """Too few values must be reported rather than silently mis-assigned."""
    def bare(dens, temp):
      return [_strip_ctx(_const_gdata(dens), "mass", "charge"),
              _strip_ctx(_const_gdata(temp), "mass", "charge")]

    with pytest.raises(ValueError, match="only 2 values"):
      ff.fetch_c_s([bare(_N_E, _T_E), bare(_N_I1, _T_I1), bare(_N_I2, _T_I2)],
                   species=["elc", "ion1", "ion2"],
                   mass=[_M_E, _M_I1, _M_I2],
                   charge=[-_E_CHARGE, _Z_I1*_E_CHARGE])

def _linear_gdata(coeff0: float, coeff1: float) -> GData:
  """A single-component p1 field with the given two modal coefficients."""
  values = np.zeros((_NUM_CELLS, _NUM_BASIS))
  values[:, 0] = coeff0
  values[:, 1] = coeff1
  gdata = GData(ctx={"poly_order": _POLY_ORDER, "basis_type": _BASIS_TYPE,
                     "mass": _MASS, "charge": 1.0})
  gdata.push([np.linspace(0.0, 1.0, _NUM_CELLS + 1)], values)
  return gdata


def _project_powsqrt_reference(coeff0: float, coeff1: float, exponent: float,
                               num_quad: int = _POLY_ORDER + 1) -> np.ndarray:
  """Independent numpy reference for pow(sqrt(f), exponent) projected on p1 1D.

  Reimplements the quadrature the gkeyll updater performs, from the definition
  rather than from its code: the 1D p1 modal basis orthonormal on [-1,1] is
  psi0 = 1/sqrt(2), psi1 = sqrt(3/2)*xi, and the projection of g onto it is
  coeff_k = integral of g*psi_k over [-1,1], evaluated by Gauss-Legendre.
  """
  xi, weights = np.polynomial.legendre.leggauss(num_quad)
  psi = np.array([np.full_like(xi, 1.0/np.sqrt(2.0)), np.sqrt(1.5)*xi])

  f_at_ords = coeff0*psi[0] + coeff1*psi[1]
  g_at_ords = np.power(np.sqrt(f_at_ords), exponent)

  return np.array([np.sum(weights*g_at_ords*psi[k]) for k in range(_NUM_BASIS)])


@_needs_dgops
class TestPowSqrt:
  """The gkyl_proj_powsqrt_on_basis binding backing vth."""

  def test_sqrt_of_a_constant_field_is_exact(self):
    out = ff._powsqrt_dg(_const_gdata(4.0), 1.0)
    assert np.allclose(_cell_avg(out), 2.0, rtol=1e-12)

  def test_sqrt_keeps_the_higher_moments(self):
    """A varying field must produce a varying square root.

    This is the whole point of projecting onto the basis rather than taking
    the square root of the cell average: the slope must survive.
    """
    out = ff._powsqrt_dg(_linear_gdata(4.0/_PSI0, 0.35), 1.0)
    assert not np.allclose(out.get_values()[:, 1], 0.0), (
      "sqrt of a varying field must not be piecewise constant")

  def test_sqrt_matches_an_independent_quadrature(self):
    """Check the binding against a from-scratch numpy projection."""
    coeff0, coeff1 = 4.0/_PSI0, 0.35
    out = ff._powsqrt_dg(_linear_gdata(coeff0, coeff1),1.0)

    expected = _project_powsqrt_reference(coeff0, coeff1, 1.0)
    assert np.allclose(out.get_values()[0, :], expected, rtol=1e-12)

  @pytest.mark.parametrize("exponent", [1.0, -1.0, 3.0])
  def test_exponents_match_an_independent_quadrature(self, exponent):
    """sqrt (1), reciprocal sqrt (-1) and the 3/2 power (3)."""
    coeff0, coeff1 = 4.0/_PSI0, 0.35
    out = ff._powsqrt_dg(_linear_gdata(coeff0, coeff1), exponent)

    expected = _project_powsqrt_reference(coeff0, coeff1, exponent)
    assert np.allclose(out.get_values()[0, :], expected, rtol=1e-12)

  def test_constant_field_exponents(self):
    """On a constant field the closed-form answers are exact."""
    field = _const_gdata(4.0)
    assert np.allclose(_cell_avg(ff._powsqrt_dg(field, 1.0)), 2.0, rtol=1e-12)
    assert np.allclose(_cell_avg(ff._powsqrt_dg(field, -1.0)), 0.5, rtol=1e-12)
    assert np.allclose(_cell_avg(ff._powsqrt_dg(field, 3.0)), 8.0, rtol=1e-12)

  def test_multi_component_input_is_rejected(self):
    """The kernel has no component index, so a vector field must not be taken."""
    from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops

    field = _const_gdata([1.0, 2.0, 3.0])  # Three physical components.
    with pytest.raises(ValueError, match="single-component"):
      GkeyllDGops().powsqrt(field, field, 1.0)





# --- Perpendicular magnetic fluctuations -------------------------------------
#
# The parallel vector potential perturbs the field by dB = curl(Apar*b), taken
# directly as pygkyl's dataparam.py does,
#   dB^i = ( d(Apar*b_k)/dx^j - d(Apar*b_j)/dx^k ) / J,  (j,k) = (i+1,i+2) mod 3,
# and then lowered with the metric, dB_i = g_ij dB^j. A curl is degenerate below
# three dimensions, so these tests use 3D p1 fields (num_basis = 8) rather than
# the 1D ones above. The geometry is uniform except where a test needs it not to
# be, which keeps the weak DG products exact.
_NUM_BASIS_3D = 8
_NUM_CELLS_3D = (2, 3, 4)
_LENGTHS_3D = (0.7, 1.3, 2.1)  # Unequal, so a swapped dx shows up.
_PSI0_3D = 2.0**-1.5

# Component of g_ij holding the (k,l) entry, stated here rather than imported so
# that the tests pin the ordering instead of inheriting it.
_G_IJ_COMP = {(0,0): 0, (0,1): 1, (0,2): 2, (1,1): 3, (1,2): 4, (2,2): 5}

# Cartesian and orthonormal: covariant and contravariant components coincide.
_EUCLIDEAN_METRIC = (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
# A skewed but positive-definite metric, for the checks that must hold whatever
# the coordinates are.
_SKEW_METRIC = (2.0, 0.3, -0.5, 3.0, 0.7, 1.5)


def _grid_3d() -> list:
  return [np.linspace(0.0, length, cells + 1)
          for length, cells in zip(_LENGTHS_3D, _NUM_CELLS_3D)]

def _gdata_3d(values) -> GData:
  gdata = GData(ctx={"poly_order": _POLY_ORDER, "basis_type": _BASIS_TYPE,
                     "mass": _MASS, "charge": 1.0})
  gdata.push(_grid_3d(), values)
  return gdata

def _const_gdata_3d(comp_avgs) -> GData:
  """A 3D field constant in space, with the given cell averages."""
  avgs = np.atleast_1d(np.asarray(comp_avgs, dtype=float))
  values = np.zeros((*_NUM_CELLS_3D, _NUM_BASIS_3D*avgs.size))
  for comp, avg in enumerate(avgs):
    values[..., comp*_NUM_BASIS_3D] = avg/_PSI0_3D
  return _gdata_3d(values)

def _cell_avg_3d(gdata: GData, comp: int = 0) -> np.ndarray:
  return gdata.get_values()[..., comp*_NUM_BASIS_3D]*_PSI0_3D

def _metric_matrix(metric) -> np.ndarray:
  return np.array([[metric[_G_IJ_COMP[(min(i,j), max(i,j))]] for j in range(3)]
                   for i in range(3)])

def _b_hat_for(metric) -> tuple:
  """b_i = g_i3/sqrt(g_33), which is how gkeyll builds it: b is along e_3."""
  return tuple(_metric_matrix(metric)[i, 2]/np.sqrt(metric[_G_IJ_COMP[(2,2)]])
               for i in range(3))

def _linear_gdata_3d(comps) -> GData:
  """A field whose comp-th component is offset + sum_d slope_d*x_d, given as
  (offset, slopes) pairs. The p1 basis represents such a field exactly.

  The 3D basis is the tensor product of the 1D pair (1/sqrt(2), sqrt(3/2)*xi),
  ordered 1,x,y,z,xy,xz,yz,xyz, so the mode linear in direction d is
  sqrt(3)*psi0*xi_d and carries the whole slope. The cell-average mode is set
  per cell, so the field is globally linear and not just linear in each cell.
  """
  centers = np.meshgrid(*[0.5*(n[:-1] + n[1:]) for n in _grid_3d()], indexing="ij")

  values = np.zeros((*_NUM_CELLS_3D, _NUM_BASIS_3D*len(comps)))
  for comp, (offset, slopes) in enumerate(comps):
    base = comp*_NUM_BASIS_3D
    values[..., base] = (offset + sum(s*c for s, c in zip(slopes, centers)))/_PSI0_3D
    for direction, slope in enumerate(slopes):
      dx = _LENGTHS_3D[direction]/_NUM_CELLS_3D[direction]
      values[..., base + 1 + direction] = slope*dx/(2.0*np.sqrt(3.0)*_PSI0_3D)
  return _gdata_3d(values)

def _linear_apar(slopes) -> GData:
  """Apar = sum_d slope_d*x_d."""
  return _linear_gdata_3d([(0.0, slopes)])

def _arbitrary_apar() -> GData:
  """An Apar with every mode populated and no symmetry to exploit."""
  return _gdata_3d(np.random.default_rng(20260904).standard_normal(
    (*_NUM_CELLS_3D, _NUM_BASIS_3D)))

def _sources_3d(apar, metric=_EUCLIDEAN_METRIC, b_hat=None, jacob=1.0) -> list:
  """The [Apar, 1/J, b_i, g_ij] sources, the geometry uniform in space.

  b defaults to the unit vector along e_3 that the metric implies.
  """
  return [apar, _const_gdata_3d(1.0/jacob),
          _const_gdata_3d(_b_hat_for(metric) if b_hat is None else b_hat),
          _const_gdata_3d(metric)]

def _B_sources_3d(apar, bmag, metric=_EUCLIDEAN_METRIC, jacob=1.0) -> list:
  """The [Apar, B, 1/J, b_i, g_ij] sources of B_tot over the geometry of _sources_3d."""
  _, jacobgeo_inv, b_i, g_ij = _sources_3d(apar, metric=metric, jacob=jacob)
  return [apar, _const_gdata_3d(bmag), jacobgeo_inv, b_i, g_ij]

def _equilibrium_srcs(b_srcs) -> list:
  """The [B, b_i] sources of B_equilibrium, out of a _B_sources_3d list."""
  return [b_srcs[1], b_srcs[3]]

def _dual_equilibrium_srcs(b_srcs) -> list:
  """The [B, g_ij] sources of B_dual_equilibrium, out of a _B_sources_3d list."""
  return [b_srcs[1], b_srcs[4]]

def _grad_apar(apar: GData) -> list:
  """d(Apar)/dx^d in every cell, as DG fields."""
  dgops = GkeyllDGops()
  lower, upper = apar.get_bounds()
  cells = apar.get_num_cells()

  grad = []
  for direction in range(3):
    deriv = ff._empty_gdata_from_gdata(apar)
    dgops.differentiate(direction, 1, (upper[direction] - lower[direction])/cells[direction],
                        0, deriv, 0, apar)
    grad.append(deriv)
  return grad


@_needs_dgops
class TestPerpMagneticFluctuations:
  """The covariant components of dB = curl(Apar*b)."""

  def test_linear_apar_has_the_slopes_it_claims(self):
    """Pin the basis ordering the closed-form tests below rely on."""
    slopes = (2.5, -1.75, 0.5)
    for deriv, slope in zip(_grad_apar(_linear_apar(slopes)), slopes):
      assert np.allclose(_cell_avg_3d(deriv), slope, rtol=1e-12)
      assert np.allclose(deriv.get_values()[..., 1:], 0.0, atol=1e-12)

  def test_dB_for_a_uniform_field_along_z(self):
    """In Cartesian coordinates with b = z and J = 1, Apar = a*x + c*y gives
    dB = (c, -a, 0).

    Pins the sign, the cyclic wiring and which coordinate each component
    belongs to, all at once. The metric being the identity, the lowering is a
    no-op here and the covariant components are the physical ones.
    """
    a, c = 2.5, -1.75
    srcs = _sources_3d(_linear_apar((a, c, 0.0)))

    for comp, expected in enumerate((c, -a, 0.0)):
      dB = ff.fetch_dB_perp(srcs, dir=comp)
      assert np.allclose(_cell_avg_3d(dB), expected, atol=1e-12*max(abs(a), abs(c)))

  def test_the_gradient_along_b_does_not_contribute(self):
    """What makes dB perpendicular: with b = z, tilting Apar in z changes nothing."""
    flat = _sources_3d(_linear_apar((2.5, -1.75, 0.0)))
    tilted = _sources_3d(_linear_apar((2.5, -1.75, 9.0)))

    for comp in range(3):
      assert np.allclose(_cell_avg_3d(ff.fetch_dB_perp(flat, dir=comp)),
                         _cell_avg_3d(ff.fetch_dB_perp(tilted, dir=comp)), atol=1e-12)

  def test_the_third_covariant_component_vanishes_for_a_uniform_b(self):
    """Where b is uniform, dB = (grad(Apar) x b)/J is perpendicular to b, and in
    covariant components that reads dB_3 = 0.

    b is along e_3, so dB.bhat = dB_3/sqrt(g_33): the whole parallel part sits
    in the third covariant component and nowhere else. This is what makes the
    covariant components the readable ones - the contravariant dB^3 is instead
    the *largest* of the three, e_3 being the shortest basis vector. It holds
    for any Apar and any coordinates, so it also pins the cyclic index wiring
    and the metric lowering together.
    """
    srcs = _sources_3d(_arbitrary_apar(), metric=_SKEW_METRIC, jacob=1.7)

    dB = [ff.fetch_dB_perp(srcs, dir=comp).get_values() for comp in range(3)]
    scale = max(np.abs(dB_i).max() for dB_i in dB)

    assert np.all(np.abs(dB[2]) < 1e-12*scale)
    assert np.abs(dB[0]).max() > 0.01*scale, "the perpendicular part must not vanish too"

  def test_dB_perp_dual_gives_the_contravariant_components(self):
    """dB_perp_dual is the bare curl, before any lowering: (c, -a, 0) here."""
    a, c = 2.5, -1.75
    srcs = _sources_3d(_linear_apar((a, c, 0.0)))[:3]

    for comp, expected in enumerate((c, -a, 0.0)):
      dB = ff.fetch_dB_perp_dual(srcs, dir=comp)
      assert np.allclose(_cell_avg_3d(dB), expected, atol=1e-12*max(abs(a), abs(c)))

  def test_the_metric_lowers_the_index(self):
    """dB_i = g_ij dB^j, with g_ij read as g_11,g_12,g_13,g_22,g_23,g_33.

    b = z-hat and J = 1 hold the contravariant components at (c, -a, 0) while
    the metric is skewed, so only the lowering can produce the answer.
    """
    a, c = 2.5, -1.75
    srcs = _sources_3d(_linear_apar((a, c, 0.0)), metric=_SKEW_METRIC,
                       b_hat=(0.0, 0.0, 1.0))

    expected = _metric_matrix(_SKEW_METRIC) @ np.array([c, -a, 0.0])
    for comp in range(3):
      assert np.allclose(_cell_avg_3d(ff.fetch_dB_perp(srcs, dir=comp)),
                         expected[comp], rtol=1e-12)

  def test_dB_perp_is_dB_perp_dual_lowered(self):
    """The two registered quantities must be the same field, index up or down.

    Checked on the real skewed geometry of the other tests rather than on a
    hand-written expectation, so it holds whatever the curl turns out to be.
    """
    srcs = _sources_3d(_arbitrary_apar(), metric=_SKEW_METRIC, jacob=1.7)
    dB_up = [_cell_avg_3d(ff.fetch_dB_perp_dual(srcs[:3], dir=j)) for j in range(3)]

    g_kl = _metric_matrix(_SKEW_METRIC)
    for comp in range(3):
      expected = sum(g_kl[comp, j]*dB_up[j] for j in range(3))
      assert np.allclose(_cell_avg_3d(ff.fetch_dB_perp(srcs, dir=comp)), expected,
                         rtol=1e-11, atol=1e-13*max(np.abs(u).max() for u in dB_up))

  def test_a_twisting_b_contributes_even_for_a_uniform_apar(self):
    """dB is the full curl, so with grad(Apar) = 0 what is left is Apar*curl(b).

    Taking b_1 = slope*y with the other components uniform leaves the whole
    perturbation in dB_3 = -Apar*slope/J.
    """
    apar_val, slope, jacob = 0.35, 1.9, 1.7
    b_i = _linear_gdata_3d([(0.0, (0.0, slope, 0.0)), (0.0, (0.0,)*3), (1.0, (0.0,)*3)])
    srcs = [_const_gdata_3d(apar_val), _const_gdata_3d(1.0/jacob), b_i,
            _const_gdata_3d(_EUCLIDEAN_METRIC)]

    expected = -apar_val*slope/jacob
    for comp, want in enumerate((0.0, 0.0, expected)):
      assert np.allclose(_cell_avg_3d(ff.fetch_dB_perp(srcs, dir=comp)), want,
                         atol=1e-12*abs(expected))

  def test_a_vanishing_apar_gives_no_perturbation(self):
    """Guard against the checks above passing for a function that returns junk."""
    srcs = _sources_3d(_const_gdata_3d(0.0), metric=_SKEW_METRIC, jacob=1.7)
    for comp in range(3):
      assert np.all(ff.fetch_dB_perp(srcs, dir=comp).get_values() == 0.0)

  def test_dB_scales_as_one_over_J(self):
    apar = _linear_apar((2.5, -1.75, 0.0))
    unit = ff.fetch_dB_perp(_sources_3d(apar, jacob=1.0), dir=0)
    scaled = ff.fetch_dB_perp(_sources_3d(apar, jacob=4.0), dir=0)
    assert np.allclose(_cell_avg_3d(scaled), 0.25*_cell_avg_3d(unit), rtol=1e-12)

  def test_a_direction_must_be_requested(self):
    with pytest.raises(KeyError, match="dir="):
      ff.fetch_dB_perp(_sources_3d(_linear_apar((1.0, 1.0, 0.0))))


@_needs_dgops
class TestPerpMagneticFluctuationMagnitude:
  """|dB| = sqrt(g_kl*dB^k*dB^l), contracting the contravariant components,
  which pins the g_ij component order as much as the contraction itself."""

  # With b = z-hat and J = 1 these slopes give dB^k = (slope_y, -slope_x, 0).
  _SLOPE_X, _SLOPE_Y = 2.5, -1.75

  def _mag(self, metric):
    srcs = _sources_3d(_linear_apar((self._SLOPE_X, self._SLOPE_Y, 0.0)),
                       metric=metric, b_hat=(0.0, 0.0, 1.0))
    return _cell_avg_3d(ff.fetch_dB_perp_mag(srcs))

  def test_euclidean_metric_gives_the_perpendicular_gradient(self):
    """With b = z and g_ij = delta_ij, |dB| = |grad_perp(Apar)|."""
    assert np.allclose(self._mag(_EUCLIDEAN_METRIC),
                       np.hypot(self._SLOPE_X, self._SLOPE_Y), rtol=1e-12)

  def test_the_diagonal_entries_are_read_in_order(self):
    """g_11, g_22 and g_33 sit at components 0, 3 and 5, not 0, 1 and 2."""
    g_11, g_22, g_33 = 4.0, 9.0, 16.0
    expected = np.sqrt(g_11*self._SLOPE_Y**2 + g_22*self._SLOPE_X**2)  # g_33 must not enter.
    assert np.allclose(self._mag((g_11, 0.0, 0.0, g_22, 0.0, g_33)), expected, rtol=1e-12)

  def test_the_off_diagonal_entries_are_counted_twice(self):
    """g_12 sits at component 1, and the (1,2) and (2,1) pair gives 2*g_12."""
    g_12 = 0.3
    expected = np.sqrt(self._SLOPE_Y**2 + self._SLOPE_X**2
                       - 2.0*g_12*self._SLOPE_Y*self._SLOPE_X)
    assert np.allclose(self._mag((1.0, g_12, 0.0, 1.0, 0.0, 1.0)), expected, rtol=1e-12)

  def test_a_general_metric_with_all_three_components(self):
    """g_13 and g_23 too: a skewed uniform b gives dB^k = (grad(Apar) x b)/J."""
    slopes = (2.5, -1.75, 0.9)
    b_hat, jacob = _b_hat_for(_SKEW_METRIC), 1.7
    srcs = _sources_3d(_linear_apar(slopes), metric=_SKEW_METRIC, jacob=jacob)

    dB = np.cross(slopes, b_hat)/jacob
    expected = np.sqrt(dB @ _metric_matrix(_SKEW_METRIC) @ dB)

    assert np.allclose(_cell_avg_3d(ff.fetch_dB_perp_mag(srcs)), expected, rtol=1e-12)

  def test_a_vanishing_apar_gives_no_perturbation(self):
    srcs = _sources_3d(_const_gdata_3d(0.0), metric=_SKEW_METRIC, jacob=1.7)
    assert np.all(ff.fetch_dB_perp_mag(srcs).get_values() == 0.0)


@_needs_dgops
class TestTotalMagneticField:
  """B = B0*b + dB, covariant (B_tot) and contravariant (B_tot_dual)."""

  _BMAG = 1.9
  _JACOB = 1.7

  def _srcs(self, apar, metric=_SKEW_METRIC):
    return _B_sources_3d(apar, self._BMAG, metric=metric, jacob=self._JACOB)

  def test_the_equilibrium_covariant_components_are_B_times_b(self):
    srcs = self._srcs(_const_gdata_3d(0.0))

    for comp, b in enumerate(_b_hat_for(_SKEW_METRIC)):
      assert np.allclose(_cell_avg_3d(ff.fetch_B_equilibrium(_equilibrium_srcs(srcs), dir=comp)),
                         self._BMAG*b, rtol=1e-12)

  def test_the_equilibrium_contravariant_components_lie_along_e3(self):
    """b^i = delta^i_3/sqrt(g_33), so the first two components are exactly zero."""
    up = _dual_equilibrium_srcs(self._srcs(_const_gdata_3d(0.0)))

    for comp in (0, 1):
      assert np.all(ff.fetch_B_dual_equilibrium(up, dir=comp).get_values() == 0.0)

    expected = self._BMAG/np.sqrt(_SKEW_METRIC[_G_IJ_COMP[(2,2)]])
    assert np.allclose(_cell_avg_3d(ff.fetch_B_dual_equilibrium(up, dir=2)),
                       expected, rtol=1e-12)

  def test_the_two_equilibrium_representations_describe_the_same_field(self):
    """B_i B^i must come back as B^2 exactly.

    The covariant side is built from b_i and the contravariant side from
    1/sqrt(g_33), by two routes that share nothing, so this is the check that
    b^i and b_i are the same unit vector and that g_33 is read from the right
    component of g_ij.
    """
    srcs = self._srcs(_const_gdata_3d(0.0))

    mag_sq = sum(_cell_avg_3d(ff.fetch_B_equilibrium(_equilibrium_srcs(srcs), dir=comp))
                 *_cell_avg_3d(ff.fetch_B_dual_equilibrium(_dual_equilibrium_srcs(srcs), dir=comp))
                 for comp in range(3))

    assert np.allclose(np.sqrt(mag_sq), self._BMAG, rtol=1e-12)

  def test_the_perturbation_is_added_to_the_equilibrium(self):
    """B_i = B*b_i + dB_i and B^i = B*b^i + dB^i, both term by term."""
    apar = _linear_apar((2.5, -1.75, 0.9))
    dB_srcs = _sources_3d(apar, metric=_SKEW_METRIC, jacob=self._JACOB)
    srcs = self._srcs(apar)

    for comp in range(3):
      expected = (_cell_avg_3d(ff.fetch_B_equilibrium(_equilibrium_srcs(srcs), dir=comp))
                  + _cell_avg_3d(ff.fetch_dB_perp(dB_srcs, dir=comp)))
      assert np.allclose(_cell_avg_3d(ff.fetch_B_tot(srcs, dir=comp)), expected, rtol=1e-12)

      expected = (_cell_avg_3d(ff.fetch_B_dual_equilibrium(_dual_equilibrium_srcs(srcs), dir=comp))
                  + _cell_avg_3d(ff.fetch_dB_perp_dual(dB_srcs[:3], dir=comp)))
      assert np.allclose(_cell_avg_3d(ff.fetch_B_tot_dual(srcs, dir=comp)), expected, rtol=1e-12)

  def test_a_vanishing_apar_leaves_the_equilibrium_field(self):
    """With no perturbation the total field must be the equilibrium one exactly.

    This is also what the no-Apar source combination is expected to return.
    """
    srcs = self._srcs(_const_gdata_3d(0.0))

    for comp in range(3):
      assert np.allclose(
        _cell_avg_3d(ff.fetch_B_tot(srcs, dir=comp)),
        _cell_avg_3d(ff.fetch_B_equilibrium(_equilibrium_srcs(srcs), dir=comp)), rtol=1e-12)
      assert np.allclose(
        _cell_avg_3d(ff.fetch_B_tot_dual(srcs, dir=comp)),
        _cell_avg_3d(ff.fetch_B_dual_equilibrium(_dual_equilibrium_srcs(srcs), dir=comp)),
        rtol=1e-12)

  def test_a_direction_must_be_requested(self):
    srcs = self._srcs(_linear_apar((1.0, 1.0, 0.0)))
    for fetch, fetch_srcs in ((ff.fetch_B_tot, srcs),
                              (ff.fetch_B_equilibrium, _equilibrium_srcs(srcs)),
                              (ff.fetch_B_tot_dual, srcs),
                              (ff.fetch_B_dual_equilibrium, _dual_equilibrium_srcs(srcs))):
      with pytest.raises(KeyError, match="dir="):
        fetch(fetch_srcs)


# --- The same, over real electromagnetic output ------------------------------
#
# The cases above are uniform in space, which is what makes them exact. These
# run the same code over a real 3x2v electromagnetic TCV run, so that the actual
# geometry files and their component layouts are exercised too. That output is
# ~76 MB and is not tracked by git, so it is skipped when absent.
_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
_APAR_DATA_DIR = os.path.join(_TEST_DATA_DIR, "apar_data.ignore")
_APAR_DATA_PREFIX = "rt_gk_tcv_nt_iwl_adapt_src_3x2v_p1"
_APAR_DATA_FRAME = 10

_needs_apar_data = pytest.mark.skipif(
  not os.path.isdir(_APAR_DATA_DIR),
  reason="requires the untracked electromagnetic TCV output in apar_data.ignore")


def _apar_data(stem: str) -> GData:
  return GData(os.path.join(_APAR_DATA_DIR, f"{_APAR_DATA_PREFIX}-{stem}.gkyl"))

def _flattened(gdata: GData) -> GData:
  """Keep only the cell-average mode, which makes a weak product by it exact."""
  out = ff._empty_gdata_from_gdata(gdata)
  out.get_values()[..., ::_NUM_BASIS_3D] = gdata.get_values()[..., ::_NUM_BASIS_3D]
  return out


@_needs_dgops
@_needs_apar_data
class TestPerpMagneticFluctuationsOnRealData:
  """dB from a real electromagnetic TCV run."""

  def _sources(self, flatten: bool = True) -> list:
    prepare = _flattened if flatten else (lambda gdata: gdata)
    return [_apar_data(f"apar_{_APAR_DATA_FRAME}"),
            prepare(_apar_data("geo_int_jacobgeo_inv")),
            prepare(_apar_data("geo_int_b_i")),
            prepare(_apar_data("geo_int_g_ij"))]

  def test_the_components_match_an_independent_evaluation(self):
    """dB_i against a from-scratch numpy evaluation on the real geometry.

    The geometry is flattened to its cell averages, which both makes the weak
    products exact and reduces the curl to (b_k dA/dx^j - b_j dA/dx^k)/J, so the
    two paths must agree to round-off. Apar keeps all eight of its coefficients,
    so the derivatives, the index wiring and the lowering still see varying 3D
    data.
    """
    srcs = self._sources()
    apar, jacobgeo_inv, b_i, g_ij = srcs
    grad_apar = [deriv.get_values() for deriv in _grad_apar(apar)]

    # Piecewise-constant weights; keep a trailing axis to broadcast over the modes.
    def weight(gdata, comp=0):
      return gdata.get_values()[..., comp*_NUM_BASIS_3D, None]*_PSI0_3D

    inv_J = weight(jacobgeo_inv)
    b = [weight(b_i, k) for k in range(3)]
    dB_up = [inv_J*(b[(i+2) % 3]*grad_apar[(i+1) % 3] - b[(i+1) % 3]*grad_apar[(i+2) % 3])
             for i in range(3)]

    for comp in range(3):
      expected = sum(weight(g_ij, _G_IJ_COMP[(min(comp,j), max(comp,j))])*dB_up[j]
                     for j in range(3))

      # g_ij spans four orders of magnitude here and the lowering all but
      # cancels for comp 2, so the residual is measured against the size of the
      # component rather than entry by entry.
      got = ff.fetch_dB_perp(srcs, dir=comp).get_values()
      assert np.allclose(got, expected, rtol=1e-9, atol=1e-11*np.abs(expected).max())

  def test_the_parallel_component_is_the_small_one(self):
    """dB_3 carries the whole parallel part, and dB is nearly perpendicular.

    What survives is Apar*bhat.curl(b), which magnetic shear makes nonzero but
    small; the point is that the perturbation shows up in dB_1 and dB_2, as it
    would not in the contravariant components.
    """
    srcs = self._sources(flatten=False)
    rms = lambda field: np.sqrt(np.mean(field**2))
    dB = [rms(_cell_avg_3d(ff.fetch_dB_perp(srcs, dir=comp))) for comp in range(3)]

    assert dB[2] < 0.2*dB[0], "the third covariant component must be the small one"
    assert dB[2] > 0.0

  def test_the_perturbation_is_a_small_fraction_of_the_field(self):
    """A units slip would put |dB|/B nowhere near the per-mille level it sits at."""
    ratio = (_cell_avg_3d(ff.fetch_dB_perp_mag(self._sources(flatten=False)))
             /_cell_avg_3d(_apar_data("geo_int_bmag")))

    assert np.all(np.isfinite(ratio))
    assert np.sqrt(np.mean(ratio**2)) > 1e-5
    assert ratio.max() < 0.05

  def test_the_total_field_barely_departs_from_the_equilibrium_one(self):
    """sqrt(B_i B^i) against bmag on the real geometry.

    Both representations of the equilibrium have to agree for this to land on
    bmag at all, and the perturbation may only shift it by the per-mille amount
    that dB/B allows. The residual is the parallel part of dB, which is what
    lifts |B| above B rather than leaving it unchanged.
    """
    apar, jacobgeo_inv, b_i, g_ij = self._sources(flatten=False)
    bmag = _apar_data("geo_int_bmag")
    srcs = [apar, bmag, jacobgeo_inv, b_i, g_ij]

    mag_sq = sum(_cell_avg_3d(ff.fetch_B_tot(srcs, dir=comp))
                 *_cell_avg_3d(ff.fetch_B_tot_dual(srcs, dir=comp)) for comp in range(3))
    ratio = np.sqrt(mag_sq)/_cell_avg_3d(bmag)

    assert np.all(np.isfinite(ratio))
    assert np.allclose(ratio, 1.0, atol=0.02)
    assert not np.allclose(ratio, 1.0, atol=1e-9), "the perturbation must move it a little"
