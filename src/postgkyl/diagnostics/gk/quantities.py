"""Gyrokinetic derived-quantity physics -- the ``fetch_*`` functions behind the
quantity registry.

All formulas preserve their inputs' representation. Modal sources stay native:
products and inverses use Gkeyll weak kernels, square roots use its quadrature
projection, and derivatives act on each cell's polynomial. Interpolation is
an explicit downstream operation, after the quantity has been computed.

Weak products are not associative. Keep the intermediate projections and the
inverse-then-multiply order of the original GK quantity definitions: ``a * (1/b)``
uses a weak inverse and product, whereas ``a/b`` solves a weak division directly.
``**0.5`` uses Gkeyll's quadrature projection, including its negative-value floor.
Inputs are the operator-enabled ``GData`` objects returned by the GK loader.

Naming keys (matching ``src_bak`` so the registry mapping in ``registry.py``
stays recognizable):
  s#: source #, c#: component #, add/sub/mul/div: the combining operator,
  pos/neg: the plus/minus term of a curvilinear cross product.
"""

from __future__ import annotations

import operator
import os
from typing import TYPE_CHECKING
import warnings

import numpy as np
from scipy import constants

from postgkyl import operations

if TYPE_CHECKING:
  from postgkyl.gdata import GData


def _get_ctx_val(gdata: "GData", key: str, **kwargs):
  """A value (or one value per species) for ``key``: ``kwargs[key]``
  (an explicit ``load_quantity`` option) wins over ``gdata.ctx[key]`` (the
  file's own attribute), which wins over raising.

  ``kwargs[key]`` may be a single value (applies to every species) or a
  list/tuple with one entry per species, indexed by ``kwargs
  ['species_idx']`` -- the position :meth:`~postgkyl.diagnostics.
  gk.quantity.GkQuantity.fetch_multi`/``load_quantity`` stamp
  onto ``extra`` for the species currently being resolved.
  """
  if key in kwargs:
    val = kwargs[key]
    if not isinstance(val, (list, tuple)):
      return val
    species_idx = kwargs.get("species_idx")
    if species_idx is None:
      raise KeyError(
          f"fetch function: '{key}' was given {len(val)} values "
          "but this quantity is not resolved per species here, so there is "
          "no way to tell which one to use. Pass a single value instead.")
    if species_idx >= len(val):
      species = kwargs.get("species")
      raise ValueError(
          f"fetch function: '{key}' was given only {len(val)} "
          f"values but species #{species_idx}"
          f"{f' ({species})' if species else ''} was requested. Give one "
          "value per species, in the order of '--species'.")
    return val[species_idx]
  if gdata.ctx.get(key) is not None:
    return gdata.ctx[key]
  raise KeyError(
      f"fetch function: context key '{key}' not found in the dataset; "
      f"pass the '{key}' option, either one value or one value per "
      "species in the order of the species list.")


def _component(d: "GData", comp: int | None) -> "GData":
  """Select a physical field while retaining its complete DG expansion."""
  return d.clone() if comp is None else operations.select(d, comp=comp)


def _direction(kwargs, quantity: str) -> int:
  """The direction ``k`` (0, 1 or 2 for x, y, z) selected with ``dir=<k>``:
  the component of a vector, or the direction of a gradient length, flux or
  diffusivity. No quantity assumes a direction."""
  if "dir" not in kwargs:
    raise ValueError(f"{quantity}: choose the direction with direction=<k> "
                     "(CLI: -d <k>), 0: x, 1: y, 2: z.")
  comp = int(kwargs["dir"])
  if not 0 <= comp < 3:
    raise ValueError(f"{quantity}: direction must be 0, 1 or 2, got {comp}.")
  return comp


# Component of the symmetric metric tensors g_ij and g^ij holding the (k, l)
# entry, stored as 11, 12, 13, 22, 23, 33.
_METRIC_COMP = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (1, 1): 3,
    (1, 2): 4,
    (2, 2): 5
}


def _config_axis(cdim: int) -> tuple[int | None, int | None, int]:
  """The data dimension holding each of ``x, y, z``, ``None`` where a reduced
  simulation does not carry it: 3x holds ``(x, y, z)``, 2x ``(x, z)`` and 1x
  only ``z``."""
  return (0 if cdim > 1 else None, 1 if cdim > 2 else None, cdim - 1)


def _derivative(field: "GData", comp: int, quantity: str) -> "GData":
  """``d(field)/dx^comp`` along configuration-space direction ``comp`` (0, 1,
  2 for x, y, z), which the simulation must carry."""
  dim = _config_axis(field.num_dims)[comp]
  if dim is None:
    raise ValueError(
        f"{quantity}: a {field.num_dims}x simulation has no {'xyz'[comp]} "
        "direction (1x runs carry z only, 2x runs x and z).")
  return operations.differentiate(field, direction=dim)


def _metric_diagonal(metric: "GData", comp: int) -> "GData":
  """The diagonal entry ``(comp, comp)`` of a symmetric metric tensor."""
  return _component(metric, _METRIC_COMP[(comp, comp)])


def _metric_contract(metric: "GData", comp: int, vec) -> "GData":
  """``sum_j g_{comp j} V_j``: component ``comp`` of the vector ``V`` (whose
  ``j``-th component is ``vec(j)``) with its index raised or lowered by the
  six-component symmetric ``metric``."""
  terms = (_component(metric, _METRIC_COMP[(min(comp, j), max(comp, j))]) *
           vec(j) for j in range(3))
  total = next(terms)
  for term in terms:
    total = total + term
  return total


def _vector_magnitude(cov, contra) -> "GData":
  """``sqrt(V_i V^i)`` of a vector whose covariant and contravariant
  components are ``cov(i)`` and ``contra(i)``. The product is of higher
  order than the basis, so the result carries projection aliasing."""
  total = cov(0) * contra(0)
  for i in (1, 2):
    total = total + cov(i) * contra(i)
  return total**0.5


# --------------------------------------------------- generic fetch factories
def _make_fetch_comp(icomp: int | None):
  """A fetch function that extracts the ``icomp``-th physical component."""

  def fetch(gdatas: list["GData"], **kwargs):
    return _component(gdatas[0], icomp)

  fetch.__name__ = f"fetch_comp{icomp}" if icomp is not None else "fetch_compAll"
  return fetch


def _make_fetch_binop(si: int, ci: int, sj: int, cj: int, op):
  """A fetch function combining component ``ci`` of source ``si`` with
  component ``cj`` of source ``sj`` via a representation-aware operation."""

  def fetch(gdatas: list["GData"], **kwargs):
    a = _component(gdatas[si], ci)
    b = _component(gdatas[sj], cj)
    return a * (1.0 / b) if op is operator.truediv else op(a, b)

  fetch.__name__ = f"fetch_s{si}c{ci}_{op.__name__}_s{sj}c{cj}"
  return fetch


# Extract a single component.
fetch_s0cAll = _make_fetch_comp(None)
fetch_s0c0 = _make_fetch_comp(0)
fetch_s0c1 = _make_fetch_comp(1)
fetch_s0c2 = _make_fetch_comp(2)
fetch_s0c3 = _make_fetch_comp(3)

# Combine components across (possibly different) sources.
fetch_s0c0_add_s1c0 = _make_fetch_binop(0, 0, 1, 0, operator.add)
fetch_s0c2_add_s0c3 = _make_fetch_binop(0, 2, 0, 3, operator.add)
fetch_s0c0_sub_s1c0 = _make_fetch_binop(0, 0, 1, 0, operator.sub)
fetch_s0c0_mul_s1c0 = _make_fetch_binop(0, 0, 1, 0, operator.mul)
fetch_s0c0_mul_s0c1 = _make_fetch_binop(0, 0, 0, 1, operator.mul)
fetch_s1c0_div_s0c0 = _make_fetch_binop(1, 0, 0, 0, operator.truediv)


# ------------------------------------------------------------------ moments
def fetch_M1_from_H(gdatas: list["GData"], **kwargs):
  """M1 from the Hamiltonian moments: ``mass**-1 * (comp0 * comp1)``."""
  hmom = gdatas[0]
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return _component(hmom, 0) * _component(hmom, 1) / mass


def _make_fetch_M2_from_Max(par: bool, t_comp: int):
  """Second parallel (``par``) or perpendicular velocity moment from
  (Bi)Maxwellian moments ``(n, u_par, T/m, ...)``, with the temperature over
  mass in component ``t_comp``::

    M2par  = n*T_par/m + n*u_par^2,
    M2perp = 2*n*T_perp/m.
  """

  def fetch(gdatas: list["GData"], **kwargs):
    mom = gdatas[0]
    n = _component(mom, 0)
    thermal = n * _component(mom, t_comp)
    if not par:
      return thermal * 2.0
    upar = _component(mom, 1)
    return thermal + (n * upar) * upar

  fetch.__name__ = f"fetch_M2{'par' if par else 'perp'}_from_Max_c{t_comp}"
  return fetch


fetch_M2par_from_Max = _make_fetch_M2_from_Max(True, 2)
fetch_M2perp_from_Max = _make_fetch_M2_from_Max(False, 2)
fetch_M2par_from_BiMax = _make_fetch_M2_from_Max(True, 2)
fetch_M2perp_from_BiMax = _make_fetch_M2_from_Max(False, 3)


def fetch_Tpar_from_BiMax(gdatas: list["GData"], **kwargs):
  """Tpar from BiMaxwellian moments: ``mass * comp2``."""
  Tpar = fetch_s0c2(gdatas)
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return Tpar * mass


def fetch_Tpar_from_M0_M1_M2par(gdatas: list["GData"], **kwargs):
  """``upar*M1 + M0*Tpar/m = M2par`` => ``Tpar = m*(M2par - upar*M1)/M0``."""
  m0, m1, m2par = gdatas
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  m0_inv = 1.0 / m0
  upar = m1 * m0_inv
  thermal = (m2par - upar * m1) * mass
  return thermal * m0_inv


def fetch_Tperp_from_BiMax(gdatas: list["GData"], **kwargs):
  """Tperp from BiMaxwellian moments: ``mass * comp3``."""
  Tperp = fetch_s0c3(gdatas)
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return Tperp * mass


def fetch_Tperp_from_M0_M2perp(gdatas: list["GData"], **kwargs):
  """``Tperp = 0.5 * mass * (M2perp / M0)``."""
  Tperp = fetch_s1c0_div_s0c0(gdatas)
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return Tperp * (0.5 * mass)


def fetch_temp_from_Max(gdatas: list["GData"], **kwargs):
  """temp from Maxwellian moments: ``mass * comp2``."""
  temp = fetch_s0c2(gdatas)
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return temp * mass


def fetch_temp_from_Tpar_Tperp(gdatas: list["GData"], **kwargs):
  """``temp = (Tpar + 2*Tperp) / 3``."""
  Tpar, Tperp = gdatas
  return (Tpar + 2.0 * Tperp) / 3.0


def fetch_press_from_Max(gdatas: list["GData"], **kwargs):
  """Pressure from Maxwellian moments: ``press = mass * comp0 * comp2``."""
  maxmom = gdatas[0]
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return _component(maxmom, 0) * _component(maxmom, 2) * mass


def fetch_press_from_BiMax(gdatas: list["GData"], **kwargs):
  """Pressure from BiMaxwellian moments: ``press = comp0 * mass*(Tpar+2Tperp)/3``."""
  bimax = gdatas[0]
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  temp = (_component(bimax, 2) + _component(bimax, 3) * 2.0) * (mass / 3.0)
  return _component(bimax, 0) * temp


def fetch_press_p(gdatas: list["GData"], **kwargs):
  """Perpendicular/parallel pressure in J/m^3: ``p_p = n * T_p``."""
  m0, Tp = gdatas
  return m0 * Tp


def _make_fetch_q(name: str):
  """Return a fetch function for the lab-frame parallel flux of the
  parallel (``name='par'``) or perpendicular (``name='perp'``) kinetic
  energy::

    q_par  = (m/2)*M3par  = (m/2) int(vpar^3 f) dv,
    q_perp = (m/2)*M3perp = (m/2) int(vpar*vperp^2 f) dv,

  so that ``q_par + q_perp`` is the parallel flux of the total kinetic
  energy. Both are in W/m^2 (kg/s^3). ``gdatas``: ``[M3par]`` or
  ``[M3perp]``.
  """

  def fetch(gdatas: list["GData"], **kwargs):
    m3 = gdatas[0]
    mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
    return m3 * (0.5 * mass)

  fetch.__name__ = f"fetch_q{name}"
  return fetch


fetch_qpar = _make_fetch_q("par")
fetch_qperp = _make_fetch_q("perp")


def _make_fetch_q_fluid(name: str):
  """Return a fetch function for the parallel/perpendicular heat flux in
  the fluid (drift) frame -- the energy carried by the random part of the
  motion, ``u = M1/M0`` being the parallel drift speed::

    q_par  = (m/2) int (vpar-u)^3 f dv
           = (m/2) [M3par - 3*u*M2par + 3*u^2*M1 - u^3*M0]
           = (m/2) [M3par - 3*u*M2par + 2*u^2*M1],
    q_perp = (m/2) int (vpar-u)*vperp^2 f dv
           = (m/2) [M3perp - u*M2perp].

  ``gdatas`` (in this order): ``[M0, M1, M2par, M3par]`` or
  ``[M0, M1, M2perp, M3perp]``.
  """
  is_par = name == "par"

  def fetch(gdatas: list["GData"], **kwargs):
    m0, m1, m2, m3 = gdatas
    mass = _get_ctx_val(gdatas[0], "mass", **kwargs)

    upar = m1 * (1.0 / m0)
    u_m2 = upar * m2

    if is_par:
      u_sq = upar**2
      u_sq_m1 = u_sq * m1
      out = m3 - u_m2 * 3.0 + u_sq_m1 * 2.0
    else:
      out = m3 - u_m2

    return out * (0.5 * mass)

  fetch.__name__ = f"fetch_q{name}_fluid"
  return fetch


fetch_qpar_fluid = _make_fetch_q_fluid("par")
fetch_qperp_fluid = _make_fetch_q_fluid("perp")


def fetch_vt(gdatas: list["GData"], **kwargs):
  """Thermal speed ``vt = sqrt(T/m)`` (m/s), ``m`` the requested species'
  mass. ``gdatas``: ``[temp]`` (temperature, in Joules)."""
  temp = gdatas[0]
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return (temp / mass)**0.5


def fetch_larmor_radius(gdatas: list["GData"], **kwargs):
  """Species Larmor (gyro-)radius: ``rho = sqrt(m*T)/(|q|*B)``. ``gdatas``:
  ``[temp, bmag]``."""
  temp, bmag = gdatas
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  charge = abs(_get_ctx_val(gdatas[0], "charge", **kwargs))
  return (temp * mass)**0.5 * (1.0 / (bmag * charge))


def fetch_debye_length(gdatas: list["GData"], **kwargs):
  """Species-wise Debye length: ``lambda_D = sqrt(eps0*T/(n*q^2))``.
  ``gdatas``: ``[temp, M0]``."""
  temp, m0 = gdatas
  charge = _get_ctx_val(gdatas[0], "charge", **kwargs)
  return (temp * constants.epsilon_0 * (1.0 / (m0 * charge**2)))**0.5


def _split_elc_ions(gdatas, quantity: str, **kwargs):
  """Split the per-species sources of a multi-species quantity into the
  electron entry and the ion entries, by the sign of each species' charge.

  ``gdatas[i]`` is species ``i``'s resolved source list, starting with
  ``[M0, temp]`` (as
  :meth:`~postgkyl.diagnostics.gk.quantity.GkQuantity.fetch_multi`
  hands it to an ``is_multi_species`` fetch function); each entry's
  ``mass``/``charge`` is resolved with ``species_idx=i`` so a per-species
  array picks the right one. With only ion species listed (e.g. adiabatic
  electrons) the electron entry is :func:`_adiabatic_electrons`.
  """
  species_names = kwargs.get("species", [])
  if len(species_names) != len(gdatas):
    species_names = [f"#{i}" for i in range(len(gdatas))]

  elcs, ions = [], []
  for species_idx, (name, srcs) in enumerate(zip(species_names, gdatas)):
    species_kwargs = dict(kwargs, species_idx=species_idx, species=name)
    entry = {
        "name": name,
        "srcs": srcs,
        "mass": _get_ctx_val(srcs[0], "mass", **species_kwargs),
        "charge": _get_ctx_val(srcs[0], "charge", **species_kwargs),
    }
    (elcs if entry["charge"] < 0.0 else ions).append(entry)

  if not ions:
    raise ValueError(
        f"{quantity}: found no positively charged (ion) species in "
        f"{list(species_names)}.")
  if len(elcs) > 1:
    raise ValueError(
        f"{quantity}: expected at most one negatively charged (electron) "
        f"species but found {len(elcs)} in {list(species_names)}.")
  if not elcs:
    return _adiabatic_electrons(ions, **kwargs), ions
  return elcs[0], ions


def _adiabatic_electrons(ions, **kwargs) -> dict:
  """The electron entry when only ions are listed, with sources
  ``[n_e, T_e]``:

    n_e = sum_j(n_j*Z_j)      (quasineutrality, Z_j = q_j/e),
    T_e = T_i/ti_over_te      (T_i the first ion species' temperature),

  ``ti_over_te`` defaulting to 1.
  """
  e = constants.elementary_charge
  ti_over_te = float(kwargs.get("ti_over_te") or 1.0)
  return {
      "name":
      "adiabatic electrons",
      "srcs": [
          _weighted_sum(ions, [ion["charge"] / e for ion in ions], 0),
          ions[0]["srcs"][1] * (1.0 / ti_over_te)
      ],
      "mass":
      constants.electron_mass,
      "charge":
      -e,
  }


def _weighted_sum(entries, weights, comp: int) -> "GData":
  """Sum the ``comp``-th source of each species,
  each scaled by a scalar weight."""
  terms = iter(e["srcs"][comp] * w for e, w in zip(entries, weights))
  total = next(terms)
  for term in terms:
    total = total + term
  return total


# The sound speeds combine the electrons and every ion species: ``gdatas``
# holds one ``[M0, temp]`` list per species, in the order requested, e.g.
# ``pgkyl gk_load_quantity --quantity c_s_cold_i --species elc,ion1,ion2``.
# Electrons and ions are told apart by the sign of each species' charge, so
# the species may be named anything.


def fetch_c_s_cold_i(gdatas: list[list["GData"]], **kwargs):
  """Cold-ion (ion-acoustic) sound speed (m/s), the wave perspective of the
  Bohm criterion and sheath/presheath matching::

    c_s = sqrt( T_e * sum_j(n_j*Z_j^2/m_j) / sum_j(n_j*Z_j) )

  summing over the ion species ``j``, with ``Z_j = q_j/e`` the ion charge
  state.
  """
  elc, ions = _split_elc_ions(gdatas, "fetch_c_s_cold_i", **kwargs)

  e = constants.elementary_charge
  charge_states = [ion["charge"] / e for ion in ions]

  numer = _weighted_sum(
      ions, [z**2 / ion["mass"] for z, ion in zip(charge_states, ions)], 0)
  denom = _weighted_sum(ions, charge_states, 0)

  temp_e = elc["srcs"][1]
  return (numer * (1.0 / denom) * temp_e)**0.5


def fetch_c_s_hot_i(gdatas: list[list["GData"]], **kwargs):
  """Hot-ion (thermodynamic) sound speed (m/s), the bulk-fluid perspective
  of Mach numbers and acoustic propagation in the core/SOL::

    c_s = sqrt( (gamma_e*n_e*T_e + sum_j(gamma_j*n_j*T_j)) / sum_j(n_j*m_j) )

  summing over the ion species ``j``. ``gamma_e`` and ``gamma_i`` default to
  1 and 3.
  """
  elc, ions = _split_elc_ions(gdatas, "fetch_c_s_hot_i", **kwargs)

  gamma_e = float(kwargs.get("gamma_e") or 1.0)
  gamma_i = float(kwargs.get("gamma_i") or 3.0)

  m0_e, temp_e = elc["srcs"][:2]
  numer = m0_e * temp_e * gamma_e
  for ion in ions:
    m0_i, temp_i = ion["srcs"][:2]
    numer = numer + m0_i * temp_i * gamma_i

  denom = _weighted_sum(ions, [ion["mass"] for ion in ions], 0)
  return (numer * (1.0 / denom))**0.5


def _fetch_mach(gdatas, fetch_c_s, **kwargs):
  """Parallel Mach number ``u_par/c_s`` of the first requested species, with
  ``c_s`` from ``fetch_c_s`` combining every listed species. ``gdatas`` has
  one ``[M0, temp, upar]`` list per species."""
  c_s = fetch_c_s([srcs[:2] for srcs in gdatas], **kwargs)
  return gdatas[0][2] * (1.0 / c_s)


def fetch_mach_cold_i(gdatas: list[list["GData"]], **kwargs):
  """Parallel Mach number ``u_par/c_s`` of the first requested species with
  the cold-ion sound speed (:func:`fetch_c_s_cold_i`): ``--species ion,elc``
  gives the ion Mach number, ``--species elc,ion`` the electron one, and
  ``--species ion`` uses adiabatic electrons at ``T_i/ti_over_te``."""
  return _fetch_mach(gdatas, fetch_c_s_cold_i, **kwargs)


def fetch_mach_hot_i(gdatas: list[list["GData"]], **kwargs):
  """Parallel Mach number ``u_par/c_s`` of the first requested species with
  the hot-ion sound speed (:func:`fetch_c_s_hot_i`); species are listed as
  for :func:`fetch_mach_cold_i`."""
  return _fetch_mach(gdatas, fetch_c_s_hot_i, **kwargs)


def _coulomb_log(ns, nr, ms, mr, Ts, Tr, qs, qr, bmag):
  """Coulomb logarithm of species ``s`` colliding with ``r``, as defined in
  the Gkeyll docs. Scalar inputs in SI units::

    log(Lambda_sr) = (1/2) log(1 + 1/(inner1 * inner2^2)),
    inner1 = (w_ps^2 + w_cs^2)/(v_ts^2 + 3 v_ts^2)
             + (w_pr^2 + w_cr^2)/(v_tr^2 + 3 v_ts^2),
    inner2 = max(|q_s q_r|/(4 pi eps0 m_sr u^2), hbar/(2 exp(0.5) m_sr u)),
    u^2 = 3 (v_ts^2 + v_tr^2),

  with ``w_p^2 = n q^2/(eps0 m)``, ``w_c = q B/m``, ``v_t^2 = T/m`` and
  ``m_sr = m_s m_r/(m_s + m_r)``: ``inner1`` is an inverse squared screening
  length, ``inner2`` the larger of the classical and quantum minimum impact
  parameters."""
  eps0, hbar = constants.epsilon_0, constants.hbar
  e_sqrt = np.exp(0.5)
  vts_sq, vtr_sq = Ts / ms, Tr / mr
  wps_sq = ns * qs * qs / ms / eps0
  wpr_sq = nr * qr * qr / mr / eps0
  wcs, wcr = qs * bmag / ms, qr * bmag / mr
  inner1 = ((wps_sq + wcs * wcs) / (Ts / ms + 3 * Ts / ms) +
            (wpr_sq + wcr * wcr) / (Tr / mr + 3 * Ts / ms))
  usq = 3 * (vts_sq + vtr_sq)
  msr = ms * mr / (ms + mr)
  inner2 = max(
      abs(qs * qr) / (4 * np.pi * eps0 * msr * usq),
      hbar / (2 * e_sqrt * msr * np.sqrt(usq)))
  return 0.5 * np.log(1 / inner1 / inner2 / inner2 + 1)


def fetch_collision_freq(gdatas: list[list["GData"]], **kwargs):
  """Collision frequency (1/s) of species ``s`` with species ``r``, the two
  requested species in order (``--species elc,ion``; ``ion,ion`` for
  self-collisions), as the gyrokinetic app computes it for LBO/BGK
  collisions with a normalized ``nu``::

    nu_sr = norm_nu_sr * n_r / (v_ts^2 + v_tr^2)^(3/2),
    norm_nu_sr = nu_frac * (1/m_s)*(1/m_s + 1/m_r) * q_s^2*q_r^2*log(Lambda_sr)
                 / (3*(2*pi)^(3/2)*eps0^2),

  with ``v_t^2 = T/m``. The Coulomb logarithm (:func:`_coulomb_log`) is
  symmetrized over ``(s, r)`` and evaluated, like Gkeyll's, at the reference
  values ``den_ref`` and ``temp_ref`` (one per species) and ``bmag_ref``;
  ``nu_frac`` defaults to 1. ``gdatas`` has one ``[M0, temp]`` list per
  species.

  One expression covers both of Gkeyll's branches, which start from Morse's
  ``alpha_E = 2 q_s^2 q_r^2 log(Lambda_sr)/(3 (2 pi)^(3/2) eps0^2 m_s m_r)``
  (``gkyl_calc_Morse_alpha_E_const``): self collisions use ``alpha_E``,
  cross collisions ``alpha_E (m_s + m_r)/(delta_sr (1 + beta) m_s)`` with
  ``delta_sr = 2`` and Greene's ``beta = 0``. Both equal ``norm_nu_sr``
  above; with ``m_s = m_r`` the cross factor is 1.
  """
  if len(gdatas) != 2:
    raise ValueError(
        f"fetch_collision_freq: expected two species (s,r) but got "
        f"{len(gdatas)}. Use e.g. '--species elc,ion', or '--species ion,ion' "
        "for self-collisions.")

  species_names = kwargs.get("species", [])
  if len(species_names) != len(gdatas):
    species_names = ["s", "r"]

  attrs = []
  for species_idx, (name, srcs) in enumerate(zip(species_names, gdatas)):
    species_kwargs = dict(kwargs, species_idx=species_idx, species=name)
    attrs.append({
        key: float(_get_ctx_val(srcs[1], key, **species_kwargs))
        for key in ("charge", "mass", "den_ref", "temp_ref", "bmag_ref")
    })
  attrs_s, attrs_r = attrs
  nu_frac = float(kwargs.get("nu_frac") or 1.0)

  coulomb_log = 0.5 * (
      _coulomb_log(attrs_s["den_ref"], attrs_r["den_ref"], attrs_s["mass"],
                   attrs_r["mass"], attrs_s["temp_ref"], attrs_r["temp_ref"],
                   attrs_s["charge"], attrs_r["charge"], attrs_s["bmag_ref"]) +
      _coulomb_log(attrs_r["den_ref"], attrs_s["den_ref"], attrs_r["mass"],
                   attrs_s["mass"], attrs_r["temp_ref"], attrs_s["temp_ref"],
                   attrs_r["charge"], attrs_s["charge"], attrs_s["bmag_ref"]))
  norm_nu = (nu_frac / attrs_s["mass"] *
             (1.0 / attrs_s["mass"] + 1.0 / attrs_r["mass"]) *
             (attrs_s["charge"] * attrs_r["charge"])**2 * coulomb_log /
             (3.0 * (2.0 * np.pi)**1.5 * constants.epsilon_0**2))

  (_m0_s, temp_s), (m0_r, temp_r) = gdatas
  vt_sq_sum = (temp_s * (1.0 / attrs_s["mass"]) + temp_r *
               (1.0 / attrs_r["mass"]))
  return vt_sq_sum**-1.5 * m0_r * norm_nu


def fetch_beta_from_bmag_press(gdatas: list["GData"], **kwargs):
  """``beta = 2*mu_0*press / bmag**2``."""
  bmag, press = gdatas
  return press * (1.0 / bmag**2) * (2.0 * constants.mu_0)


# --------------------------------------------------------- gradient lengths
def _make_fetch_inv_grad_length(name: str):
  """Inverse gradient length of a scalar field ``X`` along the direction
  ``x^k`` selected with ``dir``, ``1/L_X = -(dX/dx^k)/X``: ``dir=0`` is the
  radial one, ``dir=2`` the parallel one. The derivative is along the
  computational coordinate, not normalized by ``|grad x^k|``. ``gdatas``:
  ``[X]`` (e.g. ``M0`` or ``temp``)."""
  quantity = f"fetch_inv_L_{name}"

  def fetch(gdatas: list["GData"], **kwargs):
    field = gdatas[0]
    grad = _derivative(field, _direction(kwargs, quantity), quantity)
    return grad * (1.0 / field) * -1.0

  fetch.__name__ = quantity
  return fetch


fetch_inv_L_n = _make_fetch_inv_grad_length("n")
fetch_inv_L_T = _make_fetch_inv_grad_length("T")


# ------------------------------------------------------------ drift speeds
def _b_cross_grad_div_b_component(scalar: "GData", jacobtot_inv: "GData",
                                  b_i: "GData", comp: int) -> "GData":
  """The ``comp``-th component of ``b x grad(f) / (J B)``.

  ``(b x grad f)_k / B = epsilon_{ijk} * b_i * d(f)/dx^j / (J B)``, where
  ``epsilon_{ijk}`` is the Levi-Civita tensor, ``f`` a scalar field, ``b_i``
  the covariant components of a vector field. Modal derivatives are local
  to each cell, followed by weak products with the geometry fields.

  Args:
    scalar: Scalar field ``f`` to differentiate in its current representation.
    jacobtot_inv: Inverse of the total-coordinate-transformation Jacobian.
    b_i: Covariant components of the unit vector field ``b``.
    comp: 0-based component ``k`` of the cross product (``< 3``).

  Raises:
    KeyError: if ``comp`` is not 0, 1, or 2.
  """
  f = scalar
  cdim = f.num_dims

  diff_dir_pos = bi_c_pos = 0
  diff_dir_neg = bi_c_neg = 0
  calc_term = [True, True]
  if comp == 0:
    diff_dir_neg = bi_c_pos = 1
    diff_dir_pos = bi_c_neg = cdim - 1
    if cdim < 3:
      calc_term = [True, False]
  elif comp == 1:
    bi_c_pos, bi_c_neg = 2, 0
    diff_dir_neg, diff_dir_pos = cdim - 1, 0
    if cdim == 1:
      calc_term = [False, True]
  elif comp == 2:
    diff_dir_neg = bi_c_pos = 0
    diff_dir_pos = bi_c_neg = 1
    if cdim == 1:
      calc_term = [False, False]
    elif cdim == 2:
      calc_term = [False, True]
  else:
    raise KeyError("comp must be 0, 1, or 2.")

  pos_term = f * 0.0
  neg_term = f * 0.0
  if calc_term[0]:
    d_pos = operations.differentiate(f, direction=diff_dir_pos)
    pos_term = _component(b_i, bi_c_pos) * d_pos
  if calc_term[1]:
    d_neg = operations.differentiate(f, direction=diff_dir_neg)
    neg_term = _component(b_i, bi_c_neg) * (-d_neg)

  return (pos_term + neg_term) * jacobtot_inv


def fetch_ExB_vel(gdatas: list["GData"], **kwargs):
  """``v_{E,k} = epsilon_{ijk}/(J B) * b_i * d(phi)/dx^j`` (``dir`` selects k).

  ``gdatas``: ``(jacobtot_inv, bmag, b_i, phi)``.
  """
  jacobtot_inv, _bmag, b_i, phi = gdatas
  comp = _direction(kwargs, "fetch_ExB_vel")
  return _b_cross_grad_div_b_component(phi, jacobtot_inv, b_i, comp)


def fetch_gradB_vel(gdatas: list["GData"], **kwargs):
  """``v_gradB,k = Tperp/(q B) * epsilon_{ijk} * b_i * d(B)/dx^j / (J B)``.

  ``gdatas``: ``(jacobtot_inv, bmag, b_i, Tperp)``.
  """
  jacobtot_inv, bmag, b_i, Tperp = gdatas
  comp = _direction(kwargs, "fetch_gradB_vel")
  out = _b_cross_grad_div_b_component(bmag, jacobtot_inv, b_i, comp)
  charge = _get_ctx_val(Tperp, "charge", **kwargs)
  return Tperp * out * (1.0 / bmag) / charge


def fetch_diamag_vel(gdatas: list["GData"], **kwargs):
  """``v_diamag,k = 1/(q n) epsilon_{ijk} b_i * d(pperp)/dx^j / (J B)``.

  ``gdatas``: ``(jacobtot_inv, bmag, b_i, m0, pressperp)``.
  """
  jacobtot_inv, bmag, b_i, m0, pressperp = gdatas
  comp = _direction(kwargs, "fetch_diamag_vel")
  out = _b_cross_grad_div_b_component(pressperp, jacobtot_inv, b_i, comp)
  charge = _get_ctx_val(pressperp, "charge", **kwargs)
  return out * (1.0 / m0) / charge


# --------------------------------------------- magnetic field perturbations
def fetch_dB_perp_contra(gdatas: list["GData"], **kwargs):
  """A contravariant component of the magnetic perturbation
  ``dB = curl(A_par b)``::

    dB^i = ( d(A_par b_k)/dx^j - d(A_par b_j)/dx^k ) / J,

  ``(j, k) = (i+1, i+2)`` cyclically; a direction a reduced simulation does
  not carry contributes no derivative. ``dir`` selects ``i``.

  ``gdatas``: ``(apar, jacobgeo_inv, b_i)``.
  """
  comp = _direction(kwargs, "fetch_dB_perp_contra")
  apar, jacobgeo_inv, b_i = gdatas
  axis = _config_axis(apar.num_dims)
  out = apar * 0.0
  for diff_dir, b_comp, sign in (((comp + 1) % 3, (comp + 2) % 3, 1.0),
                                 ((comp + 2) % 3, (comp + 1) % 3, -1.0)):
    if axis[diff_dir] is None:
      continue
    apar_b = apar * _component(b_i, b_comp)
    out = out + operations.differentiate(apar_b,
                                         direction=axis[diff_dir]) * sign
  return out * jacobgeo_inv


def fetch_dB_perp_cov(gdatas: list["GData"], **kwargs):
  """A covariant component of the magnetic perturbation,
  ``dB_i = g_ij dB^j``. ``dir`` selects ``i``.

  ``gdatas``: ``(apar, jacobgeo_inv, b_i, g_ij)``.
  """
  comp = _direction(kwargs, "fetch_dB_perp_cov")
  return _metric_contract(gdatas[3], comp,
                          lambda j: fetch_dB_perp_contra(gdatas[:3], dir=j))


def fetch_dB_perp_mag(gdatas: list["GData"], **kwargs):
  """Magnitude of the magnetic perturbation, ``|dB| = sqrt(dB_i dB^i)``.

  ``gdatas``: ``(apar, jacobgeo_inv, b_i, g_ij)``.
  """
  return _vector_magnitude(lambda i: fetch_dB_perp_cov(gdatas, dir=i),
                           lambda i: fetch_dB_perp_contra(gdatas[:3], dir=i))


def fetch_B_equilibrium_cov(gdatas: list["GData"], **kwargs):
  """A covariant component of the equilibrium field, ``B_i = B b_i``: what
  ``B_tot_cov`` falls back to without ``apar`` output. ``dir`` selects ``i``.

  ``gdatas``: ``(bmag, b_i)``.
  """
  comp = _direction(kwargs, "fetch_B_equilibrium_cov")
  bmag, b_i = gdatas
  return _component(b_i, comp) * bmag


def fetch_B_tot_cov(gdatas: list["GData"], **kwargs):
  """A covariant component of the total field, ``B_i = B b_i + dB_i``.

  ``gdatas``: ``(apar, bmag, jacobgeo_inv, b_i, g_ij)``.
  """
  apar, bmag, jacobgeo_inv, b_i, g_ij = gdatas
  return (fetch_B_equilibrium_cov([bmag, b_i], **kwargs) +
          fetch_dB_perp_cov([apar, jacobgeo_inv, b_i, g_ij], **kwargs))


def fetch_B_equilibrium_contra(gdatas: list["GData"], **kwargs):
  """A contravariant component of the equilibrium field. ``b`` lies along
  ``e_3``, so ``b^i = delta^i_3/sqrt(g_33)`` and
  ``B^i = (B/sqrt(g_33)) delta^i_3``: the first two components vanish. This
  is what ``B_tot_contra`` falls back to without ``apar`` output.

  ``gdatas``: ``(bmag, g_ij)``.
  """
  comp = _direction(kwargs, "fetch_B_equilibrium_contra")
  bmag, g_ij = gdatas
  if comp != 2:
    return bmag * 0.0
  return bmag * _metric_diagonal(g_ij, 2)**-0.5


def fetch_B_tot_contra(gdatas: list["GData"], **kwargs):
  """A contravariant component of the total field, ``B^i = B b^i + dB^i``.

  ``gdatas``: ``(apar, bmag, jacobgeo_inv, b_i, g_ij)``.
  """
  apar, bmag, jacobgeo_inv, b_i, g_ij = gdatas
  return (fetch_B_equilibrium_contra([bmag, g_ij], **kwargs) +
          fetch_dB_perp_contra([apar, jacobgeo_inv, b_i], **kwargs))


def fetch_B_tot_mag(gdatas: list["GData"], **kwargs):
  """Magnitude of the total field, ``|B| = sqrt(B_i B^i)``.

  ``gdatas``: ``(apar, bmag, jacobgeo_inv, b_i, g_ij)``.
  """
  return _vector_magnitude(lambda i: fetch_B_tot_cov(gdatas, dir=i),
                           lambda i: fetch_B_tot_contra(gdatas, dir=i))


# ----------------------------------------------------------- electric field
def fetch_E_field_cov(gdatas: list["GData"], **kwargs):
  """A covariant component of the electrostatic field, ``E_i = -dphi/dx^i``
  (V per unit of ``x^i``); zero along a direction a reduced simulation does
  not carry. ``dir`` selects ``i``.

  ``gdatas``: ``(phi,)``.
  """
  comp = _direction(kwargs, "fetch_E_field_cov")
  phi = gdatas[0]
  dim = _config_axis(phi.num_dims)[comp]
  if dim is None:
    return phi * 0.0
  return operations.differentiate(phi, direction=dim) * -1.0


def fetch_E_field_contra(gdatas: list["GData"], **kwargs):
  """A contravariant component of the electrostatic field,
  ``E^i = g^ij E_j``. ``dir`` selects ``i``.

  ``gdatas``: ``(phi, gij)``.
  """
  comp = _direction(kwargs, "fetch_E_field_contra")
  phi, gij = gdatas
  return _metric_contract(gij, comp, lambda j: fetch_E_field_cov([phi], dir=j))


def fetch_E_field_mag(gdatas: list["GData"], **kwargs):
  """Magnitude of the electrostatic field (V/m), ``|E| = sqrt(E_i E^i)``.

  ``gdatas``: ``(phi, gij)``.
  """
  return _vector_magnitude(lambda i: fetch_E_field_cov(gdatas[:1], dir=i),
                           lambda i: fetch_E_field_contra(gdatas, dir=i))


# ------------------------------------------------------- cross-field fluxes
# Contravariant components (. grad x^k) of the ExB and magnetic flutter fluxes
# of 3x data, along the cross-field direction k selected with ``dir``: 0 (x,
# radial) or 1 (y, binormal). Along z the flux is dominated by the parallel
# streaming M1 b^z, which these drift fluxes omit, so z is refused.
# ``fluct='y'`` or ``'yz'`` keeps the turbulent part: the correlation of the
# fluctuations about the Jacobian-weighted y or (y, z) average.
_FLUCT_DIMS = {"y": [1], "yz": [1, 2]}


def _flux_direction(kwargs, quantity: str) -> int:
  """The cross-field direction ``k`` of a flux or diffusivity: ``dir`` 0
  (x) or 1 (y)."""
  comp = _direction(kwargs, quantity)
  if comp == 2:
    raise ValueError(
        f"{quantity}: the z component of a flux is dominated by the parallel "
        "streaming M1 b^z, which the ExB and flutter fluxes omit; use dir=0 "
        "(x) or dir=1 (y).")
  return comp


def _maybe_fluct(field: "GData", jacobgeo: "GData", quantity: str,
                 **kwargs) -> "GData":
  """``field``, or its fluctuation about the average selected by
  ``fluct``."""
  key = str(kwargs.get("fluct") or "none").lower()
  if key == "none":
    return field
  if key not in _FLUCT_DIMS:
    raise ValueError(f"{quantity}: unknown fluct={key!r}; use one of: none, "
                     f"{', '.join(_FLUCT_DIMS)}.")
  return operations.fluctuation(field, _FLUCT_DIMS[key], weight=jacobgeo)


def _cross_field_flux(moment: "GData", velocity, jacobgeo: "GData",
                      factor: float, quantity: str, **kwargs) -> "GData":
  """``factor * moment * v^k``, either factor optionally replaced by its
  fluctuation. ``velocity(k)`` builds ``v^k`` once the data are known to be
  3x and ``k`` a cross-field direction: turbulent fluxes need the binormal
  direction."""
  if moment.num_dims != 3:
    raise ValueError(f"{quantity}: cross-field fluxes need 3x (x,y,z) data, "
                     f"got {moment.num_dims} dimensions.")
  vel = _maybe_fluct(velocity(_flux_direction(kwargs, quantity)), jacobgeo,
                     quantity, **kwargs)
  return _maybe_fluct(moment, jacobgeo, quantity, **kwargs) * vel * factor


def _dB_over_B(apar: "GData", bmag: "GData", jacobgeo_inv: "GData",
               b_i: "GData", comp: int) -> "GData":
  """Contravariant magnetic flutter ``dB^k/B``."""
  return fetch_dB_perp_contra([apar, jacobgeo_inv, b_i],
                              dir=comp) * (1.0 / bmag)


def fetch_flux_particle_ExB(gdatas: list["GData"], **kwargs):
  """ExB particle flux along ``x^k`` (``dir``), ``Gamma_E^k = n v_E^k``.

  ``gdatas``: ``(M0, phi, jacobgeo, jacobtot_inv, b_i)``.
  """
  m0, phi, jacobgeo, jacobtot_inv, b_i = gdatas
  return _cross_field_flux(
      m0,
      lambda comp: _b_cross_grad_div_b_component(phi, jacobtot_inv, b_i, comp),
      jacobgeo, 1.0, "fetch_flux_particle_ExB", **kwargs)


def fetch_flux_energy_ExB(gdatas: list["GData"], **kwargs):
  """ExB energy flux along ``x^k`` (``dir``), ``Q_E^k = (m/2) M2 v_E^k``
  with ``M2 = M2par + M2perp`` (exact in long-wavelength gyrokinetics, where
  ``v_E`` does not depend on velocity).

  ``gdatas``: ``(M2, phi, jacobgeo, jacobtot_inv, b_i)``.
  """
  m2, phi, jacobgeo, jacobtot_inv, b_i = gdatas
  mass = _get_ctx_val(m2, "mass", **kwargs)
  return _cross_field_flux(
      m2,
      lambda comp: _b_cross_grad_div_b_component(phi, jacobtot_inv, b_i, comp),
      jacobgeo, 0.5 * mass, "fetch_flux_energy_ExB", **kwargs)


def fetch_flux_particle_dB(gdatas: list["GData"], **kwargs):
  """Magnetic flutter particle flux along ``x^k`` (``dir``),
  ``Gamma_dB^k = M1 dB^k/B``.

  ``gdatas``: ``(M1, apar, bmag, jacobgeo, jacobgeo_inv, b_i)``.
  """
  m1, apar, bmag, jacobgeo, jacobgeo_inv, b_i = gdatas
  return _cross_field_flux(
      m1, lambda comp: _dB_over_B(apar, bmag, jacobgeo_inv, b_i, comp),
      jacobgeo, 1.0, "fetch_flux_particle_dB", **kwargs)


def fetch_flux_energy_dB(gdatas: list["GData"], **kwargs):
  """Magnetic flutter energy flux along ``x^k`` (``dir``),
  ``Q_dB^k = (m/2) M3 dB^k/B`` with ``M3 = M3par + M3perp``.

  ``gdatas``: ``(M3, apar, bmag, jacobgeo, jacobgeo_inv, b_i)``.
  """
  m3, apar, bmag, jacobgeo, jacobgeo_inv, b_i = gdatas
  mass = _get_ctx_val(m3, "mass", **kwargs)
  return _cross_field_flux(
      m3, lambda comp: _dB_over_B(apar, bmag, jacobgeo_inv, b_i, comp),
      jacobgeo, 0.5 * mass, "fetch_flux_energy_dB", **kwargs)


def _warn_if_apar_dropped(quantity: str, **kwargs) -> None:
  """Warn when the electrostatic fallback is used although ``apar`` exists."""
  path, sim, frame = kwargs.get("path"), kwargs.get("name"), kwargs.get("frame")
  if path is None or sim is None or frame is None:
    return
  if os.path.isfile(os.path.join(path, f"{sim}-apar_{frame}.gkyl")):
    warnings.warn(
        f"{quantity}: apar output found but the moments needed for the "
        "magnetic flutter flux are missing; only the ExB contribution is "
        "included.",
        stacklevel=2)


def fetch_flux_particle_em(gdatas: list["GData"], **kwargs):
  """Total cross-field particle flux, ``Gamma^k = Gamma_E^k + Gamma_dB^k``.

  ``gdatas``: ``(M0, M1, apar, phi, bmag, jacobgeo, jacobgeo_inv,
  jacobtot_inv, b_i)``.
  """
  m0, m1, apar, phi, bmag, jacobgeo, jacobgeo_inv, jacobtot_inv, b_i = gdatas
  return (fetch_flux_particle_ExB([m0, phi, jacobgeo, jacobtot_inv, b_i], **
                                  kwargs) +
          fetch_flux_particle_dB([m1, apar, bmag, jacobgeo, jacobgeo_inv, b_i],
                                 **kwargs))


def fetch_flux_particle_es(gdatas: list["GData"], **kwargs):
  """Total cross-field particle flux of an electrostatic run,
  ``Gamma^k = Gamma_E^k``.

  ``gdatas``: ``(M0, phi, jacobgeo, jacobtot_inv, b_i)``.
  """
  _warn_if_apar_dropped("flux_particle", **kwargs)
  return fetch_flux_particle_ExB(gdatas, **kwargs)


def fetch_flux_energy_em(gdatas: list["GData"], **kwargs):
  """Total cross-field energy flux, ``Q^k = Q_E^k + Q_dB^k``.

  ``gdatas``: ``(M2, M3, apar, phi, bmag, jacobgeo, jacobgeo_inv,
  jacobtot_inv, b_i)``.
  """
  m2, m3, apar, phi, bmag, jacobgeo, jacobgeo_inv, jacobtot_inv, b_i = gdatas
  return (
      fetch_flux_energy_ExB([m2, phi, jacobgeo, jacobtot_inv, b_i], **kwargs) +
      fetch_flux_energy_dB([m3, apar, bmag, jacobgeo, jacobgeo_inv, b_i], **
                           kwargs))


def fetch_flux_energy_es(gdatas: list["GData"], **kwargs):
  """Total cross-field energy flux of an electrostatic run,
  ``Q^k = Q_E^k``.

  ``gdatas``: ``(M2, phi, jacobgeo, jacobtot_inv, b_i)``.
  """
  _warn_if_apar_dropped("flux_energy", **kwargs)
  return fetch_flux_energy_ExB(gdatas, **kwargs)


# --------------------------------------------------- transport coefficients
# Local (pointwise) diffusivities along the cross-field direction x^k selected
# with ``dir``: the local flux over the local gradient, normalized by
# g^kk = |grad x^k|^2 so they are in m^2/s for any coordinate x^k::
#
#   D^k = -Gamma^k/(g^kk dn/dx^k),  chi^k = -q^k/(n g^kk dT/dx^k),
#   q^k = Q^k - conv*T*Gamma^k,
#
# conv defaulting to 3/2.


def _heat_flux(temp: "GData", flux_particle: "GData", flux_energy: "GData",
               **kwargs) -> "GData":
  """Heat flux ``q = Q - conv*T*Gamma`` (``conv`` defaults to 3/2)."""
  conv = kwargs.get("conv")
  conv = 1.5 if conv is None else float(conv)
  return flux_energy + temp * flux_particle * -conv


def fetch_particle_D(gdatas: list["GData"], **kwargs):
  """Local particle diffusivity along ``x^k`` (``dir``),
  ``D^k = -Gamma^k/(g^kk dn/dx^k)`` (m^2/s).

  ``gdatas``: ``(M0, flux_particle, gij)``.
  """
  m0, flux_particle, gij = gdatas
  comp = _flux_direction(kwargs, "fetch_particle_D")
  denom = _metric_diagonal(gij, comp) * _derivative(m0, comp,
                                                    "fetch_particle_D")
  return flux_particle * (1.0 / denom) * -1.0


def fetch_heat_chi(gdatas: list["GData"], **kwargs):
  """Local heat diffusivity along ``x^k`` (``dir``),
  ``chi^k = -q^k/(n g^kk dT/dx^k)`` (m^2/s).

  ``gdatas``: ``(M0, temp, flux_particle, flux_energy, gij)``.
  """
  m0, temp, flux_particle, flux_energy, gij = gdatas
  comp = _flux_direction(kwargs, "fetch_heat_chi")
  heat_flux = _heat_flux(temp, flux_particle, flux_energy, **kwargs)
  denom = m0 * _metric_diagonal(gij, comp) * _derivative(
      temp, comp, "fetch_heat_chi")
  return heat_flux * (1.0 / denom) * -1.0


def _gyro_bohm_factors(gdatas, **kwargs):
  """``rho_s^2 c_s = T_e^(3/2) m_i^(1/2)/(q_i^2 B^2)``, with
  ``c_s = sqrt(T_e/m_i)`` and ``rho_s = c_s/Omega_i``, as a list of fields
  and a scalar whose product it is.

  ``gdatas`` has one source list per species, starting with ``[M0, temp]``
  and ending with ``bmag``. ``m_i``/``q_i`` are those of the first ion
  species; ``T_e`` is the electron temperature, or ``T_i/ti_over_te``
  without an electron species. ``te_ref`` and ``bmag_ref`` replace ``T_e``
  and ``B`` by constants.
  """
  elc, ions = _split_elc_ions([srcs[:2] for srcs in gdatas],
                              "gyro-Bohm diffusivity", **kwargs)
  scalar = np.sqrt(ions[0]["mass"]) / ions[0]["charge"]**2
  fields = []
  if kwargs.get("te_ref") is not None:
    scalar *= float(kwargs["te_ref"])**1.5
  else:
    fields.append(elc["srcs"][1]**1.5)
  if kwargs.get("bmag_ref") is not None:
    scalar /= float(kwargs["bmag_ref"])**2
  else:
    bmag = gdatas[0][-1]
    fields.append(1.0 / (bmag * bmag))
  return fields, scalar


def _gyro_bohm_normalized(num: "GData", grad: "GData", g_kk: "GData",
                          other_den: "GData | None", gdatas,
                          **kwargs) -> "GData":
  """``num / (other_den g^kk^(3/2) grad^2 rho_s^2 c_s)``: ``-X L_X/(rho_s^2
  c_s)`` for a diffusivity ``X = -num/(other_den g^kk grad)`` and gradient
  length ``L_X = -field/(sqrt(g^kk) grad)``, ``num`` carrying the field."""
  fields, scalar = _gyro_bohm_factors(gdatas, **kwargs)
  den = g_kk**1.5 * (grad * grad)
  if other_den is not None:
    den = den * other_den
  for field in fields:
    den = den * field
  return num * (1.0 / den) * (1.0 / scalar)


def fetch_particle_D_gB(gdatas: list[list["GData"]], **kwargs):
  """Local particle diffusivity along ``x^k`` (``dir``) over the gyro-Bohm
  one, of the first requested species::

    D^k/D_gB = D^k L_n/(rho_s^2 c_s)
             = Gamma^k n/(g^kk^(3/2) (dn/dx^k)^2 rho_s^2 c_s),

  ``L_n = -n/(sqrt(g^kk) dn/dx^k)``. ``--species ion,elc`` uses kinetic
  electrons; ``--species ion`` adiabatic ones (see
  :func:`_gyro_bohm_factors`). ``gdatas`` has one
  ``[M0, temp, flux_particle, gij, bmag]`` list per species.
  """
  m0, _temp, flux_particle, gij, _bmag = gdatas[0]
  comp = _flux_direction(kwargs, "fetch_particle_D_gB")
  return _gyro_bohm_normalized(flux_particle * m0,
                               _derivative(m0, comp, "fetch_particle_D_gB"),
                               _metric_diagonal(gij, comp), None, gdatas,
                               **kwargs)


def fetch_heat_chi_gB(gdatas: list[list["GData"]], **kwargs):
  """Local heat diffusivity along ``x^k`` (``dir``) over the gyro-Bohm one,
  of the first requested species::

    chi^k/chi_gB = chi^k L_T/(rho_s^2 c_s)
                 = q^k T/(n g^kk^(3/2) (dT/dx^k)^2 rho_s^2 c_s),

  ``L_T = -T/(sqrt(g^kk) dT/dx^k)``, species listed as for
  :func:`fetch_particle_D_gB`. ``gdatas`` has one
  ``[M0, temp, flux_particle, flux_energy, gij, bmag]`` list per species.
  """
  m0, temp, flux_particle, flux_energy, gij, _bmag = gdatas[0]
  comp = _flux_direction(kwargs, "fetch_heat_chi_gB")
  heat_flux = _heat_flux(temp, flux_particle, flux_energy, **kwargs)
  return _gyro_bohm_normalized(heat_flux * temp,
                               _derivative(temp, comp, "fetch_heat_chi_gB"),
                               _metric_diagonal(gij, comp), m0, gdatas,
                               **kwargs)


# --------------------------------------------------------- phase space (f)
def load_distf(gdatas, **kwargs):
  """Loader for the registry ``distf`` quantity: wraps
  :func:`~postgkyl.diagnostics.gk.distf.load_distf` with
  defaults tailored to registry use (never interpolate further, convert
  velocity coordinates by default). Extra keyword overrides (via
  ``**extra`` on :func:`~postgkyl.diagnostics.gk.load_quantity.
  load_quantity`): ``suffix``, ``c2p_vel``, ``mc2nu``, ``mapc2p``,
  ``block``.
  """
  from .distf import load_distf
  from .utils import dict_get_bool

  prefix = kwargs.get("path", "").rstrip("/") + "/" + kwargs.get("name", "")
  extra = {
      k: v
      for k, v in kwargs.items()
      if k not in ("path", "name", "species", "frame")
  }

  return load_distf(
      name=prefix,
      species=kwargs.get("species", ""),
      frame=int(kwargs.get("frame", 0)),
      suffix=str(extra.get("suffix", "")),
      use_c2p_vel=dict_get_bool(extra, "c2p_vel", True),
      use_mc2nu=dict_get_bool(extra, "mc2nu", False),
      use_mapc2p=dict_get_bool(extra, "mapc2p", False),
      block_idx=extra.get("block", None),
      num_interp=0,
  )


# ----------------------------------------------------- normalized quantities
def _make_fetch_q_norm(name: str):
  """Return a fetch function for a heat flux normalized by the
  free-streaming estimate ``n*T*c_s``: ``q_norm = q / (n*T*c_s)``.
  ``gdatas`` (in this order): ``[q, M0, temp, c_s]``.
  """

  def fetch(gdatas: list["GData"], **kwargs):
    q, m0, temp, c_s = gdatas
    return q * (1.0 / (m0 * temp * c_s))

  fetch.__name__ = f"fetch_q{name}_norm"
  return fetch


fetch_qpar_norm = _make_fetch_q_norm("par")
fetch_qperp_norm = _make_fetch_q_norm("perp")


def fetch_rho_over_lambda(gdatas: list["GData"], **kwargs):
  """Ratio of the species Larmor radius to its Debye length:
  ``rho/lambda_D``. ``gdatas``: ``[rho, lambda_D]``."""
  rho, lambda_d = gdatas
  return rho * (1.0 / lambda_d)


def fetch_phi_norm(gdatas: list["GData"], **kwargs):
  """Normalized electrostatic potential: ``phi_norm = e*phi/T_e``.
  ``gdatas``: ``[phi, temp]``."""
  phi, temp = gdatas
  return phi * (1.0 / temp) * constants.elementary_charge
