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
from typing import TYPE_CHECKING

from scipy import constants

from postgkyl import operations

if TYPE_CHECKING:
  from postgkyl.gdata import GData


def _get_ctx_val(gdata: "GData", key: str, **kwargs):
  """A value (or one value per species) for ``key``: ``kwargs[key]``
  (an explicit ``--extra`` override) wins over ``gdata.ctx[key]`` (the
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
          f"fetch function: '--extra {key}=' was given {len(val)} values "
          "but this quantity is not resolved per species here, so there is "
          "no way to tell which one to use. Pass a single value instead.")
    if species_idx >= len(val):
      species = kwargs.get("species")
      raise ValueError(
          f"fetch function: '--extra {key}=' was given only {len(val)} "
          f"values but species #{species_idx}"
          f"{f' ({species})' if species else ''} was requested. Give one "
          "value per species, in the order of '--species'.")
    return val[species_idx]
  if gdata.ctx.get(key) is not None:
    return gdata.ctx[key]
  raise KeyError(
      f"fetch function: context key '{key}' not found in the dataset; "
      f"pass it as '--extra {key}=<value>', or as one value per species "
      f"with '--extra {key}=<value1>,<value2>,...'.")


def _component(d: "GData", comp: int | None) -> "GData":
  """Select a physical field while retaining its complete DG expansion."""
  return d.clone() if comp is None else operations.select(d, comp=comp)


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

  ``gdatas[i]`` is species ``i``'s resolved source list (as
  :meth:`~postgkyl.diagnostics.gk.quantity.GkQuantity.fetch_multi`
  hands it to an ``is_multi_species`` fetch function); each entry's
  ``mass``/``charge`` is resolved with ``species_idx=i`` so a per-species
  ``--extra`` array picks the right one.
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

  if len(elcs) != 1:
    raise ValueError(
        f"{quantity}: expected exactly one negatively charged (electron) "
        f"species but found {len(elcs)} in {list(species_names)}.")
  if not ions:
    raise ValueError(
        f"{quantity}: found no positively charged (ion) species in "
        f"{list(species_names)}.")
  return elcs[0], ions


def _weighted_sum(entries, weights, comp: int) -> "GData":
  """Sum the ``comp``-th source of each species,
  each scaled by a scalar weight."""
  terms = iter(e["srcs"][comp] * w for e, w in zip(entries, weights))
  total = next(terms)
  for term in terms:
    total = total + term
  return total


def _fetch_c_s_ion_acoustic(gdatas, **kwargs):
  """Ion-acoustic sound speed (wave perspective), for the Bohm criterion
  and sheath/presheath matching::

    c_s = sqrt( T_e * sum_j(n_j*Z_j^2/m_j) / sum_j(n_j*Z_j) )

  summing over the ion species ``j``, with ``Z_j = q_j/e`` the ion charge
  state.
  """
  elc, ions = _split_elc_ions(gdatas, "fetch_c_s(kind=ion_acoustic)", **kwargs)

  e = constants.elementary_charge
  charge_states = [ion["charge"] / e for ion in ions]

  numer = _weighted_sum(
      ions, [z**2 / ion["mass"] for z, ion in zip(charge_states, ions)], 0)
  denom = _weighted_sum(ions, charge_states, 0)

  temp_e = elc["srcs"][1]
  return (numer * (1.0 / denom) * temp_e)**0.5


def _fetch_c_s_thermo(gdatas, **kwargs):
  """Thermodynamic sound speed (bulk fluid perspective), for Mach numbers
  and acoustic propagation in the core/SOL::

    c_s = sqrt( (gamma_e*n_e*T_e + sum_j(gamma_j*n_j*T_j)) / sum_j(n_j*m_j) )

  summing over the ion species ``j``. Default ``gamma_e=1``, ``gamma_i=3``,
  overridable via ``--extra``.
  """
  elc, ions = _split_elc_ions(gdatas, "fetch_c_s(kind=thermo)", **kwargs)

  gamma_e = float(kwargs.get("gamma_e", 1.0))
  gamma_i = float(kwargs.get("gamma_i", 3.0))

  m0_e, temp_e = elc["srcs"]
  numer = m0_e * temp_e * gamma_e
  for ion in ions:
    m0_i, temp_i = ion["srcs"]
    numer = numer + m0_i * temp_i * gamma_i

  denom = _weighted_sum(ions, [ion["mass"] for ion in ions], 0)
  return (numer * (1.0 / denom))**0.5


def fetch_c_s(gdatas: list[list["GData"]], **kwargs):
  """Sound speed (m/s), combining the electrons and every ion species.
  ``gdatas`` has one ``[M0, temp]`` source list per species, in the order
  requested, e.g. ``pgkyl gk_load_quantity --quantity c_s
  --species elc,ion1,ion2 ...``.
  Electrons and ions are told apart by the sign of each species' charge
  attribute, so the species may be named anything.

  Two definitions are available through ``--extra kind=<kind>``:
    ``ion_acoustic``: the wave/Bohm-criterion sound speed,
      ``c_s = sqrt(T_e*sum_j(n_j*Z_j^2/m_j)/sum_j(n_j*Z_j))``.
    ``thermo`` (default): the bulk-fluid sound speed,
      ``c_s = sqrt((gamma_e*n_e*T_e + sum_j(gamma_j*n_j*T_j))/sum_j(n_j*m_j))``,
      with ``gamma_e``/``gamma_i`` settable via ``--extra`` (default 1, 3).
  """
  c_s_kinds = {
      "ion_acoustic": _fetch_c_s_ion_acoustic,
      "thermo": _fetch_c_s_thermo,
  }
  kind = str(kwargs.get("kind", "thermo"))
  if kind not in c_s_kinds:
    raise ValueError(
        f"fetch_c_s: unknown kind '{kind}'. Select one with '--extra "
        f"kind=<kind>' from: {', '.join(sorted(c_s_kinds))}.")
  return c_s_kinds[kind](gdatas, **kwargs)


def fetch_beta_from_bmag_press(gdatas: list["GData"], **kwargs):
  """``beta = 2*mu_0*press / bmag**2``."""
  bmag, press = gdatas
  return press * (1.0 / bmag**2) * (2.0 * constants.mu_0)


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
  if "dir" not in kwargs:
    raise KeyError("fetch_ExB_vel: select the k-th component with dir=<int>.")
  jacobtot_inv, _bmag, b_i, phi = gdatas
  return _b_cross_grad_div_b_component(phi, jacobtot_inv, b_i, kwargs["dir"])


def fetch_gradB_vel(gdatas: list["GData"], **kwargs):
  """``v_gradB,k = Tperp/(q B) * epsilon_{ijk} * b_i * d(B)/dx^j / (J B)``.

  ``gdatas``: ``(jacobtot_inv, bmag, b_i, Tperp)``.
  """
  if "dir" not in kwargs:
    raise KeyError("fetch_gradB_vel: select the k-th component with dir=<int>.")
  jacobtot_inv, bmag, b_i, Tperp = gdatas
  out = _b_cross_grad_div_b_component(bmag, jacobtot_inv, b_i, kwargs["dir"])
  charge = _get_ctx_val(Tperp, "charge", **kwargs)
  return Tperp * out * (1.0 / bmag) / charge


def fetch_diamag_vel(gdatas: list["GData"], **kwargs):
  """``v_diamag,k = 1/(q n) epsilon_{ijk} b_i * d(pperp)/dx^j / (J B)``.

  ``gdatas``: ``(jacobtot_inv, bmag, b_i, m0, pressperp)``.
  """
  if "dir" not in kwargs:
    raise KeyError(
        "fetch_diamag_vel: select the k-th component with dir=<int>.")
  jacobtot_inv, bmag, b_i, m0, pressperp = gdatas
  out = _b_cross_grad_div_b_component(pressperp, jacobtot_inv, b_i,
                                      kwargs["dir"])
  charge = _get_ctx_val(pressperp, "charge", **kwargs)
  return out * (1.0 / m0) / charge


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
