"""Gyrokinetic derived-quantity physics -- the ``fetch_*`` functions behind the
quantity registry.

Ported from ``src_bak/postgkyl/gk/gk_quantities/fetch_funcs.py``. Every
``fetch_*`` there computed through ``GkeyllDGops`` -- a ``ctypes`` binding
that is dead in this tree (rule #2). Rewired here onto the new surface:
every fetch function **interpolates its inputs first**
(:meth:`~postgkyl.gdata.gdata.GData.interpolate`, the sanctioned "evaluation"
bridge -- REFACTOR_GKEYLL_FFI.md's field domain) and then computes with
plain NumPy on the interpolated values, exactly like every sibling equation
module (``five_moment``, ``ten_moment``, ``mhd``, ...). This is a deliberate
divergence from a literal "stay modal and call the weak kernels" port:
extracting one physical field's coefficients out of a *packed* multi-field
source file (``M0M1M2``, ``BiMaxwellianMoments``, ``HamiltonianMoments``, ...)
has no primitive reachable from this layer's allowed imports (``gdatastate``,
``operations``, ``numerics``, ``api`` -- not ``dg``/``gpython``; only ``operations.select``
could slice a component, and it unconditionally refuses gkyl-backed data).
Interpolating first sidesteps that gap entirely and matches the one
established working pattern in this codebase; see the layer-12 report for
the full trade-off discussion. Physical constants come from
``scipy.constants`` (rule #13), not a re-typed ``gk/gkeyll_const.py`` table.

Naming keys (matching ``src_bak`` so the registry mapping in ``registry.py``
stays recognizable):
  s#: source #, c#: component #, add/sub/mul/div: the combining operator,
  pos/neg: the plus/minus term of a curvilinear cross product.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np
from scipy import constants

from postgkyl import operations

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def _get_ctx_val(gdata: "GDataState", key: str, **kwargs):
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


def _ensure_interpolated(d: "GDataState") -> "GDataState":
  """Interpolate ``d`` onto the field domain unless it already is.

  Uses the ``operations.interpolate`` verb directly (rather than the fluent
  ``GData.interpolate()``) so this works on any ``GDataState``, not just the
  fluent subclass -- these functions receive whatever
  ``GkQuantity.get_src_gdata`` hands them.
  """
  if d.ctx.get("interpolated"):
    return d
  return operations.interpolate(d)


def _component(d: "GDataState", comp: int | None) -> "GDataState":
  """Interpolate ``d`` and select physical component ``comp`` (all if None)."""
  interpolated = _ensure_interpolated(d)
  return interpolated if comp is None else operations.select(interpolated,
                                                             comp=comp)


# --------------------------------------------------- generic fetch factories
def _make_fetch_comp(icomp: int | None):
  """A fetch function that extracts the ``icomp``-th physical component."""

  def fetch(gdatas, **kwargs):
    return _component(gdatas[0], icomp)

  fetch.__name__ = f"fetch_comp{icomp}" if icomp is not None else "fetch_compAll"
  return fetch


def _make_fetch_binop(si: int, ci: int, sj: int, cj: int, op):
  """A fetch function combining component ``ci`` of source ``si`` with
  component ``cj`` of source ``sj`` via ``op`` (both interpolated first)."""

  def fetch(gdatas, **kwargs):
    a = _component(gdatas[si], ci)
    b = _component(gdatas[sj], cj)
    return a._result(a.grid, op(a.values, b.values))

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
def fetch_M1_from_H(gdatas, **kwargs):
  """M1 from the Hamiltonian moments: ``mass**-1 * (comp0 * comp1)``."""
  hmom = _ensure_interpolated(gdatas[0])
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  values = hmom.values[..., 0, np.newaxis] * hmom.values[..., 1, np.newaxis]
  return hmom._result(hmom.grid, values / mass)


def fetch_Tpar_from_BiMax(gdatas, **kwargs):
  """Tpar from BiMaxwellian moments: ``mass * comp2``."""
  Tpar = fetch_s0c2(gdatas)
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return Tpar._result(Tpar.grid, mass * Tpar.values)


def fetch_Tpar_from_M0_M1_M2par(gdatas, **kwargs):
  """``upar*M1 + M0*Tpar/m = M2par`` => ``Tpar = m*(M2par - upar*M1)/M0``."""
  m0, m1, m2par = (_ensure_interpolated(g) for g in gdatas)
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  upar = m1.values / m0.values
  values = mass * (m2par.values - upar * m1.values) / m0.values
  return m0._result(m0.grid, values)


def fetch_Tperp_from_BiMax(gdatas, **kwargs):
  """Tperp from BiMaxwellian moments: ``mass * comp3``."""
  Tperp = fetch_s0c3(gdatas)
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return Tperp._result(Tperp.grid, mass * Tperp.values)


def fetch_Tperp_from_M0_M2perp(gdatas, **kwargs):
  """``Tperp = 0.5 * mass * (M2perp / M0)``."""
  Tperp = fetch_s1c0_div_s0c0(gdatas)
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return Tperp._result(Tperp.grid, 0.5 * mass * Tperp.values)


def fetch_temp_from_Max(gdatas, **kwargs):
  """temp from Maxwellian moments: ``mass * comp2``."""
  temp = fetch_s0c2(gdatas)
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return temp._result(temp.grid, mass * temp.values)


def fetch_temp_from_Tpar_Tperp(gdatas, **kwargs):
  """``temp = (Tpar + 2*Tperp) / 3``."""
  Tpar, Tperp = (_ensure_interpolated(g) for g in gdatas)
  values = (Tpar.values + 2.0 * Tperp.values) / 3.0
  return Tpar._result(Tpar.grid, values)


def fetch_press_from_Max(gdatas, **kwargs):
  """Pressure from Maxwellian moments: ``press = mass * comp0 * comp2``."""
  maxmom = _ensure_interpolated(gdatas[0])
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  values = mass * maxmom.values[..., 0, np.newaxis] * maxmom.values[..., 2,
                                                                    np.newaxis]
  return maxmom._result(maxmom.grid, values)


def fetch_press_from_BiMax(gdatas, **kwargs):
  """Pressure from BiMaxwellian moments: ``press = comp0 * mass*(Tpar+2Tperp)/3``."""
  bimax = _ensure_interpolated(gdatas[0])
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  Tpar_vals = bimax.values[..., 2, np.newaxis]
  Tperp_vals = bimax.values[..., 3, np.newaxis]
  temp_vals = mass * (Tpar_vals + 2.0 * Tperp_vals) / 3.0
  values = bimax.values[..., 0, np.newaxis] * temp_vals
  return bimax._result(bimax.grid, values)


def fetch_press_p(gdatas, **kwargs):
  """Perpendicular/parallel pressure in J/m^3: ``p_p = n * T_p``."""
  m0, Tp = (_ensure_interpolated(g) for g in gdatas)
  return m0._result(m0.grid, m0.values * Tp.values)


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

  def fetch(gdatas, **kwargs):
    m3 = _ensure_interpolated(gdatas[0])
    mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
    return m3._result(m3.grid, 0.5 * mass * m3.values)

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

  def fetch(gdatas, **kwargs):
    m0, m1, m2, m3 = (_ensure_interpolated(g) for g in gdatas)
    mass = _get_ctx_val(gdatas[0], "mass", **kwargs)

    upar = m1.values / m0.values
    u_m2 = upar * m2.values

    if is_par:
      values = m3.values - 3.0 * u_m2 + 2.0 * upar**2 * m1.values
    else:
      values = m3.values - u_m2

    return m0._result(m0.grid, 0.5 * mass * values)

  fetch.__name__ = f"fetch_q{name}_fluid"
  return fetch


fetch_qpar_fluid = _make_fetch_q_fluid("par")
fetch_qperp_fluid = _make_fetch_q_fluid("perp")


def fetch_vt(gdatas, **kwargs):
  """Thermal speed ``vt = sqrt(T/m)`` (m/s), ``m`` the requested species'
  mass. ``gdatas``: ``[temp]`` (temperature, in Joules)."""
  temp = _ensure_interpolated(gdatas[0])
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return temp._result(temp.grid, np.sqrt(temp.values / mass))


def fetch_larmor_radius(gdatas, **kwargs):
  """Species Larmor (gyro-)radius: ``rho = sqrt(m*T)/(|q|*B)``. ``gdatas``:
  ``[temp, bmag]``."""
  temp, bmag = (_ensure_interpolated(g) for g in gdatas)
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  charge = abs(_get_ctx_val(gdatas[0], "charge", **kwargs))
  values = np.sqrt(mass * temp.values) / (charge * bmag.values)
  return temp._result(temp.grid, values)


def fetch_debye_length(gdatas, **kwargs):
  """Species-wise Debye length: ``lambda_D = sqrt(eps0*T/(n*q^2))``.
  ``gdatas``: ``[temp, M0]``."""
  temp, m0 = (_ensure_interpolated(g) for g in gdatas)
  charge = _get_ctx_val(gdatas[0], "charge", **kwargs)
  values = np.sqrt(constants.epsilon_0 * temp.values / (m0.values * charge**2))
  return temp._result(temp.grid, values)


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
        "srcs": [_ensure_interpolated(s) for s in srcs],
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


def _weighted_sum(entries, weights, comp: int) -> "GDataState":
  """Sum the ``comp``-th (already-interpolated) source of each species,
  each scaled by a scalar weight."""
  base = entries[0]["srcs"][comp]
  total = sum(w * e["srcs"][comp].values for e, w in zip(entries, weights))
  return base._result(base.grid, total)


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
  values = np.sqrt(temp_e.values * numer.values / denom.values)
  return temp_e._result(temp_e.grid, values)


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
  numer_vals = gamma_e * m0_e.values * temp_e.values
  for ion in ions:
    m0_i, temp_i = ion["srcs"]
    numer_vals = numer_vals + gamma_i * m0_i.values * temp_i.values

  denom = _weighted_sum(ions, [ion["mass"] for ion in ions], 0)
  values = np.sqrt(numer_vals / denom.values)
  return temp_e._result(temp_e.grid, values)


def fetch_c_s(gdatas, **kwargs):
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


def fetch_beta_from_bmag_press(gdatas, **kwargs):
  """``beta = 2*mu_0*press / bmag**2``."""
  bmag, press = (_ensure_interpolated(g) for g in gdatas)
  values = 2.0 * constants.mu_0 * press.values / bmag.values**2
  return bmag._result(bmag.grid, values)


# ------------------------------------------------------------ drift speeds
def _b_cross_grad_div_b_component(scalar: "GDataState",
                                  jacobtot_inv: "GDataState", b_i: "GDataState",
                                  comp: int) -> "GDataState":
  """The ``comp``-th component of ``b x grad(f) / (J B)``.

  ``(b x grad f)_k / B = epsilon_{ijk} * b_i * d(f)/dx^j / (J B)``, where
  ``epsilon_{ijk}`` is the Levi-Civita tensor, ``f`` a scalar field, ``b_i``
  the covariant components of a vector field. The gradient is the numerical
  (post-``interpolate()``) one (``operations.differentiate``); see
  ``differentiate-decision.md`` -- an exact modal derivative needs a shim
  addition out of scope for this layer.

  Args:
    scalar: Scalar field ``f`` to differentiate; interpolated internally.
    jacobtot_inv: Inverse of the total-coordinate-transformation Jacobian.
    b_i: Covariant components of the unit vector field ``b``.
    comp: 0-based component ``k`` of the cross product (``< 3``).

  Raises:
    KeyError: if ``comp`` is not 0, 1, or 2.
  """
  f = _ensure_interpolated(scalar)
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

  b_i_i = _ensure_interpolated(b_i)
  jacobtot_inv_i = _ensure_interpolated(jacobtot_inv)

  pos_term = np.zeros_like(f.values)
  neg_term = np.zeros_like(f.values)
  if calc_term[0]:
    d_pos = operations.differentiate(f, direction=diff_dir_pos)
    pos_term = d_pos.values * b_i_i.values[..., bi_c_pos, np.newaxis]
  if calc_term[1]:
    d_neg = operations.differentiate(f, direction=diff_dir_neg)
    neg_term = -d_neg.values * b_i_i.values[..., bi_c_neg, np.newaxis]

  values = (pos_term + neg_term) * jacobtot_inv_i.values
  return f._result(f.grid, values)


def fetch_ExB_vel(gdatas, **kwargs):
  """``v_{E,k} = epsilon_{ijk}/(J B) * b_i * d(phi)/dx^j`` (``dir`` selects k).

  ``gdatas``: ``(jacobtot_inv, bmag, b_i, phi)``.
  """
  if "dir" not in kwargs:
    raise KeyError("fetch_ExB_vel: select the k-th component with dir=<int>.")
  jacobtot_inv, _bmag, b_i, phi = gdatas
  return _b_cross_grad_div_b_component(phi, jacobtot_inv, b_i, kwargs["dir"])


def fetch_gradB_vel(gdatas, **kwargs):
  """``v_gradB,k = Tperp/(q B) * epsilon_{ijk} * b_i * d(B)/dx^j / (J B)``.

  ``gdatas``: ``(jacobtot_inv, bmag, b_i, Tperp)``.
  """
  if "dir" not in kwargs:
    raise KeyError("fetch_gradB_vel: select the k-th component with dir=<int>.")
  jacobtot_inv, bmag, b_i, Tperp = gdatas
  out = _b_cross_grad_div_b_component(bmag, jacobtot_inv, b_i, kwargs["dir"])
  bmag_i = _ensure_interpolated(bmag)
  Tperp_i = _ensure_interpolated(Tperp)
  charge = _get_ctx_val(Tperp, "charge", **kwargs)
  values = out.values * Tperp_i.values / bmag_i.values / charge
  return out._result(out.grid, values)


def fetch_diamag_vel(gdatas, **kwargs):
  """``v_diamag,k = 1/(q n) epsilon_{ijk} b_i * d(pperp)/dx^j / (J B)``.

  ``gdatas``: ``(jacobtot_inv, bmag, b_i, m0, pressperp)``.
  """
  if "dir" not in kwargs:
    raise KeyError(
        "fetch_diamag_vel: select the k-th component with dir=<int>.")
  jacobtot_inv, bmag, b_i, m0, pressperp = gdatas
  out = _b_cross_grad_div_b_component(pressperp, jacobtot_inv, b_i,
                                      kwargs["dir"])
  m0_i = _ensure_interpolated(m0)
  charge = _get_ctx_val(pressperp, "charge", **kwargs)
  values = out.values / m0_i.values / charge
  return out._result(out.grid, values)


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

  def fetch(gdatas, **kwargs):
    q, m0, temp, c_s = (_ensure_interpolated(g) for g in gdatas)
    values = q.values / (m0.values * temp.values * c_s.values)
    return q._result(q.grid, values)

  fetch.__name__ = f"fetch_q{name}_norm"
  return fetch


fetch_qpar_norm = _make_fetch_q_norm("par")
fetch_qperp_norm = _make_fetch_q_norm("perp")


def fetch_rho_over_lambda(gdatas, **kwargs):
  """Ratio of the species Larmor radius to its Debye length:
  ``rho/lambda_D``. ``gdatas``: ``[rho, lambda_D]``."""
  rho, lambda_d = (_ensure_interpolated(g) for g in gdatas)
  return rho._result(rho.grid, rho.values / lambda_d.values)


def fetch_phi_norm(gdatas, **kwargs):
  """Normalized electrostatic potential: ``phi_norm = e*phi/T_e``.
  ``gdatas``: ``[phi, temp]``."""
  phi, temp = (_ensure_interpolated(g) for g in gdatas)
  values = constants.elementary_charge * phi.values / temp.values
  return phi._result(phi.grid, values)
