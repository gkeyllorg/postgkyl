"""
Functions for for fetching (loading and computing) quantities in the
gk_quantities registry.

Each fetch function takes a list of loaded GData objects (matching the
corresponding 'files' entry in the registry) and returns (grid, values) for
the derived quantity.

Naming keys for some fetch functions below:
  s#: source #
  c#: component #
  add: plus
  sub: minus
  mul: times
  div: divided by
  pow#: raised to the power of #

The helper functions come first, grouped by purpose; the fetch functions follow.
"""
import os

import numpy as np

from postgkyl.data import GData
from postgkyl.data.dg import get_num_basis
from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops
import postgkyl.utils.gkeyll_const as gkc

# ===========================================================================
# ================================ Helpers ==================================
# ===========================================================================

# --------------------------------
# --- Context and allocation ---
# --------------------------------

def _get_ctx_val(gdata : GData, key : str, **kwargs):
  """
  Read a value(s) for 'key', the '--extra' value overrides the GData's context.
  """
  if key in kwargs:
    val = kwargs[key]

    if not isinstance(val, (list, tuple)):
      # A single value applies to every species.
      return val

    species_idx = kwargs.get("species_idx", None)
    if species_idx is None:
      raise KeyError(f"fetch function: '--extra {key}=' was given {len(val)} values but this "
                     f"quantity is not computed per species, so there is no way to tell which "
                     f"one to use. Pass a single value instead.")

    if species_idx >= len(val):
      species = kwargs.get("species", None)
      raise ValueError(f"fetch function: '--extra {key}=' was given only {len(val)} values but "
                       f"species #{species_idx}{f' ({species})' if species else ''} was requested. "
                       f"Give one value per species, in the order of '--species'.")

    return val[species_idx]

  if gdata.ctx.get(key, None) is not None:
    return gdata.ctx[key]

  raise KeyError(f"fetch function: context key '{key}' not found in GData. Pass it as "
                 f"'--extra {key}=<value>', or as one value per species with "
                 f"'--extra {key}=<value1>,<value2>,...'.")

def _get_num_basis_from_gdata(gdata) -> int:
  ndim = gdata.get_num_dims()
  poly_order = int(gdata.ctx["poly_order"])
  basis_type = gdata.ctx["basis_type"]
  return get_num_basis(ndim, poly_order, basis_type)

def _empty_gdata_from_gdata(gdata) -> GData:
  """Allocate a zero-valued GData with the same grid/ctx as gdata."""
  out = GData(ctx=gdata.ctx)
  out.push(gdata.get_grid(), np.zeros_like(gdata.get_values()))
  return out

def _empty_scalar_like(gdata) -> GData:
  """Allocate a zero-valued single-component GData with the grid/ctx of gdata."""
  num_basis = _get_num_basis_from_gdata(gdata)
  vals = gdata.get_values()
  out = GData(ctx=gdata.ctx)
  out.push(gdata.get_grid(), np.zeros((*vals.shape[:-1], num_basis), dtype=vals.dtype))
  return out

def _comp(gdata, comp: int) -> GData:
  """
  The comp-th physical component of a DG field, as a contiguous copy (the DG
  operators read the buffer, not a strided view).
  """
  num_basis = _get_num_basis_from_gdata(gdata)
  out = GData(ctx=gdata.ctx)
  out.push(gdata.get_grid(), gdata.get_values()[..., comp*num_basis:(comp+1)*num_basis].copy())
  return out

def _scaled(gdata, factor: float) -> GData:
  """factor*f, a new GData."""
  out = _empty_gdata_from_gdata(gdata)
  out.set_values(factor*gdata.get_values())
  return out

def _add(*fields) -> GData:
  """Sum of fields, with the grid/ctx of the first one."""
  out = _empty_gdata_from_gdata(fields[0])
  out.set_values(sum(field.get_values() for field in fields))
  return out

def _sub(lop, rop) -> GData:
  """lop - rop, with the grid/ctx of lop."""
  out = _empty_gdata_from_gdata(lop)
  out.set_values(lop.get_values() - rop.get_values())
  return out

# ---------------------------
# --- Weak DG algebra ---
# ---------------------------

def _mul_scalar(lop, rop, factor: float = 1.0, c_lop: int = 0, c_rop: int = 0, like=None):
  """
  Weak DG product factor*lop[c_lop]*rop[c_rop], a single-component field with
  the grid/ctx of like (default lop).
  """
  out = _empty_scalar_like(lop if like is None else like)
  GkeyllDGops().multiply(0, out, c_lop, lop, c_rop, rop)
  if factor != 1.0:
    out.set_values(factor*out.get_values())
  return out

def _inv(gdata):
  """
  Weak DG inverse 1/f of a single-component field, computed as s/(s*f) with s
  the power of two scaling f to order one. The scaling is exact, and prevents
  products of SI quantities (e.g. n*T^(3/2)*(dT/dx)^2 ~ 1e-39) from under- or
  overflowing in the inversion.
  """
  vals = gdata.get_values()
  vmax = np.max(np.abs(vals))
  scale = np.ldexp(1.0, -np.frexp(vmax)[1]) if np.isfinite(vmax) and vmax > 0.0 else 1.0
  out = _empty_gdata_from_gdata(gdata)
  GkeyllDGops().invert(0, out, 0, _scaled(gdata, scale))
  out.set_values(scale*out.get_values())
  return out

def _div(num, den, factor: float = 1.0, like=None):
  """Weak DG quotient factor*num/den, with the grid/ctx of like (default num)."""
  return _mul_scalar(num, _inv(den), factor, like=like)

def _powsqrt_dg(gdata, exponent: float) -> GData:
  """
  pow(sqrt(f), exponent) of a single-component DG field. negative values are set to 1e-40.
  """
  out = _empty_gdata_from_gdata(gdata)
  GkeyllDGops().powsqrt(out, gdata, exponent)
  return out

def _derivative(field, dim: int):
  """d(f)/dx^dim of a single-component DG field (cell-local)."""
  lower, upper = field.get_bounds()
  cells = field.get_num_cells()
  out = _empty_gdata_from_gdata(field)
  GkeyllDGops().differentiate(dim, 1, (upper[dim] - lower[dim])/cells[dim], 0, out, 0, field)
  return out

def _radial_derivative(field, qname: str):
  """d(f)/dx, x being the radial (first) configuration space coordinate."""
  if field.get_num_dims() < 2:
    raise ValueError(f"{qname}: a 1x simulation has no radial coordinate x.")
  return _derivative(field, 0)

# ----------------
# --- Geometry ---
# ----------------

# Component of the metric tensor g_ij holding the (k,l) entry.
_G_IJ_COMP = {(0,0): 0, (0,1): 1, (0,2): 2, (1,1): 3, (1,2): 4, (2,2): 5}

def _select_dir(kwargs, qname: str, index: str = "k") -> int:
  """The vector component selected with '--extra dir=', checked to be 0, 1 or 2."""
  if "dir" not in kwargs:
    raise KeyError(f"{qname}: select the {index}-th component with '--extra dir={index}' (0-index).")
  comp = int(kwargs["dir"])
  if not 0 <= comp < 3:
    raise KeyError(f"{qname}: component must be >= 0 and < 3.")
  return comp

def _config_axis(cdim: int) -> tuple:
  """
  Dimension holding each of x, y and z, or None where a reduced simulation
  does not carry it: a 2x run holds (x,z) and a 1x run only z.
  """
  return (0 if cdim > 1 else None, 1 if cdim > 2 else None, cdim-1)

def _metric_contract(metric, comp: int, vec, like):
  """
  sum_j g_<comp>j * V_j: component comp of the vector V (whose j-th component
  is vec(j)) with its index raised or lowered by the symmetric metric g
  (6 components, see _G_IJ_COMP), with the grid/ctx of like.
  """
  out = _empty_gdata_from_gdata(like)
  for j in range(3):
    term = _mul_scalar(metric, vec(j), c_lop=_G_IJ_COMP[(min(comp,j), max(comp,j))])
    out.set_values(out.get_values() + term.get_values())
  return out

def _b_cross_grad_div_B_component(scalar, jacobtot_inv, b_i, comp):
  """
  The comp-th component of the cross product b x grad(f)
    (b x grad f)_k / B = epsilon_{ijk} * b_i * d(f)/dx^j / (J B)
  where epsilon_{ijk} is the Levi-Civitta tensor, f is a scalar field
  and b_i are the covariant components of a vector field.

  Note: the 1/Jacobian factor of the curvilinear cross product is NOT
  included here and must be applied by the caller.

  Inputs:
    scalar: scalar field f to be differentiated.
    jacobtot_inv: inverse of the Jacobian of the total coordinate transformation.
    b_i:    covariant components of the vector field b.
    comp:   component k of the cross product to compute (0-index, < 3).
  """
  cdim = scalar.get_num_dims()

  # Components of the quantities in the cross product AxB.
  diff_dir_pos = bi_c_pos = 0
  diff_dir_neg = bi_c_neg = 0
  calc_term = [True,True] # Whether to compute pos and neg term in component of AxB.
  if comp == 0:
    diff_dir_neg = bi_c_pos = 1
    diff_dir_pos = bi_c_neg = cdim-1
    if cdim < 3:
      calc_term = [True,False]
    # end
  elif comp == 1:
    bi_c_pos = 2
    bi_c_neg = 0
    diff_dir_neg = cdim-1
    diff_dir_pos = 0
    if cdim == 1:
      calc_term = [False,True]
    # end
  elif comp == 2:
    diff_dir_neg = bi_c_pos = 0
    diff_dir_pos = bi_c_neg = 1
    if cdim == 1:
      calc_term = [False,False]
    elif cdim == 2:
      calc_term = [False,True]
    # end
  else:
    raise KeyError("_b_cross_grad_component: component must be >= 0 and < 3.")

  # b_i * d(f)/dx^j for the positive and negative terms of the comp-th component of AxB.
  out = _empty_gdata_from_gdata(scalar)
  for calc, diff_dir, bi_c, sign in ((calc_term[0], diff_dir_pos, bi_c_pos,  1.0),
                                     (calc_term[1], diff_dir_neg, bi_c_neg, -1.0)):
    if calc:
      term = _mul_scalar(b_i, _derivative(scalar, diff_dir), c_lop=bi_c)
      out.set_values(out.get_values() + sign*term.get_values())

  # Divide by the Jacobian factor of the curvilinear cross product.
  return _mul_scalar(out, jacobtot_inv)

def _vector_magnitude(cov, contra):
  """
  sqrt(V_i V^i) of a vector whose covariant and contravariant components are
  returned by cov(i) and contra(i), with the grid/ctx of cov(0).
  """
  return _powsqrt_dg(_add(*(_mul_scalar(cov(i), contra(i)) for i in range(3))), 1.0)

# ---------------
# --- Species ---
# ---------------

def _mass_times_comp(gdatas, comp: int, **kwargs):
  """mass * (comp-th component of the first source), e.g. T = m*(T/m)."""
  src = gdatas[0]
  return _scaled(_comp(src, comp), _get_ctx_val(src, "mass", **kwargs))

def _split_elc_ions(gdatas, quantity: str, **kwargs):
  """
  Split the per-species sources of a multi-species quantity into the electron
  entry and the ion entries, by the sign of each species' charge..
  """
  species_names = kwargs.get("species", [])
  if len(species_names) != len(gdatas):
    species_names = [f"#{i}" for i in range(len(gdatas))]

  elcs, ions = [], []
  for species_idx, (name, srcs) in enumerate(zip(species_names, gdatas)):
    # Resolve each species' attributes against its own slot in a '--extra' array.
    species_kwargs = dict(kwargs, species_idx=species_idx, species=name)
    entry = {
      "name": name,
      "srcs": srcs,
      "mass": _get_ctx_val(srcs[0], "mass", **species_kwargs),
      "charge": _get_ctx_val(srcs[0], "charge", **species_kwargs),
    }
    (elcs if entry["charge"] < 0.0 else ions).append(entry)

  if not ions:
    raise ValueError(f"{quantity}: found no positively charged (ion) species in {list(species_names)}.")

  if not elcs:
    # Only ions (e.g. a simulation with adiabatic electrons).
    return _adiabatic_elc(ions, **kwargs), ions

  if len(elcs) != 1:
    raise ValueError(f"{quantity}: expected at most one negatively charged (electron) species "
                     f"but found {len(elcs)} in {list(species_names)}.")

  return elcs[0], ions

def _adiabatic_elc(ions, **kwargs):
  """
  Electron entry for a multi-species quantity when only ions are requested
  (e.g. adiabatic electrons), with sources [n_e, T_e]:
    n_e = sum_j(n_j*Z_j), by quasineutrality,
    T_e = T_i/Ti_over_Te, with T_i the temperature of the first ion species
  and Ti_over_Te set via '--extra Ti_over_Te=<value>' (default 1).
  """
  e = gkc.GKYL_ELEMENTARY_CHARGE
  ti_over_te = float(kwargs.get("Ti_over_Te", 1.0))

  den = _weighted_sum(ions, [ion["charge"]/e for ion in ions], 0)
  temp = _scaled(ions[0]["srcs"][1], 1.0/ti_over_te)

  return {
    "name": "adiabatic electrons",
    "srcs": [den, temp],
    "mass": gkc.GKYL_ELECTRON_MASS,
    "charge": -e,
  }

def _weighted_sum(entries, weights, comp: int):
  """
  Sum the comp-th source of each species, each scaled by a scalar weight.
  """
  out = _empty_gdata_from_gdata(entries[0]["srcs"][comp])
  total = sum(w*e["srcs"][comp].get_values() for e, w in zip(entries, weights))
  out.set_values(total)
  return out

def _gkyl_coulomb_log(ns, nr, ms, mr, Ts, Tr, qs, qr, bmag, eps0, hbar, eV):
  """
  Coulomb logarithm, transcribed literally from gkeyll's coulomb_log
  (vlasov/zero/spitzer_coll_freq.c) so that it matches what a simulation used.
  """
  vts = np.sqrt(Ts/ms)
  vtr = np.sqrt(Tr/mr)
  wps = np.sqrt(ns*eV*eV/ms/eps0)
  wpr = np.sqrt(nr*eV*eV/mr/eps0)
  wcs = qs*bmag/ms
  wcr = qr*bmag/mr
  inner1 = (wps*wps + wcs*wcs)/(Ts/ms + 3*Ts/ms) + (wpr*wpr + wcr*wcr)/(Tr/mr + 3*Ts/ms)
  u = 3*(vts*vts + vtr*vtr)
  msr = ms*mr/(ms+mr)
  inner2 = max(abs(qs*qr)/(4*np.pi*eps0*msr*u*u), hbar/(2*np.sqrt(eV)*msr*u))
  inner = (1/inner1)*(1/inner2/inner2) + 1
  return 0.5*np.log(inner)

# ---------------------
# --- Radial fluxes ---
# ---------------------

# Directions averaged over to define a fluctuation, for '--extra fluct=<key>'.
_FLUCT_DIRS = {"y": [1], "yz": [1, 2]}

def _maybe_fluct(gdata, jacobgeo, qname: str, **kwargs):
  """
  Return gdata, or its fluctuation about the Jacobian-weighted average over
  the directions selected with '--extra fluct=y|yz' (fluct=none disables it).
  """
  key = str(kwargs.get("fluct", "none")).lower()
  if key in ("none", "0", "false", ""):
    return gdata
  if key not in _FLUCT_DIRS:
    raise ValueError(f"{qname}: unknown '--extra fluct={key}'. Use one of: "
                     f"none, {', '.join(_FLUCT_DIRS)}.")
  return GkeyllDGops().fluctuation(_FLUCT_DIRS[key], gdata, weight=jacobgeo)

def _radial_ExB_vel(phi, jacobtot_inv, b_i):
  """Radial contravariant ExB velocity v_E^x = v_E.grad(x)."""
  return _b_cross_grad_div_B_component(phi, jacobtot_inv, b_i, 0)

def _radial_dB_over_B(apar, bmag, jacobgeo_inv, b_i):
  """Radial contravariant magnetic flutter dB^x/B."""
  return _div(fetch_dB_perp_dual([apar, jacobgeo_inv, b_i], dir=0), bmag)

def _radial_flux(moment, radial_vel, jacobgeo, factor, qname, **kwargs):
  """
  factor * moment * v^x, with both factors optionally replaced by fluctuations.
  radial_vel() returns v^x; it is only evaluated once the data is known to be 3x,
  since radial turbulent fluxes need the binormal direction.
  """
  if moment.get_num_dims() != 3:
    raise ValueError(f"{qname}: radial fluxes need 3x (x,y,z) data, got "
                     f"{moment.get_num_dims()} dimensions.")
  vel_x = _maybe_fluct(radial_vel(), jacobgeo, qname, **kwargs)
  moment = _maybe_fluct(moment, jacobgeo, qname, **kwargs)
  return _mul_scalar(moment, vel_x, factor)

def _warn_if_apar_dropped(qname: str, **kwargs):
  """Warn when an electrostatic fallback is used although apar output exists."""
  path, sim, frame = kwargs.get("path"), kwargs.get("name"), kwargs.get("frame")
  if path is None or sim is None or frame is None:
    return
  if os.path.isfile(os.path.join(path, f"{sim}-apar_{frame}.gkyl")):
    print(f"Warning: {qname}: apar output found but the moments needed for the magnetic "
          f"flutter flux are missing; only the ExB contribution is included.")

# ------------------------------
# --- Transport coefficients ---
# ------------------------------

def _heat_flux(temp, part_flux, energy_flux, **kwargs):
  """Heat flux q = Q - conv*T*Gamma ('--extra conv=', default 3/2)."""
  conv = float(kwargs.get("conv", 1.5))
  return _add(energy_flux, _mul_scalar(temp, part_flux, -conv))

def _gyro_bohm_factors(gdatas, **kwargs):
  """
  rho_s^2 c_s = T_e^(3/2) m_i^(1/2)/(q_i^2 B^2), with c_s = sqrt(T_e/m_i) and
  rho_s = c_s/Omega_i, as a list of DG fields and a scalar whose product it is.
  gdatas has one source list per species, starting with [M0, temp] and ending
  with B. m_i and q_i are those of the first ion species; T_e is the electron
  temperature, or T_i/Ti_over_Te without an electron species (see
  _split_elc_ions). '--extra Te_ref=,bmag_ref=' replace T_e and B by constants.
  """
  elc, ions = _split_elc_ions([srcs[:2] for srcs in gdatas], "gyro-Bohm diffusivity", **kwargs)
  scalar = np.sqrt(ions[0]["mass"])/ions[0]["charge"]**2
  fields = []
  if kwargs.get("Te_ref") is not None:
    scalar *= float(kwargs["Te_ref"])**1.5
  else:
    fields.append(_powsqrt_dg(elc["srcs"][1], 3.0))
  if kwargs.get("bmag_ref") is not None:
    scalar /= float(kwargs["bmag_ref"])**2
  else:
    bmag = gdatas[0][-1]
    fields.append(_inv(_mul_scalar(bmag, bmag)))
  return fields, scalar

def _gyro_bohm_normalized(num, grad, gxx, other_den, gdatas, **kwargs):
  """
  num / (other_den g^xx^(3/2) grad^2 rho_s^2 c_s), which is -X L_X/(rho_s^2 c_s)
  for a diffusivity X = -num/(other_den g^xx grad) and gradient length
  L_X = -field/(sqrt(g^xx) grad), num carrying the field.
  """
  fields, scalar = _gyro_bohm_factors(gdatas, **kwargs)
  den = _mul_scalar(_powsqrt_dg(gxx, 3.0), _mul_scalar(grad, grad))
  if other_den is not None:
    den = _mul_scalar(den, other_den)
  for field in fields:
    den = _mul_scalar(den, field)
  return _div(num, den, 1.0/scalar)

# ---------------------------------
# --- Generic fetch factories ---
# ---------------------------------

def _make_fetch_comp(icomp: int):
  """Return a fetch function that extracts the comp-th physical component (all if None)."""
  def fetch(gdatas, **kw):
    src = gdatas[0]
    if icomp is not None:
      return _comp(src, icomp)
    out = GData(ctx=src.ctx)
    out.push(src.get_grid(), src.get_values().copy())
    return out
  # end
  fetch.__name__ = f"fetch_comp{icomp}" if icomp is not None else f"fetch_compAll"
  return fetch

# Weak DG binary operations of _make_fetch_sick_op_sjcl.
_BINARY_OPS = {"add": _add, "sub": _sub, "mul": _mul_scalar, "div": _div}

def _make_fetch_sick_op_sjcl(si: int, ck: int, op: str, sj: int, cl: int):
  """
  Return a fetch function that does
    (k-th component of the i-th source) op (l-th component of the j-th source)
  with op one of add, sub, mul (weak product) or div (weak quotient).
  """
  def fetch(gdatas, **kwargs):
    gd_l, gd_r = gdatas[si], gdatas[sj]
    if _get_num_basis_from_gdata(gd_l) != _get_num_basis_from_gdata(gd_r):
      raise ValueError(f"Datasets have different basis")
    return _BINARY_OPS[op](_comp(gd_l, ck), _comp(gd_r, cl))
  # end
  fetch.__name__ = f"fetch_s{si}c{ck}_{op}_s{sj}c{cl}"
  return fetch

# Functions to extract a components.
fetch_s0cAll = _make_fetch_comp(None)
fetch_s0c0 = _make_fetch_comp(0)
fetch_s0c1 = _make_fetch_comp(1)
fetch_s0c2 = _make_fetch_comp(2)
fetch_s0c3 = _make_fetch_comp(3)

# Functions to add two components.
fetch_s0c0_add_s1c0 = _make_fetch_sick_op_sjcl(0,0,"add",1,0)
fetch_s0c2_add_s0c3 = _make_fetch_sick_op_sjcl(0,2,"add",0,3)

# Functions to subtract two components.
fetch_s0c0_sub_s1c0 = _make_fetch_sick_op_sjcl(0,0,"sub",1,0)

# Functions to multiply two components.
fetch_s0c0_mul_s1c0 = _make_fetch_sick_op_sjcl(0,0,"mul",1,0)
fetch_s0c0_mul_s0c1 = _make_fetch_sick_op_sjcl(0,0,"mul",0,1)

# Functions to divide two components.
fetch_s1c0_div_s0c0 = _make_fetch_sick_op_sjcl(1,0,"div",0,0)

# ===========================================================================
# ============================ Fetch functions ==============================
# ===========================================================================

# ------------------------------------------
# --- Plasma moments (species-dependent) ---
# ------------------------------------------

def fetch_M1_from_H(gdatas, **kwargs):
  """
  M1 from the Hamiltonian moments (Hmom).
  """
  hmom = gdatas[0]
  mass = _get_ctx_val(hmom, "mass", **kwargs)
  return _mul_scalar(hmom, hmom, 1.0/mass, c_rop=1)

def _make_fetch_M2_from_Max(par: bool, t_comp: int):
  """
  Return a fetch function for the second parallel (par=True) or perpendicular
  moment from (Bi)Maxwellian moments.
  """
  def fetch(gdatas, **kwargs):
    mom = gdatas[0]
    out = _mul_scalar(mom, mom, c_rop=t_comp)  # n*T/m.
    if not par:
      return _scaled(out, 2.0)
    # n*T/m + n*upar^2.
    return _add(out, _mul_scalar(_mul_scalar(mom, mom, c_rop=1), mom, c_rop=1))
  # end
  fetch.__name__ = f"fetch_M2{'par' if par else 'perp'}_from_Max_c{t_comp}"
  return fetch

fetch_M2par_from_Max = _make_fetch_M2_from_Max(True, 2)
fetch_M2perp_from_Max = _make_fetch_M2_from_Max(False, 2)
fetch_M2par_from_BiMax = _make_fetch_M2_from_Max(True, 2)
fetch_M2perp_from_BiMax = _make_fetch_M2_from_Max(False, 3)

def fetch_Tpar_from_BiMax(gdatas, **kwargs):
  """
  Tpar from BiMaxwellian moments.
  """
  return _mass_times_comp(gdatas, 2, **kwargs)

def fetch_Tpar_from_M0_M1_M2par(gdatas, **kwargs):
  """
  upar*M1 + M0*Tpar/m = M2par.
  Tpar = m * (M2par - upar*M1) / M0.
  """
  m0, m1, m2par = gdatas
  m0_inv = _inv(m0)
  u_m1 = _mul_scalar(_mul_scalar(m1, m0_inv), m1)
  mass = _get_ctx_val(m0, "mass", **kwargs)
  return _mul_scalar(_scaled(_sub(m2par, u_m1), mass), m0_inv, like=m0)

def fetch_Tperp_from_BiMax(gdatas, **kwargs):
  """
  Tperp from BiMaxwellian moments.
  """
  return _mass_times_comp(gdatas, 3, **kwargs)

def fetch_Tperp_from_M0_M2perp(gdatas, **kwargs):
  """
  Tperp = 0.5 * mass * (M2perp / M0).
  """
  mass = _get_ctx_val(gdatas[0], "mass", **kwargs)
  return _scaled(fetch_s1c0_div_s0c0(gdatas), 0.5*mass)

def fetch_temp_from_Max(gdatas, **kwargs):
  """
  temp from Maxwellian moments.
  """
  return _mass_times_comp(gdatas, 2, **kwargs)

def fetch_temp_from_Tpar_Tperp(gdatas, **kwargs):
  """
  temp = (Tpar + 2*Tperp) / 3.
  """
  Tpar, Tperp = gdatas
  temp = _empty_gdata_from_gdata(Tpar)
  temp.set_values((Tpar.get_values() + 2.0*Tperp.get_values())/3.0)
  return temp

# ---------------------------------------------------
# --- Combined plasma moments (species-dependent) ---
# ---------------------------------------------------

def fetch_press_from_Max(gdatas, **kwargs):
  """
  Pressure from Maxwellian moments.
  press = den * temp.
  """
  maxmom = gdatas[0]
  mass = _get_ctx_val(maxmom, "mass", **kwargs)
  return _mul_scalar(maxmom, maxmom, mass, c_rop=2)

def fetch_press_from_BiMax(gdatas, **kwargs):
  """
  Pressure from BiMaxwellian moments.
  press = den * (Tpar + 2*Tperp) / 3.
  """
  bimax = gdatas[0]
  mass = _get_ctx_val(bimax, "mass", **kwargs)
  temp = _comp(bimax, 2)
  temp.set_values(mass*(temp.get_values() + 2.0*_comp(bimax, 3).get_values())/3.0)
  return _mul_scalar(bimax, temp)

def fetch_press_p(gdatas, **kwargs):
  """
  Perpendicular/parallel pressure in J/m^3.
  p_p = n * T_p.
  """
  m0, Tp = gdatas
  return _mul_scalar(m0, Tp)

def _make_fetch_q(name: str):
  """
  Return a fetch function for the lab-frame parallel flux of the parallel
  (name='par') or perpendicular (name='perp') kinetic energy:
    q_par  = (m/2)*M3par  = (m/2) int(vpar^3 f) dv,
    q_perp = (m/2)*M3perp = (m/2) int(vpar*vperp^2 f) dv,
  so that q_par + q_perp is the parallel flux of the total kinetic energy.
  Both are in W/m^2 (kg/s^3). gdatas has:
    1. M3par (name='par') or M3perp (name='perp').
  """
  def fetch(gdatas, **kwargs):
    m3 = gdatas[0]
    return _scaled(m3, 0.5*_get_ctx_val(m3, "mass", **kwargs))
  # end
  fetch.__name__ = f"fetch_q{name}"
  return fetch

fetch_qpar = _make_fetch_q("par")
fetch_qperp = _make_fetch_q("perp")

def _make_fetch_q_fluid(name: str):
  """
  Return a fetch function for the parallel heat flux in the fluid (drift)
  frame, i.e. the energy carried by the random part of the parallel motion,
  u = M1/M0 being the parallel drift speed:
    q_par  = (m/2) int (vpar-u)^3 f dv
           = (m/2) [M3par - 3*u*M2par + 3*u^2*M1 - u^3*M0]
           = (m/2) [M3par - 3*u*M2par + 2*u^2*M1],
    q_perp = (m/2) int (vpar-u)*vperp^2 f dv
           = (m/2) [M3perp - u*M2perp].
  gdatas has (in this order):
    1. M0: zeroth moment (density).
    2. M1: first moment.
    3. M2par (name='par') or M2perp (name='perp').
    4. M3par (name='par') or M3perp (name='perp').
  """
  is_par = name == "par"

  def fetch(gdatas, **kwargs):
    m0, m1, m2, m3 = gdatas
    mass = _get_ctx_val(m0, "mass", **kwargs)

    upar = _div(m1, m0)
    u_m2 = _mul_scalar(upar, m2)  # u*M2par or u*M2perp.

    if is_par:
      # u^2*M1, which equals u^3*M0.
      u_sq_m1 = _mul_scalar(_mul_scalar(upar, upar), m1)
      vals = m3.get_values() - 3.0*u_m2.get_values() + 2.0*u_sq_m1.get_values()
    else:
      vals = m3.get_values() - u_m2.get_values()

    out = _empty_gdata_from_gdata(m0)
    out.set_values(0.5*mass*vals)
    return out

  fetch.__name__ = f"fetch_q{name}_fluid"
  return fetch

fetch_qpar_fluid = _make_fetch_q_fluid("par")
fetch_qperp_fluid = _make_fetch_q_fluid("perp")

def fetch_vt(gdatas, **kwargs):
  """
  Thermal speed vt = sqrt(T/m) (m/s), where T is the temperature of the
  requested species and m its mass. gdatas has:
    1. temp: temperature (in Joules).
  """
  temp = gdatas[0]
  return _powsqrt_dg(_scaled(temp, 1.0/_get_ctx_val(temp, "mass", **kwargs)), 1.0)

def fetch_larmor_radius(gdatas, **kwargs):
  """
  Species Larmor (gyro-)radius: rho = sqrt(m*T)/(|q|*B). gdatas has:
    1. B: magnetic field magnitude (bmag).
    2. temp: temperature (in Joules).
  """
  bmag, temp = gdatas
  mass = _get_ctx_val(temp, "mass", **kwargs)
  charge = abs(_get_ctx_val(temp, "charge", **kwargs))
  sqrt_mT = _powsqrt_dg(_scaled(temp, mass), 1.0)
  return _div(sqrt_mT, _scaled(bmag, charge), like=bmag)

def fetch_debye_length(gdatas, **kwargs):
  """
  Species-wise Debye length: lambda_D = sqrt(eps0*T/(n*q^2)). gdatas has:
    1. M0: zeroth moment (density).
    2. temp: temperature (in Joules).
  """
  m0, temp = gdatas
  charge = _get_ctx_val(temp, "charge", **kwargs)
  sq = _div(_scaled(temp, gkc.GKYL_EPSILON0), _scaled(m0, charge**2))
  return _powsqrt_dg(sq, 1.0)

# The sound speeds combine the electrons and every ion species: gdatas has one
# [M0, temp] pair per species, in the order they were requested, e.g.
#   pgkyl gk-load-quantity -q c_s_cold_i -s elc,ion1,ion2 ...
# Electrons and ions are told apart by the sign of each species' charge
# attribute, so the species may be named anything. With only ion species listed
# (e.g. adiabatic electrons), the electrons are taken quasineutral,
# n_e = sum_j(n_j*Z_j), with T_e = T_i/Ti_over_Te, T_i the first ion's
# temperature and '--extra Ti_over_Te=<value>' (default 1).

def fetch_c_s_cold_i(gdatas, **kwargs):
  """
  Cold-ion (ion-acoustic) sound speed (m/s), the wave perspective, for the
  Bohm criterion and sheath/presheath matching:
    c_s = sqrt( T_e * sum_j(n_j*Z_j^2/m_j) / sum_j(n_j*Z_j) )
  summing over the ion species j, with Z_j = q_j/e the ion charge state.
  """
  elc, ions = _split_elc_ions(gdatas, "fetch_c_s_cold_i", **kwargs)

  e = gkc.GKYL_ELEMENTARY_CHARGE
  charge_states = [ion["charge"]/e for ion in ions]

  # sum_j n_j*Z_j^2/m_j and sum_j n_j*Z_j, both linear in the densities (M0).
  numer = _weighted_sum(ions, [z**2/ion["mass"] for z, ion in zip(charge_states, ions)], 0)
  denom = _weighted_sum(ions, charge_states, 0)

  # T_e * numer/denom.
  return _powsqrt_dg(_mul_scalar(_div(numer, denom), elc["srcs"][1]), 1.0)

def fetch_c_s_hot_i(gdatas, **kwargs):
  """
  Hot-ion (thermodynamic) sound speed, the bulk fluid perspective, for
  Mach numbers and acoustic propagation in the core/SOL:
    c_s = sqrt( (gamma_e*n_e*T_e + sum_j(gamma_j*n_j*T_j)) / sum_j(n_j*m_j) )
  summing over the ion species j.
  Default: gamma_e=1, gamma_i=3, but these can be set via '--extra'.
  """
  elc, ions = _split_elc_ions(gdatas, "fetch_c_s_hot_i", **kwargs)

  gamma_e = float(kwargs.get("gamma_e", 1.0))
  gamma_i = float(kwargs.get("gamma_i", 3.0))

  # gamma_e*n_e*T_e + sum_j gamma_j*n_j*T_j. Each n*T is a weak product.
  numer = _add(_mul_scalar(*elc["srcs"], gamma_e),
               *(_mul_scalar(*ion["srcs"][:2], gamma_i) for ion in ions))

  # sum_j n_j*m_j, the ion mass density; linear in the densities.
  denom = _weighted_sum(ions, [ion["mass"] for ion in ions], 0)

  return _powsqrt_dg(_div(numer, denom), 1.0)

def _fetch_mach(gdatas, fetch_c_s, **kwargs):
  """
  Parallel Mach number M = upar/c_s of the first requested species, with c_s
  from fetch_c_s combining every listed species. gdatas has one
  [M0, temp, upar] triplet per species, in the order they were requested.
  """
  c_s = fetch_c_s([srcs[:2] for srcs in gdatas], **kwargs)
  return _div(gdatas[0][2], c_s)

def fetch_mach_cold_i(gdatas, **kwargs):
  """
  Parallel Mach number upar/c_s of the first requested species, with the
  cold-ion sound speed (fetch_c_s_cold_i):
    pgkyl gk-load-quantity -q mach_cold_i -s ion,elc ...  (ion Mach number)
    pgkyl gk-load-quantity -q mach_cold_i -s elc,ion ...  (electron Mach number)
    pgkyl gk-load-quantity -q mach_cold_i -s ion -e Ti_over_Te=1 ...  (adiabatic electrons)
  """
  return _fetch_mach(gdatas, fetch_c_s_cold_i, **kwargs)

def fetch_mach_hot_i(gdatas, **kwargs):
  """
  Parallel Mach number upar/c_s of the first requested species, with the
  hot-ion sound speed (fetch_c_s_hot_i); species are listed as for
  fetch_mach_cold_i.
  """
  return _fetch_mach(gdatas, fetch_c_s_hot_i, **kwargs)

def fetch_collision_freq(gdatas, **kwargs):
  """
  Collision frequency of species s with species r (1/s), as computed by the
  gyrokinetic app (LBO/BGK collisions with a normalized nu):
    nu_sr = norm_nu_sr * n_r/(v_ts^2+v_tr^2)^(3/2),
    norm_nu_sr = nu_frac * (1/m_s)*(1/m_s+1/m_r) * q_s^2*q_r^2*log(Lambda_sr)
                 / (3*(2*pi)^(3/2)*eps0^2),
  """
  if len(gdatas) != 2:
    raise ValueError(f"fetch_collision_freq: expected two species (s,r) but got {len(gdatas)}. "
                     f"Use e.g. '--species elc,ion', or '--species ion,ion' for self-collisions.")

  species_names = kwargs.get("species", [])
  if len(species_names) != len(gdatas):
    species_names = ["s", "r"]

  # Species attributes and reference values, resolved against each species' slot in a '--extra' array.
  q, m, den_ref, temp_ref, bmag_ref = [], [], [], [], []
  for species_idx, (name, srcs) in enumerate(zip(species_names, gdatas)):
    species_kwargs = dict(kwargs, species_idx=species_idx, species=name)
    q.append(_get_ctx_val(srcs[1], "charge", **species_kwargs))
    m.append(_get_ctx_val(srcs[1], "mass", **species_kwargs))
    den_ref.append(_get_ctx_val(srcs[1], "den_ref", **species_kwargs))
    temp_ref.append(_get_ctx_val(srcs[1], "temp_ref", **species_kwargs))
    bmag_ref.append(_get_ctx_val(srcs[1], "bmag_ref", **species_kwargs))

  eps0 = gkc.GKYL_EPSILON0
  hbar = gkc.GKYL_PLANCKS_CONSTANT_H/(2.0*gkc.GKYL_PI)
  eV = gkc.GKYL_ELEMENTARY_CHARGE
  nu_frac = float(kwargs.get("nu_frac", 1.0))

  # Symmetrized Coulomb logarithm from the reference values (bmag_ref of species s).
  coul_log = 0.5*(_gkyl_coulomb_log(den_ref[0], den_ref[1], m[0], m[1], temp_ref[0], temp_ref[1],
                                    q[0], q[1], bmag_ref[0], eps0, hbar, eV)
                 +_gkyl_coulomb_log(den_ref[1], den_ref[0], m[1], m[0], temp_ref[1], temp_ref[0],
                                    q[1], q[0], bmag_ref[0], eps0, hbar, eV))

  norm_nu = nu_frac/m[0]*(1.0/m[0] + 1.0/m[1])*(q[0]*q[1])**2*coul_log \
            /(3.0*(2.0*gkc.GKYL_PI)**1.5*eps0**2)

  (m0_s, temp_s), (m0_r, temp_r) = gdatas

  # norm_nu * n_r * (v_ts^2 + v_tr^2)^(-3/2).
  vtsq_sum = _empty_gdata_from_gdata(temp_s)
  vtsq_sum.set_values(temp_s.get_values()/m[0] + temp_r.get_values()/m[1])
  return _mul_scalar(_powsqrt_dg(vtsq_sum, -3.0), m0_r, norm_nu)

def fetch_beta_from_bmag_press(gdatas, **kwargs):
  """
  beta = 2*mu_0*press/bmag^2
  """
  bmag, press = gdatas
  return _div(press, _mul_scalar(bmag, bmag), 2.0*gkc.GKYL_MU0, like=bmag)

# ------------------------
# --- Gradient lengths ---
# ------------------------

def _get_fetch_inv_grad_length(name: str):
  """
  Return a fetch function for the radial inverse gradient length of a scalar field X,
    1/L_X = -(dX/dx)/X
  where x is the radial (first) configuration space coordinate. gdatas has:
    1. X: the scalar field (e.g. M0 or temp).
  """
  def fetch(gdatas, **kwargs):
    field = gdatas[0]
    return _div(_radial_derivative(field, f"fetch_inv_L_{name}"), field, -1.0)
  # end
  fetch.__name__ = f"fetch_inv_L_{name}"
  return fetch

fetch_inv_L_n = _get_fetch_inv_grad_length("n")
fetch_inv_L_T = _get_fetch_inv_grad_length("T")

# ------------------------
# --- Drift velocities ---
# ------------------------

def fetch_ExB_vel(gdatas, **kwargs):
  """
  A component of the ExB drift velocity
    v_{E,k} = epsilon_{ijk}/(J B) * b_i * d(phi)/dx^j
  where epsilon_{ijk} is the Levi-Civitta tensor
  and gdatas has (in this order):
    B: magnetic field magnitude (bmag).
    1/(J*B): inv. total Jacobian (jacobtot_inv).
    phi: electrostatic potential.
    b_i: covariant components of the magnetic field unit vector.

  The k-th component is selected by the 'dir' optional argument.
  """
  comp = _select_dir(kwargs, "fetch_ExB_vel", "j")
  _, jacobtot_inv, phi, b_i = gdatas

  # k-th component of b x grad(phi)/B.
  return _b_cross_grad_div_B_component(phi, jacobtot_inv, b_i, comp)

def fetch_gradB_vel(gdatas, **kwargs):
  """
  A component of the grad-B drift velocity
    v_gradB,k = Tperp/(q B) * epsilon_{ijk} * b_i * d(B)/dx^j / (J B)
  where epsilon_{ijk} is the Levi-Civitta tensor, q the species charge,
  and gdatas has (in this order):
    B: magnetic field magnitude (bmag).
    1/(J*B): inv. total Jacobian (jacobtot_inv).
    Tperp: perpendicular temperature (in Joules).
    b_i: covariant components of the magnetic field unit vector.

  The k-th component is selected by the 'dir' optional argument.
  """
  comp = _select_dir(kwargs, "fetch_gradB_vel", "j")
  bmag, jacobtot_inv, Tperp, b_i = gdatas
  charge = _get_ctx_val(Tperp, "charge", **kwargs)

  # Tperp/(q B) times the k-th component of b x grad(B)/B.
  out = _b_cross_grad_div_B_component(bmag, jacobtot_inv, b_i, comp)
  return _div(_mul_scalar(out, Tperp), bmag, 1.0/charge)

def fetch_diamag_vel(gdatas, **kwargs):
  """
  A component of the diamagnetic drift velocity
    v_diamag,k = 1 / (q n) epsilon_{ijk} b_i * d(pperp)/dx^j / (J B)
  where epsilon_{ijk} is the Levi-Civitta tensor, q the species charge,
  and gdatas has (in this order):
    B: magnetic field magnitude (bmag).
    1/(J*B): inv. total Jacobian (jacobtot_inv).
    M0: zeroth moment (density).
    p_perp: perpendicular pressure (in Joules/m^3).
    b_i: covariant components of the magnetic field unit vector.
  The k-th component is selected by the 'dir' optional argument.
  """
  comp = _select_dir(kwargs, "fetch_diamag_vel", "j")
  _, jacobtot_inv, m0, pressperp, b_i = gdatas
  charge = _get_ctx_val(pressperp, "charge", **kwargs)

  # 1/(q n) times the k-th component of b x grad(p) / B.
  out = _b_cross_grad_div_B_component(pressperp, jacobtot_inv, b_i, comp)
  return _div(out, m0, 1.0/charge)

# -------------------------------------
# --- Magnetic field perturbations ----
# -------------------------------------

def fetch_dB_perp_dual(gdatas, **kwargs):
  """
  A contravariant component of the magnetic field perturbation dB = curl(Apar*b),
    dB^i = ( d(Apar*b_k)/dx^j - d(Apar*b_j)/dx^k ) / J
  with (j,k) = (i+1,i+2) cyclically.

  gdatas has (in this order):
    Apar: parallel magnetic vector potential (T*m).
    1/J: reciprocal configuration space Jacobian (jacobgeo_inv).
    b_i: covariant components of the magnetic field unit vector.

  The i-th component is selected by the 'dir' optional argument.
  """
  comp = _select_dir(kwargs, "fetch_dB_perp_dual")
  apar, jacobgeo_inv, b_i = gdatas

  axis = _config_axis(apar.get_num_dims())

  out = _empty_gdata_from_gdata(apar)
  for diff_dir, b_comp, sign in (((comp+1) % 3, (comp+2) % 3,  1.0),
                                 ((comp+2) % 3, (comp+1) % 3, -1.0)):
    dim = axis[diff_dir]
    if dim is None:
      continue
    # d(Apar*b_<b_comp>)/dx^<diff_dir>.
    term = _derivative(_mul_scalar(apar, b_i, c_rop=b_comp), dim)
    out.set_values(out.get_values() + sign*term.get_values())

  # Divide by the Jacobian factor of the curvilinear curl.
  return _mul_scalar(out, jacobgeo_inv)

def fetch_dB_perp(gdatas, **kwargs):
  """
  A covariant component of the magnetic field perturbation dB = curl(Apar*b),
    dB_i = g_ij * dB^j.

  gdatas has (in this order):
    Apar: parallel magnetic vector potential (T*m).
    1/J: reciprocal configuration space Jacobian (jacobgeo_inv).
    b_i: covariant components of the magnetic field unit vector.
    g_ij: covariant metric coefficients, in the order g_11,g_12,g_13,g_22,g_23,g_33.

  The i-th component is selected by the 'dir' optional argument.
  """
  comp = _select_dir(kwargs, "fetch_dB_perp")
  apar, g_ij = gdatas[0], gdatas[3]
  return _metric_contract(g_ij, comp, lambda j: fetch_dB_perp_dual(gdatas[:3], dir=j), apar)

def fetch_dB_perp_mag(gdatas, **kwargs):
  """
  Magnitude of the magnetic field perturbation, each covariant component paired
  with its contravariant counterpart,
    |dB| = sqrt(dB_i * dB^i) = sqrt(g_ij * dB^i * dB^j).
  Warning: this product is of higher order and may introduce DG basis aliasing.

  gdatas has (in this order):
    Apar: parallel magnetic vector potential (T*m).
    1/J: reciprocal configuration space Jacobian (jacobgeo_inv).
    b_i: covariant components of the magnetic field unit vector.
    g_ij: covariant metric coefficients, in the order g_11,g_12,g_13,g_22,g_23,g_33.
  """
  return _vector_magnitude(lambda i: fetch_dB_perp(gdatas, dir=i),
                           lambda i: fetch_dB_perp_dual(gdatas[:3], dir=i))

# ------------------------------
# --- Total magnetic field -----
# ------------------------------

def fetch_B_equilibrium(gdatas, **kwargs):
  """
  A covariant component of the equilibrium magnetic field, B_i = B*b_i.
  This is what 'B_tot' falls back to on a run that carries no Apar.

  gdatas has (in this order):
    B: magnetic field magnitude (bmag).
    b_i: covariant components of the magnetic field unit vector.

  The i-th component is selected by the 'dir' optional argument.
  """
  comp = _select_dir(kwargs, "fetch_B_equilibrium")
  bmag, b_i = gdatas
  return _mul_scalar(b_i, bmag, c_lop=comp, like=bmag)

def fetch_B_tot(gdatas, **kwargs):
  """
  A covariant component of the total magnetic field, the equilibrium plus the
  perturbation carried by the parallel vector potential,
    B_i = B*b_i + dB_i.

  gdatas has (in this order):
    Apar: parallel magnetic vector potential (T*m).
    B: magnetic field magnitude (bmag).
    1/J: reciprocal configuration space Jacobian (jacobgeo_inv).
    b_i: covariant components of the magnetic field unit vector.
    g_ij: covariant metric coefficients, in the order g_11,g_12,g_13,g_22,g_23,g_33.

  The i-th component is selected by the 'dir' optional argument.
  """
  apar, bmag, jacobgeo_inv, b_i, g_ij = gdatas
  return _add(fetch_B_equilibrium([bmag, b_i], **kwargs),
              fetch_dB_perp([apar, jacobgeo_inv, b_i, g_ij], **kwargs))

def fetch_B_dual_equilibrium(gdatas, **kwargs):
  """
  A contravariant component of the equilibrium magnetic field. b is the unit
  vector along e_3, so b^i = delta^i_3/sqrt(g_33) and
    B^i = B*b^i = (B/sqrt(g_33)) * delta^i_3,
  the first two components vanishing identically. This is what 'B_tot_dual'
  falls back to on a run that carries no Apar.

  gdatas has (in this order):
    B: magnetic field magnitude (bmag).
    g_ij: covariant metric coefficients, in the order g_11,g_12,g_13,g_22,g_23,g_33.

  The i-th component is selected by the 'dir' optional argument.
  """
  comp = _select_dir(kwargs, "fetch_B_dual_equilibrium")
  bmag, g_ij = gdatas
  if comp != 2:
    return _empty_gdata_from_gdata(bmag)
  return _mul_scalar(bmag, _powsqrt_dg(_comp(g_ij, _G_IJ_COMP[(2,2)]), -1.0))

def fetch_B_tot_dual(gdatas, **kwargs):
  """
  A contravariant component of the total magnetic field,
    B^i = B*b^i + dB^i.

  gdatas has (in this order):
    Apar: parallel magnetic vector potential (T*m).
    B: magnetic field magnitude (bmag).
    1/J: reciprocal configuration space Jacobian (jacobgeo_inv).
    b_i: covariant components of the magnetic field unit vector.
    g_ij: covariant metric coefficients, in the order g_11,g_12,g_13,g_22,g_23,g_33.

  The i-th component is selected by the 'dir' optional argument.
  """
  apar, bmag, jacobgeo_inv, b_i, g_ij = gdatas
  return _add(fetch_B_dual_equilibrium([bmag, g_ij], **kwargs),
              fetch_dB_perp_dual([apar, jacobgeo_inv, b_i], **kwargs))

def fetch_B_tot_mag(gdatas, **kwargs):
  """
  Magnitude of the total magnetic field, each covariant component paired with
  its contravariant counterpart,
    |B| = sqrt(B_i * B^i) = sqrt(g_ij * B^i * B^j).
  Warning: this product is of higher order and may introduce DG basis aliasing.

  gdatas has (in this order):
    Apar: parallel magnetic vector potential (T*m).
    B: magnetic field magnitude (bmag).
    1/J: reciprocal configuration space Jacobian (jacobgeo_inv).
    b_i: covariant components of the magnetic field unit vector.
    g_ij: covariant metric coefficients, in the order g_11,g_12,g_13,g_22,g_23,g_33.
  """
  return _vector_magnitude(lambda i: fetch_B_tot(gdatas, dir=i),
                           lambda i: fetch_B_tot_dual(gdatas, dir=i))

# ----------------------
# --- Electric field ---
# ----------------------

def fetch_E_field(gdatas, **kwargs):
  """
  A covariant component of the electrostatic field, E_i = -d(phi)/dx^i, in
  V per unit of the coordinate x^i. A direction that a reduced simulation does
  not carry (y in 2x, x and y in 1x) has E_i = 0.

  gdatas has:
    phi: electrostatic potential.

  The i-th component is selected by the 'dir' optional argument.
  """
  comp = _select_dir(kwargs, "fetch_E_field")
  phi = gdatas[0]
  dim = _config_axis(phi.get_num_dims())[comp]
  if dim is None:
    return _empty_gdata_from_gdata(phi)
  return _scaled(_derivative(phi, dim), -1.0)

def fetch_E_field_dual(gdatas, **kwargs):
  """
  A contravariant component of the electrostatic field, E^i = g^ij * E_j.

  gdatas has (in this order):
    phi: electrostatic potential.
    g^ij: contravariant metric coefficients, in the order g^11,g^12,g^13,g^22,g^23,g^33.

  The i-th component is selected by the 'dir' optional argument.
  """
  comp = _select_dir(kwargs, "fetch_E_field_dual")
  phi, gij = gdatas
  return _metric_contract(gij, comp, lambda j: fetch_E_field([phi], dir=j), phi)

def fetch_E_field_mag(gdatas, **kwargs):
  """
  Magnitude of the electrostatic field (V/m),
    |E| = sqrt(E_i * E^i) = sqrt(g^ij * E_i * E_j).
  Warning: this product is of higher order and may introduce DG basis aliasing.

  gdatas has (in this order):
    phi: electrostatic potential.
    g^ij: contravariant metric coefficients, in the order g^11,g^12,g^13,g^22,g^23,g^33.
  """
  return _vector_magnitude(lambda i: fetch_E_field(gdatas[:1], dir=i),
                           lambda i: fetch_E_field_dual(gdatas, dir=i))

# ---------------------
# --- Radial fluxes ---
# ---------------------

def fetch_part_flux_ExB(gdatas, **kwargs):
  """
  Radial ExB particle flux, Gamma_E^x = n * v_E^x, where v_E^x = v_E.grad(x).
  gdatas has (in this order):
    M0: density.
    phi: electrostatic potential.
    J: configuration space Jacobian (jacobgeo), weight of the fluctuation average.
    1/(J*B): inv. total Jacobian (jacobtot_inv).
    b_i: covariant components of the magnetic field unit vector.
  With '--extra fluct=y|yz' the turbulent part <dn dv_E^x> is computed instead.
  """
  m0, phi, jacobgeo, jacobtot_inv, b_i = gdatas
  return _radial_flux(m0, lambda: _radial_ExB_vel(phi, jacobtot_inv, b_i), jacobgeo, 1.0,
                      "fetch_part_flux_ExB", **kwargs)

def fetch_energy_flux_ExB(gdatas, **kwargs):
  """
  Radial ExB energy flux, Q_E^x = (m/2) * M2 * v_E^x, where M2 = M2par + M2perp.
  Exact for long-wavelength gyrokinetics, since v_E does not depend on velocity.
  gdatas has (in this order):
    M2: second velocity moment.
    phi: electrostatic potential.
    J: configuration space Jacobian (jacobgeo).
    1/(J*B): inv. total Jacobian (jacobtot_inv).
    b_i: covariant components of the magnetic field unit vector.
  With '--extra fluct=y|yz' the turbulent part <dM2 dv_E^x> is computed instead.
  """
  m2, phi, jacobgeo, jacobtot_inv, b_i = gdatas
  mass = _get_ctx_val(m2, "mass", **kwargs)
  return _radial_flux(m2, lambda: _radial_ExB_vel(phi, jacobtot_inv, b_i), jacobgeo, 0.5*mass,
                      "fetch_energy_flux_ExB", **kwargs)

def fetch_part_flux_dB(gdatas, **kwargs):
  """
  Radial magnetic flutter particle flux, Gamma_dB^x = M1 * dB^x/B.
  gdatas has (in this order):
    M1: first velocity moment (n*upar).
    Apar: parallel magnetic vector potential.
    B: magnetic field magnitude (bmag).
    J: configuration space Jacobian (jacobgeo).
    1/J: reciprocal configuration space Jacobian (jacobgeo_inv).
    b_i: covariant components of the magnetic field unit vector.
  With '--extra fluct=y|yz' the turbulent part <dM1 d(dB^x/B)> is computed instead.
  """
  m1, apar, bmag, jacobgeo, jacobgeo_inv, b_i = gdatas
  return _radial_flux(m1, lambda: _radial_dB_over_B(apar, bmag, jacobgeo_inv, b_i), jacobgeo,
                      1.0, "fetch_part_flux_dB", **kwargs)

def fetch_energy_flux_dB(gdatas, **kwargs):
  """
  Radial magnetic flutter energy flux, Q_dB^x = (m/2) * M3 * dB^x/B, where
  M3 = M3par + M3perp.
  gdatas has (in this order):
    M3: third velocity moment.
    Apar: parallel magnetic vector potential.
    B: magnetic field magnitude (bmag).
    J: configuration space Jacobian (jacobgeo).
    1/J: reciprocal configuration space Jacobian (jacobgeo_inv).
    b_i: covariant components of the magnetic field unit vector.
  With '--extra fluct=y|yz' the turbulent part <dM3 d(dB^x/B)> is computed instead.
  """
  m3, apar, bmag, jacobgeo, jacobgeo_inv, b_i = gdatas
  mass = _get_ctx_val(m3, "mass", **kwargs)
  return _radial_flux(m3, lambda: _radial_dB_over_B(apar, bmag, jacobgeo_inv, b_i), jacobgeo,
                      0.5*mass, "fetch_energy_flux_dB", **kwargs)

def fetch_part_flux_em(gdatas, **kwargs):
  """
  Total radial particle flux, Gamma^x = Gamma_E^x + Gamma_dB^x.
  gdatas has (in this order): M0, M1, Apar, phi, B, J, 1/J, 1/(J*B), b_i.
  """
  m0, m1, apar, phi, bmag, jacobgeo, jacobgeo_inv, jacobtot_inv, b_i = gdatas
  return _add(fetch_part_flux_ExB([m0, phi, jacobgeo, jacobtot_inv, b_i], **kwargs),
              fetch_part_flux_dB([m1, apar, bmag, jacobgeo, jacobgeo_inv, b_i], **kwargs))

def fetch_part_flux_es(gdatas, **kwargs):
  """
  Total radial particle flux of an electrostatic simulation, Gamma^x = Gamma_E^x.
  gdatas has (in this order): M0, phi, J, 1/(J*B), b_i.
  """
  _warn_if_apar_dropped("part_flux", **kwargs)
  return fetch_part_flux_ExB(gdatas, **kwargs)

def fetch_energy_flux_em(gdatas, **kwargs):
  """
  Total radial energy flux, Q^x = Q_E^x + Q_dB^x.
  gdatas has (in this order): M2, M3, Apar, phi, B, J, 1/J, 1/(J*B), b_i.
  """
  m2, m3, apar, phi, bmag, jacobgeo, jacobgeo_inv, jacobtot_inv, b_i = gdatas
  return _add(fetch_energy_flux_ExB([m2, phi, jacobgeo, jacobtot_inv, b_i], **kwargs),
              fetch_energy_flux_dB([m3, apar, bmag, jacobgeo, jacobgeo_inv, b_i], **kwargs))

def fetch_energy_flux_es(gdatas, **kwargs):
  """
  Total radial energy flux of an electrostatic simulation, Q^x = Q_E^x.
  gdatas has (in this order): M2, phi, J, 1/(J*B), b_i.
  """
  _warn_if_apar_dropped("energy_flux", **kwargs)
  return fetch_energy_flux_ExB(gdatas, **kwargs)

# ------------------------------
# --- Transport coefficients ---
# ------------------------------
# Local (pointwise) radial diffusivities, the ratio of the local radial flux to
# the local radial gradient, normalized by <|grad x|^2> = g^xx so they are in
# m^2/s whatever the radial coordinate x is:
#   D = -Gamma/(g^xx dn/dx),  chi = -q/(n g^xx dT/dx),  q = Q - conv*T*Gamma,
# with conv set by '--extra conv=' (default 3/2). gk-transport computes the same
# coefficients from flux-surface and time averaged fluxes and profiles instead.

def fetch_D(gdatas, **kwargs):
  """
  Local radial particle diffusivity D = -Gamma/(g^xx dn/dx) (m^2/s).
  gdatas has: M0, part_flux, gij.
  """
  m0, part_flux, gij = gdatas
  den = _mul_scalar(_comp(gij, 0), _radial_derivative(m0, "fetch_D"))
  return _div(part_flux, den, -1.0)

def fetch_chi(gdatas, **kwargs):
  """
  Local radial heat diffusivity chi = -q/(n g^xx dT/dx) (m^2/s), with
  q = Q - conv*T*Gamma ('--extra conv=', default 3/2).
  gdatas has: M0, temp, part_flux, energy_flux, gij.
  """
  m0, temp, part_flux, energy_flux, gij = gdatas
  q = _heat_flux(temp, part_flux, energy_flux, **kwargs)
  den = _mul_scalar(_mul_scalar(m0, _comp(gij, 0)), _radial_derivative(temp, "fetch_chi"))
  return _div(q, den, -1.0)

def fetch_D_gB(gdatas, **kwargs):
  """
  Local radial particle diffusivity normalized by the gyro-Bohm diffusivity,
    D/D_gB = D L_n/(rho_s^2 c_s) = Gamma n/(g^xx^(3/2) (dn/dx)^2 rho_s^2 c_s),
  with L_n = -n/(sqrt(g^xx) dn/dx), of the first requested species:
    pgkyl gk-load-quantity -q D_gB -s ion,elc ...  (ions, kinetic electrons)
    pgkyl gk-load-quantity -q D_gB -s ion -e Ti_over_Te=1 ...  (adiabatic electrons)
  See _gyro_bohm_factors for rho_s^2 c_s. gdatas has one
  [M0, temp, part_flux, gij, bmag] list per species.
  """
  m0, _, part_flux, gij, _ = gdatas[0]
  return _gyro_bohm_normalized(_mul_scalar(part_flux, m0), _radial_derivative(m0, "fetch_D_gB"),
                               _comp(gij, 0), None, gdatas, **kwargs)

def fetch_chi_gB(gdatas, **kwargs):
  """
  Local radial heat diffusivity normalized by the gyro-Bohm diffusivity,
    chi/chi_gB = chi L_T/(rho_s^2 c_s) = q T/(n g^xx^(3/2) (dT/dx)^2 rho_s^2 c_s),
  with L_T = -T/(sqrt(g^xx) dT/dx) and q = Q - conv*T*Gamma ('--extra conv=',
  default 3/2), of the first requested species (listed as for fetch_D_gB).
  gdatas has one [M0, temp, part_flux, energy_flux, gij, bmag] list per species.
  """
  m0, temp, part_flux, energy_flux, gij, _ = gdatas[0]
  num = _mul_scalar(_heat_flux(temp, part_flux, energy_flux, **kwargs), temp)
  return _gyro_bohm_normalized(num, _radial_derivative(temp, "fetch_chi_gB"), _comp(gij, 0),
                               m0, gdatas, **kwargs)

# ------------------------------
# --- Phase space quantities ---
# ------------------------------

def load_distf(gdatas, **kwargs) -> GData:
  """
  Loader for the registry 'distf' quantity. Wraps load_gk_distf with defaults
  tailored to registry use: never interpolate (interp=0) and convert velocity
  coordinates (c2p_vel) on by default.

  Defaults can be overridden via --extra, e.g.:
    -e suffix=source      use <name>-<species>_source_<frame>.gkyl as input
    -e c2p_vel=0          disable velocity-space mapping
    -e mc2nu=1            apply non-uniform -> field-aligned position mapping
    -e mapc2p=1           apply position-space -> Cartesian/cylindrical mapping
    -e block=2            load only the 2nd block of a multi-block file
  """
  from postgkyl.commands.gk_distf import load_gk_distf
  from postgkyl.utils.gk_utils import dict_get_bool

  prefix = kwargs.get("path", "").rstrip("/") + "/" + kwargs.get("name", "")
  extra = kwargs.get("extra", {})

  return load_gk_distf(
    name=prefix, species=kwargs.get("species", ""), frame=int(kwargs.get("frame", 0)),
    suffix=str(extra.get("suffix", "")),
    use_c2p_vel=dict_get_bool(extra, "c2p_vel", True),
    use_mc2nu=dict_get_bool(extra, "mc2nu", False),
    use_mapc2p=dict_get_bool(extra, "mapc2p", False),
    block_idx=extra.get("block", None),
    interp=0,  # registry distf always works with non-interpolated DG data
  )

# -----------------------------
# --- Normalized quantities ---
# -----------------------------

def _make_fetch_q_norm(name: str):
  """
  Return a fetch function for a heat flux normalized by the free-streaming
  estimate n*T*vt:
    q_norm = q / (n*T*vt).
  gdatas has (in this order):
    1. M0: zeroth moment (density).
    2. q: the heat flux to normalize (in W/m^2).
    3. temp: temperature (in Joules).
    4. vt: thermal speed (in m/s).
  """
  def fetch(gdatas, **kwargs):
    m0, q, temp, vt = gdatas
    return _div(q, _mul_scalar(_mul_scalar(m0, temp), vt), like=m0)

  fetch.__name__ = f"fetch_q{name}_norm"
  return fetch

fetch_qpar_norm = _make_fetch_q_norm("par")
fetch_qperp_norm = _make_fetch_q_norm("perp")

def fetch_rho_over_lambda(gdatas, **kwargs):
  """
  Ratio of the species Larmor radius to its Debye length: rho/lambda_D.
  gdatas has:
    1. lambda_D: Debye length (m).
    2. rho: Larmor radius (m).
  """
  lambda_d, rho = gdatas
  return _div(rho, lambda_d)

def fetch_phi_norm(gdatas, **kwargs):
  """
  Normalized electrostatic potential.
  phi_norm = e*phi/T_e. Gdatas has:
    1. phi: electrostatic potential (phi).
    2. temp: temperature (temp).
  """
  phi, temp = gdatas
  return _div(phi, temp, gkc.GKYL_ELEMENTARY_CHARGE)
