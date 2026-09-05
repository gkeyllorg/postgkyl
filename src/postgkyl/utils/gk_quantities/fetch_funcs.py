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

"""
import numpy as np
import operator

from postgkyl.data import GData
from postgkyl.data.dg import get_num_basis
from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops
import postgkyl.utils.gkeyll_const as gkc

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
  from postgkyl.data.dg import get_num_basis
  ndim = gdata.get_num_dims()
  poly_order = int(gdata.ctx["poly_order"])
  basis_type = gdata.ctx["basis_type"]
  return get_num_basis(ndim, poly_order, basis_type)

def _empty_gdata_from_gdata(gdata) -> GData:
  """Allocate a zero-valued GData with the same grid/ctx as gdata."""
  out = GData(ctx=gdata.ctx)
  out.push(gdata.get_grid(), np.zeros_like(gdata.get_values()))
  return out

def _powsqrt_dg(gdata, exponent: float) -> GData:
  """
  pow(sqrt(f), exponent) of a single-component DG field. negative values are set to 1e-40.
  """

  out = _empty_gdata_from_gdata(gdata)

  dgops = GkeyllDGops()
  dgops.powsqrt(out, gdata, exponent)

  return out

def _make_fetch_comp(icomp: int):
  """Return a fetch function that extracts the comp-th physical component."""
  def fetch(gdatas, **kw):
    g = gdatas[0].get_grid()
    nb = _get_num_basis_from_gdata(gdatas[0])
    comp = [icomp,icomp] if icomp is not None else [0,int(gdatas[0].get_num_comps()/nb)]
    v = gdatas[0].get_values()[..., comp[0]*nb:(comp[1]+1)*nb].copy()
    out = GData(ctx=gdatas[0].ctx)
    out.push(g, v)
    return out
  # end
  fetch.__name__ = f"fetch_comp{icomp}" if icomp is not None else f"fetch_compAll"
  return fetch

def _make_fetch_sick_addsub_sjcl(si: int, ck: int, sj: int, cl: int, op):
  """
  Return a fetch function that does:
    (k-th component of the i-th source) op (l-th component of the j-th source)
  """
  def fetch(gdatas, **kwargs):
    gd_l = gdatas[si]
    gd_r = gdatas[sj]

    nb_l = _get_num_basis_from_gdata(gd_l)
    nb_r = _get_num_basis_from_gdata(gd_r)
    if not nb_l == nb_r:
      raise ValueError(f"Datasets have different basis")

    vals_l = gd_l.get_values()[..., ck*nb_l:(ck+1)*nb_l]
    vals_r = gd_r.get_values()[..., cl*nb_r:(cl+1)*nb_r]

    out = GData(ctx=gd_l.ctx)
    out.push(gd_l.get_grid(), op(vals_l,vals_r))

    return out
  # end
  fetch.__name__ = f"fetch_s{si}c{ck}_{op.__name__}_s{sj}c{cl}"
  return fetch

def _make_fetch_sick_mul_sjcl(si: int, ck: int, sj: int, cl: int):
  """
  Return a fetch function that multiplies the k-th component of the i-th
  source/dataset by the l-th component of the j-th source.
  """
  def fetch(gdatas, **kwargs):
    gd_l = gdatas[si]
    gd_r = gdatas[sj]

    nb_l = _get_num_basis_from_gdata(gd_l)
    nb_r = _get_num_basis_from_gdata(gd_r)
    if not nb_l == nb_r:
      raise ValueError(f"Datasets have different basis")

    vals_l = gd_l.get_values()
    out_shape = list(vals_l.shape)
    out_shape[-1] = nb_l
  
    out = GData(ctx=gd_l.ctx)
    out.push(gd_l.get_grid(), np.zeros(out_shape, dtype=vals_l.dtype))
  
    dgops = GkeyllDGops()
    dgops.multiply(0, out, ck, gd_l, cl, gd_r)
  
    return out
  # end
  fetch.__name__ = f"fetch_s{si}c{ck}_mul_s{sj}c{cl}"
  return fetch

def _make_fetch_sick_div_sjcl(si: int, ck: int, sj: int, cl: int):
  """
  Return a fetch function that divides the k-th component of the i-th
  source/dataset by the l-th component of the j-th source.
  """
  def fetch(gdatas, **kwargs):
    gd_l = gdatas[si]
    gd_r = gdatas[sj]

    nb_l = _get_num_basis_from_gdata(gd_l)
    nb_r = _get_num_basis_from_gdata(gd_r)
    if not nb_l == nb_r:
      raise ValueError(f"Datasets have different basis")

    vals_l = gd_l.get_values()
    out_shape = list(vals_l.shape)
    out_shape[-1] = nb_l
  
    out = GData(ctx=gd_l.ctx)
    out.push(gd_l.get_grid(), np.zeros(out_shape, dtype=vals_l.dtype))

    dgops = GkeyllDGops()
    dgops.invert(0, out, cl, gd_r)
    dgops.multiply(0, out, ck, gd_l, 0, out)
  
    return out
  # end
  fetch.__name__ = f"fetch_s{si}c{ck}_div_s{sj}c{cl}"
  return fetch

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

  buff = _empty_gdata_from_gdata(scalar) # Positive term in AxB.
  out = _empty_gdata_from_gdata(scalar) # Negative term in AxB.

  dgops = GkeyllDGops()
  lower, upper = scalar.get_bounds()
  cells = scalar.get_num_cells()
  if calc_term[0]:
    # Compute derivatives of the scalar field.
    dx = (upper[diff_dir_pos] - lower[diff_dir_pos])/cells[diff_dir_pos]
    dgops.differentiate(diff_dir_pos, 1,  dx, 0, buff, 0, scalar)
    # Multiply by b_i.
    dgops.multiply(0, buff, bi_c_pos, b_i, 0, buff)

  if calc_term[1]:
    # Compute derivatives of the scalar field.
    dx = (upper[diff_dir_neg] - lower[diff_dir_neg])/cells[diff_dir_neg]
    dgops.differentiate(diff_dir_neg, 1, -dx, 0, out , 0, scalar)
    # Multiply by b_i.
    dgops.multiply(0, out , bi_c_neg, b_i, 0, out )

  # Add the two terms to form the comp-th component of b x grad(f).
  out.set_values(buff.get_values() + out.get_values())

  # Divide by the Jacobian factor of the curvilinear cross product.
  dgops.multiply(0, out, 0, out, 0, jacobtot_inv)

  return out

# Functions to extract a components.
fetch_s0cAll = _make_fetch_comp(None)
fetch_s0c0 = _make_fetch_comp(0)
fetch_s0c1 = _make_fetch_comp(1)
fetch_s0c2 = _make_fetch_comp(2)
fetch_s0c3 = _make_fetch_comp(3)

# Functions to add two components.
fetch_s0c0_add_s1c0 = _make_fetch_sick_addsub_sjcl(0,0,1,0,operator.add)
fetch_s0c2_add_s0c3 = _make_fetch_sick_addsub_sjcl(0,2,0,3,operator.add)

# Functions to subtract two components.
fetch_s0c0_sub_s1c0 = _make_fetch_sick_addsub_sjcl(0,0,1,0,operator.sub)

# Functions to multiply two components.
fetch_s0c0_mul_s1c0 = _make_fetch_sick_mul_sjcl(0,0,1,0)
fetch_s0c0_mul_s0c1 = _make_fetch_sick_mul_sjcl(0,0,0,1)

# Functions to divide two components.
fetch_s1c0_div_s0c0 = _make_fetch_sick_div_sjcl(1,0,0,0)

# ------------------------------------------
# --- Plasma moments (species-dependent) ---
# ------------------------------------------

def fetch_M1_from_H(gdatas, **kwargs):
  """
  M1 from the Hamiltonian moments (Hmom).
  """
  hmom = gdatas[0]
  mass = _get_ctx_val(hmom, "mass", **kwargs)
  nb = _get_num_basis_from_gdata(hmom)
  vals = hmom.get_values()

  m1 = GData(ctx=hmom.ctx)
  m1.push(hmom.get_grid(), np.zeros_like(vals[..., :nb]))

  dgops = GkeyllDGops()
  dgops.multiply(0, m1, 0, hmom, 1, hmom)

  m1.set_values(m1.get_values() / mass)
  return m1

def fetch_Tpar_from_BiMax(gdatas, **kwargs):
  """
  Tpar from BiMaxwellian moments.
  """
  Tpar = fetch_s0c2(gdatas)

  bimax = gdatas[0]
  mass = _get_ctx_val(bimax, "mass", **kwargs)
  Tpar.set_values(mass * Tpar.get_values())
  return Tpar

def fetch_Tpar_from_M0_M1_M2par(gdatas, **kwargs):
  """
  upar*M1 + M0*Tpar/m = M2par.
  Tpar = m * (M2par - upar*M1) / M0.
  """
  m0, m1, m2par = gdatas
  dgops = GkeyllDGops()

  m0_inv = _empty_gdata_from_gdata(m0)
  upar   = _empty_gdata_from_gdata(m0)
  Tpar   = _empty_gdata_from_gdata(m0)

  dgops.invert(0, m0_inv, 0, m0)
  dgops.multiply(0, upar, 0, m1, 0, m0_inv)
  dgops.multiply(0, upar, 0, upar, 0, m1)

  m2par_val = m2par.get_values()
  um1_val = upar.get_values()
  
  mass = _get_ctx_val(m0, "mass", **kwargs)
  Tpar.set_values(mass * (m2par_val - um1_val))
  dgops.multiply(0, Tpar, 0, Tpar, 0, m0_inv)
  return Tpar

def fetch_Tperp_from_BiMax(gdatas, **kwargs):
  """
  Tperp from BiMaxwellian moments.
  """
  Tperp = fetch_s0c3(gdatas)

  bimax = gdatas[0]
  mass = _get_ctx_val(bimax, "mass", **kwargs)
  Tperp.set_values(mass * Tperp.get_values())
  return Tperp

def fetch_Tperp_from_M0_M2perp(gdatas, **kwargs):
  """
  Tperp = 0.5 * mass * (M2perp / M0).
  """
  Tperp = fetch_s1c0_div_s0c0(gdatas)

  m0 = gdatas[0]
  mass = _get_ctx_val(m0, "mass", **kwargs)
  Tperp.set_values(0.5 * mass * Tperp.get_values())
  return Tperp

def fetch_temp_from_Max(gdatas, **kwargs):
  """
  temp from Maxwellian moments.
  """
  temp = fetch_s0c2(gdatas)

  maxmom = gdatas[0]
  mass = _get_ctx_val(maxmom, "mass", **kwargs)
  temp.set_values(mass * temp.get_values())
  return temp

def fetch_temp_from_Tpar_Tperp(gdatas, **kwargs):
  """
  temp = (Tpar + 2*Tperp) / 3.
  """
  Tpar, Tperp = gdatas

  temp = _empty_gdata_from_gdata(Tpar)

  Tpar_val  = Tpar.get_values()
  Tperp_val = Tperp.get_values()
  
  temp.set_values((Tpar_val + 2.0*Tperp_val)/3.0)
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
  nb = _get_num_basis_from_gdata(maxmom)
  vals = maxmom.get_values()[..., :nb]
  
  press = GData(ctx=maxmom.ctx)
  press.push(maxmom.get_grid(), np.zeros_like(vals))

  dgops = GkeyllDGops()
  dgops.multiply(0, press, 0, maxmom, 2, maxmom)

  mass = _get_ctx_val(maxmom, "mass", **kwargs)
  press.set_values(mass * press.get_values())
  return press

def fetch_press_from_BiMax(gdatas, **kwargs):
  """
  Pressure from BiMaxwellian moments.
  press = den * (Tpar + 2*Tperp) / 3.
  """
  bimax = gdatas[0]
  nb = _get_num_basis_from_gdata(bimax)
  vals = bimax.get_values()

  mass = _get_ctx_val(bimax, "mass", **kwargs)
  Tpar_vals  = vals[..., 2*nb:3*nb]
  Tperp_vals = vals[..., 3*nb:4*nb]
  temp_vals  = mass*(Tpar_vals + 2.0 * Tperp_vals)/3.0

  press = GData(ctx=bimax.ctx)
  press.push(bimax.get_grid(), temp_vals.copy())

  dgops = GkeyllDGops()
  dgops.multiply(0, press, 0, bimax, 0, press)

  return press

def fetch_press_p(gdatas, **kwargs):
  """
  Perpendicular/parallel pressure in J/m^3.
  p_p = n * T_p.
  """
  m0 = gdatas[0]
  Tp = gdatas[1]
  
  dgops = GkeyllDGops()
  press_p = _empty_gdata_from_gdata(m0)
  dgops.multiply(0, press_p, 0, m0, 0, Tp)

  return press_p

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
    mass = _get_ctx_val(m3, "mass", **kwargs)

    out = _empty_gdata_from_gdata(m3)
    out.set_values(0.5*mass*m3.get_values())
    return out
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

    dgops = GkeyllDGops()

    m0_inv = _empty_gdata_from_gdata(m0)
    dgops.invert(0, m0_inv, 0, m0)

    upar = _empty_gdata_from_gdata(m0)
    dgops.multiply(0, upar, 0, m1, 0, m0_inv)

    # u*M2par or u*M2perp.
    u_m2 = _empty_gdata_from_gdata(m0)
    dgops.multiply(0, u_m2, 0, upar, 0, m2)

    if is_par:
      # u^2*M1, which equals u^3*M0.
      u_sq = _empty_gdata_from_gdata(m0)
      dgops.multiply(0, u_sq, 0, upar, 0, upar)

      u_sq_m1 = _empty_gdata_from_gdata(m0)
      dgops.multiply(0, u_sq_m1, 0, u_sq, 0, m1)

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
  mass = _get_ctx_val(temp, "mass", **kwargs)

  temp_over_m = _empty_gdata_from_gdata(temp)
  temp_over_m.set_values(temp.get_values()/mass)

  return _powsqrt_dg(temp_over_m, 1.0)

def fetch_larmor_radius(gdatas, **kwargs):
  """
  Species Larmor (gyro-)radius: rho = sqrt(m*T)/(|q|*B). gdatas has:
    1. B: magnetic field magnitude (bmag).
    2. temp: temperature (in Joules).
  """
  bmag, temp = gdatas
  mass = _get_ctx_val(temp, "mass", **kwargs)
  charge = abs(_get_ctx_val(temp, "charge", **kwargs))

  mT = _empty_gdata_from_gdata(temp)
  mT.set_values(temp.get_values() * mass)
  sqrt_mT = _powsqrt_dg(mT, 1.0)

  qB = _empty_gdata_from_gdata(bmag)
  qB.set_values(bmag.get_values() * charge)

  dgops = GkeyllDGops()

  qB_inv = _empty_gdata_from_gdata(bmag)
  dgops.invert(0, qB_inv, 0, qB)

  out = _empty_gdata_from_gdata(bmag)
  dgops.multiply(0, out, 0, sqrt_mT, 0, qB_inv)

  return out

def fetch_debye_length(gdatas, **kwargs):
  """
  Species-wise Debye length: lambda_D = sqrt(eps0*T/(n*q^2)). gdatas has:
    1. M0: zeroth moment (density).
    2. temp: temperature (in Joules).
  """
  m0, temp = gdatas
  charge = _get_ctx_val(temp, "charge", **kwargs)
  eps0 = gkc.GKYL_EPSILON0

  eps0T = _empty_gdata_from_gdata(temp)
  eps0T.set_values(temp.get_values() * eps0)

  nq2 = _empty_gdata_from_gdata(m0)
  nq2.set_values(m0.get_values() * charge**2)

  dgops = GkeyllDGops()

  nq2_inv = _empty_gdata_from_gdata(m0)
  dgops.invert(0, nq2_inv, 0, nq2)

  sq = _empty_gdata_from_gdata(temp)
  dgops.multiply(0, sq, 0, eps0T, 0, nq2_inv)

  return _powsqrt_dg(sq, 1.0)

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

  if len(elcs) != 1:
    raise ValueError(f"{quantity}: expected exactly one negatively charged (electron) species "
                     f"but found {len(elcs)} in {list(species_names)}.")
    
  if not ions:
    raise ValueError(f"{quantity}: found no positively charged (ion) species in {list(species_names)}.")

  return elcs[0], ions

def _weighted_sum(entries, weights, comp: int):
  """
  Sum the comp-th source of each species, each scaled by a scalar weight.
  """
  out = _empty_gdata_from_gdata(entries[0]["srcs"][comp])
  total = sum(w*e["srcs"][comp].get_values() for e, w in zip(entries, weights))
  out.set_values(total)
  return out

def _fetch_c_s_ion_acoustic(gdatas, **kwargs):
  """
  Ion-acoustic sound speed (wave perspective), for the Bohm criterion and
  sheath/presheath matching:
    c_s = sqrt( T_e * sum_j(n_j*Z_j^2/m_j) / sum_j(n_j*Z_j) )
  summing over the ion species j, with Z_j = q_j/e the ion charge state.
  """
  elc, ions = _split_elc_ions(gdatas, "fetch_c_s(kind=ion_acoustic)", **kwargs)

  e = gkc.GKYL_ELEMENTARY_CHARGE
  charge_states = [ion["charge"]/e for ion in ions]

  # sum_j n_j*Z_j^2/m_j and sum_j n_j*Z_j, both linear in the densities (M0).
  numer = _weighted_sum(ions, [z**2/ion["mass"] for z, ion in zip(charge_states, ions)], 0)
  denom = _weighted_sum(ions, charge_states, 0)

  dgops = GkeyllDGops()

  denom_inv = _empty_gdata_from_gdata(denom)
  dgops.invert(0, denom_inv, 0, denom)

  # T_e * numer/denom.
  c_s_sq = _empty_gdata_from_gdata(numer)
  dgops.multiply(0, c_s_sq, 0, numer, 0, denom_inv)
  dgops.multiply(0, c_s_sq, 0, c_s_sq, 0, elc["srcs"][1])

  return _powsqrt_dg(c_s_sq, 1.0)

def _fetch_c_s_thermo(gdatas, **kwargs):
  """
  Thermodynamic sound speed (bulk fluid perspective), for Mach numbers and
  acoustic propagation in the core/SOL:
    c_s = sqrt( (gamma_e*n_e*T_e + sum_j(gamma_j*n_j*T_j)) / sum_j(n_j*m_j) )
  summing over the ion species j. 
  Default: gamma_e=1, gamma_i=3, but these can be set via '--extra'.
  """
  elc, ions = _split_elc_ions(gdatas, "fetch_c_s(kind=thermo)", **kwargs)

  gamma_e = float(kwargs.get("gamma_e", 1.0))
  gamma_i = float(kwargs.get("gamma_i", 3.0))

  dgops = GkeyllDGops()

  # gamma_e*n_e*T_e + sum_j gamma_j*n_j*T_j. Each n*T is a weak product.
  numer = _empty_gdata_from_gdata(elc["srcs"][0])
  dgops.multiply(0, numer, 0, elc["srcs"][0], 0, elc["srcs"][1])
  numer.set_values(gamma_e*numer.get_values())

  press_j = _empty_gdata_from_gdata(elc["srcs"][0])
  for ion in ions:
    dgops.multiply(0, press_j, 0, ion["srcs"][0], 0, ion["srcs"][1])
    numer.set_values(numer.get_values() + gamma_i*press_j.get_values())

  # sum_j n_j*m_j, the ion mass density; linear in the densities.
  denom = _weighted_sum(ions, [ion["mass"] for ion in ions], 0)

  denom_inv = _empty_gdata_from_gdata(denom)
  dgops.invert(0, denom_inv, 0, denom)

  c_s_sq = _empty_gdata_from_gdata(numer)
  dgops.multiply(0, c_s_sq, 0, numer, 0, denom_inv)

  return _powsqrt_dg(c_s_sq, 1.0)

def fetch_c_s(gdatas, **kwargs):
  """
  Sound speed (m/s), combining the electrons and every ion species. gdatas has
  one [M0, temp] pair per species, in the order they were requested, e.g.
    pgkyl gk-load-quantity -q c_s -s elc,ion1,ion2 ...
  Electrons and ions are told apart by the sign of each species' charge
  attribute, so the species may be named anything.

  Two definitions are available through '--extra kind=<kind>':
    ion_acoustic (default): the wave/Bohm-criterion sound speed,
      c_s = sqrt(T_e*sum_j(n_j*Z_j^2/m_j)/sum_j(n_j*Z_j)).
    thermo: the bulk-fluid sound speed,
      c_s = sqrt((gamma_e*n_e*T_e + sum_j(gamma_j*n_j*T_j))/sum_j(n_j*m_j)),
      with gamma_e and gamma_i settable via '--extra' (default 1 and 3).
  """
  c_s_kinds = {
    "ion_acoustic": _fetch_c_s_ion_acoustic,
    "thermo": _fetch_c_s_thermo,
  }
  kind = str(kwargs.get("kind", "thermo"))
  if kind not in c_s_kinds:
    raise ValueError(f"fetch_c_s: unknown kind '{kind}'. Select one with '--extra kind=<kind>' "
                     f"from: {', '.join(sorted(c_s_kinds))}.")
  # end
  return c_s_kinds[kind](gdatas, **kwargs)

def fetch_beta_from_bmag_press(gdatas, **kwargs):
  """
  beta = 2*mu_0*press/bmag^2
  """
  bmag, press = gdatas

  dgops = GkeyllDGops()

  bmag_sq = _empty_gdata_from_gdata(bmag)
  out = _empty_gdata_from_gdata(bmag)

  dgops.multiply(0, bmag_sq, 0, bmag, 0, bmag)

  dgops.invert(0, out, 0, bmag_sq)
  dgops.multiply(0, out, 0, press, 0, out)

  out_val = out.get_values()
  
  mu0 = gkc.GKYL_MU0
  out.set_values(2.0*mu0*out_val)
  return out

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
  if "dir" not in kwargs:
    raise KeyError("fetch_ExB_vel: select the j-th component with '--extra dir=j' (0-index).")

  bmag = gdatas[0]
  jacobtot_inv = gdatas[1]
  phi = gdatas[2]
  b_i = gdatas[3]

  # k-th component of b x grad(phi)/B.
  out = _b_cross_grad_div_B_component(phi, jacobtot_inv, b_i, kwargs["dir"])

  return out

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
  if "dir" not in kwargs:
    raise KeyError("fetch_gradB_vel: select the j-th component with '--extra dir=j' (0-index).")

  bmag = gdatas[0]
  jacobtot_inv = gdatas[1]
  Tperp = gdatas[2]
  b_i = gdatas[3]

  # k-th component of b x grad(B)/B.
  out = _b_cross_grad_div_B_component(bmag, jacobtot_inv, b_i, kwargs["dir"])

  dgops = GkeyllDGops()
  # Multiply by Tperp.
  dgops.multiply(0, out, 0, Tperp, 0, out)

  # Divide by B.
  denom_inv = _empty_gdata_from_gdata(bmag)
  dgops.invert(0, denom_inv, 0, bmag)
  dgops.multiply(0, out, 0, out, 0, denom_inv)

  # Divide by the species charge.
  charge = _get_ctx_val(Tperp, "charge", **kwargs)
  out.set_values(out.get_values()/charge)

  return out

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
  if "dir" not in kwargs:
    raise KeyError("fetch_diamag_vel: select the j-th component with '--extra dir=j' (0-index).")

  bmag = gdatas[0]
  jacobtot_inv = gdatas[1]
  m0 = gdatas[2]
  pressperp = gdatas[3]
  b_i = gdatas[4]

  # k-th component of b x grad(p) / B.
  out = _b_cross_grad_div_B_component(pressperp, jacobtot_inv, b_i, kwargs["dir"])

  dgops = GkeyllDGops()
  # Divide by n
  denom_inv = _empty_gdata_from_gdata(bmag)
  dgops.invert(0, denom_inv, 0, m0)
  dgops.multiply(0, out, 0, out, 0, denom_inv)

  # Divide by the species charge.
  charge = _get_ctx_val(pressperp, "charge", **kwargs)
  out.set_values(out.get_values()/charge)

  return out

# -------------------------------------
# --- Magnetic field perturbations ----
# -------------------------------------

# Component of the metric tensor g_ij holding the (k,l) entry.
_G_IJ_COMP = {(0,0): 0, (0,1): 1, (0,2): 2, (1,1): 3, (1,2): 4, (2,2): 5}

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
  if "dir" not in kwargs:
    raise KeyError("fetch_dB_perp_dual: select the k-th component with '--extra dir=k' (0-index).")

  apar, jacobgeo_inv, b_i = gdatas
  comp = int(kwargs["dir"])
  if not 0 <= comp < 3:
    raise KeyError("fetch_dB_perp_dual: component must be >= 0 and < 3.")

  # Dimension holding each of x, y and z, or None where a reduced simulation
  # does not carry it: a 2x run holds (x,z) and a 1x run only z.
  cdim = apar.get_num_dims()
  axis = (0 if cdim > 1 else None, 1 if cdim > 2 else None, cdim-1)

  dgops = GkeyllDGops()
  lower, upper = apar.get_bounds()
  cells = apar.get_num_cells()

  prod = _empty_gdata_from_gdata(apar)
  term = _empty_gdata_from_gdata(apar)
  out = _empty_gdata_from_gdata(apar)
  for diff_dir, b_comp, sign in (((comp+1) % 3, (comp+2) % 3,  1.0),
                                 ((comp+2) % 3, (comp+1) % 3, -1.0)):
    dim = axis[diff_dir]
    if dim is None:
      continue
    # d(Apar*b_<b_comp>)/dx^<diff_dir>.
    dgops.multiply(0, prod, 0, apar, b_comp, b_i)
    dgops.differentiate(dim, 1, (upper[dim] - lower[dim])/cells[dim], 0, term, 0, prod)
    out.set_values(out.get_values() + sign*term.get_values())

  # Divide by the Jacobian factor of the curvilinear curl.
  dgops.multiply(0, out, 0, out, 0, jacobgeo_inv)
  return out

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
  if "dir" not in kwargs:
    raise KeyError("fetch_dB_perp: select the k-th component with '--extra dir=k' (0-index).")

  apar, g_ij = gdatas[0], gdatas[3]
  comp = int(kwargs["dir"])
  if not 0 <= comp < 3:
    raise KeyError("fetch_dB_perp: component must be >= 0 and < 3.")

  dgops = GkeyllDGops()

  term = _empty_gdata_from_gdata(apar)
  out = _empty_gdata_from_gdata(apar)
  for j in range(3):
    dB_up = fetch_dB_perp_dual(gdatas[:3], dir=j)
    dgops.multiply(0, term, _G_IJ_COMP[(min(comp,j), max(comp,j))], g_ij, 0, dB_up)
    out.set_values(out.get_values() + term.get_values())

  return out

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
  apar = gdatas[0]

  dgops = GkeyllDGops()

  buff = _empty_gdata_from_gdata(apar)
  mag_sq = _empty_gdata_from_gdata(apar)
  for comp in range(3):
    dgops.multiply(0, buff, 0, fetch_dB_perp(gdatas, dir=comp),
                   0, fetch_dB_perp_dual(gdatas[:3], dir=comp))
    mag_sq.set_values(mag_sq.get_values() + buff.get_values())

  return _powsqrt_dg(mag_sq, 1.0)

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
  if "dir" not in kwargs:
    raise KeyError("fetch_B_equilibrium: select the k-th component with '--extra dir=k' (0-index).")

  bmag, b_i = gdatas
  comp = int(kwargs["dir"])
  if not 0 <= comp < 3:
    raise KeyError("fetch_B_equilibrium: component must be >= 0 and < 3.")

  out = _empty_gdata_from_gdata(bmag)
  GkeyllDGops().multiply(0, out, comp, b_i, 0, bmag)
  return out

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

  out = fetch_B_equilibrium([bmag, b_i], **kwargs)
  dB = fetch_dB_perp([apar, jacobgeo_inv, b_i, g_ij], **kwargs)

  out.set_values(out.get_values() + dB.get_values())
  return out

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
  if "dir" not in kwargs:
    raise KeyError("fetch_B_dual_equilibrium: select the k-th component with "
                   "'--extra dir=k' (0-index).")

  bmag, g_ij = gdatas
  comp = int(kwargs["dir"])
  if not 0 <= comp < 3:
    raise KeyError("fetch_B_dual_equilibrium: component must be >= 0 and < 3.")

  out = _empty_gdata_from_gdata(bmag)
  if comp != 2:
    return out

  nb = _get_num_basis_from_gdata(g_ij)
  g_33_comp = _G_IJ_COMP[(2,2)]
  g_33 = _empty_gdata_from_gdata(bmag)
  g_33.set_values(g_ij.get_values()[..., g_33_comp*nb:(g_33_comp + 1)*nb])

  GkeyllDGops().multiply(0, out, 0, bmag, 0, _powsqrt_dg(g_33, -1.0))
  return out

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

  out = fetch_B_dual_equilibrium([bmag, g_ij], **kwargs)
  dB = fetch_dB_perp_dual([apar, jacobgeo_inv, b_i], **kwargs)

  out.set_values(out.get_values() + dB.get_values())
  return out

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
  bmag = gdatas[1]

  dgops = GkeyllDGops()

  buff = _empty_gdata_from_gdata(bmag)
  mag_sq = _empty_gdata_from_gdata(bmag)
  for comp in range(3):
    dgops.multiply(0, buff, 0, fetch_B_tot(gdatas, dir=comp),
                   0, fetch_B_tot_dual(gdatas, dir=comp))
    mag_sq.set_values(mag_sq.get_values() + buff.get_values())

  return _powsqrt_dg(mag_sq, 1.0)

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

    dgops = GkeyllDGops()

    # n*T*vt.
    denom = _empty_gdata_from_gdata(m0)
    dgops.multiply(0, denom, 0, m0, 0, temp)
    dgops.multiply(0, denom, 0, denom, 0, vt)

    denom_inv = _empty_gdata_from_gdata(m0)
    dgops.invert(0, denom_inv, 0, denom)

    out = _empty_gdata_from_gdata(m0)
    dgops.multiply(0, out, 0, q, 0, denom_inv)
    return out

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

  dgops = GkeyllDGops()

  lambda_d_inv = _empty_gdata_from_gdata(lambda_d)
  dgops.invert(0, lambda_d_inv, 0, lambda_d)

  out = _empty_gdata_from_gdata(rho)
  dgops.multiply(0, out, 0, rho, 0, lambda_d_inv)

  return out

def fetch_phi_norm(gdatas, **kwargs):
  """
  Normalized electrostatic potential.
  phi_norm = e*phi/T_e. Gdatas has:
    1. phi: electrostatic potential (phi).
    2. temp: temperature (temp).
  """
  phi, temp = gdatas
  e = gkc.GKYL_ELEMENTARY_CHARGE

  dgops = GkeyllDGops()

  temp_inv = _empty_gdata_from_gdata(temp)
  dgops.invert(0, temp_inv, 0, temp)

  out = _empty_gdata_from_gdata(phi)
  dgops.multiply(0, out, 0, phi, 0, temp_inv)

  out.set_values(out.get_values() * e)

  return out