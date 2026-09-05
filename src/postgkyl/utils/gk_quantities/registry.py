"""
Registry of pre-named gyrokinetic quantities.

You can access the "gk_quant_registry" instance to get a list
of all available gyrokinetic quantities and manipulate them.

Each entry is an instance of the GkQuantity class.
"""

import postgkyl.utils.gk_quantities.fetch_funcs as ff
from .gkquantity import GkQuantity, GkQuantityRegistry

# Instance that will hold all available gyrokinetic quantities.
gk_quant_registry: GkQuantityRegistry = GkQuantityRegistry()

# ------------------- Register quantities -------------------

# -----------------------------------
# --- Scalar geometric quantities ---
# -----------------------------------

# Configuration space Jacobian (interior).
_geo_int_jacobgeo : GkQuantity = GkQuantity(
  name = "geo_int_jacobgeo",
  source = [["geo_int_jacobgeo"],],
  fetch_func = [ff.fetch_s0c0],
  label = r"$J$",
  is_geo = True
)
gk_quant_registry.register(_geo_int_jacobgeo)

# Reciprocal of configuration space Jacobian (interior).
_geo_int_jacobgeo_inv : GkQuantity = GkQuantity(
  name = "geo_int_jacobgeo_inv",
  source = [["geo_int_jacobgeo_inv"],],
  fetch_func = [ff.fetch_s0c0],
  label = r"$J^{-1}$",
  is_geo = True
)
gk_quant_registry.register(_geo_int_jacobgeo_inv)

# Total Jacobian (interior).
_geo_int_jacobtot : GkQuantity = GkQuantity(
  name = "geo_int_jacobtot",
  source = [["geo_int_jacobtot"],],
  fetch_func = [ff.fetch_s0c0],
  label = r"$J$",
  is_geo = True
)
gk_quant_registry.register(_geo_int_jacobtot)

# Reciprocal of Jacobian times bmag (interior).
_geo_int_jacobtot_inv : GkQuantity = GkQuantity(
  name = "geo_int_jacobtot_inv",
  source = [["geo_int_jacobtot_inv"],],
  fetch_func = [ff.fetch_s0c0],
  label = r"$(J B)^{-1}$",
  is_geo = True
)
gk_quant_registry.register(_geo_int_jacobtot_inv)

# Magnetic field magnitude (interior).
_geo_int_bmag : GkQuantity = GkQuantity(
  name = "geo_int_bmag",
  source = [["geo_int_bmag"],],
  fetch_func = [ff.fetch_s0c0],
  label = r"$B$ (T)",
  is_geo = True
)
gk_quant_registry.register(_geo_int_bmag)

# -----------------------------------
# --- Vector geometric quantities ---
# -----------------------------------

# Covariant components of magnetic field unit vector (interior).
_geo_int_b_i : GkQuantity = GkQuantity(
  name = "geo_int_b_i",
  source = [["geo_int_b_i"],],
  fetch_func = [ff.fetch_s0cAll],
  label = r"$b_%s$",
  is_vector = True,
  is_geo = True
)
gk_quant_registry.register(_geo_int_b_i)

# -----------------------------------
# --- Tensor geometric quantities ---
# -----------------------------------

# Covariant metric coefficients, in the order g_11,g_12,g_13,g_22,g_23,g_33 (interior).
_geo_int_g_ij : GkQuantity = GkQuantity(
  name = "geo_int_g_ij",
  source = [["geo_int_g_ij"],],
  fetch_func = [ff.fetch_s0cAll],
  label = r"$g_{ij}$",
  is_tensor = True,
  is_geo = True
)
gk_quant_registry.register(_geo_int_g_ij)

# --------------------------------------------
# --- Field quantities (species-dependent) ---
# --------------------------------------------

# Electrostatic potential.
_field : GkQuantity = GkQuantity(
  name = "field",
  source = [["field"],],
  fetch_func = [ff.fetch_s0c0],
  label = r"$\phi$ (V)",
  is_time_dep = True,
)
gk_quant_registry.register(_field)

# ------------------------------------------
# --- Plasma moments (species-dependent) ---
# ------------------------------------------

# Zeroth velocity moment.
_M0 : GkQuantity = GkQuantity(
  name = "M0",
  source = [["M0"], ["M0M1M2"], ["M0M1M2parM2perp"], ["MaxwellianMoments"], ["BiMaxwellianMoments"], ["HamiltonianMoments"],],
  fetch_func = [ff.fetch_s0c0, ff.fetch_s0c0, ff.fetch_s0c0, ff.fetch_s0c0, ff.fetch_s0c0, ff.fetch_s0c0],
  label = r"$M_{0%s}$ (m$^{-3}$)",
  is_species_dep = True,
  is_time_dep = True
)
gk_quant_registry.register(_M0)

# First velocity moment.
_M1 : GkQuantity = GkQuantity(
  name = "M1",
  source = [["M1"], ["M0M1M2"], ["M0M1M2parM2perp"], ["MaxwellianMoments"], ["BiMaxwellianMoments"], ["HamiltonianMoments"],],
  fetch_func = [ff.fetch_s0c0, ff.fetch_s0c1, ff.fetch_s0c1, ff.fetch_s0c0_mul_s0c1, ff.fetch_s0c0_mul_s0c1, ff.fetch_M1_from_H],
  label = r"$M_{1%s}$ (m$^{-2}$/s)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_M1)

# Second parallel velocity moment.
_M2par : GkQuantity = GkQuantity(
  name = "M2par",
  source = [["M2par"], ["M0M1M2parM2perp"], ["M2","M2perp"]],
  fetch_func = [ff.fetch_s0c0, ff.fetch_s0c2, ff.fetch_s0c0_sub_s1c0],
  label = r"$M_{2\parallel%s}$ (m$^{-1}$/s$^2$)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_M2par)

# Second perpendicular velocity moment.
_M2perp : GkQuantity = GkQuantity(
  name = "M2perp",
  source = [["M2perp"], ["M0M1M2parM2perp"], ["M2","M2par"]],
  fetch_func = [ff.fetch_s0c0, ff.fetch_s0c3, ff.fetch_s0c0_sub_s1c0],
  label = r"$M_{2\perp%s}$ (m$^{-1}$/s$^2$)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_M2perp)

# Second velocity moment.
_M2 : GkQuantity = GkQuantity(
  name = "M2",
  source = [["M2"], ["M0M1M2"], ["M0M1M2parM2perp"], [_M2par,_M2perp],],
  fetch_func = [ff.fetch_s0c0, ff.fetch_s0c2, ff.fetch_s0c2_add_s0c3, ff.fetch_s0c0_add_s1c0,],
  label = r"$M_{2%s}$ (m$^{-1}$/s$^2$)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_M2)

# Third parallel velocity moment.
_M3par : GkQuantity = GkQuantity(
  name = "M3par",
  source = [["M3par"]],
  fetch_func = [ff.fetch_s0c0],
  label = r"$M_{3\parallel%s}$ (1/s$^3$)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_M3par)

# Third perpendicular velocity moment.
_M3perp : GkQuantity = GkQuantity(
  name = "M3perp",
  source = [["M3perp"]],
  fetch_func = [ff.fetch_s0c0],
  label = r"$M_{3\perp%s}$ (1/s$^3$)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_M3perp)

# Third velocity moment.
_M3 : GkQuantity = GkQuantity(
  name = "M3",
  source = [["M3"],[_M3par,_M3perp],],
  fetch_func = [ff.fetch_s0c0,ff.fetch_s0c0_add_s1c0],
  label = r"$M_{3%s}$ (1/s$^3$)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_M3)

# Parallel drift speed.
_upar : GkQuantity = GkQuantity(
  name = "upar",
  source = [["MaxwellianMoments"], ["BiMaxwellianMoments"], [_M0, _M1]],
  fetch_func = [ff.fetch_s0c1, ff.fetch_s0c1, ff.fetch_s1c0_div_s0c0],
  label = r"$u_{\parallel %s}$ (m/s)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_upar)

# Parallel temperature.
_Tpar : GkQuantity = GkQuantity(
  name = "Tpar",
  source = [["BiMaxwellianMoments"],[_M0,_M1,_M2par],],
  fetch_func = [ff.fetch_Tpar_from_BiMax, ff.fetch_Tpar_from_M0_M1_M2par],
  label = r"$T_{\parallel %s}$ (J)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_Tpar)

# Perpendicular temperature.
_Tperp : GkQuantity = GkQuantity(
  name = "Tperp",
  source = [["BiMaxwellianMoments"], [_M0,_M2perp]],
  fetch_func = [ff.fetch_Tperp_from_BiMax, ff.fetch_Tperp_from_M0_M2perp],
  label = r"$T_{\perp %s}$ (J)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_Tperp)

# ---------------------------------------------------
# --- Combined plasma moments (species-dependent) ---
# ---------------------------------------------------

# Temperature.
_temp : GkQuantity = GkQuantity(
  name = "temp",
  source = [["MaxwellianMoments"], [_Tpar,_Tperp]],
  fetch_func = [ff.fetch_temp_from_Max, ff.fetch_temp_from_Tpar_Tperp],
  label = r"$T_{%s}$ (J)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_temp)

# Pressure.
_press : GkQuantity = GkQuantity(
  name = "press",
  source = [["MaxwellianMoments"], ["BiMaxwellianMoments"], [_M0,_temp]],
  fetch_func = [ff.fetch_press_from_Max, ff.fetch_press_from_BiMax, ff.fetch_s0c0_mul_s1c0],
  label = r"$p_{%s}$ (Pa)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_press)

# Parallel pressure.
_presspar : GkQuantity = GkQuantity(
  name = "presspar",
  source = [[_M0,_Tpar]],
  fetch_func = [ff.fetch_press_p],
  label = r"$p_{\parallel %s}$ (Pa)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_presspar)

# Perpendicular pressure.
_pressperp : GkQuantity = GkQuantity(
  name = "pressperp",
  source = [[_M0,_Tperp]],
  fetch_func = [ff.fetch_press_p],
  label = r"$p_{\perp %s}$ (Pa)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_pressperp)

# Parallel flux of parallel energy, in the lab frame: (m/2)*M3par.
_qpar : GkQuantity = GkQuantity(
  name = "qpar",
  source = [[_M3par]],
  fetch_func = [ff.fetch_qpar],
  label = r"$q_{\parallel %s}$ (W/m$^2$)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_qpar)

# Parallel flux of perpendicular energy, in the lab frame: (m/2)*M3perp.
_qperp : GkQuantity = GkQuantity(
  name = "qperp",
  source = [[_M3perp]],
  fetch_func = [ff.fetch_qperp],
  label = r"$q_{\perp %s}$ (W/m$^2$)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_qperp)

# Parallel heat flux in the fluid (drift) frame: (m/2)*int (vpar-upar)^3 f dv.
_qpar_fluid : GkQuantity = GkQuantity(
  name = "qpar_fluid",
  source = [[_M0,_M1,_M2par,_M3par]],
  fetch_func = [ff.fetch_qpar_fluid],
  label = r"$q_{\parallel %s}^{fluid}$ (W/m$^2$)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_qpar_fluid)

# Perpendicular heat flux in the fluid (drift) frame: (m/2)*int (vpar-upar)*vperp^2 f dv.
_qperp_fluid : GkQuantity = GkQuantity(
  name = "qperp_fluid",
  source = [[_M0,_M1,_M2perp,_M3perp]],
  fetch_func = [ff.fetch_qperp_fluid],
  label = r"$q_{\perp %s}^{fluid}$ (W/m$^2$)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_qperp_fluid)

# Plasma beta.
_beta : GkQuantity = GkQuantity(
  name = "beta",
  source = [[_geo_int_bmag,_press],],
  fetch_func = [ff.fetch_beta_from_bmag_press],
  label = r"$\beta_{%s}$",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_beta)

# Thermal velocity.
_vt : GkQuantity = GkQuantity(
  name = "vt",
  source = [[_temp],],
  fetch_func = [ff.fetch_vt],
  label = r"$v_{t,%s}$ (m/s)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_vt)

# Larmor (gyro-)radius.
_larmor_radius : GkQuantity = GkQuantity(
  name = "larmor_radius",
  source = [[_geo_int_bmag, _temp],],
  fetch_func = [ff.fetch_larmor_radius],
  label = r"$\rho_{%s}$ (m)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_larmor_radius)

# Debye length.
_debye_length : GkQuantity = GkQuantity(
  name = "debye_length",
  source = [[_M0, _temp],],
  fetch_func = [ff.fetch_debye_length],
  label = r"$\lambda_{D,%s}$ (m)",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_debye_length)

# Sound speed.
_c_s : GkQuantity = GkQuantity(
  name = "c_s",
  source = [[_M0, _temp],],
  fetch_func = [ff.fetch_c_s],
  label = r"$c_{s}$ (m/s)",
  is_time_dep = True,
  is_species_dep = False,
  is_multi_species = True,
)
gk_quant_registry.register(_c_s)

# ------------------------
# --- Drift velocities ---
# ------------------------

# ExB drift velocity.
_ExB_vel : GkQuantity = GkQuantity(
  name = "ExB_vel",
  source = [[_geo_int_bmag, _geo_int_jacobtot_inv, _field, _geo_int_b_i],],
  fetch_func = [ff.fetch_ExB_vel],
  label = r"$v_{E,%s}$ (m/s)",
  is_time_dep = True,
  is_vector = True
)
gk_quant_registry.register(_ExB_vel)

# Grad B drift velocity.
_gradB_vel : GkQuantity = GkQuantity(
  name = "gradB_vel",
  source = [[_geo_int_bmag, _geo_int_jacobtot_inv, _Tperp, _geo_int_b_i]],
  fetch_func= [ff.fetch_gradB_vel],
  label = r"$v_{\nabla B,%s}$ (m/s)",
  is_time_dep = True,
  is_species_dep = True,
  is_vector = True
)
gk_quant_registry.register(_gradB_vel)

# Diamagnetic drift velocity.
_diamag_vel : GkQuantity = GkQuantity(
  name = "diamag_vel",
  source = [[_geo_int_bmag, _geo_int_jacobtot_inv, _M0, _pressperp, _geo_int_b_i]],
  fetch_func= [ff.fetch_diamag_vel],
  label = r"$v_{dia,%s}$ (m/s)",
  is_time_dep = True,
  is_species_dep = True,
  is_vector = True
)
gk_quant_registry.register(_diamag_vel)

# -------------------------------------
# --- Magnetic field perturbations ----
# -------------------------------------

# Parallel component of the perturb. magnetic vector potential.
_apar : GkQuantity = GkQuantity(
  name = "apar",
  source = [["apar"],],
  fetch_func = [ff.fetch_s0c0],
  label = r"$A_\parallel$ (T m)",
  is_time_dep = True,
)
gk_quant_registry.register(_apar)

# Contravariant components of  of the pertub. magnetic vector potential.
_dB_perp_dual : GkQuantity = GkQuantity(
  name = "dB_perp_dual",
  source = [[_apar, _geo_int_jacobgeo_inv, _geo_int_b_i],],
  fetch_func = [ff.fetch_dB_perp_dual],
  label = r"$\delta B_\perp^{%s}$ (T)",
  is_time_dep = True,
  is_vector = True
)
gk_quant_registry.register(_dB_perp_dual)

# Covariant components of the same, dB_i = g_ij dB^j.
_dB_perp : GkQuantity = GkQuantity(
  name = "dB_perp",
  source = [[_apar, _geo_int_jacobgeo_inv, _geo_int_b_i, _geo_int_g_ij],],
  fetch_func = [ff.fetch_dB_perp],
  label = r"$\delta B_{\perp %s}$ (T)",
  is_time_dep = True,
  is_vector = True
)
gk_quant_registry.register(_dB_perp)

# Magnitude of the magnetic field perturbation.
_dB_perp_mag : GkQuantity = GkQuantity(
  name = "dB_perp_mag",
  source = [[_apar, _geo_int_jacobgeo_inv, _geo_int_b_i, _geo_int_g_ij],],
  fetch_func = [ff.fetch_dB_perp_mag],
  label = r"$|\delta B_\perp|$ (T)",
  is_time_dep = True,
)
gk_quant_registry.register(_dB_perp_mag)

# Covariant components of the total magnetic field.
_B_tot : GkQuantity = GkQuantity(
  name = "B_tot",
  source = [[_apar, _geo_int_bmag, _geo_int_jacobgeo_inv, _geo_int_b_i, _geo_int_g_ij],
            [_geo_int_bmag, _geo_int_b_i],],
  fetch_func = [ff.fetch_B_tot, ff.fetch_B_equilibrium],
  label = r"$B_{%s}$ (T)",
  is_time_dep = True,
  is_vector = True
)
gk_quant_registry.register(_B_tot)

# Contravariant components of the total magnetic field.
_B_tot_dual : GkQuantity = GkQuantity(
  name = "B_tot_dual",
  source = [[_apar, _geo_int_bmag, _geo_int_jacobgeo_inv, _geo_int_b_i, _geo_int_g_ij],
            [_geo_int_bmag, _geo_int_g_ij],],
  fetch_func = [ff.fetch_B_tot_dual, ff.fetch_B_dual_equilibrium],
  label = r"$B^{%s}$ (T)",
  is_time_dep = True,
  is_vector = True
)
gk_quant_registry.register(_B_tot_dual)

# Total magnetic field magnitude.
_B_tot_mag : GkQuantity = GkQuantity(
  name = "B_tot_mag",
  source = [[_apar, _geo_int_bmag, _geo_int_jacobgeo_inv, _geo_int_b_i, _geo_int_g_ij]],
  fetch_func = [ff.fetch_B_tot_mag],
  label = r"$|B|$ (T)",
  is_time_dep = True,
)
gk_quant_registry.register(_B_tot_mag)

# ------------------------------
# --- Phase space quantities ---
# ------------------------------

# Distribution function loaded through load_gk_distf.
_distf : GkQuantity = GkQuantity(
  name = "distf",
  source = [[""]],
  fetch_func = [ff.load_distf],
  label = r"$f_{%s}$",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_distf)

# -----------------------------
# --- Normalized quantities ---
# -----------------------------

# Ratio of the Larmor radius to the Debye length.
_rho_over_lambda : GkQuantity = GkQuantity(
  name = "rho_over_lambda",
  source = [[_debye_length, _larmor_radius],],
  fetch_func = [ff.fetch_rho_over_lambda],
  label = r"$(\rho/\lambda_D)_{%s}$",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_rho_over_lambda)

# Normalized elctrostatic potential.
_phi_norm : GkQuantity = GkQuantity(
  name = "phi_norm",
  source = [[_field, _temp],],
  fetch_func = [ff.fetch_phi_norm],
  label = r"$e\phi/T_{%s}$",
  is_time_dep = True,
  is_species_dep = False,
)
gk_quant_registry.register(_phi_norm)

# Normalized parallel heatflux.
_qpar_norm : GkQuantity = GkQuantity(
  name = "qpar_norm",
  source = [[_M0, _qpar, _temp, _vt],],
  fetch_func = [ff.fetch_qpar_norm],
  label = r"$q_{\parallel %s}/(n T v_{th})$",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_qpar_norm)

# Normalized perpendicular heatflux.
_qperp_norm : GkQuantity = GkQuantity(
  name = "qperp_norm",
  source = [[_M0, _qperp, _temp, _vt],],
  fetch_func = [ff.fetch_qperp_norm],
  label = r"$q_{\perp %s}/(n T v_{th})$",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_qperp_norm)

# Normalized parallel fluid-frame heatflux.
_qpar_fluid_norm : GkQuantity = GkQuantity(
  name = "qpar_fluid_norm",
  source = [[_M0, _qpar_fluid, _temp, _vt],],
  fetch_func = [ff.fetch_qpar_norm],
  label = r"$q_{\parallel %s}^{fluid}/(n T v_{t})$",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_qpar_fluid_norm)

# Normalized perpendicular fluid-frame heatflux.
_qperp_fluid_norm : GkQuantity = GkQuantity(
  name = "qperp_fluid_norm",
  source = [[_M0, _qperp_fluid, _temp, _vt],],
  fetch_func = [ff.fetch_qperp_norm],
  label = r"$q_{\perp %s}^{fluid}/(n T v_{t})$",
  is_time_dep = True,
  is_species_dep = True,
)
gk_quant_registry.register(_qperp_fluid_norm)
