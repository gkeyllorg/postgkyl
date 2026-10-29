"""The gyrokinetic quantity registry -- populated from ``quantities.py``.

Ported from ``src_bak/postgkyl/gk/gk_quantities/registry.py``. Each entry
names its preferred source combinations (in order) and the fetch function
for each; :func:`~postgkyl.diagnostics.gk.quantity.GkQuantity.
get_avail_source` picks the first combination whose files are actually
present on disk.
"""

from __future__ import annotations

from . import quantities as ff
from .quantity import GkQuantity, GkQuantityRegistry

gk_quant_registry = GkQuantityRegistry()

# ----------------------------------------- scalar geometric quantities (geo)
_geo_int_jacobgeo = GkQuantity(name="geo_int_jacobgeo",
                               description='Configuration-space Jacobian J.',
                               source=[["geo_int_jacobgeo"]],
                               fetch_func=[ff.fetch_s0c0],
                               label=r"$J$",
                               is_geo=True)
gk_quant_registry.register(_geo_int_jacobgeo)

_geo_int_jacobgeo_inv = GkQuantity(
    name="geo_int_jacobgeo_inv",
    description='Inverse configuration-space Jacobian 1/J.',
    source=[["geo_int_jacobgeo_inv"]],
    fetch_func=[ff.fetch_s0c0],
    label=r"$J^{-1}$",
    is_geo=True)
gk_quant_registry.register(_geo_int_jacobgeo_inv)

_geo_int_jacobtot = GkQuantity(
    name="geo_int_jacobtot",
    description='Conf-space Jacobian times magnetic field amplitude J B.',
    source=[["geo_int_jacobtot"]],
    fetch_func=[ff.fetch_s0c0],
    label=r"$J$",
    is_geo=True)
gk_quant_registry.register(_geo_int_jacobtot)

_geo_int_jacobtot_inv = GkQuantity(
    name="geo_int_jacobtot_inv",
    description='Inverse geo_int_jacobtot 1/(J B).',
    source=[["geo_int_jacobtot_inv"]],
    fetch_func=[ff.fetch_s0c0],
    label=r"$(J B)^{-1}$",
    is_geo=True)
gk_quant_registry.register(_geo_int_jacobtot_inv)

_geo_int_bmag = GkQuantity(name="geo_int_bmag",
                           description='Equilibrium magnetic-field magnitude.',
                           source=[["geo_int_bmag"]],
                           fetch_func=[ff.fetch_s0c0],
                           label=r"$B$ (T)",
                           is_geo=True)
gk_quant_registry.register(_geo_int_bmag)

# ----------------------------------------- vector geometric quantities (geo)
_geo_int_b_i = GkQuantity(name="geo_int_b_i",
                          description='Covariant magnetic-field unit vector.',
                          source=[["geo_int_b_i"]],
                          fetch_func=[ff.fetch_s0cAll],
                          label=r"$b_%(dir)s$",
                          is_vector=True,
                          is_geo=True,
                          component_basis='covariant')
gk_quant_registry.register(_geo_int_b_i)

# ----------------------------------------- tensor geometric quantities (geo)
# Symmetric metrics, stored as 11, 12, 13, 22, 23, 33.
_geo_int_g_ij = GkQuantity(name="geo_int_g_ij",
                           description='Covariant metric-tensor.',
                           source=[["geo_int_g_ij"]],
                           fetch_func=[ff.fetch_s0cAll],
                           label=r"$g_{ij}$",
                           is_tensor=True,
                           is_geo=True)
gk_quant_registry.register(_geo_int_g_ij)

_geo_int_gij = GkQuantity(name="geo_int_gij",
                          description='Contravariant metric-tensor.',
                          source=[["geo_int_gij"]],
                          fetch_func=[ff.fetch_s0cAll],
                          label=r"$g^{ij}$",
                          is_tensor=True,
                          is_geo=True)
gk_quant_registry.register(_geo_int_gij)

# ------------------------------------------------------------------- field
_field = GkQuantity(name="field",
                    description='Electrostatic potential.',
                    source=[["field"]],
                    fetch_func=[ff.fetch_s0c0],
                    label=r"$\phi$ (V)",
                    is_time_dep=True)
gk_quant_registry.register(_field)

# --------------------------------------------------- plasma moments (per-sp)
_M0 = GkQuantity(name="M0",
                 description='Number density (zeroth velocity moment).',
                 source=[["M0"], ["M0M1M2"], ["M0M1M2parM2perp"],
                         ["MaxwellianMoments"], ["BiMaxwellianMoments"],
                         ["HamiltonianMoments"]],
                 fetch_func=[ff.fetch_s0c0] * 6,
                 label=r"$M_{0%(species)s}$ (m$^{-3}$)",
                 is_species_dep=True,
                 is_time_dep=True)
gk_quant_registry.register(_M0)

_M1 = GkQuantity(
    name="M1",
    description=
    'Parallel momentum density per unit mass (first velocity moment).',
    source=[["M1"], ["M0M1M2"], ["M0M1M2parM2perp"], ["MaxwellianMoments"],
            ["BiMaxwellianMoments"], ["HamiltonianMoments"]],
    fetch_func=[
        ff.fetch_s0c0, ff.fetch_s0c1, ff.fetch_s0c1, ff.fetch_s0c0_mul_s0c1,
        ff.fetch_s0c0_mul_s0c1, ff.fetch_M1_from_H
    ],
    label=r"$M_{1%(species)s}$ (m$^{-2}$/s)",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_M1)

_M2par = GkQuantity(name="M2par",
                    description='Parallel second velocity moment.',
                    source=[["M2par"], ["M0M1M2parM2perp"], ["M2", "M2perp"],
                            ["MaxwellianMoments"], ["BiMaxwellianMoments"]],
                    fetch_func=[
                        ff.fetch_s0c0, ff.fetch_s0c2, ff.fetch_s0c0_sub_s1c0,
                        ff.fetch_M2par_from_Max, ff.fetch_M2par_from_BiMax
                    ],
                    label=r"$M_{2\parallel%(species)s}$ (m$^{-1}$/s$^2$)",
                    is_time_dep=True,
                    is_species_dep=True)
gk_quant_registry.register(_M2par)

_M2perp = GkQuantity(name="M2perp",
                     description='Perpendicular second velocity moment.',
                     source=[["M2perp"], ["M0M1M2parM2perp"], ["M2", "M2par"],
                             ["MaxwellianMoments"], ["BiMaxwellianMoments"]],
                     fetch_func=[
                         ff.fetch_s0c0, ff.fetch_s0c3, ff.fetch_s0c0_sub_s1c0,
                         ff.fetch_M2perp_from_Max, ff.fetch_M2perp_from_BiMax
                     ],
                     label=r"$M_{2\perp%(species)s}$ (m$^{-1}$/s$^2$)",
                     is_time_dep=True,
                     is_species_dep=True)
gk_quant_registry.register(_M2perp)

_M2 = GkQuantity(
    name="M2",
    description='Kinetic energy density times 2/mass (second velocity moment).',
    source=[["M2"], ["M0M1M2"], ["M0M1M2parM2perp"], [_M2par, _M2perp]],
    fetch_func=[
        ff.fetch_s0c0, ff.fetch_s0c2, ff.fetch_s0c2_add_s0c3,
        ff.fetch_s0c0_add_s1c0
    ],
    label=r"$M_{2%(species)s}$ (m$^{-1}$/s$^2$)",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_M2)

_M3par = GkQuantity(name="M3par",
                    description='Parallel third velocity moment.',
                    source=[["M3par"]],
                    fetch_func=[ff.fetch_s0c0],
                    label=r"$M_{3\parallel%(species)s}$ (1/s$^3$)",
                    is_time_dep=True,
                    is_species_dep=True)
gk_quant_registry.register(_M3par)

_M3perp = GkQuantity(
    name="M3perp",
    description='Parallel-perpendicular third velocity moment.',
    source=[["M3perp"]],
    fetch_func=[ff.fetch_s0c0],
    label=r"$M_{3\perp%(species)s}$ (1/s$^3$)",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_M3perp)

_M3 = GkQuantity(
    name="M3",
    description='Energy flux times 2/mass (third velocity moment).',
    source=[["M3"], [_M3par, _M3perp]],
    fetch_func=[ff.fetch_s0c0, ff.fetch_s0c0_add_s1c0],
    label=r"$M_{3%(species)s}$ (1/s$^3$)",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_M3)

_upar = GkQuantity(
    name="upar",
    description='Parallel flow velocity.',
    source=[["MaxwellianMoments"], ["BiMaxwellianMoments"], [_M0, _M1]],
    fetch_func=[ff.fetch_s0c1, ff.fetch_s0c1, ff.fetch_s1c0_div_s0c0],
    label=r"$u_{\parallel %(species)s}$ (m/s)",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_upar)

_Tpar = GkQuantity(
    name="Tpar",
    description='Parallel temperature.',
    source=[["BiMaxwellianMoments"], [_M0, _M1, _M2par]],
    fetch_func=[ff.fetch_Tpar_from_BiMax, ff.fetch_Tpar_from_M0_M1_M2par],
    label=r"$T_{\parallel %(species)s}$ (J)",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_Tpar)

_Tperp = GkQuantity(
    name="Tperp",
    description='Perpendicular temperature.',
    source=[["BiMaxwellianMoments"], [_M0, _M2perp]],
    fetch_func=[ff.fetch_Tperp_from_BiMax, ff.fetch_Tperp_from_M0_M2perp],
    label=r"$T_{\perp %(species)s}$ (J)",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_Tperp)

# ------------------------------------------- combined plasma moments (per-sp)
_temp = GkQuantity(
    name="temp",
    description='Isotropic temperature: (Tpar + 2 Tperp)/3.',
    source=[["MaxwellianMoments"], [_Tpar, _Tperp]],
    fetch_func=[ff.fetch_temp_from_Max, ff.fetch_temp_from_Tpar_Tperp],
    label=r"$T_{%(species)s}$ (J)",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_temp)

_press = GkQuantity(name="press",
                    description='Isotropic thermal pressure.',
                    source=[["MaxwellianMoments"], ["BiMaxwellianMoments"],
                            [_M0, _temp]],
                    fetch_func=[
                        ff.fetch_press_from_Max, ff.fetch_press_from_BiMax,
                        ff.fetch_s0c0_mul_s1c0
                    ],
                    label=r"$p_{%(species)s}$ (Pa)",
                    is_time_dep=True,
                    is_species_dep=True)
gk_quant_registry.register(_press)

_presspar = GkQuantity(name="presspar",
                       description='Parallel thermal pressure.',
                       source=[[_M0, _Tpar]],
                       fetch_func=[ff.fetch_press_p],
                       label=r"$p_{\parallel %(species)s}$ (Pa)",
                       is_time_dep=True,
                       is_species_dep=True)
gk_quant_registry.register(_presspar)

_pressperp = GkQuantity(name="pressperp",
                        description='Perpendicular thermal pressure.',
                        source=[[_M0, _Tperp]],
                        fetch_func=[ff.fetch_press_p],
                        label=r"$p_{\perp %(species)s}$ (Pa)",
                        is_time_dep=True,
                        is_species_dep=True)
gk_quant_registry.register(_pressperp)

_beta = GkQuantity(name="beta",
                   description='Ratio of thermal to magnetic pressure.',
                   source=[[_geo_int_bmag, _press]],
                   fetch_func=[ff.fetch_beta_from_bmag_press],
                   label=r"$\beta_{%(species)s}$",
                   is_time_dep=True,
                   is_species_dep=True)
gk_quant_registry.register(_beta)

# --------------------------------------------------------------- heat fluxes
_qpar = GkQuantity(name="qpar",
                   description='Parallel-energy flux.',
                   source=[[_M3par]],
                   fetch_func=[ff.fetch_qpar],
                   label=r"$q_{\parallel %(species)s}$ (W/m$^2$)",
                   is_time_dep=True,
                   is_species_dep=True)
gk_quant_registry.register(_qpar)

_qperp = GkQuantity(name="qperp",
                    description='Perpendicular-energy flux.',
                    source=[[_M3perp]],
                    fetch_func=[ff.fetch_qperp],
                    label=r"$q_{\perp %(species)s}$ (W/m$^2$)",
                    is_time_dep=True,
                    is_species_dep=True)
gk_quant_registry.register(_qperp)

_qpar_fluid = GkQuantity(
    name="qpar_fluid",
    description='Parallel heat flux in the local fluid frame.',
    source=[[_M0, _M1, _M2par, _M3par]],
    fetch_func=[ff.fetch_qpar_fluid],
    label=r"$q_{\parallel %(species)s}^{fluid}$ (W/m$^2$)",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_qpar_fluid)

_qperp_fluid = GkQuantity(
    name="qperp_fluid",
    description='Perpendicular heat flux in the local fluid frame.',
    source=[[_M0, _M1, _M2perp, _M3perp]],
    fetch_func=[ff.fetch_qperp_fluid],
    label=r"$q_{\perp %(species)s}^{fluid}$ (W/m$^2$)",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_qperp_fluid)

# ------------------------------------------------ thermal speed / lengths
_vt = GkQuantity(name="vt",
                 description='Thermal speed sqrt(T/m).',
                 source=[[_temp]],
                 fetch_func=[ff.fetch_vt],
                 label=r"$v_{t,%(species)s}$ (m/s)",
                 is_time_dep=True,
                 is_species_dep=True)
gk_quant_registry.register(_vt)

_larmor_radius = GkQuantity(name="larmor_radius",
                            description='Thermal Larmor radius.',
                            source=[[_temp, _geo_int_bmag]],
                            fetch_func=[ff.fetch_larmor_radius],
                            label=r"$\rho_{%(species)s}$ (m)",
                            is_time_dep=True,
                            is_species_dep=True)
gk_quant_registry.register(_larmor_radius)

_debye_length = GkQuantity(name="debye_length",
                           description='Species Debye length.',
                           source=[[_temp, _M0]],
                           fetch_func=[ff.fetch_debye_length],
                           label=r"$\lambda_{D,%(species)s}$ (m)",
                           is_time_dep=True,
                           is_species_dep=True)
gk_quant_registry.register(_debye_length)

# Multi-species: one dataset from every listed species (--species elc,ion,...);
# with only ions listed the electrons are adiabatic (ti_over_te).
_c_s_cold_i = GkQuantity(
    name="c_s_cold_i",
    description='Sound speed in the cold-ion approximation.',
    source=[[_M0, _temp]],
    fetch_func=[ff.fetch_c_s_cold_i],
    label=r"$c_{s}$ (m/s)",
    is_time_dep=True,
    is_multi_species=True)
gk_quant_registry.register(_c_s_cold_i)

_c_s_hot_i = GkQuantity(
    name="c_s_hot_i",
    description='Sound speed including finite ion temperature.',
    source=[[_M0, _temp]],
    fetch_func=[ff.fetch_c_s_hot_i],
    label=r"$c_{s}$ (m/s)",
    is_time_dep=True,
    is_multi_species=True)
gk_quant_registry.register(_c_s_hot_i)

# The Mach numbers belong to the first listed species (--species s,...).
_mach_cold_i = GkQuantity(
    name="mach_cold_i",
    description='Parallel Mach number with cold-ion sound speed.',
    source=[[_M0, _temp, _upar]],
    fetch_func=[ff.fetch_mach_cold_i],
    label=r"$u_{\parallel %(species)s}/c_{s}$",
    is_time_dep=True,
    is_species_dep=True,
    is_multi_species=True)
gk_quant_registry.register(_mach_cold_i)

_mach_hot_i = GkQuantity(
    name="mach_hot_i",
    description='Parallel Mach number with finite ion temperature.',
    source=[[_M0, _temp, _upar]],
    fetch_func=[ff.fetch_mach_hot_i],
    label=r"$u_{\parallel %(species)s}/c_{s}$",
    is_time_dep=True,
    is_species_dep=True,
    is_multi_species=True)
gk_quant_registry.register(_mach_hot_i)

# Collision frequency of species s with species r (--species s,r).
_collision_freq = GkQuantity(name="collision_freq",
                             description='Coulomb collision frequency.',
                             source=[[_M0, _temp]],
                             fetch_func=[ff.fetch_collision_freq],
                             label=r"$\nu_{sr}$ (1/s)",
                             is_time_dep=True,
                             is_multi_species=True)
gk_quant_registry.register(_collision_freq)

# ---------------------------------------------------------- gradient lengths
# Along the direction dir (0: x, 1: y, 2: z) of the simulation.
_inv_L_n_cov = GkQuantity(
    name="inv_L_n_cov",
    description='Covariant inverse density gradient length: -(partial_i n)/n.',
    source=[[_M0]],
    fetch_func=[ff.fetch_inv_L_n_cov],
    label='$1/L_{n,%(species)s,%(dir)s}$ (1/coordinate unit)',
    is_time_dep=True,
    is_species_dep=True,
    component_basis='covariant',
    is_vector=True)
gk_quant_registry.register(_inv_L_n_cov)

_inv_L_T_cov = GkQuantity(
    name="inv_L_T_cov",
    description=
    'Covariant inverse temperature gradient length: -(partial_i T)/T.',
    source=[[_temp]],
    fetch_func=[ff.fetch_inv_L_T_cov],
    label='$1/L_{T,%(species)s,%(dir)s}$ (1/coordinate unit)',
    is_time_dep=True,
    is_species_dep=True,
    component_basis='covariant',
    is_vector=True)
gk_quant_registry.register(_inv_L_T_cov)

# ----------------------------------------------------------- drift speeds
_ExB_vel_contra = GkQuantity(
    name="ExB_vel_contra",
    description='Contravariant ExB drift velocity.',
    source=[[_geo_int_jacobtot_inv, _geo_int_bmag, _geo_int_b_i, _field]],
    fetch_func=[ff.fetch_ExB_vel_contra],
    label='$v_{E}^{%(dir)s}$ (%(units)s)',
    is_time_dep=True,
    is_vector=True,
    component_basis='contravariant',
    physical_units='m/s',
    projection_label='$v_{E,%(dir)s}$ (%(units)s)')
gk_quant_registry.register(_ExB_vel_contra)

_gradB_vel_contra = GkQuantity(
    name="gradB_vel_contra",
    description='Contravariant grad-B drift velocity.',
    source=[[_geo_int_jacobtot_inv, _geo_int_bmag, _geo_int_b_i, _Tperp]],
    fetch_func=[ff.fetch_gradB_vel_contra],
    label='$v_{\\nabla B}^{%(dir)s}$ (%(units)s)',
    is_time_dep=True,
    is_species_dep=True,
    is_vector=True,
    component_basis='contravariant',
    physical_units='m/s',
    projection_label='$v_{\\nabla B,%(dir)s}$ (%(units)s)')
gk_quant_registry.register(_gradB_vel_contra)

_diamag_vel_contra = GkQuantity(
    name="diamag_vel_contra",
    description='Contravariant diamagnetic drift velocity.',
    source=[[
        _geo_int_jacobtot_inv, _geo_int_bmag, _geo_int_b_i, _M0, _pressperp
    ]],
    fetch_func=[ff.fetch_diamag_vel_contra],
    label='$v_{dia}^{%(dir)s}$ (%(units)s)',
    is_time_dep=True,
    is_species_dep=True,
    is_vector=True,
    component_basis='contravariant',
    physical_units='m/s',
    projection_label='$v_{dia,%(dir)s}$ (%(units)s)')
gk_quant_registry.register(_diamag_vel_contra)

# ------------------------------------------- magnetic field perturbations
_apar = GkQuantity(name="apar",
                   description='Parallel magnetic vector potential.',
                   source=[["apar"]],
                   fetch_func=[ff.fetch_s0c0],
                   label=r"$A_\parallel$ (T m)",
                   is_time_dep=True)
gk_quant_registry.register(_apar)

_dB_perp_contra = GkQuantity(
    name="dB_perp_contra",
    description='Contravariant magnetic perturbation.',
    source=[[_apar, _geo_int_jacobgeo_inv, _geo_int_b_i]],
    fetch_func=[ff.fetch_dB_perp_contra],
    label='$\\delta B_\\perp^{%(dir)s}$ (%(units)s)',
    is_time_dep=True,
    is_vector=True,
    component_basis='contravariant',
    physical_units='T',
    projection_label='$\\delta B_{\\perp %(dir)s}$ (%(units)s)')
gk_quant_registry.register(_dB_perp_contra)

_dB_perp_cov = GkQuantity(
    name="dB_perp_cov",
    description='Covariant magnetic perturbation.',
    source=[[_apar, _geo_int_jacobgeo_inv, _geo_int_b_i, _geo_int_g_ij]],
    fetch_func=[ff.fetch_dB_perp_cov],
    label='$\\delta B_{\\perp %(dir)s}$ (%(units)s)',
    is_time_dep=True,
    is_vector=True,
    component_basis='covariant',
    physical_units='T',
    projection_label='$\\delta B_{\\perp %(dir)s}$ (%(units)s)')
gk_quant_registry.register(_dB_perp_cov)

_dB_perp_mag = GkQuantity(
    name="dB_perp_mag",
    description='Magnetic perturbation magnitude.',
    source=[[_apar, _geo_int_jacobgeo_inv, _geo_int_b_i, _geo_int_g_ij]],
    fetch_func=[ff.fetch_dB_perp_mag],
    label=r"$|\delta B_\perp|$ (T)",
    is_time_dep=True)
gk_quant_registry.register(_dB_perp_mag)

# Without apar output the total field falls back to the equilibrium one.
_B_tot_sources = [
    _apar, _geo_int_bmag, _geo_int_jacobgeo_inv, _geo_int_b_i, _geo_int_g_ij
]
_B_tot_cov = GkQuantity(
    name="B_tot_cov",
    description='Covariant total magnetic field.',
    source=[_B_tot_sources, [_geo_int_bmag, _geo_int_b_i]],
    fetch_func=[ff.fetch_B_tot_cov, ff.fetch_B_equilibrium_cov],
    label='$B_{%(dir)s}$ (%(units)s)',
    is_time_dep=True,
    is_vector=True,
    component_basis='covariant',
    physical_units='T',
    projection_label='$B_{%(dir)s}$ (%(units)s)')
gk_quant_registry.register(_B_tot_cov)

_B_tot_contra = GkQuantity(
    name="B_tot_contra",
    description='Contravariant total magnetic field.',
    source=[_B_tot_sources, [_geo_int_bmag, _geo_int_g_ij]],
    fetch_func=[ff.fetch_B_tot_contra, ff.fetch_B_equilibrium_contra],
    label='$B^{%(dir)s}$ (%(units)s)',
    is_time_dep=True,
    is_vector=True,
    component_basis='contravariant',
    physical_units='T',
    projection_label='$B_{%(dir)s}$ (%(units)s)')
gk_quant_registry.register(_B_tot_contra)

_B_tot_mag = GkQuantity(name="B_tot_mag",
                        description='Total magnetic-field magnitude.',
                        source=[_B_tot_sources],
                        fetch_func=[ff.fetch_B_tot_mag],
                        label=r"$|B|$ (T)",
                        is_time_dep=True)
gk_quant_registry.register(_B_tot_mag)

# ------------------------------------------------------------ electric field
_E_field_cov = GkQuantity(name="E_field_cov",
                          description='Covariant electrostatic field.',
                          source=[[_field]],
                          fetch_func=[ff.fetch_E_field_cov],
                          label='$E_{%(dir)s}$ (%(units)s)',
                          is_time_dep=True,
                          is_vector=True,
                          component_basis='covariant',
                          physical_units='V/m',
                          projection_label='$E_{%(dir)s}$ (%(units)s)')
gk_quant_registry.register(_E_field_cov)

_E_field_contra = GkQuantity(name="E_field_contra",
                             description='Contravariant electrostatic field.',
                             source=[[_field, _geo_int_gij]],
                             fetch_func=[ff.fetch_E_field_contra],
                             label='$E^{%(dir)s}$ (%(units)s)',
                             is_time_dep=True,
                             is_vector=True,
                             component_basis='contravariant',
                             physical_units='V/m',
                             projection_label='$E_{%(dir)s}$ (%(units)s)')
gk_quant_registry.register(_E_field_contra)

_E_field_mag = GkQuantity(name="E_field_mag",
                          description='Electrostatic-field magnitude.',
                          source=[[_field, _geo_int_gij]],
                          fetch_func=[ff.fetch_E_field_mag],
                          label=r"$|E|$ (V/m)",
                          is_time_dep=True)
gk_quant_registry.register(_E_field_mag)

# -------------------------------------------------------- cross-field fluxes
# Contravariant components (.grad x^k) of 3x fluxes along dir=0 (x) or 1 (y);
# fluct='y'|'yz' keeps the turbulent part about the y or (y, z) average.
_es_flux_geo = [_field, _geo_int_jacobgeo, _geo_int_jacobtot_inv, _geo_int_b_i]
_em_flux_geo = [
    _apar, _geo_int_bmag, _geo_int_jacobgeo, _geo_int_jacobgeo_inv, _geo_int_b_i
]
_total_flux_geo = [
    _apar, _field, _geo_int_bmag, _geo_int_jacobgeo, _geo_int_jacobgeo_inv,
    _geo_int_jacobtot_inv, _geo_int_b_i
]

_flux_particle_ExB_contra = GkQuantity(
    name="flux_particle_ExB_contra",
    description='Contravariant cross-field particle flux from ExB drift.',
    source=[[_M0] + _es_flux_geo],
    fetch_func=[ff.fetch_flux_particle_ExB_contra],
    label=r"$\Gamma^{%(dir)s}_{E,%(species)s}$",
    is_time_dep=True,
    is_species_dep=True,
    component_basis='contravariant',
    is_vector=True)
gk_quant_registry.register(_flux_particle_ExB_contra)

_flux_energy_ExB_contra = GkQuantity(
    name="flux_energy_ExB_contra",
    description='Contravariant cross-field energy flux from ExB drift.',
    source=[[_M2] + _es_flux_geo],
    fetch_func=[ff.fetch_flux_energy_ExB_contra],
    label=r"$Q^{%(dir)s}_{E,%(species)s}$",
    is_time_dep=True,
    is_species_dep=True,
    component_basis='contravariant',
    is_vector=True)
gk_quant_registry.register(_flux_energy_ExB_contra)

_flux_particle_dB_contra = GkQuantity(
    name="flux_particle_dB_contra",
    description='Contravariant cross-field particle flux from magnetic flutter.',
    source=[[_M1] + _em_flux_geo],
    fetch_func=[ff.fetch_flux_particle_dB_contra],
    label=r"$\Gamma^{%(dir)s}_{\delta B,%(species)s}$",
    is_time_dep=True,
    is_species_dep=True,
    component_basis='contravariant',
    is_vector=True)
gk_quant_registry.register(_flux_particle_dB_contra)

_flux_energy_dB_contra = GkQuantity(
    name="flux_energy_dB_contra",
    description='Contravariant cross-field energy flux from magnetic flutter.',
    source=[[_M3] + _em_flux_geo],
    fetch_func=[ff.fetch_flux_energy_dB_contra],
    label=r"$Q^{%(dir)s}_{\delta B,%(species)s}$",
    is_time_dep=True,
    is_species_dep=True,
    component_basis='contravariant',
    is_vector=True)
gk_quant_registry.register(_flux_energy_dB_contra)

# ExB plus flutter, or ExB only without apar output.
_flux_particle_contra = GkQuantity(
    name="flux_particle_contra",
    description='Contravariant cross-field particle flux from ExB and flutter.',
    source=[[_M0, _M1] + _total_flux_geo, [_M0] + _es_flux_geo],
    fetch_func=[
        ff.fetch_flux_particle_contra_em, ff.fetch_flux_particle_contra_es
    ],
    label=r"$\Gamma^{%(dir)s}_{%(species)s}$",
    is_time_dep=True,
    is_species_dep=True,
    component_basis='contravariant',
    is_vector=True)
gk_quant_registry.register(_flux_particle_contra)

_flux_energy_contra = GkQuantity(
    name="flux_energy_contra",
    description='Contravariant cross-field energy flux from ExB and flutter.',
    source=[[_M2, _M3] + _total_flux_geo, [_M2] + _es_flux_geo],
    fetch_func=[ff.fetch_flux_energy_contra_em, ff.fetch_flux_energy_contra_es],
    label=r"$Q^{%(dir)s}_{%(species)s}$",
    is_time_dep=True,
    is_species_dep=True,
    component_basis='contravariant',
    is_vector=True)
gk_quant_registry.register(_flux_energy_contra)

# ---------------------------------------------------- transport coefficients
# Local diffusivities along dir=0 (x) or 1 (y); conv sets the convective part
# of Q (3/2).
_particle_D = GkQuantity(name="particle_D",
                         description='Local directional particle diffusivity.',
                         source=[[_M0, _flux_particle_contra, _geo_int_gij]],
                         fetch_func=[ff.fetch_particle_D],
                         label=r"$D^{%(dir)s}_{%(species)s}$ (m$^2$/s)",
                         is_time_dep=True,
                         is_species_dep=True)
gk_quant_registry.register(_particle_D)

_heat_chi = GkQuantity(name="heat_chi",
                       description='Local directional heat diffusivity.',
                       source=[[
                           _M0, _temp, _flux_particle_contra,
                           _flux_energy_contra, _geo_int_gij
                       ]],
                       fetch_func=[ff.fetch_heat_chi],
                       label=r"$\chi^{%(dir)s}_{%(species)s}$ (m$^2$/s)",
                       is_time_dep=True,
                       is_species_dep=True)
gk_quant_registry.register(_heat_chi)

_particle_D_gB = GkQuantity(
    name="particle_D_gB",
    description='Local particle diffusivity normalized to gyro-Bohm value.',
    source=[[_M0, _temp, _flux_particle_contra, _geo_int_gij, _geo_int_bmag]],
    fetch_func=[ff.fetch_particle_D_gB],
    label=r"$D^{%(dir)s}_{%(species)s}/D_{gB}$",
    is_time_dep=True,
    is_species_dep=True,
    is_multi_species=True)
gk_quant_registry.register(_particle_D_gB)

_heat_chi_gB = GkQuantity(
    name="heat_chi_gB",
    description='Local heat diffusivity normalized to gyro-Bohm value.',
    source=[[
        _M0, _temp, _flux_particle_contra, _flux_energy_contra, _geo_int_gij,
        _geo_int_bmag
    ]],
    fetch_func=[ff.fetch_heat_chi_gB],
    label=r"$\chi^{%(dir)s}_{%(species)s}/\chi_{gB}$",
    is_time_dep=True,
    is_species_dep=True,
    is_multi_species=True)
gk_quant_registry.register(_heat_chi_gB)

# ------------------------------------------------------------- phase space
_distf = GkQuantity(name="distf",
                    description='Distribution function.',
                    source=[[""]],
                    fetch_func=[ff.load_distf],
                    label=r"$f_{%(species)s}$",
                    is_time_dep=True,
                    is_species_dep=True)
gk_quant_registry.register(_distf)

# ----------------------------------------------------------- normalized
_rho_over_lambda = GkQuantity(
    name="rho_over_lambda",
    description='Ratio of thermal Larmor radius to Debye length.',
    source=[[_larmor_radius, _debye_length]],
    fetch_func=[ff.fetch_rho_over_lambda],
    label=r"$(\rho/\lambda_D)_{%(species)s}$",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_rho_over_lambda)

_phi_norm = GkQuantity(
    name="phi_norm",
    description='Electrostatic potential normalized as e phi/T.',
    source=[[_field, _temp]],
    fetch_func=[ff.fetch_phi_norm],
    label=r"$e\phi/T_{%(species)s}$",
    is_time_dep=True,
    is_species_dep=False)
gk_quant_registry.register(_phi_norm)

_qpar_norm = GkQuantity(
    name="qpar_norm",
    description='Parallel-energy flux normalized to n T vt.',
    source=[[_qpar, _M0, _temp, _vt]],
    fetch_func=[ff.fetch_qpar_norm],
    label=r"$q_{\parallel %(species)s}/(n T v_{th})$",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_qpar_norm)

_qperp_norm = GkQuantity(
    name="qperp_norm",
    description='Perpendicular-energy flux normalized to n T vt.',
    source=[[_qperp, _M0, _temp, _vt]],
    fetch_func=[ff.fetch_qperp_norm],
    label=r"$q_{\perp %(species)s}/(n T v_{th})$",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_qperp_norm)

_qpar_fluid_norm = GkQuantity(
    name="qpar_fluid_norm",
    description='Fluid-frame parallel heat flux normalized to n T vt.',
    source=[[_qpar_fluid, _M0, _temp, _vt]],
    fetch_func=[ff.fetch_qpar_norm],
    label=r"$q_{\parallel %(species)s}^{fluid}/(n T v_{t})$",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_qpar_fluid_norm)

_qperp_fluid_norm = GkQuantity(
    name="qperp_fluid_norm",
    description='Fluid-frame perpendicular heat flux normalized to n T vt.',
    source=[[_qperp_fluid, _M0, _temp, _vt]],
    fetch_func=[ff.fetch_qperp_norm],
    label=r"$q_{\perp %(species)s}^{fluid}/(n T v_{t})$",
    is_time_dep=True,
    is_species_dep=True)
gk_quant_registry.register(_qperp_fluid_norm)
