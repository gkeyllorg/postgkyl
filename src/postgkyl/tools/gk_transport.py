"""
Radial turbulent transport of 3x gyrokinetic simulations.

The fluxes are the contravariant radial components (.grad x) of the registry
quantities 'part_flux' and 'energy_flux' (ExB, plus magnetic flutter when apar
is written). They are flux-surface averaged, i.e. averaged over (y,z) weighted
by the Jacobian, and then time averaged over a window of frames:
  Gamma = <n v^x>,  Q = <(m/2) M2 v^x>,  q = Q - conv*<T>*Gamma.
The transport coefficients are ratios of these averages to the gradients of the
averaged profiles, normalized by <|grad x|^2> = <g^xx> so they are in m^2/s
whatever the radial coordinate x is:
  D   = -Gamma / (<g^xx> d<n>/dx),
  chi = -q / (<n> <g^xx> d<T>/dx).
They can be normalized by the local gyro-Bohm diffusivities rho_s^2 c_s/L_n and
rho_s^2 c_s/L_T, with the species' own gradient lengths
  L_n = -<n>/(sqrt(<g^xx>) d<n>/dx),  L_T = -<T>/(sqrt(<g^xx>) d<T>/dx),
and c_s = sqrt(T_e/m_i), rho_s = c_s/Omega_i on the averaged profiles.
"""
import os

import numpy as np

from postgkyl.data import GData
from postgkyl.data.dg import GInterpModal
from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops
from postgkyl.utils.gk_quantities.fetch_funcs import _get_ctx_val
from postgkyl.utils.gk_quantities.registry import gk_quant_registry

# Flux-surface directions of 3x field-aligned data: y (binormal) and z (parallel).
_FSA_DIRS = [1, 2]

# Registry quantities averaged frame by frame, keyed by output name.
_FRAME_QUANTITIES = {"gamma": "part_flux", "Q": "energy_flux", "n": "M0", "T": "temp"}

# Every output of compute_transport, in order, with its LaTeX label (%s: species).
TRANSPORT_OUTPUTS = {
  "gamma": r"$\langle\Gamma^x_{%s}\rangle$",
  "Q": r"$\langle Q^x_{%s}\rangle$",
  "q": r"$\langle q^x_{%s}\rangle$",
  "D": r"$D_{%s}$ (m$^2$/s)",
  "chi": r"$\chi_{%s}$ (m$^2$/s)",
  "D_gB": r"$D_{%s}/D_{gB}$",
  "chi_gB": r"$\chi_{%s}/\chi_{gB}$",
  "n": r"$\langle n_{%s}\rangle$ (m$^{-3}$)",
  "T": r"$\langle T_{%s}\rangle$ (J)",
  "gxx": r"$\langle g^{xx}\rangle$",
}


def transport_coefficients(gamma, heat_flux_tot, n, T, gxx, dn_dx, dT_dx,
                           conv: float = 1.5, grad_tol: float = 1e-3) -> dict:
  """
  Heat flux and transport coefficients from averaged radial profiles.

  Inputs are arrays on the same radial nodes: the particle flux gamma, the
  energy flux heat_flux_tot, the density n, temperature T, the metric
  coefficient gxx = <|grad x|^2> and the radial gradients of n and T.
  The convective part conv*T*gamma is removed from the energy flux to form the
  heat flux q. D (chi) is NaN where |dn/dx| (|dT/dx|) is below grad_tol times
  its maximum, where the ratio is meaningless.

  Returns a dict with q, D and chi.
  """
  gamma, heat_flux_tot, n, T, gxx, dn_dx, dT_dx = (
    np.asarray(a, dtype=float) for a in (gamma, heat_flux_tot, n, T, gxx, dn_dx, dT_dx))

  q = heat_flux_tot - conv*T*gamma

  def _masked_ratio(num, den, grad):
    small = np.abs(grad) <= grad_tol*np.nanmax(np.abs(grad)) if np.any(grad) else np.ones_like(grad, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
      ratio = -num/den
    return np.where(small, np.nan, ratio)

  D = _masked_ratio(gamma, gxx*dn_dx, dn_dx)
  chi = _masked_ratio(q, n*gxx*dT_dx, dT_dx)
  return {"q": q, "D": D, "chi": chi}


# Outputs that need the gyro-Bohm normalization of gyro_bohm_normalize.
GYRO_BOHM_OUTPUTS = ("D_gB", "chi_gB")


def gyro_bohm_diffusivity(Te, bmag, mass_i: float, charge_i: float):
  """
  rho_s^2 c_s (m^3/s), with c_s = sqrt(T_e/m_i), rho_s = c_s/Omega_i and
  Omega_i = |q_i| B/m_i. Divided by a gradient length it is the gyro-Bohm
  diffusivity.
  """
  c_s = np.sqrt(np.asarray(Te, dtype=float)/mass_i)
  rho_s = c_s*mass_i/(abs(charge_i)*np.asarray(bmag, dtype=float))
  return rho_s**2*c_s


def _same_window(windows: list, frames: list, species: str) -> dict:
  for res in windows:
    if res["frames"] == frames:
      return res
  raise ValueError(f"gk-transport: species '{species}' has no data over frames "
                   f"{frames[0]}..{frames[-1]}, which the gyro-Bohm normalization needs.")


def gyro_bohm_normalize(results: dict, Ti_over_Te: float = 1.0, Te_ref=None, bmag_ref=None):
  """
  Add the gyro-Bohm normalized diffusivities to the compute_transport results
  of every species, in place:
    D_gB   = D / (rho_s^2 c_s/L_n)   = D L_n/(rho_s^2 c_s),
    chi_gB = chi / (rho_s^2 c_s/L_T) = chi L_T/(rho_s^2 c_s),
  with each species' own gradient lengths L_n and L_T.

  results maps each species to its list of compute_transport windows. c_s and
  rho_s use the mass and charge of the first ion (positively charged) species,
  the averaged <B>(x), and T_e from the electron species if listed, else
  T_i/Ti_over_Te with T_i the first ion's temperature (adiabatic electrons).
  Te_ref (J) and bmag_ref (T) replace the T_e and B profiles by constants.
  """
  charges = {}
  for species, windows in results.items():
    charges[species] = windows[0]["charge"]
    for key in ("mass", "charge"):
      if windows[0][key] is None:
        raise ValueError(f"gk-transport: the gyro-Bohm normalization needs the {key} of "
                         f"species '{species}'; pass it with '--extra {key}=<value>'.")
  ions = [s for s, q in charges.items() if q > 0.0]
  elcs = [s for s, q in charges.items() if q < 0.0]
  if not ions:
    raise ValueError("gk-transport: the gyro-Bohm normalization needs an ion species "
                     f"in --species, got {list(results)}.")
  if len(elcs) > 1:
    raise ValueError("gk-transport: expected at most one electron species for the "
                     f"gyro-Bohm normalization, got {elcs}.")

  for species, windows in results.items():
    for res in windows:
      ion = _same_window(results[ions[0]], res["frames"], ions[0])
      if Te_ref is not None:
        Te = float(Te_ref)
      elif elcs:
        Te = _same_window(results[elcs[0]], res["frames"], elcs[0])["T"]
      else:
        Te = ion["T"]/float(Ti_over_Te)
      if bmag_ref is not None:
        bmag = float(bmag_ref)
      elif ion["bmag"] is None:
        raise ValueError("gk-transport: the gyro-Bohm normalization needs the "
                         "<prefix>-geo_int_bmag.gkyl file, or '--extra bmag_ref=<value>'.")
      else:
        bmag = ion["bmag"]
      D0 = gyro_bohm_diffusivity(Te, bmag, ion["mass"], ion["charge"])

      with np.errstate(divide="ignore", invalid="ignore"):
        L_n = -res["n"]/(np.sqrt(res["gxx"])*res["dn_dx"])
        L_T = -res["T"]/(np.sqrt(res["gxx"])*res["dT_dx"])
        res["D_gB"] = res["D"]*L_n/D0
        res["chi_gB"] = res["chi"]*L_T/D0


def _parse_frames(frame_inp):
  if frame_inp is None:
    return None
  if isinstance(frame_inp, int):
    return str(frame_inp)
  if isinstance(frame_inp, (list, tuple)):
    return ",".join(str(f) for f in frame_inp)
  return str(frame_inp)


def common_frames(path: str, name: str, species: str, frame_inp=None) -> list:
  """Frames at which every quantity needed by compute_transport is available."""
  frames = None
  for qname in _FRAME_QUANTITIES.values():
    _, avail = gk_quant_registry.get(qname).get_avail_source(path, name, species,
                                                             _parse_frames(frame_inp))
    frames = set(avail) if frames is None else frames & set(avail)
  if not frames:
    raise FileNotFoundError(f"No frame has every quantity needed for the transport "
                            f"({', '.join(_FRAME_QUANTITIES.values())}) of species "
                            f"'{species}' (path='{path}', name='{name}').")
  return sorted(frames)


def _geo(path: str, name: str, stem: str) -> GData:
  file_name = os.path.join(path, f"{name}-{stem}.gkyl")
  if not os.path.isfile(file_name):
    raise FileNotFoundError(f"gk-transport needs the geometry file '{file_name}'.")
  return GData(file_name)


def _nodal(profile: GData, num_interp: int | None):
  """Values of a 1D DG profile on uniform nodes, with the node grid (cell edges)."""
  profile.ctx.setdefault("grid_type", "uniform")  # Averages live on the uniform grid.
  dg = GInterpModal(profile, num_interp=num_interp)
  grid, values = dg.interpolate()
  _, deriv = dg.differentiate(direction=0)
  return grid, values[..., 0], deriv[..., 0]


def _fsa_frame(path, name, species, frame, extra, jacobgeo, ops) -> dict:
  """Flux-surface averaged (1D DG) profiles of the per-frame quantities at one frame."""
  out = {}
  for key, qname in _FRAME_QUANTITIES.items():
    quant = gk_quant_registry.get(qname)
    combo_idx, _ = quant.get_avail_source(path, name, species, str(frame))
    field = quant.fetch(path, name, species, frame, combo_idx, **extra)
    if field.get_num_dims() != 3:
      raise ValueError(f"gk-transport needs 3x (x,y,z) data, got {field.get_num_dims()} "
                       f"dimensions for '{qname}'.")
    out[key] = ops.average(_FSA_DIRS, field, weight=jacobgeo)
  return out


def compute_transport(path: str, name: str, species: str, frame=None, fluct: str = "none",
                      conv: float = 1.5, grad_tol: float = 1e-3, extra: dict | None = None,
                      per_frame: bool = False, num_interp: int | None = None) -> list:
  """
  Flux-surface and time averaged radial transport of one species.

  Parameters:
    path, name, species: simulation directory, file prefix and species name.
    frame: frame selection with the gk-load-quantity syntax (int, list,
      'a,b,c' or 'start:stop[:step]'); None means every available frame.
    fluct: 'none' for the total fluxes, or 'y'/'yz' for the turbulent part only,
      the correlation of the fluctuations about the y or (y,z) average.
    conv: coefficient of the convective energy flux conv*<T>*Gamma removed from Q.
    grad_tol: relative gradient below which D and chi are set to NaN.
    extra: extra keyword arguments of the fetch functions (e.g. mass, charge).
    per_frame: if True, return the flux-surface averaged profiles of each
      frame instead of their time average.
    num_interp: number of radial nodes per cell (default poly_order+1).

  Returns:
    A list with one dict per time window (a single one unless per_frame), each
    holding the node grid under 'grid', the frame(s) under 'frames', the time
    under 'time', one nodal array per entry of TRANSPORT_OUTPUTS except the
    gyro-Bohm ones (see gyro_bohm_normalize), the radial gradients 'dn_dx' and
    'dT_dx', the averaged magnetic field 'bmag' (None without the geo_int_bmag
    file) and the species' 'mass' and 'charge' (None when unknown).
  """
  path = path.rstrip("/") + "/"
  extra = dict(extra or {})
  extra["fluct"] = fluct
  frames = common_frames(path, name, species, frame)

  ops = GkeyllDGops()
  jacobgeo = _geo(path, name, "geo_int_jacobgeo")
  gij = _geo(path, name, "geo_int_gij")
  num_basis = jacobgeo.get_values().shape[-1]
  gxx = GData(ctx=gij.ctx)
  gxx.push(gij.get_grid(), gij.get_values()[..., :num_basis])
  gxx_grid, gxx_nodal, _ = _nodal(ops.average(_FSA_DIRS, gxx, weight=jacobgeo), num_interp)
  bmag_nodal = None
  if os.path.isfile(os.path.join(path, f"{name}-geo_int_bmag.gkyl")):
    bmag = _geo(path, name, "geo_int_bmag")
    _, bmag_nodal, _ = _nodal(ops.average(_FSA_DIRS, bmag, weight=jacobgeo), num_interp)

  # Flux-surface averages of every frame, grouped into time windows.
  windows, window, times = [], None, []
  for frame in frames:
    fsa = _fsa_frame(path, name, species, frame, extra, jacobgeo, ops)
    times.append(fsa["n"].ctx.get("time", None))
    if frame == frames[0]:
      attrs = {}
      for key in ("mass", "charge"):
        try:
          attrs[key] = float(_get_ctx_val(fsa["T"], key, **extra))
        except KeyError:
          attrs[key] = None  # Only needed by the gyro-Bohm normalization.
    if per_frame:
      windows.append(([frame], fsa))
    elif window is None:
      window = fsa
    else:
      for key in window:
        window[key].set_values(window[key].get_values() + fsa[key].get_values())
  if not per_frame:
    for key in window:
      window[key].set_values(window[key].get_values()/len(frames))
    windows.append((frames, window))

  results = []
  for win_frames, profiles in windows:
    res = {"grid": gxx_grid, "frames": win_frames, "gxx": gxx_nodal, "bmag": bmag_nodal, **attrs}
    nodal = {key: _nodal(prof, num_interp) for key, prof in profiles.items()}
    for key, (_, vals, _) in nodal.items():
      res[key] = vals
    res.update(transport_coefficients(
      res["gamma"], res["Q"], res["n"], res["T"], gxx_nodal,
      dn_dx=nodal["n"][2], dT_dx=nodal["T"][2], conv=conv, grad_tol=grad_tol))
    res["dn_dx"], res["dT_dx"] = nodal["n"][2], nodal["T"][2]
    win_times = [times[frames.index(f)] for f in win_frames]
    res["time"] = (float(np.mean(win_times)) if None not in win_times else None)
    results.append(res)
  return results
