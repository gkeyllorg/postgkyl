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
"""
import os

import numpy as np

from postgkyl.data import GData
from postgkyl.data.dg import GInterpModal
from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops
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
    under 'time' and one nodal array per entry of TRANSPORT_OUTPUTS.
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

  # Flux-surface averages of every frame, grouped into time windows.
  windows, window, times = [], None, []
  for frame in frames:
    fsa = _fsa_frame(path, name, species, frame, extra, jacobgeo, ops)
    times.append(fsa["n"].ctx.get("time", None))
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
    res = {"grid": gxx_grid, "frames": win_frames, "gxx": gxx_nodal}
    nodal = {key: _nodal(prof, num_interp) for key, prof in profiles.items()}
    for key, (_, vals, _) in nodal.items():
      res[key] = vals
    res.update(transport_coefficients(
      res["gamma"], res["Q"], res["n"], res["T"], gxx_nodal,
      dn_dx=nodal["n"][2], dT_dx=nodal["T"][2], conv=conv, grad_tol=grad_tol))
    win_times = [times[frames.index(f)] for f in win_frames]
    res["time"] = (float(np.mean(win_times)) if None not in win_times else None)
    results.append(res)
  return results
