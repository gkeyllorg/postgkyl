"""Radial turbulent transport of 3x gyrokinetic simulations.

The fluxes are the contravariant radial components (``.grad x``) of the
registry quantities ``part_flux`` and ``energy_flux`` (ExB, plus magnetic
flutter when ``apar`` is written). They are flux-surface averaged, i.e.
averaged over ``(y, z)`` weighted by the Jacobian, and then time averaged
over a window of frames::

  Gamma = <n v^x>,  Q = <(m/2) M2 v^x>,  q = Q - conv <T> Gamma.

The transport coefficients are ratios of these averages to the gradients of
the averaged profiles, normalized by ``<g^xx> = <|grad x|^2>`` so they are in
m^2/s for any radial coordinate ``x``::

  D   = -Gamma / (<g^xx> d<n>/dx),
  chi = -q / (<n> <g^xx> d<T>/dx).

They can be normalized by the gyro-Bohm diffusivities ``rho_s^2 c_s/L_n`` and
``rho_s^2 c_s/L_T``, with each species' own gradient lengths
``L_n = -<n>/(sqrt(<g^xx>) d<n>/dx)`` and ``L_T = -<T>/(sqrt(<g^xx>) d<T>/dx)``,
``c_s = sqrt(T_e/m_i)`` and ``rho_s = c_s/Omega_i`` on the averaged profiles.

The averaging is native and modal; the profiles are interpolated onto radial
points only to form the ratios. The local, unaveraged counterparts are the
registry quantities ``D``, ``chi``, ``D_gB`` and ``chi_gB``.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

from postgkyl import operations
from postgkyl.gdata import GData

from .quantities import _get_ctx_val
from .registry import gk_quant_registry

# Flux-surface directions of 3x field-aligned data: y (binormal), z (parallel).
_FSA_DIMS = [1, 2]

# Registry quantities averaged frame by frame, keyed by output name.
_FRAME_QUANTITIES = {
    "gamma": "part_flux",
    "Q": "energy_flux",
    "n": "M0",
    "T": "temp"
}

# Every output, in order, with its label (``%s``: the species' initial).
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
TransportOutput = Literal[tuple(TRANSPORT_OUTPUTS)]

# Outputs that need the gyro-Bohm normalization of gyro_bohm_normalize.
GYRO_BOHM_OUTPUTS = ("D_gB", "chi_gB")


def transport_coefficients(gamma,
                           heat_flux_tot,
                           n,
                           T,
                           gxx,
                           dn_dx,
                           dT_dx,
                           *,
                           conv: float = 1.5,
                           grad_tol: float = 1e-3) -> dict:
  """Heat flux and diffusivities from averaged radial profiles.

  All inputs are arrays on the same radial points: the particle flux
  ``gamma``, the energy flux ``heat_flux_tot``, the density ``n``,
  temperature ``T``, metric coefficient ``gxx = <|grad x|^2>`` and the radial
  gradients of ``n`` and ``T``. The convective part ``conv*T*gamma`` is
  removed from the energy flux to form the heat flux ``q``. ``D`` (``chi``)
  is NaN where ``|dn/dx|`` (``|dT/dx|``) is at most ``grad_tol`` times its
  maximum, where the ratio is meaningless.

  Returns:
    A dict with arrays ``q``, ``D`` and ``chi``.
  """
  gamma, heat_flux_tot, n, T, gxx, dn_dx, dT_dx = (np.asarray(
      a, dtype=float) for a in (gamma, heat_flux_tot, n, T, gxx, dn_dx, dT_dx))
  q = heat_flux_tot - conv * T * gamma

  def masked_ratio(num, den, grad):
    peak = np.nanmax(np.abs(grad)) if np.any(grad) else 0.0
    flat = np.abs(grad) <= grad_tol * peak
    with np.errstate(divide="ignore", invalid="ignore"):
      ratio = -num / den
    return np.where(flat, np.nan, ratio)

  return {
      "q": q,
      "D": masked_ratio(gamma, gxx * dn_dx, dn_dx),
      "chi": masked_ratio(q, n * gxx * dT_dx, dT_dx),
  }


def gyro_bohm_diffusivity(Te, bmag, mass_i: float, charge_i: float):
  """``rho_s^2 c_s`` (m^3/s), with ``c_s = sqrt(T_e/m_i)``,
  ``rho_s = c_s/Omega_i`` and ``Omega_i = |q_i| B/m_i``. Divided by a
  gradient length it is the gyro-Bohm diffusivity."""
  c_s = np.sqrt(np.asarray(Te, dtype=float) / mass_i)
  rho_s = c_s * mass_i / (abs(charge_i) * np.asarray(bmag, dtype=float))
  return rho_s**2 * c_s


def _same_window(windows: list, frames: list, species: str) -> dict:
  for res in windows:
    if res["frames"] == frames:
      return res
  raise ValueError(
      f"transport: species '{species}' has no data over frames "
      f"{frames[0]}..{frames[-1]}, which the gyro-Bohm normalization needs.")


def gyro_bohm_normalize(results: dict,
                        *,
                        ti_over_te: float = 1.0,
                        te_ref: float | None = None,
                        bmag_ref: float | None = None) -> None:
  """Add ``D_gB`` and ``chi_gB`` to every species' results, in place::

    D_gB   = D L_n/(rho_s^2 c_s),   chi_gB = chi L_T/(rho_s^2 c_s),

  with each species' own gradient lengths. ``results`` maps each species to
  its list of :func:`compute_transport` windows. ``c_s`` and ``rho_s`` use
  the mass and charge of the first ion (positively charged) species, the
  averaged ``<B>(x)``, and ``T_e`` from the electron species if listed, else
  ``T_i/ti_over_te`` with ``T_i`` the first ion's temperature (adiabatic
  electrons). ``te_ref`` (J) and ``bmag_ref`` (T) replace the ``T_e`` and
  ``B`` profiles by constants.

  Raises:
    ValueError: a species lacks mass or charge, no ion species is listed,
      more than one electron species is listed, or ``<B>`` is unknown.
  """
  charges = {}
  for species, windows in results.items():
    for key in ("mass", "charge"):
      if windows[0][key] is None:
        raise ValueError(
            f"transport: the gyro-Bohm normalization needs the {key} of "
            f"species '{species}'; pass it with the {key} option.")
    charges[species] = windows[0]["charge"]
  ions = [s for s, q in charges.items() if q > 0.0]
  elcs = [s for s, q in charges.items() if q < 0.0]
  if not ions:
    raise ValueError("transport: the gyro-Bohm normalization needs an ion "
                     f"species, got {list(results)}.")
  if len(elcs) > 1:
    raise ValueError("transport: expected at most one electron species for "
                     f"the gyro-Bohm normalization, got {elcs}.")

  for windows in results.values():
    for res in windows:
      ion = _same_window(results[ions[0]], res["frames"], ions[0])
      if te_ref is not None:
        Te = float(te_ref)
      elif elcs:
        Te = _same_window(results[elcs[0]], res["frames"], elcs[0])["T"]
      else:
        Te = ion["T"] / float(ti_over_te)
      if bmag_ref is not None:
        bmag = float(bmag_ref)
      elif ion["bmag"] is None:
        raise ValueError(
            "transport: the gyro-Bohm normalization needs the "
            "<name>-geo_int_bmag.gkyl file, or the bmag_ref option.")
      else:
        bmag = ion["bmag"]
      D0 = gyro_bohm_diffusivity(Te, bmag, ion["mass"], ion["charge"])
      with np.errstate(divide="ignore", invalid="ignore"):
        L_n = -res["n"] / (np.sqrt(res["gxx"]) * res["dn_dx"])
        L_T = -res["T"] / (np.sqrt(res["gxx"]) * res["dT_dx"])
        res["D_gB"] = res["D"] * L_n / D0
        res["chi_gB"] = res["chi"] * L_T / D0


def _common_frames(path: str, name: str, species: str,
                   frame: str | None) -> list:
  """Frames at which every per-frame quantity is available."""
  frames = None
  for qname in _FRAME_QUANTITIES.values():
    _, avail = gk_quant_registry.get(qname).get_avail_source(
        path, name, species, frame)
    frames = set(avail) if frames is None else frames & set(avail)
  if not frames:
    raise FileNotFoundError(
        "No frame has every quantity needed for the transport "
        f"({', '.join(_FRAME_QUANTITIES.values())}) of species '{species}' "
        f"(path={path!r}, name={name!r}).")
  return sorted(frames)


def _geometry(path: str, name: str, stem: str) -> GData:
  file_name = os.path.join(path, f"{name}-{stem}.gkyl")
  if not os.path.isfile(file_name):
    raise FileNotFoundError(f"transport needs the geometry file '{file_name}'.")
  return GData(file_name)


def _points(profile: GData, num_interp: int | None):
  """A 1-D modal profile and its radial derivative at interpolation points:
  ``(points dataset, values, derivative values)``."""
  points = profile.interpolate(num_interp=num_interp)
  deriv = profile.differentiate(direction=0).interpolate(num_interp=num_interp)
  return points, points.values[..., 0], deriv.values[..., 0]


def _flux_surface_profiles(path, name, species, frame, extra, jacobgeo) -> dict:
  """Flux-surface averaged (1-D modal) profiles of the per-frame quantities."""
  out = {}
  for key, qname in _FRAME_QUANTITIES.items():
    quant = gk_quant_registry.get(qname)
    combo_idx, _ = quant.get_avail_source(path, name, species, str(frame))
    field = quant.fetch(path, name, species, frame, combo_idx, **extra)
    if field.num_dims != 3:
      raise ValueError(f"transport needs 3x (x,y,z) data, got "
                       f"{field.num_dims} dimensions for '{qname}'.")
    out[key] = field.average(_FSA_DIMS, weight=jacobgeo)
  return out


def compute_transport(path: str,
                      name: str,
                      species: str,
                      frame: str | None = None,
                      *,
                      fluct: str = "none",
                      conv: float = 1.5,
                      grad_tol: float = 1e-3,
                      extra: dict | None = None,
                      per_frame: bool = False,
                      num_interp: int | None = None) -> list[dict]:
  """Flux-surface and time averaged radial transport of one species.

  Args:
    path: Simulation directory.
    name: Simulation name prefix.
    species: Species name.
    frame: Frame, comma-separated list, or ``'start:stop[:step]'`` range;
      ``None`` selects every frame available for all the needed quantities.
    fluct: ``'none'`` for the total fluxes, or ``'y'``/``'yz'`` for the
      turbulent part only (see the ``part_flux`` quantity).
    conv: Coefficient of the convective energy flux removed from ``Q``.
    grad_tol: Relative gradient below which ``D`` and ``chi`` are NaN.
    extra: Further fetch options (e.g. ``mass``, ``charge``,
      ``species_idx``).
    per_frame: Return each frame's flux-surface averaged profiles instead
      of their time average.
    num_interp: Radial points per cell (default ``poly_order + 1``).

  Returns:
    One dict per time window (a single one unless ``per_frame``) holding
    the points dataset under ``points``, the frames under ``frames``, the
    mean time under ``time``, one array per :data:`TRANSPORT_OUTPUTS` entry
    except the gyro-Bohm ones (see :func:`gyro_bohm_normalize`), the radial
    gradients ``dn_dx``/``dT_dx``, the averaged field ``bmag`` (``None``
    without the ``geo_int_bmag`` file) and the species' ``mass``/``charge``
    (``None`` when unknown).
  """
  path = path.rstrip("/") + "/"
  extra = dict(extra or {}, fluct=fluct)
  frames = _common_frames(path, name, species, frame)

  jacobgeo = _geometry(path, name, "geo_int_jacobgeo")
  gxx = operations.select(_geometry(path, name, "geo_int_gij"), comp=0)
  _, gxx_points, _ = _points(gxx.average(_FSA_DIMS, weight=jacobgeo),
                             num_interp)
  bmag_points = None
  if os.path.isfile(os.path.join(path, f"{name}-geo_int_bmag.gkyl")):
    bmag = _geometry(path, name, "geo_int_bmag")
    _, bmag_points, _ = _points(bmag.average(_FSA_DIMS, weight=jacobgeo),
                                num_interp)

  windows, total, times, attrs = [], None, {}, {}
  for fr in frames:
    profiles = _flux_surface_profiles(path, name, species, fr, extra, jacobgeo)
    times[fr] = profiles["n"].ctx.get("time")
    if not attrs:
      for key in ("mass", "charge"):
        try:
          attrs[key] = float(_get_ctx_val(profiles["T"], key, **extra))
        except KeyError:
          attrs[key] = None  # Only the gyro-Bohm normalization needs them.
    if per_frame:
      windows.append(([fr], profiles))
    elif total is None:
      total = profiles
    else:
      total = {key: total[key] + profiles[key] for key in total}
  if not per_frame:
    windows.append((frames, {
        key: prof * (1.0 / len(frames))
        for key, prof in total.items()
    }))

  results = []
  for win_frames, profiles in windows:
    sampled = {key: _points(prof, num_interp) for key, prof in profiles.items()}
    res = {
        "points": sampled["n"][0],
        "frames": win_frames,
        "gxx": gxx_points,
        "bmag": bmag_points,
        **attrs
    }
    for key, (_, values, _) in sampled.items():
      res[key] = values
    res["dn_dx"], res["dT_dx"] = sampled["n"][2], sampled["T"][2]
    res.update(
        transport_coefficients(res["gamma"],
                               res["Q"],
                               res["n"],
                               res["T"],
                               gxx_points,
                               res["dn_dx"],
                               res["dT_dx"],
                               conv=conv,
                               grad_tol=grad_tol))
    win_times = [times[f] for f in win_frames]
    res["time"] = None if None in win_times else float(np.mean(win_times))
    results.append(res)
  return results


def transport(name: str,
              species: str,
              frame: str | None = None,
              *,
              path: str = "./",
              outputs: list[TransportOutput] | None = None,
              fluct: Literal["none", "y", "yz"] = "none",
              conv: float = 1.5,
              grad_tol: float = 1e-3,
              per_frame: bool = False,
              num_interp: int | None = None,
              mass: list[float] | None = None,
              charge: list[float] | None = None,
              ti_over_te: float = 1.0,
              te_ref: float | None = None,
              bmag_ref: float | None = None,
              tag: str = "transport",
              label: str | None = None) -> list:
  """Radial turbulent transport of a 3x gyrokinetic simulation.

  Flux-surface (``y``, ``z``, Jacobian-weighted) and time averaged radial
  profiles, one dataset per species, window and output:

  * ``gamma``: particle flux ``<Gamma^x>`` (ExB, plus flutter with ``apar``).

  * ``Q``: energy flux ``<Q^x> = <(m/2) M2 v^x>``.

  * ``q``: heat flux ``Q - conv <T> Gamma``.

  * ``D``: particle diffusivity ``-Gamma/(<g^xx> d<n>/dx)`` (m^2/s).

  * ``chi``: heat diffusivity ``-q/(<n> <g^xx> d<T>/dx)`` (m^2/s).

  * ``D_gB``, ``chi_gB``: those over the gyro-Bohm diffusivity
    ``rho_s^2 c_s/L``, with the species' own gradient length.

  * ``n``, ``T``, ``gxx``: the averaged density, temperature and
    ``<|grad x|^2>``.

  The fluxes are contravariant radial components (``.grad x``). For the
  gyro-Bohm outputs, ``c_s = sqrt(T_e/m_i)`` and ``rho_s = c_s/Omega_i`` use
  the first ion species' mass and charge and the averaged ``<B>(x)``;
  ``T_e`` is the electron species' ``<T>(x)`` if listed, else
  ``T_i/ti_over_te`` (adiabatic electrons).

  Args:
    name: Simulation name prefix (e.g. ``'gk_tcv_3x2v_p1'``).
    species: Species name, or a comma-separated list (e.g. ``'elc,ion'``).
    frame: Frames to average over: a frame, a comma-separated list, or a
      ``'start:stop[:step]'`` range; every available frame by default.
    path: Directory containing the simulation files.
    outputs: Profiles to return (repeat the option); default
      ``gamma, q, D, chi``.
    fluct: ``none`` for the total fluxes, ``y`` or ``yz`` for the turbulent
      part only, the correlation of the fluctuations about the ``y`` or
      ``(y, z)`` average.
    conv: Coefficient ``c`` of the convective energy flux ``c <T> Gamma``
      removed from ``Q`` to form ``q`` (e.g. 0, 3/2 or 5/2).
    grad_tol: ``D`` (``chi``) is NaN where ``|d<n>/dx|`` (``|d<T>/dx|``) is
      at most this fraction of its maximum.
    per_frame: Return each frame's flux-surface averaged profiles instead of
      their time average (e.g. to collect a space-time diagram).
    num_interp: Radial points per cell (default ``poly_order + 1``).
    mass: Species mass, one value or one per species, when the files lack
      it.
    charge: Species charge, one value or one per species, when the files
      lack it.
    ti_over_te: Ion-to-electron temperature ratio of adiabatic electrons
      (gyro-Bohm outputs without an electron species).
    te_ref: Constant electron temperature (J) replacing ``T_e(x)`` in the
      gyro-Bohm outputs.
    bmag_ref: Constant magnetic field (T) replacing ``<B>(x)`` in the
      gyro-Bohm outputs.
    tag: Tag prefix of the datasets, ``<tag>_<output>[_<species>]``.
    label: Label override for the datasets.

  Returns:
    A list of NumPy-backed 1-D datasets on the radial points.

  Raises:
    ValueError: an output name is unknown, or the data are not 3x.
    FileNotFoundError: a needed geometry file or quantity is missing.
  """
  outputs = list(outputs or ("gamma", "q", "D", "chi"))
  unknown = [o for o in outputs if o not in TRANSPORT_OUTPUTS]
  if unknown:
    raise ValueError(f"transport: unknown output(s) {', '.join(unknown)}; "
                     f"choose among {', '.join(TRANSPORT_OUTPUTS)}.")
  species_list = [s.strip() for s in species.split(",") if s.strip()]
  extra = {}
  for key, value in (("mass", mass), ("charge", charge)):
    if value:
      extra[key] = value[0] if len(value) == 1 else list(value)

  results = {
      sp:
      compute_transport(path,
                        name,
                        sp,
                        frame,
                        fluct=fluct,
                        conv=conv,
                        grad_tol=grad_tol,
                        extra=dict(extra, species_idx=idx),
                        per_frame=per_frame,
                        num_interp=num_interp)
      for idx, sp in enumerate(species_list)
  }
  if any(o in GYRO_BOHM_OUTPUTS for o in outputs):
    gyro_bohm_normalize(results,
                        ti_over_te=ti_over_te,
                        te_ref=te_ref,
                        bmag_ref=bmag_ref)

  datasets = []
  for sp, windows in results.items():
    for res in windows:
      for output in outputs:
        template = TRANSPORT_OUTPUTS[output]
        if label is not None:
          out_label = label
        elif "%s" in template:  # gxx belongs to no species.
          out_label = template % sp[0]
        else:
          out_label = template
        if label is not None and len(species_list) > 1:
          out_label += f" {sp}"
        if len(windows) > 1:
          out_label += f" f{res['frames'][-1]}"
        out_tag = f"{tag}_{output}" + (f"_{sp}"
                                       if len(species_list) > 1 else "")
        ctx = {"frame": res["frames"][-1], "time": res["time"]}
        datasets.append(res["points"]._result(res["points"].grid,
                                              res[output][..., np.newaxis],
                                              tag=out_tag,
                                              label=out_label,
                                              **ctx))
  return datasets
