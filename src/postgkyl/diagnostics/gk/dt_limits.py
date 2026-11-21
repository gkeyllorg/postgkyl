"""What sets a gyrokinetic run's time step, rebuilt from its frames.

The rebuild follows the library (``gyrokinetic.c`` ``gyrokinetic_rhs``,
``gk_species_kinetic.c``): per block and charged species the step is
``min(cfl_frac / max cflrate, cfl_frac_omegaH / omega_H)``, and the run's
step is the smallest over all of them. ``cflrate`` sums, per phase-space
cell, each term's rate:

- x and z faces (``gk_collisionless_flux`` surfx, surfy kernels):
  ``3 / dx max |xdot|`` over the Gauss nodes of the cell's lower face
  (v_par at its 3 nodes, mu at its 2). Along z: ``v_par b.grad z``
  (streaming) and, with the drifts, ``(m v_par^2 / q) (curl b / B)^z``
  (curvature) ``- (b_y / q J B) mu d_x B`` (grad B) ``- (b_y / J B) d_x phi``
  (E x B); along x the drifts with ``d_z`` (signs flipped), no streaming.
- v_par faces (surfvpar, interior faces): ``5 / dv_par`` (the narrower of
  the two cells) ``max |v_par dot|`` at the volume Gauss nodes:
  ``-(b.grad z / m) (q d_z phi`` (E_par) ``+ mu d_z B`` (mirror force)``)``;
  with the drifts also ``-(v_par / q) (curl b / B).grad H``, counted with the
  magnetic drifts.
- BGK collisions (``bgk_collisions.c``): the cell average of ``nu_sum``.
- Anomalous diffusion (``gk_anomalous_diffusion`` vol kernel):
  ``2 c0(D g^xx J) (2 / dx)^2``, with ``c0`` the kernel's coefficient 0.

A direction's rate is split among its terms by their sizes at its fastest
node (they add inside one ``|.|``).

``omega_H = |q| sqrt(n_max / m) gf`` (``gk_species_kinetic_omegaH_dt``):
``n_max`` the block's largest cell-average density, ``gf`` the largest cell
average of ``(cmag J_tot^-1 2 pi / dz) (k_x^2 W g^xx J)^-1/2``
(``gyrokinetic.c`` ``omegaH_init``), ``k_x = pi /`` the block's x extent,
``W = sum_s n_pol,s m_s / B_ref^2`` the polarization weight. Runs do not
write ``W``: it is the block's field energy over ``1/2 int g^xx J (d_x
phi)^2`` (the field energy is ``1/2 int W g^xx J (d_x phi)^2``,
``gkyl_array_integrate_op_eps_gradperp_sq``).

The kernels are those of 2x2v runs with a p = 1 configuration basis; other
runs are refused.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Annotated, Literal

import numpy as np

from postgkyl import dg, io
from postgkyl.cli_spec import CliType

from . import utils
from .distf import FrameSpec, resolve_frames
from .multib_layout import block_prefixes

DT_TERMS = ("streaming", "exb", "magnetic", "epar", "mirror", "collisions",
            "diffusion")
LIB_CFL_FRAC = 1.0  # the library's cfl_frac for an input of 0
LIB_CFL_FRAC_OMEGAH = 1.7  # likewise cfl_frac_omegaH
# gyrokinetic_rhs keeps the step it was given when the new limit is less than
# 1 % below it, so a faithful rebuild is reproduced to within twice that.
LIB_DT_STICK = 0.01

_CONF = ("serendipity", 2, 1)  # basis_type, ndim, poly_order of conf fields
_G2 = np.polynomial.legendre.leggauss(2)[0]  # mu and conf-space nodes
_G3 = np.polynomial.legendre.leggauss(3)[0]  # v_par nodes (p = 2 in v_par)


@dataclass(frozen=True, eq=False)
class DtLimits:
  """The rebuilt time-step limits at a run's frames (arrays over frames).

  Attributes:
    frames: The frame numbers.
    time: Each frame's time (s).
    taken: The step the run took from each frame's time (``nan`` past the
      end of its ``-dt`` file).
    dt_cfl: ``cfl_frac`` over the largest CFL rate.
    dt_omegaH: ``cfl_frac_omegaH / omega_H``: ``inf`` with one charged
      species (the library applies none), ``nan`` where the polarization
      weight cannot be recovered.
    binding: ``'omega_H'`` where ``dt_omegaH < dt_cfl``, else the CFL term
      with the largest share of the binding cell's rate.
    cfl_cell: ``(n, 6)``: block, species index, x, z, v_par and mu cell (the
      files' indices) of the phase-space cell setting ``dt_cfl``.
    omegaH_at: ``(n, 2)``: block and species index setting ``dt_omegaH``.
    v_at: ``(n, 2)``: v_par and mu at that cell's centre.
    share: Each term's share of that cell's rate (they sum to 1).
    alone: Each term alone: ``cfl_frac`` over its own largest rate.
    agreement: ``(median, smallest, largest)`` of ``taken`` over the rebuilt
      step from the second frame on (the first step starts from the initial
      condition); ``nan`` without two such frames.
    reproduced: The median agreement lies within ``2 LIB_DT_STICK`` of 1.
    species: The charged species, as indexed in ``cfl_cell``.
    vel_cells: ``(n_vpar, n_mu)`` of each ``'<block>|<species>'`` grid.
    cfl_frac: The ``cfl_frac`` used.
    cfl_frac_omegaH: The ``cfl_frac_omegaH`` used.
    drifts: Whether the magnetic and E x B drift terms were included.
    notes: Where each setting came from.
  """

  frames: np.ndarray
  time: np.ndarray
  taken: np.ndarray
  dt_cfl: np.ndarray
  dt_omegaH: np.ndarray
  binding: tuple[str, ...]
  cfl_cell: np.ndarray
  omegaH_at: np.ndarray
  v_at: np.ndarray
  share: dict[str, np.ndarray]
  alone: dict[str, np.ndarray]
  agreement: tuple[float, float, float]
  reproduced: bool
  species: tuple[str, ...]
  vel_cells: dict[str, tuple[int, int]]
  cfl_frac: float
  cfl_frac_omegaH: float
  drifts: bool
  notes: tuple[str, ...]

  def __str__(self) -> str:
    med, lo, hi = self.agreement
    verdict = "reproduces" if self.reproduced else "does NOT reproduce"
    lines = [f"note: {n}" for n in self.notes]
    lines.append(f"rebuild {verdict} the steps taken: taken / rebuilt "
                 f"{med:.4f} (from {lo:.4f} to {hi:.4f})")
    lines.append(f"{'frame':>6} {'time (s)':>11} {'taken (s)':>11} "
                 f"{'dt_cfl (s)':>11} {'dt_omegaH':>11}  binding")
    for i, k in enumerate(self.frames):
      lines.append(f"{int(k):6d} {self.time[i]:11.4e} {self.taken[i]:11.4e} "
                   f"{self.dt_cfl[i]:11.4e} {self.dt_omegaH[i]:11.4e}  "
                   f"{self.binding[i]}")
    return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class _Setup:
  """What a run's step is made of, fixed over the run."""

  cfl: float
  cfl_H: float
  species: tuple[str, ...]
  mass: dict[str, float]
  charge: dict[str, float]
  geo: dict[int, dict]
  vel: dict[tuple[int, str], tuple[np.ndarray, ...]]
  diff: dict[tuple[int, str], np.ndarray | None]
  gf: dict[int, float]


# ------------------------------------------------------ modal fields at points
def _basis(x: float, y: float) -> np.ndarray:
  """Gkeyll's 2D p = 1 serendipity basis at (x, y) of the reference cell."""
  return dg.basis_matrix(*_CONF, np.array([x, y]))[0]


def _at(c: np.ndarray, x: float, y: float) -> np.ndarray:
  """p = 1 fields ``c (..., 4)`` at (x, y) of each cell."""
  return c[..., :4] @ _basis(x, y)


def _ddx(c: np.ndarray, y: float, dx: float) -> np.ndarray:
  """``d/dx`` of p = 1 fields at the cells' y: a p = 1 field is linear in x,
  so the difference across the cell is its derivative everywhere in it."""
  return (_at(c, 1.0, y) - _at(c, -1.0, y)) / dx


def _ddz(c: np.ndarray, x: float, dz: float) -> np.ndarray:
  return (_at(c, x, 1.0) - _at(c, x, -1.0)) / dz


def _native(values: np.ndarray):
  """A conf field ``(nx, nz, ncomp)`` as a native array for the kernels."""
  return dg.rep.wrap(np.ascontiguousarray(values.reshape(-1, values.shape[-1])))


def _back(arr, shape: tuple[int, ...]) -> np.ndarray:
  return np.asarray(arr.view()).reshape(shape[:-1] + (-1, ))


def _weak_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  return _back(dg.modal.weak_mul(*_CONF, _native(a), _native(b)), a.shape)


def _pointwise(fn, *fields: np.ndarray) -> np.ndarray:
  """``fn`` of p = 1 fields at the quadrature nodes, projected back (the
  library's projection of a pointwise function)."""
  quads = [
      np.asarray(dg.rep.modal_to_quad(*_CONF, _native(f)).view())
      for f in fields
  ]
  with np.errstate(invalid="ignore", divide="ignore"):
    vals = fn(*quads)
  out = dg.rep.quad_to_modal(*_CONF, dg.rep.wrap(np.ascontiguousarray(vals)))
  return _back(out, fields[0].shape)


# ---------------------------------------------------------------- the setup
def _read(file_name: str) -> np.ndarray:
  return np.asarray(utils.read_gfile(file_name)[1])


def _geometry(prefix: str) -> dict:
  """A block's geometry the rates use: modal fields by name, ``b.grad z`` as
  ``'bz'`` (B3overB, else B3 / bmag projected), the Hamiltonian's |B| as
  ``'Bh'`` (``geo_corn_bmag``), its cell sizes and x extent."""
  g = {
      w: _read(f"{prefix}-geo_int_{w}.gkyl")
      for w in ("gxxj", "cmag", "jacobtot_inv", "bmag", "dualcurlbhatoverB",
                "bioverJB")
  }
  g["Bh"] = _read(f"{prefix}-geo_corn_bmag.gkyl")
  b3overb = f"{prefix}-geo_int_B3overB.gkyl"
  if os.path.exists(b3overb):
    g["bz"] = _read(b3overb)
  else:
    g["bz"] = _pointwise(lambda b3, b: b3 / b,
                         _read(f"{prefix}-geo_int_B3.gkyl"), g["bmag"])
  ctx = utils.read_gfile(f"{prefix}-geo_int_gxxj.gkyl")[2].ctx
  if (ctx.get("basis_type"), len(ctx["cells"]), ctx.get("poly_order")) != _CONF:
    raise ValueError(f"{prefix}: the rebuild needs 2D p = 1 serendipity "
                     "geometry (2x2v kernels)")
  lower, upper = np.asarray(ctx["lower"]), np.asarray(ctx["upper"])
  g["dx"], g["dz"] = (upper - lower) / np.asarray(ctx["cells"])
  g["Lx"] = upper[0] - lower[0]
  return g


def _velocity(prefix: str, s: str) -> tuple[np.ndarray, ...]:
  """``(v_par centre, v_par slope, mu centre, mu slope)`` of each velocity
  cell (``v_par = centre + slope eta``): the species' velocity map holds one
  p = 1 expansion per direction."""
  vm = _read(f"{prefix}-{s}_mapc2p_vel.gkyl")
  m0, m1 = dg.basis_matrix("serendipity", 1, 1, np.array([[0.0], [1.0]]))
  centre = lambda c: c @ m0  # noqa: E731
  slope = lambda c: c @ (m1 - m0)  # noqa: E731
  return (centre(vm[:, 0, 0:2]), slope(vm[:, 0, 0:2]), centre(vm[0, :, 2:4]),
          slope(vm[0, :, 2:4]))


def _energy_over_W(g: dict, phi: np.ndarray) -> float:
  """``1/2 int g^xx J (d_x phi)^2`` over the block (its field energy over the
  polarization weight). The integrand is cubic in z and linear in x, so the
  2 x 2 Gauss rule is exact; the cell volume is ``dx dz / 4`` per node."""
  total = 0.0
  for x in _G2:
    for y in _G2:
      total += np.sum(_at(g["gxxj"], x, y) * _ddx(phi, y, g["dx"])**2)
  return float(0.5 * total * g["dx"] * g["dz"] / 4)


def _frame_time(file_name: str) -> float:
  return float(io.read_metadata(file_name).get("time", np.nan))


def _polarization_weight(prefix: str, species: str, frames: list[int],
                         g: dict) -> float:
  """W of a block from its field energy at the first frame after the first
  whose potential gives it (the energy sampled at the frame's time)."""
  energy = f"{prefix}-field_energy.gkyl"
  if not os.path.exists(energy):
    return np.nan
  t, E = utils.read_time_trace(energy)
  for k in frames[1:4] or frames[:1]:
    tk = _frame_time(f"{prefix}-{species}_M0_{k}.gkyl")
    i = np.where(np.isclose(t, tk, rtol=1e-9, atol=0))[0]
    K = _energy_over_W(g, _read(f"{prefix}-field_{k}.gkyl"))
    if len(i) and K > 0:
      return float(E[i[-1], 0] / K)
  return np.nan


def _omegaH_gf(g: dict, W: float) -> float:
  par = _weak_mul(g["cmag"], g["jacobtot_inv"]) * (2 * np.pi / g["dz"])
  perp = (np.pi / g["Lx"])**2 * W * g["gxxj"]
  inv = _pointwise(lambda p: p**-0.5, perp)
  return float(np.nanmax(dg.modal.cell_means(_weak_mul(par, inv), 2, 4)))


# ------------------------------------------------------------------- rates
def _direction(pieces: dict, factor) -> tuple[dict, dict]:
  """One direction's rate per phase-space cell: ``factor max |sum|`` over
  the nodes (last axis), split among the pieces by their sizes at that
  node; and each piece's own rate ``factor max |piece|``."""
  shape = np.broadcast_shapes(*[np.shape(p) for p in pieces.values()])
  P = {n: np.broadcast_to(p, shape) for n, p in pieces.items()}
  a = np.abs(sum(P.values()))
  i = np.argmax(a, -1)[..., None]
  rate = factor * np.take_along_axis(a, i, -1)[..., 0]
  size = {n: np.abs(np.take_along_axis(p, i, -1)[..., 0]) for n, p in P.items()}
  tot = sum(size.values())
  with np.errstate(invalid="ignore", divide="ignore"):
    parts = {n: np.where(tot > 0, rate * v / tot, 0.0) for n, v in size.items()}
  own = {n: factor * np.abs(p).max(-1) for n, p in P.items()}
  return parts, own


def _rates(g: dict, vel: tuple, q: float, m: float, phi: np.ndarray,
           drifts: bool) -> tuple[dict, dict]:
  """The collisionless CFL rate per phase-space cell ``(nx, nz, nv, nmu)``:
  ``{term: its part of the rate}`` and ``{term: its own rate}``, summed over
  the directions."""
  vp0, vp1, mu0, mu1 = vel
  VQ = np.stack([vp0 + vp1 * e for e in _G3], -1)[None, None, :, None, :, None]
  MQ = np.stack([mu0 + mu1 * e for e in _G2], -1)[None, None, None, :, None, :]
  c = lambda a: a[:, :, None, None, None, None]  # noqa: E731
  dx, dz, bz, Bh = g["dx"], g["dz"], g["bz"], g["Bh"]
  cx, cz = g["dualcurlbhatoverB"][..., 0:4], g["dualcurlbhatoverB"][..., 8:12]
  by = g["bioverJB"][..., 4:8]
  flat = lambda a: a.reshape(a.shape[:4] + (-1, ))  # noqa: E731
  dirs = []
  z = {"streaming": [], "magnetic": [], "exb": []}
  for x in _G2:  # the lower z face
    z["streaming"].append(flat(VQ * c(_at(bz, x, -1)) + 0 * MQ))
    if drifts:
      z["magnetic"].append(
          flat(m * VQ**2 / q * c(_at(cz, x, -1)) -
               c(_at(by, x, -1) * _ddx(Bh, -1, dx)) / q * MQ))
      z["exb"].append(flat(-c(_at(by, x, -1) * _ddx(phi, -1, dx)) +
                           0 * VQ * MQ))
  dirs.append(({n: np.concatenate(v, -1) for n, v in z.items() if v}, 3 / dz))
  if drifts:  # the lower x face
    xs = {"magnetic": [], "exb": []}
    for y in _G2:
      xs["magnetic"].append(
          flat(m * VQ**2 / q * c(_at(cx, -1, y)) +
               c(_at(by, -1, y) * _ddz(Bh, -1, dz)) / q * MQ))
      xs["exb"].append(
          flat(c(_at(by, -1, y) * _ddz(phi, -1, dz)) + 0 * VQ * MQ))
    dirs.append(({n: np.concatenate(v, -1) for n, v in xs.items()}, 3 / dx))
  # v_par faces: each cell's lower one (none for the first), at volume nodes
  VS = (vp0 - vp1)[None, None, :, None, None]
  M2 = MQ[..., 0, :]
  cv = lambda a: a[:, :, None, None, None]  # noqa: E731
  v = {"epar": [], "mirror": [], "magnetic": []}
  for x in _G2:
    for y in _G2:
      b_z = _at(bz, x, y)
      v["epar"].append(cv(-b_z * q * _ddz(phi, x, dz) / m) + 0 * VS * M2)
      v["mirror"].append(-cv(b_z * _ddz(Bh, x, dz) / m) * M2 + 0 * VS)
      if drifts:
        v["magnetic"].append(
            -VS / q * (cv(_at(cx, x, y)) *
                       (cv(q * _ddx(phi, y, dx)) + cv(_ddx(Bh, y, dx)) * M2) +
                       cv(_at(cz, x, y)) *
                       (cv(q * _ddz(phi, x, dz)) + cv(_ddz(Bh, x, dz)) * M2)))
  dv = np.zeros(len(vp0))
  dv[1:] = 2 * np.minimum(vp1[1:], vp1[:-1])  # the narrower cell's width
  fv = np.divide(5.0, dv, out=np.zeros_like(dv), where=dv > 0)[None, None, :,
                                                               None]
  dirs.append(({n: np.concatenate(a, -1) for n, a in v.items() if a}, fv))
  parts, own = {}, {}
  for pieces, factor in dirs:
    p, o = _direction(pieces, factor)
    for n in p:
      parts[n] = parts.get(n, 0) + p[n]
      own[n] = own.get(n, 0) + o[n]
  return parts, own


def _limits_at(prefixes: tuple[str, ...], st: _Setup, k: int,
               drifts: bool) -> dict:
  """Frame k's limits (one entry of :class:`DtLimits` per key)."""
  best = (0.0, None)  # (rate, (block, species index, cell, parts there))
  own_max = dict.fromkeys(DT_TERMS, 0.0)
  dtH, H_at = np.inf, (np.nan, np.nan)

  def block(b):
    prefix = prefixes[b]
    phi = _read(f"{prefix}-field_{k}.gkyl")
    res = []
    for si, s in enumerate(st.species):
      q, m = st.charge[s], st.mass[s]
      parts, own = _rates(st.geo[b], st.vel[(b, s)], q, m, phi, drifts)
      shape = next(iter(parts.values())).shape
      nu = f"{prefix}-{s}_bgk_nu_sum_{k}.gkyl"
      if os.path.exists(nu):
        parts["collisions"] = own["collisions"] = np.broadcast_to(
            dg.modal.cell_means(_read(nu), 2, 4)[..., 0][:, :, None, None],
            shape)
      if st.diff[(b, s)] is not None:
        parts["diffusion"] = own["diffusion"] = np.broadcast_to(
            (2 * st.diff[(b, s)] * (2 / st.geo[b]["dx"])**2)[:, :, None,
                                                             None], shape)
      R = sum(parts.values())
      idx = np.unravel_index(np.argmax(R), R.shape)
      n_max = float(
          dg.modal.cell_means(_read(f"{prefix}-{s}_M0_{k}.gkyl"), 2, 4).max())
      om = abs(q) * np.sqrt(max(n_max, 0.0) / m) * st.gf[b]
      res.append((si, float(R[idx]), idx, {
          n: float(v[idx])
          for n, v in parts.items()
      }, {
          n: float(v.max())
          for n, v in own.items()
      }, st.cfl_H / om if om > 0 else np.inf))
    return b, res

  with ThreadPoolExecutor(4) as ex:  # I/O bound
    for b, res in ex.map(block, range(len(prefixes))):
      for si, rate, idx, at, own, dth in res:
        if rate > best[0]:
          best = (rate, (b, si, idx, at))
        for n, v in own.items():
          own_max[n] = max(own_max[n], v)
        if dth < dtH:
          dtH, H_at = dth, (b, si)
  rate, (b, si, idx, at) = best
  vp0, _, mu0, _ = st.vel[(b, st.species[si])]
  if len(st.species) < 2:  # adiabatic or Boltzmann closure: no omega_H limit
    dtH = np.inf
  elif not all(np.isfinite(v) for v in st.gf.values()):
    dtH = np.nan  # no polarization weight: not rebuilt
  out = {
      "time": _frame_time(f"{prefixes[0]}-{st.species[0]}_M0_{k}.gkyl"),
      "dt_cfl": st.cfl / rate,
      "dt_omegaH": dtH,
      "cfl_cell": np.array((b, si) + tuple(idx), float),
      "omegaH_at": np.array(H_at, float),
      "v_at": np.array([vp0[idx[2]], mu0[idx[3]]])
  }
  for n in DT_TERMS:
    out[f"share|{n}"] = at.get(n, 0.0) / rate
    out[f"alone|{n}"] = st.cfl / own_max[n] if own_max[n] > 0 else np.inf
  return out


def _taken_after(dt_file: str, t: np.ndarray) -> np.ndarray:
  """The step the run took from each time (its ``-dt`` file: the first step
  written after t); ``nan`` past its end."""
  if not os.path.exists(dt_file):
    return np.full(np.shape(t), np.nan)
  td, vd = utils.read_time_trace(dt_file)
  vd = vd[:, 0]
  i = np.searchsorted(td, t, side="right")
  return np.where(i < len(vd), vd[np.minimum(i, len(vd) - 1)], np.nan)


def _agreement(taken, dt_cfl, dt_omegaH) -> tuple[float, float, float]:
  r = taken[1:] / np.fmin(
      dt_cfl[1:], np.where(np.isnan(dt_omegaH[1:]), np.inf, dt_omegaH[1:]))
  r = r[np.isfinite(r)]
  return (float(np.median(r)), float(r.min()),
          float(r.max())) if len(r) else (np.nan, ) * 3


def dt_limits(
    name: str,
    species: list[str],
    *,
    path: str = "./",
    frame: Annotated[FrameSpec, CliType(str)] = ":",
    cfl_frac: float | None = None,
    cfl_frac_omegaH: float | None = None,
    drifts: Literal["auto", "on", "off"] = "auto",
    diffusivity: float | None = None,
) -> DtLimits:
  """Rebuild what limits a gyrokinetic run's time step at its frames.

  Per frame: the CFL limit with each term's share of the binding cell's rate
  (streaming, E x B, magnetic drifts, parallel electric field, mirror force,
  BGK collisions, anomalous diffusion), the omega_H limit, and the step the
  run actually took, so the rebuild can be checked against the run.

  Reads, per block (``<name>_b<k>``, or ``<name>`` for a single-block run):
  the geometry (``geo_int_{gxxj, cmag, jacobtot_inv, bmag,
  dualcurlbhatoverB, bioverJB, B3overB or B3}``, ``geo_corn_bmag``), each
  species' ``mapc2p_vel``, and per frame ``field``, ``<s>_M0`` and, if
  present, ``<s>_bgk_nu_sum``; ``<s>_anom_diff_0`` if present;
  ``field_energy`` (for omega_H) and ``<name>-dt.gkyl`` (the steps taken).

  Args:
    name: The run's file prefix (without ``_b<k>``).
    species: The species; neutral ones (charge 0) are left out.
    path: Directory holding the run's output.
    frame: Frame number, comma list, or ``'start:stop[:step]'`` range of the
      frames to rebuild (``':'``: all).
    cfl_frac: The run's ``cfl_frac``; ``None``: the library's default (1).
    cfl_frac_omegaH: The run's ``cfl_frac_omegaH``; ``None``: the library's
      default (1.7).
    drifts: Whether the run had the magnetic and E x B drifts; ``'auto'``
      picks the choice whose rebuilt step at the run's second frame is
      nearer the step it took.
    diffusivity: A uniform anomalous diffusivity D (m^2/s) for a run that
      wrote no ``<s>_anom_diff_0`` file; ``None``: no diffusion term there.

  Returns:
    The :class:`DtLimits`.
  """
  path = path.rstrip("/") or "."
  prefixes = block_prefixes(name, path)
  notes = []

  def frac(key, value, lib):
    if value is not None and value > 0:
      notes.append(f"{key} {value:g} (given)")
      return float(value)
    notes.append(f"{key} {lib:g} (the library's default)")
    return lib

  cfl = frac("cfl_frac", cfl_frac, LIB_CFL_FRAC)
  cfl_H = frac("cfl_frac_omegaH", cfl_frac_omegaH, LIB_CFL_FRAC_OMEGAH)
  base = os.path.join(path, name)
  multib = len(prefixes) > 1 or prefixes[0] != base
  all_frames = resolve_frames(":",
                              name=base,
                              species=species[0],
                              suffix="M0",
                              block_idx=0 if multib else None)
  frames = resolve_frames(frame,
                          name=base,
                          species=species[0],
                          suffix="M0",
                          block_idx=0 if multib else None)
  meta = {
      s: io.read_metadata(f"{prefixes[0]}-{s}_M0_{all_frames[0]}.gkyl")
      for s in species
  }
  mass = {s: float(meta[s]["mass"]) for s in species}
  charge = {s: float(meta[s]["charge"]) for s in species}
  charged = tuple(s for s in species if charge[s] != 0)
  if len(charged) < len(species):
    notes.append("the neutral species' own limit is not rebuilt")
  nb = len(prefixes)
  geo = {b: _geometry(prefixes[b]) for b in range(nb)}
  vel = {(b, s): _velocity(prefixes[b], s) for b in range(nb) for s in charged}
  diff = {}
  for b in range(nb):
    for s in charged:
      own = f"{prefixes[b]}-{s}_anom_diff_0.gkyl"
      if os.path.exists(own):
        diff[(b, s)] = _read(own)[..., 0]  # the kernel's c0 of D g^xx J
      elif diffusivity is not None:
        diff[(b, s)] = diffusivity * geo[b]["gxxj"][..., 0]
      else:
        diff[(b, s)] = None
  if any(v is None for v in diff.values()):
    notes.append("no anom_diff file and no diffusivity given: no diffusion "
                 "term where the run wrote none")
  gf, Ws = {}, []
  for b in range(nb):
    W = _polarization_weight(prefixes[b], charged[0], all_frames,
                             geo[b]) if len(charged) > 1 else np.nan
    gf[b] = _omegaH_gf(geo[b], W) if np.isfinite(W) else np.nan
    Ws.append(W)
  if len(charged) < 2:
    notes.append("one charged species (adiabatic or Boltzmann closure): no "
                 "omega_H limit")
  elif not np.isfinite(Ws).all():
    notes.append("omega_H not rebuilt: no field energy at a frame's time to "
                 "give the polarization weight")
  else:
    notes.append(
        f"polarization weight W = {np.median(Ws):.4g} kg m^-3 T^-2 from the "
        f"field energy (blocks agree to "
        f"{(max(Ws) - min(Ws)) / np.median(Ws):.1e})")
  st = _Setup(cfl, cfl_H, charged, mass, charge, geo, vel, diff, gf)
  dt_file = os.path.join(path, f"{name}-dt.gkyl")

  if drifts == "auto":
    k = all_frames[1] if len(all_frames) > 1 else all_frames[0]
    t_k = _frame_time(f"{prefixes[0]}-{charged[0]}_M0_{k}.gkyl")
    taken = float(_taken_after(dt_file, np.array([t_k]))[0])
    gap = {}
    for d in (True, False):
      r = _limits_at(prefixes, st, k, d)
      gap[d] = abs(taken / np.nanmin([r["dt_cfl"], r["dt_omegaH"]]) - 1)
    if np.isfinite(taken):
      use_drifts = gap[True] <= gap[False]
      notes.append(
          f"drifts {'on' if use_drifts else 'off'} (auto): the rebuild at "
          f"frame {k} is {gap[True]:.1%} from the step taken with them, "
          f"{gap[False]:.1%} without")
    else:
      use_drifts = True
      notes.append("drifts on (auto): no step taken to choose by")
  else:
    use_drifts = drifts == "on"
    notes.append(f"drifts {drifts} (given)")

  rows = [_limits_at(prefixes, st, k, use_drifts) for k in frames]
  col = lambda key: np.array([r[key] for r in rows])  # noqa: E731
  time = col("time")
  taken = _taken_after(dt_file, time)
  dt_cfl, dt_H = col("dt_cfl"), col("dt_omegaH")
  share = {n: col(f"share|{n}") for n in DT_TERMS}
  binding = tuple("omega_H" if dt_H[i] <
                  dt_cfl[i] else max(DT_TERMS, key=lambda n: share[n][i])
                  for i in range(len(rows)))
  agreement = _agreement(taken, dt_cfl, dt_H)
  return DtLimits(frames=np.array(frames),
                  time=time,
                  taken=taken,
                  dt_cfl=dt_cfl,
                  dt_omegaH=dt_H,
                  binding=binding,
                  cfl_cell=col("cfl_cell").astype(int),
                  omegaH_at=col("omegaH_at"),
                  v_at=col("v_at"),
                  share=share,
                  alone={n: col(f"alone|{n}")
                         for n in DT_TERMS},
                  agreement=agreement,
                  reproduced=bool(abs(agreement[0] - 1) <= 2 * LIB_DT_STICK),
                  species=charged,
                  vel_cells={
                      f"{b}|{s}": (len(v[0]), len(v[2]))
                      for (b, s), v in vel.items()
                  },
                  cfl_frac=cfl,
                  cfl_frac_omegaH=cfl_H,
                  drifts=use_drifts,
                  notes=tuple(notes))
