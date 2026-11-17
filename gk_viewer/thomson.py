"""Measured profiles for the run viewer (NSTX-U shots, through PPPL Flux), read from the slim HDF5 file of a run's shot:
runs/flux/<shot>.h5 in the data directory (written by fetch_flux.sh), else viewer.json's "thomson_fallback" (the
sim-init workflow's data/h5_slim: no beam power, no TRANSP).
- Thomson scattering (MPTS) n_e and T_e, CHERS T_i, EFIT01's midplane separatrix and the beam power WF:\\PNB: nodes
  of the PPPL MDSplus HDF5 dump, copied by the workflow's tools/extract_slim_h5.py;
- TRANSP's T_i and thermal deuterium density n_D, from the shot's run flux_transp.py chose on Flux (its group
  /<shot>/TRANSP says which run, why, and which profiles the run predicts instead of fitting them to measurements).

Definitions follow nstxu_sim_init_workflow (docs/02, docs/03, docs/08):
  MPTS branch: TS3, then TS2, then BEST, the first holding every node read here; the per-channel fits FIT_NE (cm^-3)
  and FIT_TE (keV) with their errors on FIT_RADII (cm) at TS_TIMES (s); a channel counts at a frame only where
  VALID == 1.
  CHERS: CT1 and CT2 are two analyses of the same views, not independent measurements; TI ± DTI (keV) on RADIUS (cm,
  each view's radius, taken at the midplane) at TIME (s); a point counts where VALID == 1, TI > 0 and DTI >= 0 (the
  workflow's audit). An impurity temperature, taken as the deuterium one (T_D = T_CHERS, as the workflow does). A
  window uses the analysis with more such points in it, CT2 on a tie (the audit's rule at its one time).
  TRANSP: TI (eV) and ND (cm^-3) on its zone centres X at TIME3 (s); RMAJM (cm), the midplane major radius of the
  zone boundaries XB on both sides, inboard edge to outboard edge through the axis (2 N + 1 points). TRANSP models
  the plasma inside its boundary only: nothing in the SOL.
  Beam flattop (upstream/elong_tri_range_scan.py): the WF:\\PNB samples within 0.2 MW of its maximum whose
  neighbours, 15 samples on either side, all lie within 0.2 MW of them; the window spans the first to the last such
  sample.
  The scan's equilibrium time: the EFIT01 time in that window with the smallest |DRSEP| (the run's geqdsk time).
  Separatrix: EFIT01 RMIDOUT / RMIDIN (m), linear in time between the EFIT01 times where both are set.
Placement, as a distance from the separatrix (positive into the SOL on either side): Thomson and CHERS channels from
the run's own midplane separatrix moved by EFIT01's separatrix shift between the run's equilibrium time and each
sample. EFIT01 serves only for the shift: at the run's time its separatrix sits mm away from the run's (Sim 2, 203899
@ 325 ms: outboard +7.1 mm, inboard +2.5 mm), and the run's profiles were mapped on the run's own geqdsk; a plasma that
moves during the window does not smear the profile. TRANSP's zones by their midplane distance from TRANSP's own
boundary at each of its times (both on TRANSP's equilibrium): its boundary is put on the run's separatrix.
Window: the Thomson frames chosen; CHERS and TRANSP average their own samples within the span those frames stand for
(band: half a Thomson frame interval past each end frame).
"""
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import flux_transp
import gkview as gv


def flux_dir():
  """Where fetch_flux.sh puts the shots' Flux copies: the data directory's runs/flux. Read at each call, as
    fallback_dir: gkview's configuration may be reloaded."""
  return os.path.join(gv.RUNS, 'flux')


def fallback_dir():
  """viewer.json's "thomson_fallback" (the sim-init workflow's data/h5_slim, read only), or None."""
  return gv.data_path('thomson_fallback')


MPTS = '\\ACTIVESPEC::TOP.MPTS.OUTPUT_DATA.'
BRANCHES = ('TS3', 'TS2', 'BEST')
MPTS_NODES = ('TS_TIMES', 'TS_LD', 'FIT_RADII', 'FIT_NE', 'FIT_NE_ERR',
              'FIT_TE', 'FIT_TE_ERR', 'VALID')
CHERS = ('NSTX', '\\NSTX::TOP.ACTIVESPEC.CHERS.ANALYSIS.')
CHERS_SYSTEMS = ('CT1', 'CT2')
CHERS_TIE = 'CT2'  # the analysis a window takes when both hold as many points (the workflow's audit)
# TRANSP's profiles drawn: (quantity, the run quantity it is drawn with, Transp field, the namelist flag that makes
# TRANSP predict it: n_D follows the electron density)
TRANSP_PROFILES = (('T_i', 'T_ion', 'ti', 'lpredict_ti'), ('n_D', 'n_ion', 'nd',
                                                           'lpredict_ne'))
PREDICTS = {
    'lpredict_te': 'T_e',
    'lpredict_ti': 'T_i',
    'lpredict_ne': 'n_e (so n_D)'
}  # what each flag makes TRANSP predict
AEQDSK = (('EFIT01', '\\EFIT01::TOP.RESULTS.AEQDSK:'),
          ('EFIT', '\\EFIT::TOP.EFIT01.RESULTS.AEQDSK:'))
PNB = ('WF', '\\WF::TOP:PNB')
FLAT_MW, FLAT_SAMPLES = 0.2, 15  # the scan's flattop rule (upstream/elong_tri_range_scan.py)


@dataclass(frozen=True)
class Channels:
  """One quantity a diagnostic measures on fixed midplane channels: its analysis (an MPTS branch, a CHERS analysis),
    channel major radii R (m), sample times t (s), values v (channel, sample; m^-3 or eV) NaN where not valid, and
    their errors err."""
  name: str
  R: np.ndarray
  t: np.ndarray
  v: np.ndarray
  err: np.ndarray


@dataclass(frozen=True)
class Transp:
  """A TRANSP run's profiles: its id, sample times t (TIME3, s), per side ('outboard', 'inboard') the midplane major
    radius of each zone centre R[side] (zone, sample; m) and of TRANSP's boundary edge[side] (sample; m), T_i (eV) and
    n_D (m^-3) per (zone, sample); predicts: {lpredict_te / _ti / _ne: the time (s; -inf from the start) from which the
    run predicts that profile with a transport model instead of fitting it to measurements}."""
  run: str
  t: np.ndarray
  R: dict
  edge: dict
  ti: np.ndarray
  nd: np.ndarray
  predicts: dict


@dataclass(frozen=True)
class Shot:
  """A shot's measured data (its slim file): Thomson n_e (m^-3) and T_e (eV) at frames t (s) and its line density ld
    (m^-2); EFIT01's midplane separatrix; the CHERS analyses (T_i, eV) and, when there are none, why; the TRANSP run
    (None: none) and why it, or none, was taken (None: the file has no TRANSP record: fetched before TRANSP was added,
    or the workflow's copy); the beam power (MW) and its flattop (s), None without WF:\\PNB or a flattop; t_eq: the
    scan's equilibrium time in the flattop (s)."""
  shot: int
  path: str
  branch: str
  t: np.ndarray
  ne: Channels
  te: Channels
  ld: np.ndarray
  efit_t: np.ndarray  # EFIT01 times with both separatrix radii set (s)
  r_out: np.ndarray  # outboard / inboard midplane separatrix radius there (m)
  r_in: np.ndarray
  chers: tuple
  chers_why: str
  transp: Transp
  transp_why: str
  pnb_t: np.ndarray = None
  pnb: np.ndarray = None
  flat: tuple = None
  t_eq: float = None

  def separatrix(self, side, t):
    return np.interp(t, self.efit_t,
                     self.r_out if side == 'outboard' else self.r_in)

  def frame_time(self, t):
    """The time of the frame nearest t (s): a window given in rounded ms keeps its end frames."""
    return float(self.t[np.argmin(np.abs(self.t - t))])


def shot_file(shot):
  """The slim HDF5 file of a shot: the viewer's Flux copy, else the sim-init workflow's; None if neither."""
  for d in (flux_dir(), fallback_dir()):
    if d and os.path.exists(os.path.join(d, f'{shot}.h5')):
      return os.path.join(d, f'{shot}.h5')
  return None


def _text(v):
  return (v.decode() if isinstance(v, bytes) else str(v)).strip()


def _node(grp, name, units=None):
  """A node's data and dims; its units must be one of `units` (a mismatch raises)."""
  g = grp[name]
  u = _text(g['units'][()])
  if units is not None and u not in units:
    raise ValueError(f'{name}: units {u!r}, expected one of {units}')
  return np.asarray(g['data'][()]), {
      k: np.asarray(g[k][()])
      for k in ('dim0', 'dim1') if k in g
  }


def _axis_is(dim, values):
  """A node's stored dimension holds these values (any other length: no)."""
  return dim.shape == values.shape and np.allclose(dim, values)


def flattop(t, p):
  """The beam flattop window (t0, t1) by the scan's rule, or None when no sample qualifies (the scan then took the whole
    beam time range)."""
  w = FLAT_SAMPLES
  ok = [
      i for i in range(w,
                       len(p) - w) if p[i] >= np.max(p) -
      FLAT_MW and np.all(np.abs(p[i - w:i + w + 1] - p[i]) <= FLAT_MW)
  ]
  return (float(t[ok].min()), float(t[ok].max())) if ok else None


def read(shot):
  """The shot's measured data, or None when no file holds it (re-read when the file is rewritten)."""
  p = shot_file(shot)
  return None if p is None else load(p)


def load(path):
  """A slim shot file's measured data (re-read when the file is rewritten)."""
  return _load(path, os.stat(path).st_mtime_ns)


def _chers(s):
  """The file's CHERS analyses (Channels of T_i, eV, (channel, sample)), and why there are none ('' when there are)."""
  tree, pre = CHERS
  out = []
  for sy in CHERS_SYSTEMS:
    if tree not in s or f'{pre}{sy}:TI' not in s[tree]:
      continue
    g = s[tree]
    ti, dims = _node(g, f'{pre}{sy}:TI', ('keV', ))
    dti, _ = _node(g, f'{pre}{sy}:DTI', ('keV', ))
    valid, _ = _node(g, f'{pre}{sy}:VALID')
    Rcm, _ = _node(g, f'{pre}{sy}:RADIUS', ('cm', ))
    t, _ = _node(g, f'{pre}{sy}:TIME', ('sec', 's'))
    # stored (sample, channel), dim0 the channels' radii and dim1 the times: told apart by the dims' values
    if (ti.shape != (len(t), len(Rcm)) or not _axis_is(dims['dim0'], Rcm)
        or not _axis_is(dims['dim1'], t) or dti.shape != ti.shape
        or valid.shape != ti.shape):
      raise ValueError(
          f'CHERS {sy} arrays are not (sample, channel) = {(len(t), len(Rcm))}')
    ok = (valid == 1) & np.isfinite(ti) & np.isfinite(dti) & (ti > 0) & (dti
                                                                         >= 0)
    out.append(
        Channels(sy,
                 Rcm.astype(float) / 100, t.astype(float),
                 np.where(ok,
                          ti.astype(float) * 1e3, np.nan).T,
                 (dti.astype(float) * 1e3).T))
  return tuple(out), '' if out else 'none in this file'


def _midplane_zones(x, xb, rmajm):
  """(R, edge) of Transp from TRANSP's zone centres x and boundaries xb (sample, zone) and RMAJM (sample, 2 N + 1; m):
    each zone centre's midplane radius interpolated in x between the boundaries' (the axis at x = 0)."""
  n = xb.shape[1]
  if (rmajm.shape != (len(xb), 2 * n + 1)
      or np.any(np.diff(rmajm, axis=1) <= 0) or not np.allclose(xb[:, -1], 1)):
    raise ValueError(
        f'RMAJM {rmajm.shape} is not the midplane radius of the {n} zone boundaries on each side and the axis, '
        'inboard to outboard')
  bounds = {'outboard': rmajm[:, n + 1:], 'inboard': rmajm[:, n - 1::-1]}
  R = {
      side:
      np.array([
          np.interp(x[i], np.r_[0.0, xb[i]], np.r_[rmajm[i, n], b[i]])
          for i in range(len(x))
      ]).T
      for side, b in bounds.items()
  }
  return R, {side: b[:, -1] for side, b in bounds.items()}


def _transp(s):
  """(Transp or None, why) of the file's TRANSP record; why None when the file has none."""
  if 'TRANSP' not in s:
    return None, None
  g = s['TRANSP']
  run, why = _text(g.attrs.get('run', '')), _text(g.attrs.get('why', ''))
  if not run:
    return None, why
  v = {}
  for name, (dims, units) in flux_transp.VARIABLES.items():
    v[name], _ = _node(g, name, (units, ))
    got = tuple(_text(d) for d in g[name]['dims'][()])
    if len(got) != len(dims) or any(w not in (None, d)
                                    for w, d in zip(dims, got)):
      raise ValueError(f'TRANSP {name} on {got}, expected {dims}')
  R, edge = _midplane_zones(v['X'].astype(float), v['XB'].astype(float),
                            v['RMAJM'].astype(float) / 100)
  on = dict(
      zip((_text(p) for p in g.attrs.get('predicts', [])),
          np.asarray(g.attrs.get('predicts_from', []), float)))
  return Transp(run, v['TIME3'].astype(float), R, edge, v['TI'].astype(float).T,
                v['ND'].astype(float).T * 1e6, on), why


@lru_cache(maxsize=8)
def _load(path, stamp):
  import h5py
  with h5py.File(path, 'r') as f:
    key = next(k for k in f.keys() if k.isdigit())  # the top group is the shot
    s = f[key]
    a = s['ACTIVESPEC'] if 'ACTIVESPEC' in s else {}
    branch = next((b for b in BRANCHES if all(f'{MPTS}{b}:{n}' in a
                                              for n in MPTS_NODES)), None)
    if branch is None:
      raise KeyError(
          f'{path}: no MPTS branch of {BRANCHES} holds all of {MPTS_NODES}')
    t, _ = _node(a, f'{MPTS}{branch}:TS_TIMES', ('sec', 's'))
    Rcm, _ = _node(a, f'{MPTS}{branch}:FIT_RADII', ('cm', ))
    ne, dims = _node(a, f'{MPTS}{branch}:FIT_NE', ('cm-3', ))
    ne_err, _ = _node(a, f'{MPTS}{branch}:FIT_NE_ERR', ('cm-3', ))
    te, te_dims = _node(a, f'{MPTS}{branch}:FIT_TE', ('keV', ))
    te_err, _ = _node(a, f'{MPTS}{branch}:FIT_TE_ERR', ('keV', ))
    valid, _ = _node(
        a, f'{MPTS}{branch}:VALID'
    )  # its stored dim1 has 101 points: shapes are checked against FIT_NE
    ld, _ = _node(a, f'{MPTS}{branch}:TS_LD', ('cm-2', ))
    # (channel, frame), told apart by the dims' values: 42 channels and 42 frames would look alike by shape
    if (ne.shape != (len(Rcm), len(t)) or not _axis_is(dims['dim0'], t)
        or not _axis_is(dims['dim1'], Rcm) or te.shape != ne.shape
        or not _axis_is(te_dims['dim0'], t)
        or not _axis_is(te_dims['dim1'], Rcm) or ne_err.shape != ne.shape
        or te_err.shape != ne.shape or valid.shape != ne.shape
        or ld.shape != t.shape):
      raise ValueError(
          f'{path}: MPTS {branch} arrays are not (channel, frame) = {(len(Rcm), len(t))}'
      )
    tree, pre = next(
        ((tr, p) for tr, p in AEQDSK if tr in s and f'{p}ATIME' in s[tr]),
        (None, None))
    if tree is None:
      raise KeyError(f'{path}: no EFIT01 AEQDSK')
    g = s[tree]
    at, _ = _node(g, f'{pre}ATIME', ('s', ))
    rout, _ = _node(g, f'{pre}RMIDOUT', ('m', ))
    rin, _ = _node(g, f'{pre}RMIDIN', ('m', ))
    drsep, _ = _node(g, f'{pre}DRSEP', ('m', ))
    pnb_t = pnb = None
    if PNB[0] in s and PNB[1] in s[PNB[0]]:
      pnb, d = _node(s[PNB[0]], PNB[1], ('MW', ))
      pnb_t = d['dim0']
    # CHERS and TRANSP: a record that does not read leaves Thomson standing and says why
    try:
      chers, chers_why = _chers(s)
    except (KeyError, ValueError) as e:
      chers, chers_why = (), f'unreadable ({type(e).__name__}: {e})'
    try:
      transp, transp_why = _transp(s)
    except (KeyError, ValueError) as e:
      transp, transp_why = None, f'unreadable ({type(e).__name__}: {e})'
  t, R = t.astype(float), Rcm.astype(float) / 100
  ok = valid == 1
  set_ = (rout > 0) & (rin > 0)
  flat = t_eq = None
  if pnb is not None:
    flat = flattop(pnb_t, pnb)
    if flat is not None:
      inside = np.flatnonzero((at >= flat[0]) & (at <= flat[1]))
      if len(inside):
        t_eq = float(at[inside[np.argmin(np.abs(drsep[inside]))]])
  return Shot(
      int(key), path, branch, t,
      Channels(branch, R, t, np.where(ok, ne * 1e6, np.nan), ne_err * 1e6),
      Channels(branch, R, t, np.where(ok, te * 1e3, np.nan),
               te_err * 1e3), ld * 1e4, at[set_].astype(float),
      rout[set_].astype(float), rin[set_].astype(float), chers, chers_why,
      transp, transp_why, pnb_t, pnb, flat, t_eq)


# ---------------------------------------------------------------------------------------- window, placement
def band(th, t0, t1):
  """The time span (s) the Thomson frames t0..t1 stand for: half the median Thomson frame interval past each end
    frame (the timeline's shaded window). CHERS and TRANSP average their samples in it."""
  half = 0.5 * float(np.median(np.diff(th.t)))
  return t0 - half, t1 + half


def chers_in(th, lo, hi):
  """The CHERS analysis a window lo..hi (s) uses: the one with more valid points in it, CHERS_TIE on a tie; None
    without CHERS."""
  if not th.chers:
    return None
  return max(th.chers,
             key=lambda c:
             (np.isfinite(c.v[:, (c.t >= lo) &
                              (c.t <= hi)]).sum(), c.name == CHERS_TIE))


def samples(t, lo, hi):
  """The samples of times t (s) in lo..hi."""
  return np.flatnonzero((t >= lo) & (t <= hi))


@dataclass(frozen=True)
class Profile:
  """One source's profile on one side, averaged over a window, per channel (or TRANSP zone) with a value in it:
    quantity: what it is (n_e, T_e, T_i, n_D); key: the run quantity it is drawn with (gkview's n_elc, T_elc, T_ion,
    n_ion); source: Thomson, CHERS or TRANSP; label: the source with its analysis or run; samples: the source's
    samples in the window; per row the mean major radius R (m), distance d (m, from the separatrix, positive into the
    SOL) and value v (m^-3 or eV), their spreads over the samples d_sd, v_sd (NaN from one sample), the mean error err
    (NaN: none), and n, the samples with a value; predicted: TRANSP predicts it (a transport model, not a fit to
    measurements) at some of the window's samples."""
  quantity: str
  key: str
  source: str
  label: str
  samples: int
  R: np.ndarray
  d: np.ndarray
  d_sd: np.ndarray
  v: np.ndarray
  v_sd: np.ndarray
  err: np.ndarray
  n: np.ndarray
  predicted: bool = False


def _average(R, d, v, err):
  """Per row (a channel or a zone) of (row, sample) arrays, over the samples where v and d are finite: the means and
    spreads (sample standard deviation, NaN from one sample) of d and v, the mean R and error, the count; rows with no
    such sample left out."""
  ok = np.isfinite(v) & np.isfinite(d)
  n = ok.sum(1)
  keep = n > 0
  pick = lambda X: np.where(ok, X, np.nan)[keep]
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)  # one sample: no spread
    sd = lambda X: np.where(n[keep] > 1, np.nanstd(pick(X), axis=1, ddof=1), np.
                            nan)
    return dict(R=np.nanmean(pick(R), axis=1),
                d=np.nanmean(pick(d), axis=1),
                d_sd=sd(d),
                v=np.nanmean(pick(v), axis=1),
                v_sd=sd(v),
                err=np.nanmean(pick(err), axis=1),
                n=n[keep])


def _on_channels(th, ch, side, k, t_ref, rsep_ref):
  """_average of a diagnostic's channels on one side over its samples k: each sample's channels measured from the
    separatrix rsep_ref (m, the run's at its time t_ref) moved by EFIT01's shift from t_ref to that sample. A channel's
    side at a sample: R beyond the midpoint of that sample's two EFIT01 separatrix radii."""
  t = ch.t[k]
  rin, rout = th.separatrix('inboard', t), th.separatrix('outboard', t)
  sign = 1.0 if side == 'outboard' else -1.0
  rsep = rsep_ref + th.separatrix(side, t) - th.separatrix(side, t_ref)
  on = (ch.R[:, None] > 0.5 * (rin + rout)[None, :]) == (side == 'outboard')
  d = np.where(on, sign * (ch.R[:, None] - rsep[None, :]), np.nan)
  return _average(np.broadcast_to(ch.R[:, None], d.shape), d, ch.v[:, k],
                  ch.err[:, k])


def _on_zones(tr, side, k, v):
  """_average of a TRANSP profile v (zone, sample) on one side over the samples k: each zone at its midplane distance
    from TRANSP's boundary."""
  sign = 1.0 if side == 'outboard' else -1.0
  R = tr.R[side][:, k]
  return _average(R, sign * (R - tr.edge[side][None, k]), v[:, k],
                  np.full(R.shape, np.nan))


def profiles(th, side, t0, t1, t_ref, rsep_ref):
  """Every source's Profile on one side (those with no point there left out): Thomson over its frames t0..t1 (s),
    CHERS and TRANSP over their samples in band(th, t0, t1); Thomson and CHERS placed from the run's separatrix
    rsep_ref (m) at its equilibrium time t_ref (s)."""
  lo, hi = band(th, t0, t1)
  out = []
  k = samples(th.t, t0, t1)
  for q, key, ch in (('n_e', 'n_elc', th.ne), ('T_e', 'T_elc', th.te)):
    out.append(
        Profile(q, key, 'Thomson', f'Thomson (MPTS {ch.name})', len(k),
                **_on_channels(th, ch, side, k, t_ref, rsep_ref)))
  ch = chers_in(th, lo, hi)
  if ch is not None:
    k = samples(ch.t, lo, hi)
    out.append(
        Profile('T_i', 'T_ion', 'CHERS', f'CHERS {ch.name}', len(k),
                **_on_channels(th, ch, side, k, t_ref, rsep_ref)))
  tr = th.transp
  if tr is not None:
    k = samples(tr.t, lo, hi)
    for q, key, v, flag in TRANSP_PROFILES:
      out.append(
          Profile(q,
                  key,
                  'TRANSP',
                  f'TRANSP {tr.run}',
                  len(k),
                  **_on_zones(tr, side, k, getattr(tr, v)),
                  predicted=tr.predicts.get(flag, np.inf) <= hi))
  return [p for p in out if len(p.d)]


def default_window(th, eq_s):
  """The frames the viewer opens on (s): the later half of the flattop after the run's equilibrium time, i.e. from the
    midpoint of (equilibrium time, flattop end) to the flattop's last frame; without a flattop, the frame nearest the
    equilibrium time."""
  if th.flat is None:
    t = float(th.t[np.argmin(np.abs(th.t - eq_s))])
    return t, t
  fr = th.t[(th.t >= th.flat[0]) & (th.t <= th.flat[1])]
  late = fr[fr >= 0.5 * (eq_s + th.flat[1])]
  return (float(late[0]), float(late[-1])) if len(late) else (float(fr[-1]),
                                                              float(fr[-1]))
