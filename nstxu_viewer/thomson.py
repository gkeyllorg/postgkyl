"""Measured profiles for the NSTX-U run viewer: Thomson scattering (MPTS) n_e of a run's shot, read from a slim copy of
the PPPL MDSplus HDF5 dump (runs/flux/<shot>.h5 in the data directory, written by fetch_flux.sh; else viewer.json's
"thomson_fallback", the sim-init workflow's data/h5_slim, which has no beam power).

Definitions follow nstxu_sim_init_workflow (docs/02, docs/08):
  MPTS branch: TS3, then TS2, then BEST, the first holding every node read here; the per-channel fits FIT_NE (cm^-3) and
  FIT_NE_ERR on FIT_RADII (cm) at TS_TIMES (s); a channel counts at a frame only where VALID == 1.
  Beam flattop (upstream/elong_tri_range_scan.py): the WF:\\PNB samples within 0.2 MW of its maximum whose neighbours,
  15 samples on either side, all lie within 0.2 MW of them; the window spans the first to the last such sample.
  The scan's equilibrium time: the EFIT01 time in that window with the smallest |DRSEP| (the run's geqdsk time).
  Separatrix: EFIT01 RMIDOUT / RMIDIN (m), linear in time between the EFIT01 times where both are set.
Each frame's channels are placed by their distance (positive into the SOL on either side) from the run's own midplane
separatrix moved by EFIT01's separatrix shift between the run's equilibrium time and that frame. EFIT01 serves only
for the shift: at the run's time its separatrix sits mm away from the run's (Sim 2, 203899 @ 325 ms: outboard +7.1 mm,
inboard +2.5 mm), and the run's profiles were mapped on the run's own geqdsk; a plasma that moves during the window
does not smear the profile.
"""
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import gkview as gv

CACHE = os.path.join(gv.RUNS, 'flux')
WORKFLOW = gv.data_path('thomson_fallback')   # viewer.json: the sim-init workflow's data/h5_slim (read only)

MPTS = '\\ACTIVESPEC::TOP.MPTS.OUTPUT_DATA.'
BRANCHES = ('TS3', 'TS2', 'BEST')
MPTS_NODES = ('TS_TIMES', 'TS_LD', 'FIT_RADII', 'FIT_NE', 'FIT_NE_ERR', 'VALID')
AEQDSK = (('EFIT01', '\\EFIT01::TOP.RESULTS.AEQDSK:'), ('EFIT', '\\EFIT::TOP.EFIT01.RESULTS.AEQDSK:'))
PNB = ('WF', '\\WF::TOP:PNB')
FLAT_MW, FLAT_SAMPLES = 0.2, 15   # the scan's flattop rule (upstream/elong_tri_range_scan.py)


@dataclass
class Thomson:
    shot: int
    path: str
    branch: str
    t: np.ndarray        # frame times (s)
    R: np.ndarray        # channel major radii (m)
    ne: np.ndarray       # (channel, frame) m^-3, NaN where VALID != 1
    err: np.ndarray      # its fit error (m^-3)
    ld: np.ndarray       # line density per frame (m^-2)
    efit_t: np.ndarray   # EFIT01 times with both separatrix radii set (s)
    r_out: np.ndarray    # outboard / inboard midplane separatrix radius there (m)
    r_in: np.ndarray
    pnb_t: np.ndarray = None   # beam power (MW) and its times; None when the file has no WF:\PNB
    pnb: np.ndarray = None
    flat: tuple = None         # beam flattop window (s); None without beam power or when no sample qualifies
    t_eq: float = None         # the scan's equilibrium time in that window (s)

    def separatrix(self, side, t):
        return np.interp(t, self.efit_t, self.r_out if side == 'outboard' else self.r_in)

    def frame_time(self, t):
        """The time of the frame nearest t (s): a window given in rounded ms keeps its end frames."""
        return float(self.t[np.argmin(np.abs(self.t - t))])


def shot_file(shot):
    """The slim HDF5 file of a shot: the viewer's Flux copy, else the sim-init workflow's; None if neither."""
    for d in (CACHE, WORKFLOW):
        if d and os.path.exists(os.path.join(d, f'{shot}.h5')):
            return os.path.join(d, f'{shot}.h5')
    return None


def _node(grp, name, units=None):
    """A node's data and dims; its units must be one of `units` (a mismatch raises)."""
    g = grp[name]
    u = g['units'][()]
    u = (u.decode() if isinstance(u, bytes) else str(u)).strip()
    if units is not None and u not in units:
        raise ValueError(f'{name}: units {u!r}, expected one of {units}')
    return np.asarray(g['data'][()]), {k: np.asarray(g[k][()]) for k in ('dim0', 'dim1') if k in g}


def flattop(t, p):
    """The beam flattop window (t0, t1) by the scan's rule, or None when no sample qualifies (the scan then took the whole
    beam time range)."""
    w = FLAT_SAMPLES
    ok = [i for i in range(w, len(p) - w)
          if p[i] >= np.max(p) - FLAT_MW and np.all(np.abs(p[i - w:i + w + 1] - p[i]) <= FLAT_MW)]
    return (float(t[ok].min()), float(t[ok].max())) if ok else None


def read(shot):
    """The shot's Thomson data, or None when no file holds it (re-read when the file is rewritten)."""
    p = shot_file(shot)
    return None if p is None else _load(p, os.stat(p).st_mtime_ns)


@lru_cache(maxsize=8)
def _load(path, stamp):
    import h5py
    with h5py.File(path, 'r') as f:
        key = next(iter(f.keys()))   # the top group is the shot
        s = f[key]
        a = s['ACTIVESPEC'] if 'ACTIVESPEC' in s else {}
        branch = next((b for b in BRANCHES if all(f'{MPTS}{b}:{n}' in a for n in MPTS_NODES)), None)
        if branch is None:
            raise KeyError(f'{path}: no MPTS branch of {BRANCHES} holds all of {MPTS_NODES}')
        t, _ = _node(a, f'{MPTS}{branch}:TS_TIMES', ('sec', 's'))
        Rcm, _ = _node(a, f'{MPTS}{branch}:FIT_RADII', ('cm',))
        ne, dims = _node(a, f'{MPTS}{branch}:FIT_NE', ('cm-3',))
        err, _ = _node(a, f'{MPTS}{branch}:FIT_NE_ERR', ('cm-3',))
        valid, _ = _node(a, f'{MPTS}{branch}:VALID')   # its stored dim1 has 101 points: shapes are checked against FIT_NE
        ld, _ = _node(a, f'{MPTS}{branch}:TS_LD', ('cm-2',))
        # (channel, frame), told apart by the dims' values: 42 channels and 42 frames would look alike by shape
        if (ne.shape != (len(Rcm), len(t)) or not np.allclose(dims['dim0'], t) or not np.allclose(dims['dim1'], Rcm)
                or err.shape != ne.shape or valid.shape != ne.shape or ld.shape != t.shape):
            raise ValueError(f'{path}: MPTS {branch} arrays are not (channel, frame) = {(len(Rcm), len(t))}')
        tree, pre = next(((tr, p) for tr, p in AEQDSK if tr in s and f'{p}ATIME' in s[tr]), (None, None))
        if tree is None:
            raise KeyError(f'{path}: no EFIT01 AEQDSK')
        g = s[tree]
        at, _ = _node(g, f'{pre}ATIME', ('s',))
        rout, _ = _node(g, f'{pre}RMIDOUT', ('m',))
        rin, _ = _node(g, f'{pre}RMIDIN', ('m',))
        drsep, _ = _node(g, f'{pre}DRSEP', ('m',))
        pnb_t = pnb = None
        if PNB[0] in s and PNB[1] in s[PNB[0]]:
            pnb, d = _node(s[PNB[0]], PNB[1], ('MW',))
            pnb_t = d['dim0']
    ne = np.where(valid == 1, ne * 1e6, np.nan)
    set_ = (rout > 0) & (rin > 0)
    th = Thomson(int(key), path, branch, t.astype(float), Rcm.astype(float) / 100, ne, err * 1e6, ld * 1e4,
                 at[set_].astype(float), rout[set_].astype(float), rin[set_].astype(float), pnb_t, pnb)
    if pnb is not None:
        th.flat = flattop(pnb_t, pnb)
        if th.flat is not None:
            inside = np.flatnonzero((at >= th.flat[0]) & (at <= th.flat[1]))
            if len(inside):
                th.t_eq = float(at[inside[np.argmin(np.abs(drsep[inside]))]])
    return th


def window_profile(th, side, t0, t1, t_ref, rsep_ref):
    """Per channel on one side, over the frames with t0 <= t <= t1 (s): distance (m, positive into the SOL) from the
    separatrix rsep_ref (m, the run's at its time t_ref) moved by EFIT01's shift from t_ref to each frame, and n_e
    (m^-3) where valid. Mean and spread (sample standard deviation over the frames, NaN from one frame) per channel;
    channels with no valid frame are left out. A channel's side at a frame: R beyond the midpoint of that frame's two
    EFIT01 separatrix radii."""
    k = np.flatnonzero((th.t >= t0) & (th.t <= t1))
    rin, rout = th.separatrix('inboard', th.t[k]), th.separatrix('outboard', th.t[k])
    sign = 1.0 if side == 'outboard' else -1.0
    rsep = rsep_ref + th.separatrix(side, th.t[k]) - th.separatrix(side, t_ref)
    on = (th.R[:, None] > 0.5 * (rin + rout)[None, :]) == (side == 'outboard')
    N = np.where(on, th.ne[:, k], np.nan)
    ok = np.isfinite(N)
    D = np.where(ok, sign * (th.R[:, None] - rsep[None, :]), np.nan)
    E = np.where(ok, th.err[:, k], np.nan)
    n = ok.sum(1)
    keep = n > 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # one frame: no spread
        sd = lambda X: np.where(n[keep] > 1, np.nanstd(X[keep], axis=1, ddof=1), np.nan)
        return dict(R=th.R[keep], d=np.nanmean(D[keep], axis=1), d_sd=sd(D), ne=np.nanmean(N[keep], axis=1), ne_sd=sd(N),
                    err=np.nanmean(E[keep], axis=1), n=n[keep], frames=len(k), t=th.t[k])


def default_window(th, eq_s):
    """The frames the viewer opens on (s): the later half of the flattop after the run's equilibrium time, i.e. from the
    midpoint of (equilibrium time, flattop end) to the flattop's last frame; without a flattop, the frame nearest the
    equilibrium time."""
    if th.flat is None:
        t = float(th.t[np.argmin(np.abs(th.t - eq_s))])
        return t, t
    fr = th.t[(th.t >= th.flat[0]) & (th.t <= th.flat[1])]
    late = fr[fr >= 0.5 * (eq_s + th.flat[1])]
    return (float(late[0]), float(late[-1])) if len(late) else (float(fr[-1]), float(fr[-1]))
