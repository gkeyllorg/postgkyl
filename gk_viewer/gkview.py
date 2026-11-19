"""Data layer of the Gkeyll run viewer (gk_viewer.py): finds the runs of each simulation, reads their frames, works out
each run's block layout from its own topology file, and computes what the progress posters and movies show.

Layout (derived, never typed in): a block whose psi-upper face is physical is SOL, one whose psi-lower face is physical
is core or private flux (PF: it has a theta face on a target; core: it does not). A physical theta face lying on Z = 0
(span-relative) is the reflecting midplane of a half domain, any other is a target plate. SOL chains run along theta
from one physical face to the next; two chains meeting at the mirror are joined into one unrolled strip. Each chain's
block that reaches Z = 0 gives the midplane radius of every SOL flux surface in that chain, and its psi-lower partner is
the core block below it (the radial stack of the midplane profile). Inner (R below the core's mid-R) or outer by that
block's separatrix radius at Z = 0; upper or lower by the target's Z.

Definitions follow sims_20261022/poster_progress.py and movie_density.py:
  cell averages of the p = 1 modal coefficients (c0 / 2 in 2D); masses and charges from the files' metadata;
  T = m (M2 - M1^2/M0) / (3 M0), T_par = m (M2par - M1^2/M0) / M0, T_perp = m M2perp / (2 M0) (M2perp = 2 p_perp / m);
  q_par = sum_s m_s (M3par + M3perp) / 2, the energy flux along B; q_perp = sum_s m_s M3perp / 2, its part carried by
  perpendicular energy (M3perp = the parallel flux of v_perp^2 = 2 mu B / m, as M2perp); at a target, |q_par| in the plate cell of each SOL flux surface, the surface
  placed at its midplane radius (node rows interpolated to Z = 0); lambda_q = the distance from the separatrix holding
  63 % of the integral of |q_par| over that radius (piecewise constant per cell, linear inside the crossing cell);
  midplane profiles: per cell row, linear in theta between the two cells whose centres bracket Z = 0 (at a reflecting
  midplane: the cell next to it), at the row's Z = 0 radius.
"""
import os
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gkyl_read import read_gkyl  # noqa: E402

# The data directory: viewer.json (what to offer, below) and runs/ (everything the viewer writes: cache, saved views,
# exports, Flux copies, logs). $GK_VIEWER_DATA (view.sh exports it), else this directory.
DATA = os.path.abspath(
    os.environ.get('GK_VIEWER_DATA')
    or os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(DATA, 'runs')
CONFIG_FILE = os.path.join(DATA, 'viewer.json')
CACHE_DIR = os.path.join(RUNS, 'cache')
CACHE_VERSION = 3  # bump when a summary definition changes: old cache files are then ignored (2: + fields, 3: + q_perp)


@dataclass(frozen=True)
class SimSettings:
  """A simulation's own settings: a "sims" entry's optional fifth element, {"device": ..., "geqdsk": ...}."""
  device: str  # the device its runs simulate (else viewer.json's "device"; '': none)
  geqdsk: object = None  # the equilibrium file its runs read (absolute), for runs whose own files name none; or None


SIM_KEYS = ('device', 'geqdsk')


def load_config(path):
  """viewer.json at path, read: (its settings, base (absolute), the simulations offered [(label, directory under
  base, default run, one-line description)], their SimSettings by label, problems). Every path in it is relative to
  base, itself relative to the file's directory (absolute paths too). A sims entry is [label, directory, default run,
  description] or those and {"device": <name>, "geqdsk": <file>} (both optional; the device defaults to the
  top-level "device"). No file offers nothing; a file that cannot be read, or an entry that does not parse, offers
  nothing from it and is named in problems, so the page opens anyway and says why."""
  if not os.path.exists(path):
    return {}, os.path.dirname(path), [], {}, []
  try:
    with open(path) as f:
      cfg = json.load(f)
    if not isinstance(cfg, dict):
      raise ValueError('not a JSON object')
  except (OSError, ValueError) as e:
    return {}, os.path.dirname(path), [], {}, [f'{path}: {e}']
  base = os.path.normpath(
      os.path.join(os.path.dirname(path),
                   cfg.get('base') or '.'))
  device = cfg.get('device') or ''
  sims, settings, problems = [], {}, []
  for e in cfg.get('sims') or []:
    own = e[4] if isinstance(e, list) and len(e) == 5 else {}
    ok = (isinstance(e, list) and len(e) in (4, 5)
          and all(isinstance(v, str) for v in e[:4]) and isinstance(own, dict)
          and set(own) <= set(SIM_KEYS)
          and all(isinstance(v, str) for v in own.values()))
    if not ok:
      problems.append(
          f'"sims": {json.dumps(e)} left out: an entry is [label, directory, default run, description], '
          'optionally followed by {"device": <name>, "geqdsk": <file>}')
      continue
    sims.append(tuple(e[:4]))
    settings[e[0]] = SimSettings(
        own.get('device') or device,
        os.path.normpath(os.path.join(base, own['geqdsk']))
        if own.get('geqdsk') else None)
  return cfg, base, sims, settings, problems


def reload():
  """Read viewer.json (CONFIG_FILE) into this module's settings, the one place it is read: CONFIG (the file's
  settings), BASE, SIMS, SIM_SETTINGS, DEVICE (the default device) and CONFIG_PROBLEMS. A form that writes
  viewer.json calls this after (thomson reads its directories at each call; pgkyl_cmd's pgkyl is read at its
  import)."""
  global CONFIG, BASE, SIMS, SIM_SETTINGS, DEVICE, CONFIG_PROBLEMS
  CONFIG, BASE, SIMS, SIM_SETTINGS, CONFIG_PROBLEMS = load_config(CONFIG_FILE)
  DEVICE = CONFIG.get('device') or ''


# CONFIG: viewer.json's settings. BASE: what simulation directories and cache names are relative to. SIMS: the
# simulations the viewer offers, (label, directory under base, default run, one-line description); the shot and
# equilibrium time come from each run's own files. SIM_SETTINGS: label -> SimSettings. DEVICE: the device runs
# simulate unless their simulation names its own, named in a run's title before its shot ('': none).
reload()


def data_path(key):
  """A file or directory viewer.json names under `key` (relative to base), or None when it does not name one."""
  return os.path.normpath(os.path.join(
      BASE, CONFIG[key])) if CONFIG.get(key) else None


def device_settings(device):
  """viewer.json's "devices" entry of a device: {"device_image": ..., "wall": ...} ({} when it has none)."""
  d = (CONFIG.get('devices') or {}).get(device) or {}
  if not isinstance(d, dict):
    raise ValueError(f'viewer.json "devices": "{device}" must be an object')
  return d


def device_image(device):
  """The picture of a device's poloidal cross-section to draw behind the poloidal plane: its own ("devices":
  {device: {"device_image": ...}}), else the top-level "device_image" when the device is viewer.json's default
  ("device", or none: the picture of the device every simulation without one of its own simulates; another device's
  runs do not get it). {"file": <path, relative to base>, "extent": [R_left, R_right, Z_bottom, Z_top] (m: where the
  picture's edges lie, so that it sits on the R-Z axes), "opacity": <0 to 1, default 1>}. Returns (path, extent,
  opacity), or None when there is none; raises ValueError when the one named cannot be used."""
  own = device_settings(device).get('device_image')
  d = own if own else CONFIG.get('device_image') if device == DEVICE else None
  where = f'viewer.json "devices": "{device}": "device_image"' if own else 'viewer.json "device_image"'
  if not d:
    return None
  if not isinstance(d, dict) or not d.get('file'):
    raise ValueError(f'{where} needs a "file"')
  path = os.path.normpath(os.path.join(BASE, d['file']))
  if not os.path.isfile(path):
    raise ValueError(f'{where}: no file {path}')
  e = d.get('extent')
  if not (isinstance(e, list) and len(e) == 4
          and all(isinstance(v, (int, float))
                  for v in e) and e[1] > e[0] and e[3] > e[2]):
    raise ValueError(
        f'{where}: "extent" must be [R_left, R_right, Z_bottom, Z_top] (m), '
        f'R_right > R_left and Z_top > Z_bottom; it is {e!r}')
  opacity = float(d.get('opacity', 1.0))
  if not 0 < opacity <= 1:
    raise ValueError(f'{where}: "opacity" must be in (0, 1]; it is {opacity}')
  return path, tuple(float(v) for v in e), opacity


PHYSICAL = 5  # gkyl_oriented_edge (core/zero/gkyl_block_topo.h)
LOWER_POS, LOWER_NEG, UPPER_POS, UPPER_NEG = 1, 2, 3, 4
GEOMETRY_MAPC2P = 3  # gkyl_geometry_id (core/zero/gkyl_eqn_type.h): a user's mapc2p, positions Cartesian X, Y, Z
MOMS = ('M0', 'M1', 'M2', 'M2par', 'M2perp', 'M3par', 'M3perp')
SPECIES_TAG = {'elc': ('electron', 'e'), 'ion': ('ion', 'i')}
TARGET_ORDER = [
    'lower outer target', 'upper outer target', 'lower inner target',
    'upper inner target'
]
LAMBDA_FRACTION = 0.63


# ----------------------------------------------------------------------------------------------------------- reading
def cav(path):
  """Cell averages of a 2D p = 1 modal field (orthonormal basis: c0 / 2)."""
  return np.asarray(read_gkyl(path)['data'])[..., 0] / 2.0


def meta(path):
  return read_gkyl(path)['meta']


_DYNVEC = {}  # path -> (size, mtime_ns, t, d): the last read of each file


def read_dynvec(path):
  """(time, data[n, ncomp]) of a gkyl dynvector file (file_type 2): every appended chunk repeats the full header.
    Copies of the last read while the file's size and mtime are unchanged (a run appends to its files)."""
  st = os.stat(path)
  hit = _DYNVEC.get(path)
  if hit is None or hit[:2] != (st.st_size, st.st_mtime_ns):
    hit = (st.st_size, st.st_mtime_ns) + _read_dynvec(path)
    _DYNVEC[path] = hit
  return hit[2].copy(), hit[3].copy()


def _read_dynvec(path):
  buf = open(path, 'rb').read()
  off, ts, ds = 0, [], []
  while off < len(buf):
    if buf[off:off + 5] != b'gkyl0':
      raise ValueError(f'{path}: bad chunk header at byte {off}')
    version, ftype, msize = struct.unpack_from('<QQQ', buf, off + 5)
    if version != 1 or ftype != 2:
      raise ValueError(
          f'{path}: not a v1 dynvector (version {version}, type {ftype})')
    off += 29 + msize
    real_type, esznc, size = struct.unpack_from('<QQQ', buf, off)
    if real_type != 2:
      raise ValueError(f'{path}: only float64 dynvectors are supported')
    off += 24
    ts.append(np.frombuffer(buf, '<f8', size, off))
    off += 8 * size
    ds.append(
        np.frombuffer(buf, '<f8', size * esznc // 8,
                      off).reshape(size, esznc // 8))
    off += size * esznc
  t, d = np.concatenate(ts), np.concatenate(ds)
  # a restart appends from its restart time: keep the latest sample written for each time
  _, keep = np.unique(t[::-1], return_index=True)
  keep = len(t) - 1 - keep
  return t[keep], d[keep]


def read_options(run):
  """key -> value of options_used.txt (the driver reads the first occurrence of a key)."""
  out = {}
  path = os.path.join(run, 'options_used.txt')
  if os.path.exists(path):
    for line in open(path):
      if '=' in line and not line.lstrip().startswith('#'):
        k, v = line.strip().split('=', 1)
        out.setdefault(k, v)
  return out


def _pick(k, j_near, offset):
  """The sign change of a row to use: the first, or (a row crossing Z = 0 more than once, as the main SOL block of a
    single null does at both midplanes) the one nearest the theta node position j_near."""
  return int(k[0]) if j_near is None else int(k[np.argmin(
      np.abs(k + offset - j_near))])


def closes(nd):
  """Whether a block of nodes closes on itself in theta: its first and last node columns are one."""
  return np.allclose(nd[:, 0], nd[:, -1], rtol=0, atol=1e-9 * np.ptp(nd))


def at_seam(nd, j_near):
  """Whether j_near (a theta node position) asks for the crossing at the seam of a block closing on itself: within
    half a node of either end."""
  nt = nd.shape[1] - 1
  return j_near is not None and min(j_near, nt - j_near) < 0.5 and closes(nd)


def mid_edges(nd, j_near=None):
  """Radial node positions where each psi row of nodes crosses Z = 0 (linear in theta between the bracketing nodes);
    j_near: which crossing, for a block crossing Z = 0 more than once (default the first); at_seam: the crossing
    across its seam (the last node, the first, the second)."""
  out = []
  seam = at_seam(nd, j_near)
  for R, Z in zip(nd[..., 0], nd[..., 1]):
    if seam:
      R, Z = R[[-2, 0, 1]], Z[[-2, 0, 1]]
      j_near = 1
    k = np.where(np.diff(np.sign(Z)) != 0)[0]
    if len(k) == 0:
      out.append(R[np.argmin(np.abs(Z))])
      continue
    k = _pick(k, j_near, 0.5)
    w = Z[k] / (Z[k] - Z[k + 1])
    out.append(R[k] + w * (R[k + 1] - R[k]))
  return np.array(out)


def cell_centres(nd):
  return 0.25 * (nd[:-1, :-1] + nd[1:, :-1] + nd[:-1, 1:] + nd[1:, 1:])


def midplane_weights(nd, j_near=None):
  """Per cell row (j0, j1, w): the value at Z = 0 is (1 - w) v[i, j0] + w v[i, j1] (j_near: as mid_edges). A block
    closing on itself in theta (its first and last node columns one) crosses Z = 0 between its last and first cells
    too: j_near at either end picks that crossing."""
  Zc = cell_centres(nd)[..., 1]
  nt = Zc.shape[1]
  seam = at_seam(nd, j_near)
  out = []
  for z in Zc:
    if seam and z[-1] * z[0] < 0:
      out.append((nt - 1, 0, float(z[-1] / (z[-1] - z[0]))))
      continue
    k = np.where(np.diff(np.sign(z)) != 0)[0]
    if len(
        k
    ) == 0:  # Z = 0 is the block's edge (reflecting midplane): the cell next to it
      j = int(np.argmin(np.abs(z)))
      out.append((j, j, 0.0))
      continue
    k = _pick(k, j_near, 1.0)
    out.append((k, k + 1, float(z[k] / (z[k] - z[k + 1]))))
  return out


def row_crossings(Z, ztol):
  """Theta node positions (fractional) where one row of nodes meets Z = 0, in order: each sign change once (a node
    on Z = 0 makes two), else the nearest node when it lies within ztol of Z = 0 (a reflecting midplane's edge)."""
  k = np.where(np.diff(np.sign(Z)) != 0)[0]
  out = []
  for kk in k:
    p = kk + Z[kk] / (Z[kk] - Z[kk + 1])
    if not out or p - out[-1] > 1e-9:
      out.append(float(p))
  if not out:
    j = int(np.argmin(np.abs(Z)))
    if abs(Z[j]) <= ztol:
      out.append(float(j))
  return out


# ------------------------------------------------------------------------------------------------------------ layout
@dataclass
class Target:
  name: str  # 'lower outer target', ...
  leg: int  # SOL block holding the plate
  pf: object  # the block under it (private flux), or None
  j: int  # theta index of the plate cell (0 or -1)
  mid: object  # SOL block of this chain at the midplane (the surfaces' midplane radii), or None: no block of
  # the chain reaches Z = 0, the surfaces are placed along the plate instead
  updown: str = ''  # 'lower' or 'upper'
  side: str = ''  # 'outer' or 'inner'
  mid_j: object = None  # theta node position of that block's midplane crossing, where it crosses Z = 0 twice
  open_row: int = 0  # the leg's first row on the plate: its LCFS row in a block holding closed rows too (Layout.lcfs)


@dataclass
class Stack:
  side: str  # 'outboard' or 'inboard'
  core: object  # the core block under the SOL block at the midplane, or None
  sol: object  # the SOL block reaching the midplane; None only in empty_run (a midplane holding no block)
  mirror: bool  # the midplane is the reflecting boundary of a half domain
  j: object = None  # theta node position of the crossing, where the block crosses Z = 0 more than once
  sep_row: int = 0  # the SOL block's node row on the separatrix (its LCFS in a block holding closed rows too)


# strip bands, y in 0..8: (y0, height). A chain's SOL blocks over the blocks under them, or one band for a chain
# with nothing under it (a single-block run, a core ring)
BANDS = {'sol': (4.0, 4.0), 'in': (0.0, 4.0), 'all': (0.0, 8.0)}
TWO_BAND_LABELS = (('core edge', 0.4), ('core / PF', 2.2), ('separatrix', 4.0),
                   ('SOL', 6.0))


@dataclass
class Strip:
  title: str
  items: list  # (block, x0, band: 'sol' | 'in' | 'all')
  segments: list  # (x0, x1, label)
  mirror_x: object  # x of the reflecting midplane, or None
  width: int
  ylabels: tuple = TWO_BAND_LABELS  # (label, y) down the strip's left side
  sep: object = 4.0  # y of the separatrix line, or None (one band)


@dataclass
class Layout:
  region: dict  # block -> 'sol' | 'core' | 'pf'
  strips: list
  targets: list
  stacks: dict  # side -> Stack
  half: bool
  flip: bool = False  # the files' psi index grows inward: rows were reversed on reading (Run.rows)
  under: dict = field(
      default_factory=dict
  )  # block -> the aligned block under it (psi-lower partner), or None
  lcfs: dict = field(
      default_factory=dict
  )  # block -> its LCFS node row, in a block holding closed rows (below it) and open ones (a limited run)


def theta_chains(conn):
  """Blocks joined along theta, each in theta order: dict(blocks, ring) (ring: it closes on itself, a core)."""
  nb = len(conn)

  def link(
      b, e
  ):  # the block across theta face e (0 lower, 1 upper) continuing the chain, or None
    if conn[b, 1, e, 2] == PHYSICAL:
      return None
    p, d, edge = (int(v) for v in conn[b, 1, e])
    return p if d == 1 and edge == (
        LOWER_POS if e == 1 else UPPER_POS) else None

  seen, out = set(), []
  for b0 in range(nb):
    if b0 in seen:
      continue
    b, path = b0, [b0]
    while (p := link(b, 0)) is not None and p not in path:
      b = p
      path.append(b)
    ring = link(b, 0) is not None and link(b, 0) in path
    start = b0 if ring else b
    chain, b = [start], start
    while (p := link(b, 1)) is not None and p != start and p not in chain:
      b = p
      chain.append(b)
    seen.update(chain)
    out.append(dict(blocks=chain, ring=ring))
  return out


def flip_conn(conn):
  """The topology with the psi direction reversed: each block's psi faces swapped, and every connection into a psi
    face pointed at the other one (lower <-> upper, orientation kept: the theta direction does not change)."""
  c = conn.copy()
  c[:, 0, 0], c[:, 0, 1] = conn[:, 0, 1], conn[:, 0, 0]
  swap = {
      LOWER_POS: UPPER_POS,
      UPPER_POS: LOWER_POS,
      LOWER_NEG: UPPER_NEG,
      UPPER_NEG: LOWER_NEG
  }
  for b, d, e in np.ndindex(c.shape[:3]):
    if c[b, d, e, 2] != PHYSICAL and c[b, d, e, 1] == 0:
      c[b, d, e, 2] = swap[int(c[b, d, e, 2])]
  return c


def psi_grows_outward(conn, nodes, r_axis=None):
  """True when the psi index grows outward (row 0 = a block's inner edge, as NSTX-U's files), False when it grows
    inward (STEP's). A closed chain (the core) with one physical psi face says it: that face is the core's inner edge.
    Without one, the geometry: where a block's rows cross Z = 0, whether R moves away from the magnetic axis (r_axis,
    else the nodes' mean R) as the psi index grows."""
  phys = lambda b, e: conn[b, 0, e, 2] == PHYSICAL  # noqa: E731
  for c in theta_chains(conn):
    faces = {(bool(phys(b, 0)), bool(phys(b, 1))) for b in c['blocks']}
    closed = c['ring'] or all(
        conn[b, 1, e, 2] == PHYSICAL and np.abs(nodes[b][:, 0 if e == 0 else -1,
                                                         1]).max() <= 1e-9 *
        np.ptp(nodes[b][..., 1])
        for b, e in ((c['blocks'][0], 0), (c['blocks'][-1], 1)))
    if closed and faces == {(True, False)}:
      return True
    if closed and faces == {(False, True)}:
      return False
  allR = np.concatenate([nodes[b][..., 0].ravel() for b in nodes])
  r0 = float(r_axis) if r_axis is not None else float(allR.mean())
  for b in sorted(nodes):
    nd = nodes[b]
    if nd[..., 1].min() < 0 < nd[..., 1].max():
      a, z = mid_edges(nd[[0, -1]])
      return bool((z - a) * (1 if a > r0 else -1) > 0)
  return True


def derive_layout(conn, nodes, r_axis=None, lcfs=None):
  """The block layout from the topology (psi growing outward: flip_conn first where it does not) and the nodes.
    r_axis: R of the magnetic axis, for the inner/outer side where the domain has no core block. lcfs: block -> its
    LCFS node row, for a block closing on itself in theta whose rows from there on end on a sheath at its theta
    ends instead (a limited run, sheath_open_rows): those rows are its SOL, their ends its targets."""
  nb = len(conn)
  lcfs = lcfs or {}
  allZ = np.concatenate([nodes[b][..., 1].ravel() for b in range(nb)])
  ztol = 1e-9 * np.ptp(allZ)

  def phys(b, d, e):
    return conn[b, d, e, 2] == PHYSICAL

  def face_kind(b,
                e):  # a physical theta face: 'mirror' on Z = 0, else 'target'
    col = nodes[b][:, 0 if e == 0 else -1, 1]
    return 'mirror' if np.abs(col).max() <= ztol else 'target'

  chains = theta_chains(conn)
  for c in chains:
    for which, b, e in (('start', c['blocks'][0], 0), ('end', c['blocks'][-1],
                                                       1)):
      c[which] = 'ring' if c['ring'] else face_kind(b, e) if phys(b, 1,
                                                                  e) else 'open'
    c['closed'] = c['ring'] or (c['start'] == c['end'] == 'mirror')
  chain_of = {b: c for c in chains for b in c['blocks']}

  region = {}
  for b in range(nb):
    lo, hi = phys(b, 0, 0), phys(b, 0, 1)
    if hi and not lo:
      region[b] = 'sol'
    elif lo and not hi:
      tgt = any(phys(b, 1, e) and face_kind(b, e) == 'target' for e in (0, 1))
      region[b] = 'pf' if tgt else 'core'
    else:  # both psi faces physical (a single-block run) or neither: closed field lines are core
      region[b] = 'core' if chain_of[b]['closed'] else 'sol'

  def inner_of(b):  # the aligned block under b (its psi-lower partner), or None
    if phys(b, 0, 0):
      return None
    p, d, e = (int(v) for v in conn[b, 0, 0])
    return p if d == 0 and e == UPPER_POS and nodes[p].shape[1] == nodes[
        b].shape[1] else None

  cores = [b for b in range(nb) if region[b] == 'core']
  if cores:
    core_nodes = np.concatenate([nodes[b][..., 0].ravel() for b in cores])
    r_ref = 0.5 * (core_nodes.min() + core_nodes.max())
  else:
    r_ref = float(r_axis) if r_axis is not None else float(
        np.mean(np.concatenate([nodes[b][..., 0].ravel() for b in range(nb)])))
  side_of = lambda R: 'outboard' if R > r_ref else 'inboard'  # noqa: E731

  def crossings(b, row):
    """(theta node position, R) where one node row of block b crosses Z = 0."""
    R0, Z0 = nodes[b][row, :, 0], nodes[b][row, :, 1]
    out = []
    for p in row_crossings(Z0, ztol):
      k = min(int(p), len(R0) - 2)
      out.append((p, R0[k] + (p - k) * (R0[k + 1] - R0[k])))
    return out

  # SOL chains (open, every block SOL), in the order of their first block; their midplane crossings (row 0)
  sol_chains = sorted([
      c for c in chains if not c['closed'] and all(region[b] == 'sol'
                                                   for b in c['blocks'])
  ],
                      key=lambda c: c['blocks'][0])
  span = np.ptp(np.concatenate([nodes[b][..., 0].ravel() for b in range(nb)]))
  stacks, targets = {}, []
  for c in sol_chains:
    cross = [
    ]  # (block, theta node position, R), in chain order; a crossing on a shared node once
    for b in c['blocks']:
      for p, R in crossings(b, 0):
        if not cross or abs(R - cross[-1][2]) > 1e-9 * span:
          cross.append((b, p, R))
    nby = {b: sum(1 for x in cross if x[0] == b) for b in c['blocks']}
    c['cross'] = [(b, p if nby[b] > 1 else None, side_of(R))
                  for b, p, R in cross]
    sides = list(dict.fromkeys(s for _, _, s in c['cross']))
    c['side'] = sides[0] if len(sides) == 1 else 'SOL'
    for b, j, side in c['cross']:
      if side in stacks:
        continue
      core = inner_of(b)
      core = core if core is not None and region[core] == 'core' else None
      at_mirror = 'mirror' in (c['start'], c['end'])
      if len(
          c['cross']
      ) > 1:  # at the mirror only if this crossing is the chain's end on it
        at_mirror = ((c['start'] == 'mirror' and b == c['blocks'][0])
                     or (c['end'] == 'mirror' and b == c['blocks'][-1]))
      stacks[side] = Stack(side, core, b, at_mirror, j)
    lab, last = [], len(c['blocks']) - 1
    for k, b in enumerate(c['blocks']):
      ends = [
          e for e, first in ((0, k == 0), (1, k == last))
          if first and c['start' if e == 0 else 'end'] == 'target'
      ]
      for e in ends:
        zt = nodes[b][:, 0 if e == 0 else -1, 1].mean()
        ud = 'upper' if zt > 0 else 'lower'
        near = (
            c['cross'][0] if e == 0 else c['cross'][-1]) if c['cross'] else None
        if near is not None:
          io = 'outer' if near[2] == 'outboard' else 'inner'
        else:
          io = 'outer' if nodes[b][:, 0 if e == 0 else -1,
                                   0].mean() > r_ref else 'inner'
        name = f'{ud} {io} target'
        if any(t.name == name for t in targets):
          name += f' (block {b})'
        targets.append(
            Target(name, b, inner_of(b), 0 if e == 0 else -1,
                   near[0] if near else None, ud, io,
                   near[1] if near else None))
      own = list(dict.fromkeys(s for bb, _, s in c['cross'] if bb == b))
      if len(ends) == 1:
        lab.append(f"{targets[-1].updown} {targets[-1].side} leg")
      else:
        lab.append(own[0] if len(own) == 1 else 'main SOL' if len(own) >
                   1 else c['side'] if c['side'] != 'SOL' else 'SOL')
    c['labels'] = lab
  # a block closing on itself whose rows from its LCFS on end on a sheath at its theta ends (a limited run): its
  # LCFS row's midplane crossings make the stacks (core rows then SOL rows), its open rows at each theta end a target
  # (named by the side its cells reach it from: the theta ends meet where the field lines end)
  for b, m in lcfs.items():
    if not chain_of[b]['ring'] or len(chain_of[b]['blocks']) != 1:
      continue
    cross = crossings(b, m)
    nt = nodes[b].shape[1] - 1
    if abs(nodes[b][m, 0, 1]) <= ztol and not any(
        min(p, nt - p) < 0.5 for p, _ in cross):  # its seam on Z = 0
      cross.insert(0, (0.0, float(nodes[b][m, 0, 0])))
    for p, R in cross:
      stacks.setdefault(
          side_of(R),
          Stack(side_of(R), None, b, False, p if len(cross) > 1 else None, m))
    mid = max(cross, key=lambda x: x[1])[0] if cross else None
    for e in (0, -1):
      ud = 'upper' if cell_centres(nodes[b][m:])[:, e,
                                                 1].mean() > 0 else 'lower'
      io = 'outer' if nodes[b][m:, e, 0].mean() > r_ref else 'inner'
      targets.append(
          Target(f'{ud} {io} target', b, None, e, b if cross else None, ud, io,
                 mid, m))

  # strips: a target-to-target chain alone; chains meeting at the mirror joined there (target -> mirror first)
  strips = []
  into = [c for c in sol_chains if c['end'] == 'mirror']
  outof = [c for c in sol_chains if c['start'] == 'mirror']
  pairs = list(zip(into, outof))
  paired = {id(c) for p in pairs for c in p}
  groups = ([[c]
             for c in sol_chains if 'mirror' not in (c['start'], c['end'])] +
            [list(p) for p in pairs] +
            [[c] for c in sol_chains
             if 'mirror' in (c['start'], c['end']) and id(c) not in paired])
  groups.sort(key=lambda g: 0 if g[0]['side'] == 'outboard' else 1)
  for g in groups:
    items, segs, x, mx = [], [], 0, None
    for c in g:
      if c['start'] == 'mirror':
        mx = x
      for b, lab in zip(c['blocks'], c['labels']):
        nt = nodes[b].shape[1] - 1
        items.append((b, x, 'sol'))
        if inner_of(b) is not None:
          items.append((inner_of(b), x, 'in'))
        segs.append((x, x + nt, lab))
        x += nt
      if c['end'] == 'mirror' and len(g) == 1:
        mx = x
    side = g[0]['side'] if all(c['side'] == g[0]['side'] for c in g) else 'SOL'
    title = ('every cell, unrolled (midplane: reflecting boundary)'
             if mx is not None and len(g) == 2 else
             f'every cell, unrolled: {side} strip')
    if any(band == 'in' for _, _, band in items):
      strips.append(Strip(title, items, segs, mx, x))
    else:
      strips.append(
          Strip(title, [(b, x0, 'all') for b, x0, _ in items],
                segs,
                mx,
                x,
                ylabels=(('inner edge', 0.4), ('SOL', 4.0), ('outer edge',
                                                             7.6)),
                sep=None))
  # every other block (a core ring, a PF chain under nothing, a single-block run): its theta chain, one band
  covered = {b for st in strips for b, _, _ in st.items}
  for c in chains:
    rest = [b for b in c['blocks'] if b not in covered]
    if not rest:
      continue
    items, segs, x = [], [], 0
    for b in rest:
      nt = nodes[b].shape[1] - 1
      items.append((b, x, 'all'))
      segs.append((x, x + nt, f'block {b}'))
      x += nt
    regs = ', '.join(dict.fromkeys(REGION_LABEL[region[b]] for b in rest))
    mx = 0 if c['start'] == 'mirror' else (x if c['end'] == 'mirror' else None)
    m = lcfs.get(rest[0]) if len(rest) == 1 else None
    if m is not None:  # a limited block: its core rows under its LCFS, its SOL rows over it, in one band
      ys = 8.0 * m / (nodes[rest[0]].shape[0] - 1)
      strips.append(
          Strip('every cell, unrolled: core (closes on itself) and SOL',
                items,
                segs,
                mx,
                x,
                ylabels=(('core edge', 0.4), ('core', ys / 2), ('LCFS', ys),
                         ('SOL', (ys + 8) / 2)),
                sep=ys))
      continue
    strips.append(
        Strip(f'every cell, unrolled: {regs}' +
              (' (closes on itself)' if c['ring'] else ''),
              items,
              segs,
              mx,
              x,
              ylabels=(('inner edge', 0.4), (regs, 4.0), ('outer edge', 7.6)),
              sep=None))
  targets.sort(key=lambda t: (TARGET_ORDER.index(t.name) if t.name in
                              TARGET_ORDER else len(TARGET_ORDER), t.name))
  half = any(c['start'] == 'mirror' or c['end'] == 'mirror' for c in chains)
  return Layout(region,
                strips,
                targets,
                stacks,
                half=half,
                under={b: inner_of(b)
                       for b in range(nb)},
                lcfs=dict(lcfs))


def select_strips(layout, blocks=None):
  """The layout's strips holding only the chosen blocks (None: every strip as it is): the segments whose SOL block
    or the block under it is chosen, packed together in their order; a strip left empty is dropped."""
  if blocks is None:
    return layout.strips
  out = []
  for st in layout.strips:
    by_x = {}
    for b, x0, band in st.items:
      by_x.setdefault(x0, []).append((b, band))
    items, segs, x, mx = [], [], 0, None
    for xa, xb, lab in st.segments:
      members = [(b, band) for b, band in by_x.get(xa, []) if b in blocks]
      if not members:
        continue
      if st.mirror_x is not None and st.mirror_x == xa:
        mx = x
      items += [(b, x, band) for b, band in members]
      segs.append((x, x + xb - xa, lab))
      x += xb - xa
      if st.mirror_x is not None and st.mirror_x == xb:
        mx = x
    if items:
      out.append(Strip(st.title, items, segs, mx, x, st.ylabels, st.sep))
  return out


REGION_LABEL = {'core': 'core', 'sol': 'SOL', 'pf': 'private flux'}


# ---------------------------------------------------------------------------------------------------------------- run
@dataclass
class Run:
  path: str
  sim: str = ''
  stem: str = ''
  nb: int = 0
  species: list = field(default_factory=list)
  mass: dict = field(default_factory=dict)
  charge: dict = field(default_factory=dict)
  e: float = 0.0
  nodes: dict = field(default_factory=dict)
  # blocks whose psi index runs against their own topology entry (contrary_blocks): read reversed
  contrary: tuple = ()
  layout: Layout = None
  options: dict = field(default_factory=dict)
  shot: int = 0
  eq_ms: int = 0
  half_meta: int = -1
  single: bool = False  # a single-block run (the gyrokinetic app's own files: no _b<k> in the names, no topology)
  flip: bool = False  # its psi index grows inward: every block's rows are reversed on reading (rows())
  device: str = ''  # the device it simulates (its simulation's SimSettings; '': none)
  geqdsk: object = None  # its simulation's equilibrium file (SimSettings), for when its own files name none
  conn: object = None  # its block topology [nb, 2, 2, 3] as the layout reads it (rows flipped with flip; None: no run)

  @property
  def name(self):
    return os.path.basename(self.path.rstrip('/'))

  def f(self, b, what, k=None):
    tail = f'_{k}' if k is not None else ''
    return f'{self.path}/{self.stem}-{what}{tail}.gkyl' if self.single else \
        f'{self.path}/{self.stem}_b{b}-{what}{tail}.gkyl'

  def frames(self):
    """Frame numbers written so far (from block 0's potential files; dir_index)."""
    return list(
        dir_index(self.path)['frames'].get(
            self.stem if self.single else f'{self.stem}_b0', ()))

  def time(self, k):
    return float(
        meta(self.f(0, f'{self.species[0]}_M0', k)).get('time', np.nan))

  def stamp(self, k):
    """Changes when frame k is rewritten (a restart from an earlier frame rewrites the later ones)."""
    st = os.stat(self.f(0, f'{self.species[0]}_M0', k))
    return st.st_mtime_ns ^ st.st_size

  def reversed(self, b):
    """Whether block b's rows are read reversed: the run's psi index grows inward (flip), or the block's own runs
        against its topology entry (contrary), not both."""
    return self.flip != (b in self.contrary)

  def rows(self, b, a):
    """Block b's array (psi rows first) in the viewer's order, row 0 innermost: the file's, or reversed (reversed()).
        Only for cell averages and node values (a modal coefficient odd in psi changes sign)."""
    return a[::-1] if self.reversed(b) else a

  def file_row(self, b, i):
    """The file's psi index of the viewer's row i of block b, and back (the same map both ways)."""
    return self.nodes[b].shape[0] - 2 - i if self.reversed(b) else i

  @property
  def title(self):
    """'NSTX-U #203899 @ 325 ms' (its device, then the shot) when the run's geqdsk names a shot, else its
    directory."""
    if not self.shot:
      return self.name
    return (f'{self.device} '
            if self.device else '') + f'#{self.shot} @ {self.eq_ms} ms'


_DIRS = {}  # path -> (mtime_ns, dir_index)
_FIELD_FILE = re.compile(r'(.+)-field_(\d+)\.gkyl$')
_M0_FILE = re.compile(r'(\w+?)_M0_(\d+)\.gkyl$')


def dir_index(path):
  """What run discovery reads of a directory, from one listing, kept while its mtime is unchanged (a file added or
    removed changes it): 'topo' the stems of its block-topology files, 'geo' those of its single-block geometry
    (<stem>-geo_corn_nodes.gkyl), 'frames' {prefix: sorted frame numbers} of its <prefix>-field_<k>.gkyl files
    (prefix <stem>_b0 for a multiblock run's block 0, <stem> for a single-block run), 'm0' {'<file prefix>-': {frame:
    the names <name> of its <file prefix>-<name>_M0_<frame>.gkyl files}} (the species, and their sources), 'logs' its
    job logs' names (slurm-*.out, *.log, stdout*). Names starting with '.' are left out, as glob leaves them."""
  if not os.path.isdir(
      path
  ):  # no such directory (the empty run's ''): nothing in it, as glob finds nothing
    return dict(topo=[], geo=[], frames={}, m0={}, logs=[])
  mt = os.stat(path).st_mtime_ns
  hit = _DIRS.get(path)
  if hit is not None and hit[0] == mt:
    return hit[1]
  topo, geo, frames, m0, logs = [], [], {}, {}, []
  with os.scandir(path) as it:
    for e in it:
      n = e.name
      if n.startswith('.'):
        continue
      if (n.startswith('slurm-') and
          n.endswith('.out')) or n.endswith('.log') or n.startswith('stdout'):
        logs.append(n)
      elif n.endswith('-block_topo.gkyl'):
        topo.append(n[:-len('-block_topo.gkyl')])
      elif n.endswith('-geo_corn_nodes.gkyl'):
        geo.append(n[:-len('-geo_corn_nodes.gkyl')])
      elif '-field_' in n:
        m = _FIELD_FILE.match(n)
        if m:
          frames.setdefault(m.group(1), []).append(int(m.group(2)))
      elif '_M0_' in n:
        head, _, tail = n.rpartition('-')
        m = _M0_FILE.fullmatch(tail)
        if head and m:
          m0.setdefault(head + '-', {}).setdefault(int(m.group(2)),
                                                   []).append(m.group(1))
  out = dict(topo=sorted(topo),
             geo=sorted(geo),
             frames={
                 k: sorted(v)
                 for k, v in frames.items()
             },
             m0=m0,
             logs=sorted(logs))
  _DIRS[path] = (mt, out)
  return out


def find_stem(path):
  """(stem, single) of the run in a directory: the multiblock app's topology file, else the single-block app's
    geometry nodes (<stem>-geo_corn_nodes.gkyl, the stem not ending in _b<k>)."""
  idx = dir_index(path)
  topo = idx['topo']
  if len(topo) == 1:
    return topo[0], False
  if len(topo) > 1:
    raise FileNotFoundError(f'{path}: {len(topo)} block-topology files')
  stems = idx['geo']
  single = [s for s in stems if not re.search(r'_b\d+$', s)]
  if len(single) == 1:
    return single[0], True
  raise FileNotFoundError(f'{path}: no block-topology file' +
                          (' (block files without one: copy it in)'
                           if stems else ' and no single-block geometry'))


def single_block_topology(nd):
  """A single-block run's connections (the app writes none): every face physical, except theta when the block
    closes on itself (its first and last theta node columns coincide: a core ring, periodic in theta)."""
  conn = np.zeros((1, 2, 2, 3), dtype=int)
  conn[..., 2] = PHYSICAL
  if closes(nd):
    conn[0, 1, 0], conn[0, 1, 1] = (0, 1, UPPER_POS), (0, 1, LOWER_POS)
  return conn


def physical_rz(xyz, geometry_type):
  """(R, Z) of positions as a geometry file writes them (last axis): the tokamak geometry's (R, Z, phi), or a mapc2p
    geometry's Cartesian (X, Y, Z) (geometry_type GEOMETRY_MAPC2P, the file's metadata): R = hypot(X, Y)."""
  xyz = np.asarray(xyz)
  if geometry_type == GEOMETRY_MAPC2P:
    return np.stack([np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]], axis=-1)
  return xyz[..., :2]


def corner_nodes(path):
  """(R, Z) of a block's corner nodes, from its geo_corn_nodes file (physical_rz by its geometry_type)."""
  g = read_gkyl(path)
  return physical_rz(g['data'], int(g['meta'].get('geometry_type', -1)))


def _gaps(pts, poly):
  """The distance of each point to a polyline [n, (R, Z)]."""
  a, d = poly[:-1], np.diff(poly, axis=0)
  t = np.clip(((pts[:, None] - a) * d).sum(-1) / np.maximum(
      (d * d).sum(-1), 1e-300), 0, 1)
  return np.hypot(*(a + t[..., None] * d -
                    pts[:, None]).transpose(2, 0, 1)).min(1)


def contrary_blocks(conn, nodes):
  """The blocks whose psi index runs against their own topology entry: across each of their connected psi faces
    the topology names a block whose psi faces lie against their other psi face (nearer it than the named one, by
    the median distance of the face's nodes). Such a topology joins faces that do not touch (TCV x21's fixture,
    2026-11-16: blocks 0-4, 0.1 m apart; every other run here: the named face, 1e-10 cells off, the other 1-300
    cells); reading the block reversed makes its rows run as the topology says."""

  def gap(face, p):
    """How far a face of nodes lies from the nearer psi face of block p."""
    return min(np.median(_gaps(face, nodes[p][e])) for e in (0, -1))

  out = []
  for b, nd in nodes.items():
    against = [
        gap(nd[e - 1], int(conn[b, 0, e, 0]))
        < gap(nd[-e], int(conn[b, 0, e, 0])) for e in (0, 1)
        if conn[b, 0, e, 2] != PHYSICAL
    ]
    if against and all(against):
      out.append(b)
  return tuple(out)


def sheath_open_rows(run, b, k):
  """The rows of block b (the file's psi index, a range) whose field lines end on a sheath at its theta ends: the
    range the sheath boundary condition's own files of frame k cover (<species>_bc_z{lower,upper}_vcutsq, written
    where it applies: a limited run's SOL rows only, its core rows closing on themselves); None where the block
    writes none."""
  p = next((q for s in run.species for e in ('lower', 'upper')
            if os.path.exists(q := run.f(b, f'{s}_bc_z{e}_vcutsq', k))), None)
  if p is None:
    return None
  grid, g = read_gkyl(run.f(b, 'field', k)), read_gkyl(p)
  lo, dx = grid['lower'][0], (grid['upper'][0] -
                              grid['lower'][0]) / grid['cells'][0]
  return range(int(round((g['lower'][0] - lo) / dx)),
               int(round((g['upper'][0] - lo) / dx)))


def magnetic_axis_R(run):
  """R of the magnetic axis from the run's geqdsk header (rmaxis on line 4: its line-3 copy is transposed in some
    files), or None."""
  path = geqdsk_path(run)
  return geqdsk_header(path)['rmaxis'] if path else None


_GEQDSK = {}


def geqdsk_header(path):
  """The magnetic axis and the flux at it and at the boundary from a geqdsk's header: dict(rmaxis, zmaxis, simag,
    sibry), from the copies on lines 4 and 5 (line 3 is transposed in some files, step.geqdsk's; the library reads the
    later copies too); nan where the header is short."""
  if path not in _GEQDSK:
    fre = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?')
    f = [
        float(x) for line in open(path).read().split('\n')[1:6]
        for x in fre.findall(line)
    ]
    get = lambda i: f[i] if len(f) > i else np.nan  # noqa: E731
    _GEQDSK[path] = dict(rmaxis=get(13),
                         zmaxis=get(15),
                         simag=get(11),
                         sibry=get(17))
  return _GEQDSK[path]


def sim_settings(sim):
  """The SimSettings of the simulation labelled sim: its own, or for a simulation viewer.json does not offer (a
    --path run's) the default device and no geqdsk."""
  return SIM_SETTINGS.get(sim) or SimSettings(DEVICE)


def open_run(path, sim=''):
  """The run in directory path, of the simulation labelled sim (its sim_settings: the device, the geqdsk)."""
  path = os.path.abspath(path)
  own = sim_settings(sim)
  r = Run(path, sim, device=own.device, geqdsk=own.geqdsk)
  r.stem, r.single = find_stem(path)
  frames = r.frames()
  if not frames:
    raise FileNotFoundError(f'{path}: no frames')
  k0 = frames[0]
  pre = f'{r.stem}-' if r.single else f'{r.stem}_b0-'
  names = set(dir_index(path)['m0'].get(pre, {}).get(k0, ()))
  # a species' source writes its moments as <species>_source_<moment> (gk_species_source.c): not a species
  r.species = sorted(
      n for n in names
      if not (n.endswith('_source') and n.removesuffix('_source') in names))
  if not r.species:
    raise FileNotFoundError(f'{path}: no moments at frame {k0}')
  for s in r.species:
    m = meta(r.f(0, f'{s}_M0', k0))
    r.mass[s], r.charge[s] = float(m['mass']), float(m['charge'])
  r.half_meta = int(
      meta(r.f(0, f'{r.species[0]}_M0', k0)).get('half_domain', -1))
  r.e = min(
      abs(q)
      for q in r.charge.values())  # the elementary charge the run used (for eV)
  r.options = read_options(path)
  if r.single:
    r.nb = 1
    nodes = {0: corner_nodes(r.f(0, 'geo_corn_nodes'))}
    conn = single_block_topology(nodes[0])
  else:
    tm = read_gkyl(f'{path}/{r.stem}-block_topo.gkyl')['meta']
    r.nb = int(tm['num_blocks'])
    conn = np.array(tm['connections']).reshape(r.nb, int(tm['ndim']), 2, 3)
    nodes = {b: corner_nodes(r.f(b, 'geo_corn_nodes')) for b in range(r.nb)}
    r.contrary = contrary_blocks(conn, nodes)
  r_axis = magnetic_axis_R(r)
  r.flip = not psi_grows_outward(conn, {
      b: nd[::-1] if b in r.contrary else nd
      for b, nd in nodes.items()
  }, r_axis)
  if r.flip:
    conn = flip_conn(conn)
  r.nodes = {b: r.rows(b, nd) for b, nd in nodes.items()}
  # a block whose sheath covers its outer rows only (a limited run): the first of them is its LCFS row
  lcfs = {}
  for b in range(r.nb):
    open_rows = sheath_open_rows(r, b, k0)
    if open_rows is not None and len(open_rows):
      rows = sorted(r.file_row(b, i) for i in open_rows)
      nr = r.nodes[b].shape[0] - 1
      if 0 < rows[0] and rows == list(range(rows[0], nr)):
        lcfs[b] = rows[0]
  r.layout = derive_layout(conn, r.nodes, r_axis, lcfs)
  r.conn = conn
  r.layout.flip = r.flip
  # shot and equilibrium time: the header of the geqdsk the run read, else its file stem
  g = geqdsk_path(r)
  m = re.search(r'#\s*(\d+)\s+(\d+)\s*ms', open(g).readline()) if g else None
  m = m or re.search(r'shot(\d+)_ms(\d+)', r.stem)
  if m:
    r.shot, r.eq_ms = int(m.group(1)), int(m.group(2))
  return r


def empty_run():
  """The run shown when there is none (viewer.json offers no simulation, or none of its runs can be read): no
    directory, no block, no frame, nothing to read. Its species are SPECIES_TAG's (their masses and charges unknown:
    nan) and its outboard and inboard midplanes hold no block, so that each chart keeps the panels, axes and legend
    it has for a run, with nothing in them: what the page looks like, never data."""
  unknown = dict.fromkeys(SPECIES_TAG, np.nan)
  return Run(
      '',
      species=list(SPECIES_TAG),
      mass=dict(unknown),
      charge=dict(unknown),
      e=np.nan,
      layout=Layout(
          {}, [], [],
          {s: Stack(s, None, None, False)
           for s in ('outboard', 'inboard')}, False))


def is_run_dir(p):
  """A directory holding a run with at least one frame (multiblock or single-block)."""
  if not os.path.isdir(p):
    return False
  try:
    stem, single = find_stem(p)
  except FileNotFoundError:
    return False
  return bool(dir_index(p)['frames'].get(stem if single else f'{stem}_b0'))


def runs_root(simdir):
  """Where a simulation's runs are: <simdir>/runs (simdir under base, or absolute), else simdir itself."""
  base = os.path.join(BASE, simdir)
  return os.path.join(base, 'runs') if os.path.isdir(os.path.join(
      base, 'runs')) else base


def runs_in(root):
  """The run directories in a directory: those with a topology file (or a single-block run's files) and a frame."""
  return [
      d for d in sorted(os.listdir(root)) if is_run_dir(os.path.join(root, d))
  ]


def path_simulation(path):
  """The simulation a directory makes of its own (--path, the sidebar's directory field): (label, the directory of
    its runs, the run opened, description), or (None, why). A run directory (its parent's runs offered beside it), a
    directory of runs (its last opened), or a simulation's directory with runs/ in it (runs_root)."""
  p = os.path.abspath(os.path.expanduser(str(path)))

  def name(d):  # a simulation's name: its directory's, past a runs/ directory
    return os.path.basename(
        os.path.dirname(d) if os.path.basename(d) == 'runs' else d)

  if is_run_dir(p):
    return (f'{name(os.path.dirname(p))} (--path)', os.path.dirname(p),
            os.path.basename(p), f'the run directory {p}'), ''
  for root in (p, os.path.join(p, 'runs')):
    if os.path.isdir(root) and runs_in(root):
      return (f'{name(root)} (--path)', root, runs_in(root)[-1],
              f'the runs in {root}'), ''
  return None, (
      f'{p} does not exist' if not os.path.exists(p) else
      f'{p} is no run directory, directory of runs, or simulation directory with runs/ in it'
  )


def sim_runs(simdir):
  """Run directories of a simulation (runs_in its runs_root)."""
  return runs_in(runs_root(simdir))


def geqdsk_named(path, options):
  """The geqdsk the run at path read, as its files name it: options_used.txt (options), else the -o geqdsk= of
    command.txt; a relative name made absolute from the run. None when neither names one."""
  if not path:  # no directory (the empty run): no files to name one
    return None
  name = options.get('geqdsk')
  cmd = os.path.join(path, 'command.txt')
  if not name and os.path.exists(cmd):
    m = re.search(r'geqdsk=(\S+)', open(cmd).read())
    name = m.group(1) if m else None
  return os.path.join(
      path, name) if name else None  # join keeps an absolute name as it is


def geqdsk_path(run):
  """The geqdsk the run read, here (here()): the one its own files name, else (they name none) the one viewer.json
    names for its simulation (Run.geqdsk); None when that is not here, or there is none. It gives the run's shot and
    time and its wall."""
  path = geqdsk_named(run.path, run.options)
  path = here(path) if path else run.geqdsk
  return path if path and os.path.exists(path) else None


def geqdsk_disagreement(run):
  """(the geqdsk the run's own files name, the one viewer.json names for its simulation) when both are here and
    are different files, else None. The run's own is the one used."""
  named = geqdsk_named(run.path, run.options)
  named = named and here(named)
  if named and run.geqdsk and os.path.exists(named) and os.path.exists(
      run.geqdsk) and not os.path.samefile(named, run.geqdsk):
    return named, run.geqdsk
  return None


def here(path):
  """A path that a run's files name on the machine that wrote them, on this one: one under viewer.json's "remote"
    base is at the same place under the base here (sync.py mirrors it there), when it is there."""
  base = (CONFIG.get('remote') or {}).get('base', '').rstrip('/')
  if base and path.startswith(base + '/'):
    mirrored = os.path.join(BASE, os.path.relpath(path, base))
    if os.path.exists(mirrored):
      return mirrored
  return path


def wall(run):
  """The wall drawn with a run, (R, Z) closed: the limiter of its geqdsk (geqdsk_path); where that gives none (no
    geqdsk, or fewer than 3 limiter points), the limiter of the geqdsk viewer.json names as its device's "wall"
    ("devices": {device: {"wall": <file, relative to base>}}: the wall only, never the run's shot); else None."""
  own = device_settings(run.device).get('wall')
  for path in (geqdsk_path(run),
               os.path.normpath(os.path.join(BASE, own)) if own else None):
    w = limiter(path) if path and os.path.exists(path) else None
    if w is not None:
      return w
  return None


def limiter(path):
  """The limiter outline (R, Z) of a geqdsk, closed; None when it has fewer than 3 points (no vessel)."""
  lines = open(path).read().split('\n')
  fre = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?')
  floats, intline = [], None
  for line in lines[1:]:
    if intline is None and len(floats) > 20 and re.fullmatch(
        r'\s*\d+\s+\d+\s*', line):
      intline = (len(floats), [int(x) for x in line.split()])
      continue
    floats += [float(x) for x in fre.findall(line)]
  if intline is None:
    return None
  pos, (nbdy, nlim) = intline
  lim = np.array(floats[pos + 2 * nbdy:pos + 2 * nbdy + 2 * nlim]).reshape(
      -1, 2)
  return np.vstack([lim, lim[:1]]) if len(lim) >= 3 else None


# ----------------------------------------------------------------------------------------------------------- frames
def species_tag(s):
  return SPECIES_TAG.get(s, (s, s))


def quantities(run):
  """Quantity key -> (label, symbol, unit, scale to the unit, kind: 'pos' | 'signed'). Symbol and unit are LaTeX
    math without the dollars (texlabel.py sets them); descriptive subscripts upright."""
  q = {}
  for s in run.species:
    full, t = species_tag(s)
    q[f'n_{s}'] = (f'{full} density', rf'n_\mathrm{{{t}}}',
                   r'10^{19}\,\mathrm{m^{-3}}', 1e-19, 'pos')
    q[f'T_{s}'] = (f'{full} temperature', rf'T_\mathrm{{{t}}}', r'\mathrm{eV}',
                   1.0, 'pos')
    q[f'Tpar_{s}'] = (f'{full} parallel temperature',
                      rf'T_{{\parallel,\mathrm{{{t}}}}}', r'\mathrm{eV}', 1.0,
                      'pos')
    q[f'Tperp_{s}'] = (f'{full} perpendicular temperature',
                       rf'T_{{\perp,\mathrm{{{t}}}}}', r'\mathrm{eV}', 1.0,
                       'pos')
    q[f'upar_{s}'] = (f'{full} parallel flow',
                      rf'u_{{\parallel,\mathrm{{{t}}}}}', r'\mathrm{km/s}',
                      1e-3, 'signed')
  if 'elc' in run.species and 'ion' in run.species:
    q['mach'] = ('ion parallel Mach number',
                 r'u_{\parallel,\mathrm{i}}/c_\mathrm{s}', '', 1.0, 'signed')
  q['phi'] = ('electrostatic potential', r'\phi', r'\mathrm{V}', 1.0, 'signed')
  q['q_par'] = ('total parallel energy flux', r'q_\parallel',
                r'\mathrm{MW\,m^{-2}}', 1e-6, 'signed')
  q['q_perp'] = ('parallel flux of perpendicular energy', r'q_\perp',
                 r'\mathrm{MW\,m^{-2}}', 1e-6, 'signed')
  return q


def _block_fields(run, b, k):
  d = {}
  qpar, qperp = 0.0, 0.0
  for s in run.species:
    M = {m: run.rows(b, cav(run.f(b, f'{s}_{m}', k))) for m in MOMS}
    n = M['M0']
    ok = n > 0
    with np.errstate(divide='ignore', invalid='ignore'):
      u = np.where(ok, M['M1'] / n, np.nan)
      m_e = run.mass[s] / run.e
      d[f'n_{s}'] = n
      d[f'upar_{s}'] = u
      d[f'T_{s}'] = np.where(ok,
                             m_e * (M['M2'] - M['M1'] * u) / (3 * n), np.nan)
      d[f'Tpar_{s}'] = np.where(ok,
                                m_e * (M['M2par'] - M['M1'] * u) / n, np.nan)
      d[f'Tperp_{s}'] = np.where(ok, m_e * M['M2perp'] / (2 * n), np.nan)
    qpar = qpar + run.mass[s] / 2 * (M['M3par'] + M['M3perp'])
    qperp = qperp + run.mass[s] / 2 * M['M3perp']
  if 'mach' in quantities(run):
    with np.errstate(invalid='ignore'):
      cs = np.sqrt((d['T_elc'] + d['T_ion']) * run.e / run.mass['ion'])
      d['mach'] = d['upar_ion'] / cs
  d['phi'] = run.rows(b, cav(run.f(b, 'field', k)))
  d['q_par'] = qpar
  d['q_perp'] = qperp
  return d


_FRAMES = {}


def frame(run, k):
  """block -> quantity -> (nr, nt) cell averages at frame k (the last 24 frames kept in memory)."""
  key = (run.path, k, run.stamp(k))
  if key not in _FRAMES:
    with ThreadPoolExecutor(
        4) as ex:  # I/O bound; 4 threads keep a login node responsive
      blocks = list(ex.map(lambda b: _block_fields(run, b, k), range(run.nb)))
    _FRAMES[key] = dict(enumerate(blocks))
    while len(_FRAMES) > 24:
      _FRAMES.pop(next(iter(_FRAMES)))
  return _FRAMES[key]


# ------------------------------------------------------------------------------------------- profiles and targets
def stack_blocks(st):
  """The blocks of a midplane stack, inside out (the core block under the SOL block, when there is one)."""
  return [b for b in (st.core, st.sol) if b is not None]


def midplane_radii(run, side):
  """(R at Z = 0 of each cell row through the core block then the SOL block on one side, R_sep); none and nan at
    a midplane holding no block (the empty run)."""
  st = run.layout.stacks[side]
  Rs = [
      0.5 * (e[1:] + e[:-1])
      for e in (mid_edges(run.nodes[b], st.j) for b in stack_blocks(st))
  ]
  return np.concatenate([np.zeros(0)] + Rs), mid_edges(
      run.nodes[st.sol], st.j)[st.sep_row] if st.sol is not None else np.nan


def midplane_profile(run, fr, side, qty):
  """(R at Z = 0 per cell row, value) through the core block then the SOL block on one side; R_sep."""
  st = run.layout.stacks[side]
  vs = []
  for b in stack_blocks(st):
    v = fr[b][qty]
    vs.append(
        np.array([(1 - w) * v[i, j0] + w * v[i, j1]
                  for i, (j0, j1,
                          w) in enumerate(midplane_weights(run.nodes[b], st.j))
                  ]))
  R, rsep = midplane_radii(run, side)
  return R, np.concatenate([np.zeros(0)] + vs), rsep


def stack_rows(run, side, blocks=None):
  """Which rows of a midplane profile (midplane_profile's order) belong to the chosen blocks (None: all)."""
  st = run.layout.stacks[side]
  return np.concatenate([np.zeros(0, bool)] + [
      np.full(run.nodes[b].shape[0] - 1, blocks is None or b in blocks)
      for b in stack_blocks(st)
  ])


def target_heat_flux(run, fr, tgt):
  """|q_par| per SOL surface at the plate cell, the surfaces' midplane distance from the separatrix (mm), lambda_q.
    Where no block of the leg's chain reaches Z = 0 (tgt.mid None) the surfaces sit at their distance along the plate
    from its inner corner instead, and there is no lambda_q."""
  r0 = tgt.open_row
  q = np.abs(fr[tgt.leg]['q_par'][r0:, tgt.j])
  if tgt.mid is None:
    face = run.nodes[tgt.leg][r0:, tgt.j]
    dist = np.concatenate([[0.0],
                           np.cumsum(np.hypot(*np.diff(face, axis=0).T))])
    return q, 0.5 * (dist[1:] + dist[:-1]) * 1e3, np.nan
  edges = mid_edges(run.nodes[tgt.mid], tgt.mid_j)[r0:]
  dist = np.abs(edges - edges[0])
  C = np.concatenate([[0.0], np.cumsum(q * np.diff(dist))])
  lam = (np.interp(LAMBDA_FRACTION *
                   C[-1], C, dist)) * 1e3 if C[-1] > 0 else np.nan
  return q, 0.5 * (dist[1:] + dist[:-1]) * 1e3, lam


# ---------------------------------------------------------------------------------------- per-frame summary cache
def cell_offsets(run):
  """block -> index of its first cell in the flat cell order of the cached fields (blocks in turn, rows of psi)."""
  off, n = {}, 0
  for b in range(run.nb):
    off[b] = n
    n += (run.nodes[b].shape[0] - 1) * (run.nodes[b].shape[1] - 1)
  return off


def _summary_one(run, fr, t, fr0):
  """The summary entries of one frame fr (block -> fields) at time t."""
  out = {'t': t}
  for tgt in run.layout.targets:
    q, _, lam = target_heat_flux(run, fr, tgt)
    out[f'q|{tgt.name}'] = q
    out[f'lam|{tgt.name}'] = lam
  for side in run.layout.stacks:
    for qty in quantities(run):
      out[f'mid|{side}|{qty}'] = midplane_profile(run, fr, side, qty)[1]
  for qty in quantities(run):
    v = np.concatenate([np.zeros(0)] +
                       [fr[b][qty].ravel() for b in range(run.nb)])
    out[f'fld|{qty}'] = v.astype(
        np.float32)  # every cell (cell_offsets order): the browser-side movie
    v = v[np.isfinite(v)]
    out[f'pct|{qty}'] = np.percentile(
        v, [0, 1, 50, 99, 100]) if len(v) else np.full(5, np.nan)
  for reg in ('core', 'sol', 'pf'):
    bs = [(b, region_rows(run, b, reg)) for b in range(run.nb)]
    for s in run.species:
      for base in ('n', 'T'):
        qty = f'{base}_{s}'
        with np.errstate(divide='ignore', invalid='ignore'):
          rel = np.concatenate([(fr[b][qty][i] / fr0[b][qty][i] - 1).ravel()
                                for b, i in bs] or [np.zeros(0)])
        rel = rel[np.isfinite(rel)]
        out[f'med|{reg}|{qty}'] = np.median(rel) if len(rel) else np.nan
  for s in run.species:
    out[f'neg|{s}'] = float(
        sum((fr[b][f'n_{s}'] <= 0).sum() for b in range(run.nb)))
  return out


CACHES = {
    'summary': ('', CACHE_VERSION),
    'dt': ('dt__', 1
           )  # bump when a time-step limit's definition changes (dt_limits_one)
}  # per-frame cache files of a run: kind -> (name part, version)


def cache_path(run_path, kind='summary', ext='npz'):
  """The per-frame cache file of one kind (CACHES) of the run at run_path (named by its place under base: the same
    on a mirror); ext 'json': the settings file kept beside it (dt_limits)."""
  rel = os.path.relpath(run_path, BASE).replace(os.sep, '__')
  part, version = CACHES[kind]
  return os.path.join(CACHE_DIR, f'{rel}__{part}v{version}.{ext}')


def cache_files(run_path):
  """Every cache file the run at run_path may have: each kind's per-frame file, and the time-step limits' settings."""
  return [cache_path(run_path, k)
          for k in CACHES] + [cache_path(run_path, 'dt', 'json')]


def frame_cache(run, path, one, progress=None):
  """Per-frame arrays (key -> [n_frames, ...]) over every frame written, kept in the file at path: only the frames
    new or rewritten since it was written (their stamps) are computed, one(k) -> {key: value} each, and the file is
    then rewritten. progress(iterable, total) wraps the loop over them (a progress bar). 'fld|' keys are float32."""
  frames = run.frames()
  stamps = np.array([run.stamp(k) for k in frames], dtype=np.int64)
  old = {}
  if os.path.exists(path):
    with np.load(path) as z:
      old = {key: z[key] for key in z.files}
  have = {
      int(k): (int(s), i)
      for i, (k,
              s) in enumerate(zip(old.get('frames', ()), old.get('stamp', ())))
  }
  todo = [k for k, s in zip(frames, stamps) if have.get(k, (None, ))[0] != s]
  it = progress(todo, len(todo)) if (progress and todo) else todo
  new = {k: one(k) for k in it}
  keys = sorted(set(new[todo[0]]) if todo else set(old) - {'frames', 'stamp'})
  out = {'frames': np.array(frames), 'stamp': stamps}
  for key in keys:
    out[key] = np.array([
        np.asarray(new[k][key],
                   dtype=np.float32 if key.startswith('fld|') else float)
        if k in new else old[key][have[k][1]] for k in frames
    ])
  if todo:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez(tmp, **out)
    os.replace(tmp, path)
  return out


_SUMMARY = {}


def summary(run, progress=None):
  """Per-frame summary arrays (key -> [n_frames, ...]) over every frame written, cached on disk in runs/cache (and
    in memory); only new or rewritten frames are read (frame_cache). progress(iterable, total) wraps the loop over
    them (a progress bar). 'fld|<qty>' holds every cell's value (float32, cell_offsets order)."""
  frames = run.frames()
  if not frames:  # nothing written (the empty run): every entry, no row
    return dict(
        {
            key: np.zeros((0, ) + np.shape(v))
            for key, v in _summary_one(run, {}, np.nan, {}).items()
        },
        frames=np.zeros(0, int),
        stamp=np.zeros(0, np.int64))
  stamps = np.array([run.stamp(k) for k in frames], dtype=np.int64)
  memo = _SUMMARY.get(run.path)
  if memo is not None and np.array_equal(memo['stamp'], stamps):
    return memo
  fr0 = {}  # the first frame, read once if any frame is

  def one(k):
    if not fr0:
      fr0.update(frame(run, frames[0]))
    return _summary_one(run, frame(run, k), run.time(k), fr0)

  out = frame_cache(run, cache_path(run.path), one, progress)
  _SUMMARY[run.path] = out
  return out


# ------------------------------------------------------------------------------------------------------ time traces
def dt_history(run):
  """(t, dt) of every step (the run's -dt file); none for a run with no block (the empty run)."""
  if not run.nb:
    return np.zeros(0), np.zeros(0)
  t, d = read_dynvec(f'{run.path}/{run.stem}-dt.gkyl')
  return t, d[:, 0]


def integrated(run):
  """species -> (t, [N, P_par, H]) summed over blocks: the HamiltonianMoments integrated over each block."""
  out = {}
  for s in run.species:
    tot, t0 = None, None
    for b in range(run.nb):
      p = run.f(b, f'{s}_integrated_moms')
      if not os.path.exists(p):
        tot = None
        break
      t, d = read_dynvec(p)
      if tot is None:
        t0, tot = t, d.copy()
      else:
        n = min(len(t0), len(t))
        t0, tot = t0[:n], tot[:n] + d[:n]
    if tot is not None:
      out[s] = (t0, tot)
  return out


def field_energy(run):
  """(t, field energy summed over blocks), or None when the run wrote none."""
  tot, t0 = None, None
  for b in range(run.nb):
    p = run.f(b, 'field_energy')
    if not os.path.exists(p):
      return None
    t, d = read_dynvec(p)
    if tot is None:
      t0, tot = t, d[:, 0].copy()
    else:
      n = min(len(t0), len(t))
      t0, tot = t0[:n], tot[:n] + d[:n, 0]
  return (t0, tot) if tot is not None else None


# ------------------------------------------------------------------------------------------------ time-step limits
# What sets the time step, rebuilt from the frames the way the library takes it (gyrokinetic.c gyrokinetic_rhs,
# gk_species_kinetic.c): per block and species dt = min(cfl_frac / max cflrate, cfl_frac_omegaH / omega_H), the step
# the smallest over all of them. cflrate sums, per phase-space cell, each term's rate:
#   x and z (ker/gk_collisionless_flux surfx, surfy): 3 / dx max |xdot| over the Gauss nodes of the cell's lower face
#     (v_par at its 3, mu at its 2). z: v_par b.grad z (streaming), with the drifts also (m v_par^2 / q) (curl b / B)^z
#     (curvature) - (b_y / q J B) mu d_x B (grad B) - (b_y / J B) d_x phi (E x B); x: the drifts with d_z (signs
#     flipped), no streaming;
#   v_par (surfvpar, interior faces): 5 / dv_par (the narrower cell of the two) max |v_par dot| at the volume Gauss
#     nodes: -(b.grad z / m) (q d_z phi (E_par) + mu d_z B (mirror force)); with the drifts also -(v_par / q) (curl b /
#     B).grad H, their energy exchange, counted with the magnetic drifts;
#   BGK collisions (vlasov/zero/bgk_collisions.c): the cell average of nu_sum;
#   anomalous diffusion (ker/gk_anomalous_diffusion vol): 2 c0(D g^xx J) (2 / dx)^2.
# A direction's rate is split among its terms by their sizes at its fastest node (they add inside one |.|).
# omega_H = |q| sqrt(n_max / m) gf (gk_species_kinetic_omegaH_dt), n_max the block's largest cell-average density, gf
# the largest cell average of (cmag J_tot^-1 2 pi / dz) (k_x^2 W g^xx J)^-1/2 (gyrokinetic.c omegaH_init), k_x = pi /
# the block's x extent, W = sum_s n_pol,s m_s / B_ref^2 the polarization weight. The runs do not write W: it is the
# block's field energy over 1/2 int g^xx J (d_x phi)^2 (the field energy is 1/2 int W g^xx J (d_x phi)^2,
# gkyl_array_integrate_op_eps_gradperp_sq).
DT_TERMS = ('streaming', 'exb', 'magnetic', 'epar', 'mirror', 'collisions',
            'diffusion')
LIB_CFL_FRAC, LIB_CFL_FRAC_OMEGAH = 1.0, 1.7  # the library's, for an input of 0 (gyrokinetic.c)
LIB_DT_STICK = 0.01  # gyrokinetic_rhs keeps the step it was given when the new limit is less than 1 % below it
GAUSS2 = 1 / np.sqrt(3)
GAUSS3 = np.sqrt(0.6)


@dataclass(frozen=True)
class DtSetup:
  """What a run's time step is made of, fixed for the run (dt_setup)."""
  cfl: float  # cfl_frac
  cfl_H: float  # cfl_frac_omegaH
  drifts: bool  # the magnetic and E x B drifts (GKYL_GK_COLLISIONLESS_ES), else none (ES_NO_BY)
  species: tuple  # the charged species (each one's own limits)
  geo: dict  # block -> geometry (_dt_geometry)
  vel: dict  # (block, species) -> (v_par0, v_par1, mu0, mu1): each velocity cell's map, centre + slope x eta
  D: dict  # (block, species) -> per cell c0 of D g^xx J (the diffusion's rate), or None
  gf: dict  # block -> the omega_H geometric factor (nan: W unknown)
  notes: tuple  # where each setting came from


def _data(path):
  return np.asarray(read_gkyl(path)['data'])


def _at(c, x, y):
  """p = 1 modal fields c (..., 4) at (x, y) of each cell."""
  return c[..., :4] @ ser_phi(x, y)


def _ddx(c, y, dx):
  """d/dx of p = 1 modal fields c at the cells' y (their x derivative is the same at every x)."""
  return 2 / dx * (np.sqrt(3) / 2 * c[..., 1] + 1.5 * c[..., 3] * y)


def _ddz(c, x, dz):
  return 2 / dz * (np.sqrt(3) / 2 * c[..., 2] + 1.5 * c[..., 3] * x)


def _project(f):
  """The p = 1 expansion of f(x, y) from its values at the 2 x 2 Gauss nodes (as the library projects)."""
  return sum(
      np.multiply.outer(f(x, y), ser_phi(x, y)) for x in (-GAUSS2, GAUSS2)
      for y in (-GAUSS2, GAUSS2))


def _weak_mul(a, b):
  return np.einsum('kij,...i,...j->...k', SER_T, a, b)


def _dt_geometry(run, b):
  """Block b's geometry the rates use: modal fields (nx, nz, 4) by name, b.grad z as 'bz' (B3overB, or B3 / bmag
    where the run wrote no B3overB), the Hamiltonian's |B| as 'Bh' (geo_corn_bmag, gk_collisionless_flux.c), and its
    cell sizes dx, dz and x extent Lx."""
  g = {
      w: _data(run.f(b, f'geo_int_{w}'))
      for w in ('gxxj', 'cmag', 'jacobtot_inv', 'bmag', 'dualcurlbhatoverB',
                'bioverJB')
  }
  g['Bh'] = _data(run.f(b, 'geo_corn_bmag'))
  p = run.f(b, 'geo_int_B3overB')
  if os.path.exists(p):
    g['bz'] = _data(p)
  else:
    B3 = _data(run.f(b, 'geo_int_B3'))
    g['bz'] = _project(lambda x, y: _at(B3, x, y) / _at(g['bmag'], x, y))
  d = read_gkyl(run.f(b, 'geo_int_gxxj'))
  g['dx'], g['dz'] = (d['upper'] - d['lower']) / np.array(d['cells'])
  g['Lx'] = d['upper'][0] - d['lower'][0]
  return g


def _dt_velocity(run, b, s):
  """(v_par centre, v_par slope, mu centre, mu slope) of each velocity cell: v_par = centre + slope eta in the cell
    (its map, <s>_mapc2p_vel: p = 1 along each velocity direction)."""
  vm = _data(run.f(b, f'{s}_mapc2p_vel'))
  return (vm[:, 0, 0] / np.sqrt(2), vm[:, 0, 1] * np.sqrt(1.5),
          vm[0, :, 2] / np.sqrt(2), vm[0, :, 3] * np.sqrt(1.5))


def _energy_over_W(g, phi):
  """The block's field energy over its polarization weight W: 1/2 int g^xx J (d_x phi)^2 (the library's integral,
    gkyl_array_integrate_op_eps_gradperp_sq_2x_ser_p1, cell volume dx dz / 4)."""
  c1, c3, G = phi[..., 1], phi[..., 3], g['gxxj']
  return float(
      np.sum(6 / g['dx']**2 * 0.5 *
             (G[..., 0] * (c1**2 + c3**2) + 2 * G[..., 2] * c1 * c3) * g['dx'] *
             g['dz'] / 4))


def polarization_weight(run, b, geo):
  """(W, frame) of block b from its field energy at the first frame after the first whose potential gives it (the
    energy sampled at the frame's time); (nan, None) when none does."""
  p = run.f(b, 'field_energy')
  if not os.path.exists(p):
    return np.nan, None
  t, E = read_dynvec(p)
  frames = run.frames()
  for k in frames[1:4] or frames[:1]:
    tk = run.time(k)
    i = np.where(np.isclose(t, tk, rtol=1e-9, atol=0))[0]
    K = _energy_over_W(geo, _data(run.f(b, 'field', k)))
    if len(i) and K > 0:
      return float(E[i[-1], 0] / K), k
  return np.nan, None


def _omegaH_gf(g, W):
  par = _weak_mul(g['cmag'], g['jacobtot_inv']) * (2 * np.pi / g['dz'])
  perp = (np.pi / g['Lx'])**2 * W * g['gxxj']
  with np.errstate(invalid='ignore'):
    inv = _project(lambda x, y: _at(perp, x, y)**-0.5)
  return float(np.nanmax(_weak_mul(par, inv)[..., 0] / 2))


def _direction(pieces, factor):
  """One direction's rate per phase-space cell (nx, nz, nv, nmu): factor max |sum of pieces| over the nodes (the
    last axis), split among the pieces by their sizes at that node: {term: its part}; and {term: factor max |piece|}
    per cell (the term's own rate in this direction)."""
  shape = np.broadcast_shapes(*[np.shape(p) for p in pieces.values()])
  P = {n: np.broadcast_to(p, shape) for n, p in pieces.items()}
  a = np.abs(sum(P.values()))
  i = np.argmax(a, -1)[..., None]
  rate = factor * np.take_along_axis(a, i, -1)[..., 0]
  size = {n: np.abs(np.take_along_axis(p, i, -1)[..., 0]) for n, p in P.items()}
  tot = sum(size.values())
  with np.errstate(invalid='ignore', divide='ignore'):
    parts = {n: np.where(tot > 0, rate * v / tot, 0.0) for n, v in size.items()}
  own = {n: factor * np.abs(p).max(-1) for n, p in P.items()}
  return parts, own


def _rates(geo, vel, q, m, phi, drifts):
  """The CFL rate of the collisionless terms per phase-space cell (nx, nz, nv, nmu): {term: its part of the rate},
    {term: its own rate}, both summed over the directions."""
  vp0, vp1, mu0, mu1 = vel
  VQ = np.stack([vp0 + vp1 * e for e in (-GAUSS3, 0.0, GAUSS3)],
                -1)[None, None, :, None, :, None]  # (1, 1, nv, 1, 3, 1)
  MQ = np.stack([mu0 + mu1 * e for e in (-GAUSS2, GAUSS2)],
                -1)[None, None, None, :, None, :]  # (1, 1, 1, nmu, 1, 2)
  c = lambda a: a[:, :, None, None, None, None
                  ]  # noqa: E731  a conf field (nx, nz) against the velocity nodes
  dx, dz, bz, Bh = geo['dx'], geo['dz'], geo['bz'], geo['Bh']
  cx, cz = geo['dualcurlbhatoverB'][..., 0:4], geo['dualcurlbhatoverB'][...,
                                                                        8:12]
  by = geo['bioverJB'][..., 4:8]
  flat = lambda a: a.reshape(a.shape[:4] + (-1, )
                             )  # noqa: E731  (v_par, mu) nodes as one axis
  dirs = []
  z = {'streaming': [], 'magnetic': [], 'exb': []}
  for x in (-GAUSS2, GAUSS2):  # the lower z face
    z['streaming'].append(flat(VQ * c(_at(bz, x, -1)) + 0 * MQ))
    if drifts:
      z['magnetic'].append(
          flat(m * VQ**2 / q * c(_at(cz, x, -1)) -
               c(_at(by, x, -1) * _ddx(Bh, -1, dx)) / q * MQ))
      z['exb'].append(flat(-c(_at(by, x, -1) * _ddx(phi, -1, dx)) +
                           0 * VQ * MQ))
  dirs.append(({n: np.concatenate(v, -1) for n, v in z.items() if v}, 3 / dz))
  if drifts:  # the lower x face
    xs = {'magnetic': [], 'exb': []}
    for y in (-GAUSS2, GAUSS2):
      xs['magnetic'].append(
          flat(m * VQ**2 / q * c(_at(cx, -1, y)) +
               c(_at(by, -1, y) * _ddz(Bh, -1, dz)) / q * MQ))
      xs['exb'].append(
          flat(c(_at(by, -1, y) * _ddz(phi, -1, dz)) + 0 * VQ * MQ))
    dirs.append(({n: np.concatenate(v, -1) for n, v in xs.items()}, 3 / dx))
  # v_par faces: each cell's lower one (none for the first cell), at the volume nodes
  VS = (vp0 - vp1)[None, None, :, None, None]  # v_par on the face
  M2 = MQ[..., 0, :]  # (1, 1, 1, nmu, 2)
  cv = lambda a: a[:, :, None, None, None]  # noqa: E731
  v = {'epar': [], 'mirror': [], 'magnetic': []}
  for x in (-GAUSS2, GAUSS2):
    for y in (-GAUSS2, GAUSS2):
      b_z = _at(bz, x, y)
      v['epar'].append(cv(-b_z * q * _ddz(phi, x, dz) / m) + 0 * VS * M2)
      v['mirror'].append(-cv(b_z * _ddz(Bh, x, dz) / m) * M2 + 0 * VS)
      if drifts:
        v['magnetic'].append(
            -VS / q * (cv(_at(cx, x, y)) *
                       (cv(q * _ddx(phi, y, dx)) + cv(_ddx(Bh, y, dx)) * M2) +
                       cv(_at(cz, x, y)) *
                       (cv(q * _ddz(phi, x, dz)) + cv(_ddz(Bh, x, dz)) * M2)))
  dv = np.zeros(len(vp0))
  dv[1:] = 2 * np.minimum(vp1[1:],
                          vp1[:-1])  # the narrower of the two cells' widths
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


def _diffusion_rate(run, b, s, geo):
  """c0 of D g^xx J per cell (nx, nz) of block b, species s: the anom_diff_0 file, else the uniform D the run records
    times geo_int_gxxj; None without diffusion on record."""
  p = run.f(b, f'{s}_anom_diff', 0)
  if os.path.exists(p):
    return _data(p)[..., 0]
  D, _ = recorded_uniform_D(run)
  return D * geo['gxxj'][..., 0] if D is not None else None


def _dt_one(run, st, k, drifts=None):
  """Frame k's limits: {'t', 'dt_cfl', 'dt_H', 'cfl_at' (block, species index, i, j, i_vpar, i_mu of the cell whose
    rate sets dt_cfl, the files' own indices), 'H_at' (block, species index), 'v_at' (v_par, mu at that cell's
    centre), 'share|<term>' (each term's part of that cell's rate), 'alone|<term>' (cfl over the term's own largest
    rate)}. drifts: st's, or the one given (dt_setup's choice)."""
  drifts = st.drifts if drifts is None else drifts
  best = (0.0, None)  # (rate, (b, si, idx, parts at idx))
  own_max = dict.fromkeys(DT_TERMS, 0.0)
  dtH, H_at = np.inf, (np.nan, np.nan)

  def block(b):
    phi = _data(run.f(b, 'field', k))
    res = []
    for si, s in enumerate(st.species):
      q, m = run.charge[s], run.mass[s]
      parts, own = _rates(st.geo[b], st.vel[(b, s)], q, m, phi, drifts)
      shape = next(iter(parts.values())).shape
      p = run.f(b, f'{s}_bgk_nu_sum', k)
      if os.path.exists(p):
        parts['collisions'] = own['collisions'] = np.broadcast_to(
            (_data(p)[..., 0] / 2)[:, :, None, None], shape)
      if st.D[(b, s)] is not None:
        parts['diffusion'] = own['diffusion'] = np.broadcast_to(
            (2 * st.D[(b, s)] * (2 / st.geo[b]['dx'])**2)[:, :, None,
                                                          None], shape)
      R = sum(parts.values())
      idx = np.unravel_index(np.argmax(R), R.shape)
      n_max = float((_data(run.f(b, f'{s}_M0', k))[..., 0] / 2).max())
      om = abs(q) * np.sqrt(max(n_max, 0.0) / m) * st.gf[b]
      res.append((si, float(R[idx]), idx, {
          n: float(v[idx])
          for n, v in parts.items()
      }, {
          n: float(v.max())
          for n, v in own.items()
      }, st.cfl_H / om if om > 0 else np.inf))
    return b, res

  with ThreadPoolExecutor(4) as ex:
    for b, res in ex.map(block, range(run.nb)):
      for si, rate, idx, at, own, dth in res:
        if rate > best[0]:
          best = (rate, (b, si, idx, at))
        for n, v in own.items():
          own_max[n] = max(own_max[n], v)
        if dth < dtH:
          dtH, H_at = dth, (b, si)
  rate, (b, si, idx, at) = best
  vp0, vp1, mu0, _ = st.vel[(b, st.species[si])]
  if len(
      st.species
  ) < 2:  # an adiabatic or Boltzmann closure: the library applies no omega_H limit
    dtH = np.inf
  elif not all(np.isfinite(v)
               for v in st.gf.values()):  # no polarization weight: not rebuilt
    dtH = np.nan
  out = {
      't': run.time(k),
      'dt_cfl': st.cfl / rate,
      'dt_H': dtH,
      'cfl_at': np.array((b, si) + tuple(idx), float),
      'H_at': np.array(H_at, float),
      'v_at': np.array([vp0[idx[2]], mu0[idx[3]]])
  }
  for n in DT_TERMS:
    out[f'share|{n}'] = at.get(n, 0.0) / rate
    out[f'alone|{n}'] = st.cfl / own_max[n] if own_max[n] > 0 else np.inf
  return out


def taken_after(run, t):
  """The step the run took from each time t (its -dt file: the first step written after t); nan past its end."""
  td, vd = dt_history(run)
  i = np.searchsorted(td, t, side='right')
  return np.where(i < len(vd), vd[np.minimum(i, len(vd) - 1)], np.nan)


_DT_SETUP = {}


def dt_setup(run):
  """The run's DtSetup, or (None, why) when its files cannot give one. cfl_frac and cfl_frac_omegaH: its launch
    options, else the library's defaults; the drifts: its 'drifts' option, else the choice (with or without) whose
    rebuilt step at its second frame is the nearer the step it took."""
  if run.path in _DT_SETUP:
    return _DT_SETUP[run.path]
  if not run.nb or not run.frames():
    return None, 'no frames'
  opts = launch_options(run)
  notes = []

  def frac(key, lib):
    v = float(opts.get(key, 0) or 0)
    notes.append(f'{key} {v:g} (its options)' if v > 0 else
                 f'{key} {lib:g} (the library\'s default: none in its options)')
    return v if v > 0 else lib

  cfl, cfl_H = frac('cfl_frac', LIB_CFL_FRAC), frac('cfl_frac_omegaH',
                                                    LIB_CFL_FRAC_OMEGAH)
  species = tuple(s for s in run.species if run.charge[s] != 0)
  if len(species) < len(run.species):
    notes.append('the neutral species\' own limit is not rebuilt')
  try:
    geo = {b: _dt_geometry(run, b) for b in range(run.nb)}
    vel = {
        (b, s): _dt_velocity(run, b, s)
        for b in range(run.nb)
        for s in species
    }
  except FileNotFoundError as e:
    out = (None, f'the run wrote no {os.path.basename(e.filename)}')
    _DT_SETUP[run.path] = out
    return out
  D = {
      (b, s): _diffusion_rate(run, b, s, geo[b])
      for b in range(run.nb)
      for s in species
  }
  gf, Ws = {}, []
  for b in range(run.nb):
    W, kW = polarization_weight(run, b,
                                geo[b]) if len(species) > 1 else (np.nan, None)
    gf[b] = _omegaH_gf(geo[b], W) if np.isfinite(W) else np.nan
    Ws.append(W)
  if len(species) < 2:
    notes.append(
        'one charged species (adiabatic or Boltzmann closure): no omega_H limit'
    )
  elif not np.isfinite(Ws).all():
    notes.append(
        'omega_H not rebuilt: no field energy at a frame\'s time to give the polarization weight'
    )
  else:
    notes.append(
        f'polarization weight W = {np.median(Ws):.4g} kg m^-3 T^-2 from the field energy (blocks agree to '
        f'{(max(Ws) - min(Ws)) / np.median(Ws):.1e})')
  st = DtSetup(cfl, cfl_H, False, species, geo, vel, D, gf, ())
  if 'drifts' in opts:
    drifts = opts['drifts'].lower() in ('1', 'true', 'yes')
    notes.append(f'drifts {"on" if drifts else "off"} (its options)')
  else:  # not on record: the choice that rebuilds the step it took
    frames = run.frames()
    k = frames[1] if len(frames) > 1 else frames[0]
    taken = float(taken_after(run, run.time(k)))
    gap = {}
    for d in (True, False):
      r = _dt_one(run, st, k, d)
      gap[d] = abs(taken / np.nanmin([r['dt_cfl'], r['dt_H']]) - 1)
    if np.isfinite(taken):
      drifts = gap[True] <= gap[False]
      notes.append(
          f'drifts {"on" if drifts else "off"}: not in its options; the rebuild at frame {k} is '
          f'{gap[True]:.1%} from the step taken with them, {gap[False]:.1%} without'
      )
    else:
      drifts = True
      notes.append(
          'drifts on: neither in its options nor a step taken to choose by')
  out = (DtSetup(cfl, cfl_H, drifts, species, geo, vel, D, gf,
                 tuple(notes)), '')
  _DT_SETUP[run.path] = out
  return out


class _NoLimits(Exception):
  """dt_setup found no setup (its reason the message)."""


def _dt_info(st):
  """What the note on the limits needs of a DtSetup: the charged species, each '<block>|<species>' velocity grid's
    cells [n_vpar, n_mu], where each setting came from (JSON)."""
  return {
      'species': list(st.species),
      'vel_cells': {
          f'{b}|{s}': [len(v[0]), len(v[2])]
          for (b, s), v in st.vel.items()
      },
      'notes': list(st.notes)
  }


def dt_limits(run, progress=None):
  """Per-frame time-step limits (_dt_one's entries, key -> [n_frames, ...]) over every frame written, cached on disk
    (frame_cache, cache kind 'dt'), with 'taken' (the step the run took from each frame), _dt_info's entries (kept
    beside the cache, so that a run whose frames are all cached reads no geometry) and 'why' (why there are none:
    '' when there are). The setup (dt_setup) is built only for frames to compute or a missing settings file."""
  if not run.nb or not run.frames():
    return {'why': 'no frames'}
  built = {}

  def setup():
    if 'st' not in built:
      st, why = dt_setup(run)
      if st is None:
        raise _NoLimits(why)
      built['st'] = st
    return built['st']

  info_path = cache_path(run.path, 'dt', 'json')
  try:
    out = dict(
        frame_cache(run, cache_path(run.path, 'dt'),
                    lambda k: _dt_one(run, setup(), k), progress))
    info = None
    if not built and os.path.exists(info_path):
      with open(info_path) as f:
        info = json.load(f)
    if info is None:
      info = _dt_info(setup())
      os.makedirs(CACHE_DIR, exist_ok=True)
      with open(info_path + '.tmp', 'w') as f:
        json.dump(info, f)
      os.replace(info_path + '.tmp', info_path)
  except _NoLimits as e:
    return {'why': str(e)}
  out.update(info, taken=taken_after(run, out['t']), why='')
  return out


def dt_agreement(lim):
  """(median, smallest, largest) of the step taken over the rebuilt one, from the second frame on (the first step
    starts from the initial condition); nans without two such frames."""
  r = lim['taken'][1:] / np.fmin(
      lim['dt_cfl'][1:],
      np.where(np.isnan(lim['dt_H'][1:]), np.inf, lim['dt_H'][1:]))
  r = r[np.isfinite(r)]
  return (float(np.median(r)), float(r.min()),
          float(r.max())) if len(r) else (np.nan, ) * 3


def dt_reproduced(lim):
  """Whether the rebuild reproduces the run's steps: the median ratio (dt_agreement) within twice the library's
    keep-the-step margin (LIB_DT_STICK) of 1."""
  return abs(dt_agreement(lim)[0] - 1) <= 2 * LIB_DT_STICK


# --------------------------------------------------------------------------------------------------- frame ranges
def select_frames(spec, available):
  """The frame numbers a frame specification selects among the available ones, read as `pgkyl load --frame` reads
    it (postgkyl.io.naming.select_frames, ported): '7', '0,10,20', or 'start:stop[:step]' = the available frames
    start <= f < stop whose number is start plus a multiple of step (':' all, '::2' every other frame number); a
    negative number or bound counts back from the last frame ('-1' the last, '-10:' the last ten). Blank = all."""
  available = sorted(int(f) for f in available)

  def position(f):
    if f >= 0:
      return f
    if -f > len(available):
      raise ValueError(
          f'frame {f} counts back past the first of the {len(available)} frames'
      )
    return available[f]

  def bound(v, default):
    return default if v is None else (
        v if v >= 0 else available[max(len(available) + v, 0)])

  text = str(spec).strip()
  if not text:
    return list(available)
  try:
    if ':' not in text:
      return [position(int(f)) for f in text.split(',')]
    parts = text.split(':')
    if len(parts) > 3:
      raise ValueError
    start, stop, step = [int(p) if p.strip() else None
                         for p in parts] + [None] * (3 - len(parts))
  except ValueError:
    raise ValueError(
        f"frames {spec!r}: not a frame number, a comma-separated list or a 'start:stop[:step]' "
        'range') from None
  step = 1 if step is None else step
  if step <= 0:
    raise ValueError(f'frames {spec!r}: the step must be a positive integer')
  if not available:
    return []
  lo, hi = bound(start, available[0]), bound(stop, available[-1] + 1)
  return [f for f in available if lo <= f < hi and (f - lo) % step == 0]


# ------------------------------------------------------------------------------------------------------- cell edges
# where a cell edge's two Gauss-Legendre nodes lie along it (0 and 1: its corners), and where a drawn edge passes:
# the cubic through the corners and those nodes, sampled at them and evenly between
GAUSS_T = (0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3))
EDGE_T = np.sort(np.r_[np.linspace(0, 1, 7)[1:-1], GAUSS_T])


def _edge_weights():
  """[len(EDGE_T), 4]: the weights of an edge's corner, Gauss nodes and corner (Lagrange in t at 0, GAUSS_T, 1) at
    each of EDGE_T."""
  nodes = np.r_[0.0, GAUSS_T, 1.0]
  return np.array([[
      np.prod([(t - nodes[m]) / (nodes[k] - nodes[m]) for m in range(4)
               if m != k]) for k in range(4)
  ] for t in EDGE_T])


def _edge_polylines(C, s0, s1):
  """A block's cell edges as polylines along curves, each edge the cubic through its corners and its two Gauss nodes
    (EDGE_T): its node rows [nr + 1, (len(EDGE_T) + 1) nt + 1, 2] and node columns [nt + 1, (len(EDGE_T) + 1) nr + 1,
    2], from its corners C [nr + 1, nt + 1, 2] and the two Gauss nodes on each constant-psi edge s0 [nr + 1, 2 nt, 2]
    and constant-theta edge s1 [2 nr, nt + 1, 2]."""
  W = _edge_weights()

  def along(
      A, G1, G2,
      B):  # edges [..., n, 2] -> polylines [..., (len(EDGE_T) + 1) n + 1, 2]
    inner = np.einsum('sk,k...->...s', W, np.stack([A, G1, G2,
                                                    B]))  # [..., n, 2, s]
    pts = np.concatenate([A[..., None], inner],
                         axis=-1).transpose(*range(A.ndim - 2), A.ndim - 2,
                                            A.ndim,
                                            A.ndim - 1)  # [..., n, 1 + s, 2]
    flat = pts.reshape(*pts.shape[:-3], -1, 2)
    return np.concatenate([flat, B[..., -1:, :]], axis=-2)

  rows = along(C[:, :-1], s0[:, 0::2], s0[:, 1::2], C[:, 1:])
  cols = along(*(a.transpose(1, 0, 2)
                 for a in (C[:-1], s1[0::2], s1[1::2], C[1:])))
  return rows, cols


def _gauss_nodes_by_edges(C, s0, s1):
  """Whether every Gauss node lies by its edge: within one chord length of the chord's middle (a curved edge bends
    away from its chord, by up to 0.88 of it at STEP's X point, 2026-11-16; a misread or unwritten node lies
    anywhere: Sim 1's and Sim 2's files hold zeros)."""
  tol = 1e-9 * np.ptp(C)
  for A, B, G in ((C[:, :-1], C[:, 1:], (s0[:, 0::2], s0[:, 1::2])),
                  (C[:-1], C[1:], (s1[0::2], s1[1::2]))):
    L, M = np.hypot(*(B - A).transpose(2, 0, 1)), 0.5 * (A + B)
    if any((np.hypot(*(g - M).transpose(2, 0, 1)) > L + tol).any() for g in G):
      return False
  return True


def cell_edges(run):
  """Every block's cell edges as curves, for drawing: dict(rows={b: node rows}, cols={b: node columns}, curved, note),
    polylines in the viewer's orientation. A Gkeyll cell edge is a curve through its two corners and the two
    Gauss-Legendre nodes between them (GAUSS_T), drawn as the cubic through the four (_edge_polylines); the nodes'
    positions are the first three components of the block's geo_surf0 (constant-psi edges) and geo_surf1
    (constant-theta edges) mc2p_nodal_fd files (the rest are a finite-difference stencil), physical_rz as the
    corners. Where a block lacks them, or one of its nodes lies away from its edge (_gauss_nodes_by_edges), every
    edge is drawn as its chord instead (curved False) and note says so."""
  if getattr(run, '_edges', None) is not None:
    return run._edges
  rows, cols, note = {}, {}, ''
  for b in range(run.nb):
    g = read_gkyl(run.f(b, 'geo_corn_nodes'))
    gt = int(g['meta'].get('geometry_type', -1))
    C = physical_rz(g['data'], gt)
    try:
      s0, s1 = (physical_rz(
          np.asarray(read_gkyl(run.f(
              b, f'geo_surf{d}_mc2p_nodal_fd'))['data'])[..., :3], gt)
                for d in (0, 1))
    except (OSError, ValueError) as e:
      note = f'no surface nodes ({os.path.basename(str(getattr(e, "filename", None) or e))})'
      break
    if not _gauss_nodes_by_edges(C, s0, s1):
      note = f'block {b}\'s surface nodes lie off their edges'
      break
    r, c = _edge_polylines(C, s0, s1)
    rows[b], cols[b] = (r[::-1], c[:, ::-1]) if run.reversed(b) else (r, c)
  if note:  # chords: the corners alone
    rows = {b: run.nodes[b] for b in range(run.nb)}
    cols = {b: run.nodes[b].transpose(1, 0, 2) for b in range(run.nb)}
  run._edges = dict(rows=rows,
                    cols=cols,
                    curved=not note,
                    note=f'cell edges drawn as chords: {note}' if note else '')
  return run._edges


def cell_polygons(run, b):
  """Block b's cells as the regions their edges bound (cell_edges: curves, or chords): [nr, nt, 4 k, (R, Z)], each
    cell's edges in turn from its first corner (rows i, then theta j, as the viewer's cells), k points to an edge."""
  E = cell_edges(run)
  R, C = E['rows'][b], E['cols'][b]
  nr, nt = R.shape[0] - 1, C.shape[0] - 1
  k = (R.shape[1] - 1) // nt
  a = np.arange(k)
  j, i = np.arange(nt)[:, None] * k, np.arange(nr)[:, None] * k
  bottom, top = R[:-1][:, j + a], R[1:][:, j + k - a]  # [nr, nt, k, 2]
  right, left = C[1:][:, i + a], C[:-1][:, i + k - a]  # [nt, nr, k, 2]
  return np.concatenate(
      [bottom,
       right.transpose(1, 0, 2, 3), top,
       left.transpose(1, 0, 2, 3)],
      axis=2)


# ------------------------------------------------------------------------------------- volumes and region means
# --------------------------------------------------------------------------------- flux coordinates (psi_N, L_theta)
def psi_nodes(run, b):
  """psi at block b's nodes [nr + 1, nt + 1], the viewer's rows: the position map's first coordinate
    (geo_corn_mc2nu_pos, the computational x = psi to its non-uniform value) at the cells' corners (node_values)."""
  mc = _data(run.f(b, 'geo_corn_mc2nu_pos'))
  return run.rows(b, node_values(mc[..., 0:4])[0])


def flux_coords(run):
  """psi_N of every block's node rows and cell rows, and L_theta of every cell of the open chains (flux_lengths),
    kept on the run: dict(node {b: [nr + 1]}, cell {b: [nr]}, sep (psi_N of each SOL block's separatrix row: 1 when the
    run's separatrix is its geqdsk's boundary), why ('' or why there is none), chains (flux_lengths)). psi_N = (psi -
    simag) / (sibry - simag), the geqdsk's (EFIT's) normalization, so measurements mapped by the same geqdsk compare."""
  if getattr(run, '_flux', None) is not None:
    return run._flux
  out = dict(node={}, cell={}, sep={}, why='', chains=[])
  g = geqdsk_path(run) if run.nb else None
  if g is None:
    out['why'] = 'no geqdsk (its axis flux normalizes psi)' if run.nb else 'no run'
  else:
    h = geqdsk_header(g)
    try:
      for b in range(run.nb):
        P = psi_nodes(run, b)
        n = (P.mean(axis=1) - h['simag']) / (h['sibry'] - h['simag'])
        out['node'][b], out['cell'][b] = n, 0.5 * (n[1:] + n[:-1])
    except (OSError, ValueError) as e:
      out['why'] = f'no position map ({e})'
    for st in run.layout.stacks.values():
      if not out['why'] and st.sol is not None:
        out['sep'][st.sol] = float(out['node'][st.sol][st.sep_row])
  if run.nb:
    out['chains'] = flux_lengths(run)
  run._flux = out
  return out


def _row_lengths(lines):
  """Arc length along polylines [n_rows, n_points, 2] from their first point: [n_rows, n_points]."""
  d = np.hypot(*np.diff(lines, axis=1).transpose(2, 0, 1))
  return np.concatenate([np.zeros(
      (len(lines), 1)), np.cumsum(d, axis=1)],
                        axis=1)


def flux_lengths(run):
  """L_theta, the poloidal length along each flux surface (in R, Z) from one end of its chain of blocks (theta_chains,
    the open ones: SOL legs and main SOL, PF), measured along the cell-centre line of each cell row (the mean of its
    two node rows' edges, cell_edges: curves, or chords where the run lacks the surface nodes). From the chain's
    target end, an outer target before an inner one, a lower before an upper (the paper's L_theta runs from the outer
    divertor). Per chain: dict(blocks, start, end (a target's name or 'midplane' for a half domain's mirror), cells
    {b: [nr, nt] L_theta of each cell's centre}, total [nr], marks [(label, [nr] L_theta)] (the X-point: where a
    SOL chain's block over a PF block meets one over the core, or a PF chain's blocks meet; each Z = 0 crossing:
    outboard or inboard midplane, by R against the magnetic axis; a SOL chain's highest point when it lies inside the
    chain: top), kind (its blocks' region), curved). Chains whose blocks' rows do not align are left out."""
  edges = cell_edges(run)
  r_axis = magnetic_axis_R(run)
  if r_axis is None:  # no geqdsk: the core's centre
    core = [
        run.nodes[b][..., 0] for b, r in run.layout.region.items()
        if r == 'core'
    ]
    r_axis = float(np.mean([c.mean() for c in core])) if core else 0.0
  ends = {(t.leg, t.j): t for t in run.layout.targets}
  ends.update({(t.pf, t.j): t for t in run.layout.targets if t.pf is not None})
  out = []
  for ch in theta_chains(run.conn):
    bs = ch['blocks']
    if ch['ring'] or len({run.nodes[b].shape[0] for b in bs}) != 1:
      continue
    a, z = ends.get((bs[0], 0)), ends.get((bs[-1], -1))

    def rank(t):
      return (0 if t.side == 'outer' else 1, 0 if t.updown == 'lower' else 1)

    flip = z is not None and (a is None or rank(z) < rank(a))
    lines, spans = [], []
    for b in (bs[::-1] if flip else bs):
      R = edges['rows'][b]
      R = R[:, ::-1] if flip else R
      K = (R.shape[1] - 1) // (run.nodes[b].shape[1] - 1)
      mid = 0.5 * (R[1:] + R[:-1])  # [nr, K nt + 1, 2]
      lines.append(mid if not lines else mid[:, 1:])
      spans.append((b, K, mid.shape[1]))
    line = np.concatenate(lines, axis=1)
    L = _row_lengths(line)
    i0, out_cells, joins = 0, {}, []
    for b, K, n in spans:
      nt = (n - 1) // K
      idx = i0 + K * (np.arange(nt) + 0.5
                      )  # each cell's centre on the line (fractional index)
      Lc = np.array([np.interp(idx, np.arange(L.shape[1]), row) for row in L])
      out_cells[b] = Lc[:, ::-1] if flip else Lc
      joins.append((b, i0))
      i0 += n - 1
    marks = []
    under, region = run.layout.under, run.layout.region
    kind = region.get(
        bs[0],
        '')  # 'sol', 'pf' or 'core' (a half domain's core: mirror to mirror)
    for (b0, _), (b1, j1) in zip(joins, joins[1:]):
      if kind == 'pf' or {region.get(under.get(b0)),
                          region.get(under.get(b1))} == {'pf', 'core'}:
        marks.append(
            ('X-point', L[:, j1])
        )  # PF legs meet under it; a SOL chain's leg meets its main SOL there
    Z = line[..., 1]
    side = lambda R: 'outboard' if R >= r_axis else 'inboard'  # noqa: E731
    cross = [np.where(np.diff(np.sign(Zr)) != 0)[0] for Zr in Z]
    for c in range(min(len(x) for x in cross)):
      vals, Rs = [], []
      for Zr, Lr, Rr, x in zip(Z, L, line[..., 0], cross):
        k = x[c]
        w = Zr[k] / (Zr[k] - Zr[k + 1])
        vals.append(Lr[k] + w * (Lr[k + 1] - Lr[k]))
        Rs.append(Rr[k])
      marks.append((f'{side(np.mean(Rs))} midplane', np.array(vals)))
    top = np.argmax(Z, axis=1)
    if kind == 'sol' and np.all((top > 0) & (top < Z.shape[1] - 1)):
      marks.append(('top', L[np.arange(len(L)), top]))
    first, last = (z, a) if flip else (a, z)

    def name(
        t, end
    ):  # a target, else the mirror at that end: the half domain's midplane
      return t.name if t is not None else f'{side(line[:, end, 0].mean())} midplane'

    out.append(
        dict(blocks=list(bs[::-1] if flip else bs),
             kind=kind,
             start=name(first, 0),
             end=name(last, -1),
             cells=out_cells,
             total=L[:, -1],
             marks=sorted(marks, key=lambda m: float(np.mean(m[1]))),
             curved=edges['curved']))
  return out


def cell_flux_coords(run, b, i, j):
  """(psi_N, L_theta, the chain's start) of block b's cell (i, j), the viewer's indices; nan / '' where there are none
    (no geqdsk; a core ring)."""
  fc = flux_coords(run)
  psin = float(fc['cell'][b][i]) if b in fc['cell'] else np.nan
  ch = next((c for c in fc['chains'] if b in c['cells']), None)
  return psin, (float(ch['cells'][b][i, j]) if ch else np.nan), (ch['start']
                                                                 if ch else '')


# ------------------------------------------------------------------------------------------------- sheath entrance
def sheath_frames(run):
  """The frames at which the run wrote its sheath boundary condition's files (<species>_bc_z{lower,upper}_vcutsq:
    with its phase-space output), from its first target's leg block; kept while its frame list is unchanged."""
  frames = run.frames()
  memo = getattr(run, '_sheath_frames', None)
  if memo is not None and memo[0] == len(frames):
    return memo[1]
  tg = run.layout.targets[0] if run.layout.targets else None
  e = next((s for s in run.species if run.charge[s] < 0), None)
  out = [] if tg is None or e is None else [
      k for k in frames if os.path.exists(
          run.f(tg.leg, f'{e}_bc_z{"lower" if tg.j == 0 else "upper"}_vcutsq',
                k))
  ]
  run._sheath_frames = (len(frames), out)
  return out


def _face(M, j, ez):
  """The plate-face average of modal p = 1 fields M [nr, nt, 4] at theta cell j, its face ez (-1 lower, +1 upper):
    their x-linear parts average out over the face."""
  return 0.5 * M[:, j, 0] + np.sqrt(3) / 2 * M[:, j, 2] * ez


def sheath_profiles(run, k):
  """The sheath entrance of every target at the latest frame <= k with the sheath boundary condition's files
    (sheath_frames): dict(frame (None: none), why, targets {name: dict(psin, dphi_Te, expected, mach, Te, Ti)}), each
    an array over the target's open SOL rows, the viewer's order. At the plate face (its p = 1 moments averaged over
    the face, _face; T from those): dphi_Te = e (phi_se - phi_wall) / T_e, the drop the boundary condition reflects
    electrons by (its cutoff, m_e v_cut^2 / 2, the electrons' <species>_bc_z*_vcutsq cell average); expected = -1/2
    ln[2 pi (m_e / m_i) (1 + T_i / T_e)] (the textbook floating sheath, Stangeby); mach = u_par,i / c_s toward the plate,
    c_s = sqrt((T_e + T_i) / m_i). psin: flux_coords' cell rows (nan without a geqdsk)."""
  e = next((s for s in run.species if run.charge[s] < 0), None)
  i = next((s for s in run.species if run.charge[s] > 0), None)
  ks = [f for f in sheath_frames(run) if f <= k]
  out = dict(frame=ks[-1] if ks else None, why='', targets={})
  if e is None or i is None:
    out['why'] = 'no electron and ion species'
  elif not ks:
    out['why'] = (
        'the run wrote no sheath boundary-condition files (<species>_bc_z*_vcutsq)'
        if not sheath_frames(run) else
        f'no sheath files at or before frame {k}')
  if out['why']:
    return out
  kf, fc = out['frame'], flux_coords(run)
  for tg in run.layout.targets:
    face = 'lower' if tg.j == 0 else 'upper'
    ez = -1 if tg.j == 0 else 1
    b = tg.leg
    path = run.f(b, f'{e}_bc_z{face}_vcutsq', kf)
    if not os.path.exists(path):
      continue
    g = read_gkyl(path)
    grid = read_gkyl(run.f(b, 'field', kf))
    lo, dx = grid['lower'][0], (grid['upper'][0] -
                                grid['lower'][0]) / grid['cells'][0]
    r0 = int(round(
        (g['lower'][0] - lo) / dx))  # the file rows the sheath covers: r0 on
    vc = np.asarray(g['data'])[..., 0].mean(
        axis=1) / 2  # cell average (it does not vary with mu)
    mom = {
        (s, m): _face(_data(run.f(b, f'{s}_{m}', kf)), tg.j,
                      ez)[r0:r0 + len(vc)]
        for s in (e, i)
        for m in ('M0', 'M1', 'M2')
    }
    T = {
        s:
        run.mass[s] / run.e * (mom[s, 'M2'] - mom[s, 'M1']**2 / mom[s, 'M0']) /
        (3 * mom[s, 'M0'])
        for s in (e, i)
    }
    with np.errstate(divide='ignore', invalid='ignore'):
      dphi = run.mass[e] * vc / 2 / run.e  # V
      ui = mom[i, 'M1'] / mom[i, 'M0']
      cs = np.sqrt((T[e] + T[i]) * run.e / run.mass[i])
      rows = dict(dphi_Te=dphi / T[e],
                  expected=-0.5 * np.log(2 * np.pi * run.mass[e] / run.mass[i] *
                                         (1 + T[i] / T[e])),
                  mach=ez * ui / cs,
                  Te=T[e],
                  Ti=T[i])
    nr = run.nodes[b].shape[0] - 1
    full = {key: np.full(nr, np.nan) for key in rows}
    for key, v in rows.items():
      full[key][r0:r0 + len(v)] = v
    view = {key: run.rows(b, v)[tg.open_row:] for key, v in full.items()}
    psin = fc['cell'][b][tg.open_row:] if b in fc['cell'] else np.full(
        nr - tg.open_row, np.nan)
    ok = np.isfinite(view['dphi_Te'])
    out['targets'][tg.name] = dict(psin=psin[ok],
                                   **{
                                       key: v[ok]
                                       for key, v in view.items()
                                   })
  return out


def cell_volumes(run):
  """Volume of every cell (cell_offsets order): the Jacobian's cell average times the logical cell size, per radian
    of toroidal angle like every integrated diagnostic (sum(n V) matches the integrated M0 to ~0.2 %)."""
  if getattr(run, '_volumes', None) is None:
    out = []
    for b in range(run.nb):
      g = read_gkyl(run.f(b, 'geo_int_jacobgeo'))
      cell = np.prod((np.asarray(g['upper']) - np.asarray(g['lower'])) /
                     np.asarray(g['cells']))
      out.append(
          run.rows(b,
                   np.asarray(g['data'])[..., 0] / 2.0 * cell).ravel())
    run._volumes = np.concatenate(out)
  return run._volumes


def region_rows(run, b, region):
  """The rows of block b in a region ('core', 'sol', 'pf'), a slice: all or none, but in a block holding closed and
    open field lines (Layout.lcfs) its rows under the LCFS are core and the others SOL."""
  nr = run.nodes[b].shape[0] - 1
  m = run.layout.lcfs.get(b)
  if m is not None:
    return {'core': slice(0, m), 'sol': slice(m, nr)}.get(region, slice(0, 0))
  return slice(0, nr) if run.layout.region[b] == region else slice(0, 0)


def cell_region(run, b, i):
  """The region of the viewer's row i of block b."""
  return next(
      (reg for reg in ('core', 'sol', 'pf')
       if region_rows(run, b, reg).start <= i < region_rows(run, b, reg).stop),
      run.layout.region[b])


def region_cells(run, region):
  """Cell indices (cell_offsets order) of a region ('core', 'sol', 'pf': region_rows), of every cell ('all'), or of
    a list of blocks."""
  off = cell_offsets(run)
  idx = []
  for b in range(run.nb):
    nr, nt = run.nodes[b].shape[0] - 1, run.nodes[b].shape[1] - 1
    if isinstance(region, (list, tuple, set, frozenset)) or region == 'all':
      i = slice(0, nr if region == 'all' or b in region else 0)
    else:
      i = region_rows(run, b, region)
    idx.append(np.arange(off[b] + i.start * nt, off[b] + i.stop * nt))
  return np.concatenate(idx) if idx else np.zeros(0, int)


def region_mean(run, summ, qty, region):
  """Volume-weighted mean of a quantity over a region at every frame (cells without a value left out)."""
  c = region_cells(run, region)
  F = summ[f'fld|{qty}'][:, c].astype(np.float64)
  W = np.broadcast_to(cell_volumes(run)[c], F.shape)
  ok = np.isfinite(F)
  with np.errstate(invalid='ignore'):
    return np.where(ok, F * W, 0).sum(1) / np.where(ok, W, 0).sum(1)


def full_device(run):
  """Factor from the integrated diagnostics (per radian of toroidal angle) to the whole device: 2 pi, times 2 for
    a half domain (its mirror image). Checked: Sim 2's source rate x 4 pi == its configured particle rate."""
  return 2 * np.pi * (2 if run.layout.half else 1)


# -------------------------------------------------------------------------------------------- particle / energy
def _dyn_sum(run, what, comp, blocks=None):
  """(t, sum over blocks of one component) of an integrated time trace, the blocks that wrote it; (None, None)
    when none did; empty when there is no block to sum over (the empty run)."""
  bs = range(run.nb) if blocks is None else blocks
  found = [
      read_dynvec(run.f(b, what)) for b in bs if os.path.exists(run.f(b, what))
  ]
  if not found:
    return (np.zeros(0), np.zeros(0)) if not len(bs) else (None, None)
  n = min(len(t) for t, _ in found)
  return found[0][0][:n], sum(d[:n, comp] for _, d in found)


def plate_face(tgt):
  """The boundary-flux file suffix of a target's plate: the theta face of its leg and PF blocks."""
  return 'ylower' if tgt.j == 0 else 'yupper'


def balance(run):
  """Particle and energy budgets from the integrated diagnostics, per radian (multiply by full_device(run)):
    t; per species N_src, N_out (every boundary-flux face, interior faces cancelling), dN/dt (fdot); E_src =
    sum_s m_s M2/2 of the source (this driver integrates the source's M0M1M2 moments, not its Hamiltonian; its
    q phi term cancels for a quasineutral source), E_out (the Hamiltonian moment of every boundary flux),
    dE/dt = fdot of the Hamiltonian minus the field-energy rate (as postgkyl.diagnostics.gk.energy_balance);
    per target: power and particle flux through the plate (leg + PF block, both species)."""
  out = {'species': {}, 'targets': {}}
  E_src = E_out = E_dot = None
  faces = ('xlower', 'xupper', 'ylower', 'yupper')

  def add(a, b):
    return b if a is None else a[:min(len(a), len(b))] + b[:min(len(a), len(b))]

  for s in run.species:
    t, fN = _dyn_sum(run, f'{s}_fdot_integrated_moms', 0)
    if t is None:
      return None
    _, fE = _dyn_sum(run, f'{s}_fdot_integrated_moms', 2)
    _, sN = _dyn_sum(run, f'{s}_source_integrated_moms', 0)
    _, sM2 = _dyn_sum(run, f'{s}_source_integrated_moms', 2)
    oN = oE = None
    for face in faces:
      _, xN = _dyn_sum(run, f'{s}_bflux_{face}_integrated_HamiltonianMoments',
                       0)
      _, xE = _dyn_sum(run, f'{s}_bflux_{face}_integrated_HamiltonianMoments',
                       2)
      if xN is not None:
        oN, oE = add(oN, xN), add(oE, xE)
    zero = np.zeros_like(fN)
    sN = zero if sN is None else sN
    sE = zero if sM2 is None else 0.5 * run.mass[s] * sM2
    n = min(len(fN), len(sN), len(oN))
    out['species'][s] = dict(t=t[:n], src=sN[:n], out=oN[:n], dot=fN[:n])
    E_src, E_out, E_dot = add(E_src, sE), add(E_out, oE), add(E_dot, fE)
    out['t'] = t
  _, fd = _dyn_sum(run, 'field_energy_dot', 0)
  n = min(len(E_src), len(E_out), len(E_dot),
          len(fd) if fd is not None else len(E_dot))
  out['t'] = out['t'][:n]
  out['E'] = dict(src=E_src[:n],
                  out=E_out[:n],
                  dot=E_dot[:n] - (fd[:n] if fd is not None else 0))
  for tgt in run.layout.targets:
    P = G = None
    for s in run.species:
      what = f'{s}_bflux_{plate_face(tgt)}_integrated_HamiltonianMoments'
      plate = [bb for bb in (tgt.leg, tgt.pf) if bb is not None]
      _, e = _dyn_sum(run, what, 2, blocks=plate)
      _, g = _dyn_sum(run, what, 0, blocks=plate)
      if e is not None:
        P, G = add(P, e), add(G, g)
    if P is not None:
      out['targets'][tgt.name] = dict(power=P[:n], particles=G[:n])
  return out


def imbalance(src, out, dot):
  """How far a budget is from closing at each time: (src - out - dot) over the largest of |src|, |out|, |dot|. 0
    where what the sources put in, less what the boundaries take out, is what the content gains; +-1 where a whole
    term is unaccounted for. nan where the diagnostics hold nothing yet (out and dot both exactly 0: the first
    sample, before a step) or every term is 0."""
  big = np.maximum(np.maximum(np.abs(src), np.abs(out)), np.abs(dot))
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(((out == 0) & (dot == 0)) | (big == 0), np.nan,
                    (src - out - dot) / big)


def conservation(B, t_frames):
  """Each budget of balance(run) over each interval between the run's output frames (t_frames): what the sources put
    in, what the boundaries took out and what the content gained over it, and their imbalance(). Over an interval,
    not at each sample: in some runs the rate of change swings sample to sample (the 2025 re-run at 5.68 ms: +3e5,
    -8e5, +3e5 W/rad between neighbours) while what it integrates to does not. {'species': {s: (t_mid, r)},
    'energy': (t_mid, r)}; (empty, empty) where fewer than two frames fall inside the diagnostics' span."""

  def per_interval(t, d):
    if not len(t):  # no sample (the empty run)
      return np.zeros(0), np.zeros(0)
    first = np.argmax(
        (d['out'] != 0) |
        (d['dot'] != 0))  # the samples before a step hold nothing yet
    t = t[first:]
    edges = np.unique(np.clip(np.asarray(t_frames, float), t[0], t[-1]))
    if len(edges) < 2:
      return np.zeros(0), np.zeros(0)
    cum = {
        k:
        np.concatenate([[0.0],
                        np.cumsum(0.5 * (d[k][first + 1:] + d[k][first:-1]) *
                                  np.diff(t))])
        for k in ('src', 'out', 'dot')
    }
    over = {k: np.diff(np.interp(edges, t, c)) for k, c in cum.items()}
    return 0.5 * (edges[1:] + edges[:-1]), imbalance(over['src'], over['out'],
                                                     over['dot'])

  return dict(species={
      s: per_interval(d['t'], d)
      for s, d in B['species'].items()
  },
              energy=per_interval(B['t'], B['E']))


# ------------------------------------------------------------------------------------------------------ diffusivity
def ser_phi(x, y):
  """The 2D p = 1 serendipity basis on [-1, 1]^2 at (x, y) (orthonormal: 1/2, sqrt(3) x/2, sqrt(3) y/2, 3 x y/2), x
    along psi, y along theta."""
  return np.array([0.5, np.sqrt(3) / 2 * x, np.sqrt(3) / 2 * y, 1.5 * x * y])


def _ser_p1_2d():
  """The triple-product tensor T[k, i, j] = int phi_k phi_i phi_j of ser_phi, and the basis at the four corners (-,-)
    (+,-) (-,+) (+,+)."""
  g, w = np.polynomial.legendre.leggauss(
      2)  # exact for the product of three p = 1 functions
  T = sum(wx * wy *
          np.einsum('k,i,j->kij', ser_phi(x, y), ser_phi(x, y), ser_phi(x, y))
          for x, wx in zip(g, w) for y, wy in zip(g, w))
  return T, np.array(
      [ser_phi(x, y) for x, y in ((-1, -1), (1, -1), (-1, 1), (1, 1))])


SER_T, SER_CORNERS = _ser_p1_2d()
CORNER_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1)
                  )  # node (i + di, j + dj) of cell (i, j) for each corner


def weak_divide(A, G):
  """Per cell, the p = 1 expansion D whose weak product with G is A (the inverse of gkyl_dg_mul_op): modal (..., 4)."""
  M = np.einsum('kij,...j->...ki', SER_T, G)
  return np.linalg.solve(M, A[..., None])[..., 0]


def node_values(D):
  """The values of per-cell p = 1 expansions D (nr, nt, 4) at the cells' corners, as (nr + 1, nt + 1) node values;
    and the largest disagreement between the cells sharing a node (0 for a continuous field)."""
  nr, nt = D.shape[:2]
  Dc = np.einsum('ci,...i->...c', SER_CORNERS, D)
  out, gap = np.full((nr + 1, nt + 1), np.nan), 0.0
  for c, (di, dj) in enumerate(CORNER_OFFSETS):
    cur = out[di:di + nr, dj:dj + nt]
    gap = max(
        gap,
        float(
            np.nanmax(
                np.abs(np.where(np.isnan(cur), Dc[..., c], cur) - Dc[..., c]))))
    out[di:di + nr, dj:dj + nt] = Dc[..., c]
  return out, gap


def launch_options(run):
  """key -> value the run was started with: options_used.txt, else the -o key=value of its command file."""
  if run.options:
    return dict(run.options)
  out = {}
  for name in ('command.txt', 'cmd.txt'):
    p = os.path.join(run.path, name)
    if os.path.exists(p):
      for k, v in re.findall(r'-o\s+(\w+)=(\S+)', open(p).read()):
        out.setdefault(k, v)
      break
  return out


def run_logs(run):
  """The run's job logs (slurm-*.out, *.log, stdout), the newest first."""
  found = {
      p
      for p in (os.path.join(run.path, n) for n in dir_index(run.path)['logs'])
      if os.path.isfile(p) and os.path.basename(p) != 'restarts.log'
  }  # restarts.log: the job list
  return sorted(found, key=os.path.getmtime, reverse=True)


def _log_head(path, nlines=600):
  with open(path, errors='replace') as f:
    return [line for _, line in zip(range(nlines), f)]


# a uniform D in the run's own printout (each driver words it differently), the most specific first
D_LOG_LINES = [
    (r'Anomalous diffusion: D_perp \S+ m\^2/s x D_perp_mult \S+ = (\S+) m\^2/s',
     False),  # nstxu_multib, mult != 1
    (r'Physics preset/overrides:.*\bD=(\S+) m\^2/s', False),  # the 2025 driver
    (r'Applied overrides:.*D_const=(\S+) \(m\^2/s\)',
     False),  # the 2025 driver, csv
    (r'\bn0 \S+ m\^-3, Te = Ti = Tsource \S+ eV, D_perp (\S+) m\^2/s',
     True),  # nstxu_multib csv_init
]


def recorded_uniform_D(run):
  """(D in m^2/s, where it was found) for a run that wrote no anom_diff file (a uniform D), else (None, why).
    Its job log first (what the driver printed it used), then the options it was launched with."""
  opts = launch_options(run)
  mult = float(opts.get('D_perp_mult', 1.0))
  for log in run_logs(run):
    head = _log_head(log)
    for pat, times_mult in D_LOG_LINES:
      for line in head:
        m = re.search(pat, line)
        if m:
          D = float(m.group(1)) * (mult if times_mult else 1.0)
          tail = f' × D_perp_mult {mult:g}' if times_mult and mult != 1 else ''
          return D, f'{os.path.basename(log)}: "{line.strip()[:160]}"{tail}'
  if 'D_perp' in opts and not opts.get('csv_init',
                                       'false').lower() in ('1', 'true', 'yes'):
    return float(
        opts['D_perp']) * mult, f"its options: D_perp={opts['D_perp']}" + (
            f', D_perp_mult={mult:g}' if mult != 1 else '')
  if 'D_const' in opts:
    return float(opts['D_const']), f"its options: D_const={opts['D_const']}"
  return None, (
      'not recorded: no anom_diff file, no D option and no D line in its logs, so the driver\'s default '
      'applied (which value that was is not in the run\'s files)')


_DIFF = {}


def diffusivity(run):
  """The anomalous diffusivity the run used, per species and block (m^2/s): dict(kind 'file' | 'uniform' | None,
    source, species {s: dict(nodes {b: (nr+1, nt+1)}, cells {b: (nr, nt)})}, gap). 'file': the anom_diff_0 files,
    D times g^xx J as the solver uses it (the library multiplies the D_profile values, taken on the cells' corner
    nodes, by geo_int_gxxj with gkyl_dg_mul_op), divided back here per cell (weak_divide): the corner values are the
    driver's D at those nodes again (gap: the largest disagreement between cells sharing a node). 'uniform': the D its
    log or options record, everywhere. None: not recorded."""
  if run.path in _DIFF:
    return _DIFF[run.path]
  out = dict(kind=None, source='', species={}, gap=0.0)
  if not run.nb:  # the empty run: nothing to read
    return dict(out, source='no run', same=True)
  if all(
      os.path.exists(run.f(b, f'{s}_anom_diff', 0)) for b in range(run.nb)
      for s in run.species):
    for s in run.species:
      nodes, cells = {}, {}
      for b in range(run.nb):
        A = np.asarray(read_gkyl(run.f(b, f'{s}_anom_diff', 0))['data'])
        G = np.asarray(read_gkyl(run.f(b, 'geo_int_gxxj'))['data'])
        D = weak_divide(A, G)
        nv, gap = node_values(D)
        nodes[b], cells[b] = run.rows(b, nv), run.rows(b, D[..., 0] / 2.0)
        out['gap'] = max(out['gap'], gap / max(np.nanmax(np.abs(nv)), 1e-300))
      out['species'][s] = dict(nodes=nodes, cells=cells)
    out['kind'], out['source'] = 'file', (
        f'the run\'s {run.stem}{"" if run.single else "_b<k>"}-<species>_anom_diff_0 '
        'files (D × gˣˣJ as the solver uses it), divided by geo_int_gxxj per cell'
    )
  else:
    D, why = recorded_uniform_D(run)
    out['source'] = why
    if D is not None:
      out['kind'] = 'uniform'
      for s in run.species:
        out['species'][s] = dict(
            nodes={
                b: np.full(run.nodes[b].shape[:2], D)
                for b in range(run.nb)
            },
            cells={
                b:
                np.full((run.nodes[b].shape[0] - 1, run.nodes[b].shape[1] - 1),
                        D)
                for b in range(run.nb)
            })
  sp = list(out['species'])
  out['same'] = all(
      all(
          np.array_equal(out['species'][sp[0]]['nodes'][b], out['species'][s]
                         ['nodes'][b]) for b in range(run.nb)) for s in sp[1:])
  _DIFF[run.path] = out
  return out


def midplane_D(run, Dsp, side):
  """D at Z = 0 along one side's midplane stack, from the node values: (R per node row, D, the block of each row),
    inside out (rows in a block linear in theta between the nodes bracketing Z = 0, as mid_edges; a node row shared by
    two blocks once, with the inner block)."""
  st = run.layout.stacks[side]
  Rs, Ds, Bs = [], [], []
  for b in stack_blocks(st):
    nd, v = run.nodes[b], Dsp['nodes'][b]
    R = mid_edges(nd, st.j)
    Dm = []
    for i, (Rrow, Z) in enumerate(zip(nd[..., 0], nd[..., 1])):
      k = np.where(np.diff(np.sign(Z)) != 0)[0]
      if len(k) == 0:
        Dm.append(v[i, int(np.argmin(np.abs(Z)))])
        continue
      k = _pick(k, st.j, 0.5)
      w = Z[k] / (Z[k] - Z[k + 1])
      Dm.append(v[i, k] + w * (v[i, k + 1] - v[i, k]))
    if Rs and abs(R[0] - Rs[-1][-1]) <= 1e-12 * max(abs(
        R[0]), 1.0):  # the shared separatrix row
      R, Dm = R[1:], Dm[1:]
    Rs.append(np.asarray(R))
    Ds.append(np.asarray(Dm))
    Bs.append(np.full(len(R), b))
  return np.concatenate(Rs), np.concatenate(Ds), np.concatenate(Bs)


# --------------------------------------------------------------------------------------------------- comparison
def match_frames(t_a, t_b):
  """For each time of run a, the index of run b's frame nearest in time, or -1 where b has none within half of the
    coarser of the two frame spacings (before b starts, after it ends, or in a gap)."""
  t_a, t_b = np.asarray(t_a, float), np.asarray(t_b, float)
  if not len(t_b) or not len(t_a):
    return np.full(len(t_a), -1)
  o = np.argsort(t_b, kind='stable')
  tb = t_b[o]
  i = np.clip(np.searchsorted(tb, t_a), 1, max(len(tb) - 1, 1))
  lo = np.clip(i - 1, 0, len(tb) - 1)
  hi = np.clip(i, 0, len(tb) - 1)
  near = np.where(np.abs(t_a - tb[lo]) <= np.abs(t_a - tb[hi]), lo, hi)
  sp = [float(np.median(np.diff(t))) for t in (np.sort(t_a), tb) if len(t) > 1]
  tol = 0.5 * max(sp) * (1 + 1e-9) if sp else np.inf
  return np.where(np.abs(tb[near] - t_a) <= tol, o[near], -1)


class Compared:
  """The summary of run a with run b's per-frame entries next to it, as 'cmp|<key>': b's row at the frame matched to
    each of a's frames (match_frames), NaN where none matches. Everything else is a's. For the LivePlots: a trace of
    b tagged 'cmp|mid|outboard|n_elc' follows a's movie."""

  def __init__(self, summ_a, summ_b, idx):
    self.a, self.b, self.idx, self.memo = summ_a, summ_b, np.asarray(idx), {}

  def __contains__(self, key):
    return key in self.a or (key.startswith('cmp|') and key[4:] in self.b)

  def __getitem__(self, key):
    if not key.startswith('cmp|'):
      return self.a[key]
    if key not in self.memo:
      B = np.asarray(self.b[key[4:]], dtype=float)
      out = np.full((len(self.idx), ) + B.shape[1:], np.nan)
      ok = self.idx >= 0
      out[ok] = B[self.idx[ok]]
      self.memo[key] = out
    return self.memo[key]

  def get(self, key, default=None):
    return self[key] if key in self else default


def nearest_cell(run, R, Z, region=None):
  """The cell (cell_offsets order) of run whose centre is nearest (R, Z), preferring the given region; and the
    distance (m)."""
  off = cell_offsets(run)
  best = (np.inf, -1)
  for pass_region in ([region, None] if region else [None]):
    for b in range(run.nb):
      c = cell_centres(run.nodes[b])
      d = np.hypot(c[..., 0] - R, c[..., 1] - Z)
      if pass_region:  # the other rows out of the running
        keep = np.zeros(len(d), bool)
        keep[region_rows(run, b, pass_region)] = True
        d = np.where(keep[:, None], d, np.inf)
        if not keep.any():
          continue
      i, j = np.unravel_index(int(np.argmin(d)), d.shape)
      if d[i, j] < best[0]:
        best = (float(d[i, j]), off[b] + i * d.shape[1] + j)
    if best[1] >= 0:
      break
  return best[1], best[0]
