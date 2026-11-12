"""Data layer of the NSTX-U run viewer (nstxu_viewer.py): finds the runs of each simulation, reads their frames, works out
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
import glob
import os
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gkyl_read import read_gkyl, unpack_meta  # noqa: E402

# The data directory: viewer.json (what to offer, below) and runs/ (everything the viewer writes: cache, saved views,
# exports, Flux copies, logs). $NSTXU_VIEWER_DATA (view.sh exports it), else this directory.
DATA = os.path.abspath(os.environ.get('NSTXU_VIEWER_DATA') or os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(DATA, 'runs')
CONFIG_FILE = os.path.join(DATA, 'viewer.json')
CONFIG = json.load(open(CONFIG_FILE)) if os.path.exists(CONFIG_FILE) else {}
# Simulation directories and cache names are relative to 'base' (relative to the data directory; absolute paths too).
NSTXU = os.path.normpath(os.path.join(DATA, CONFIG.get('base', '.')))


def data_path(key):
    """A file or directory viewer.json names under `key` (relative to base), or None when it does not name one."""
    return os.path.normpath(os.path.join(NSTXU, CONFIG[key])) if CONFIG.get(key) else None


CACHE_DIR = os.path.join(RUNS, 'cache')
CACHE_VERSION = 3   # bump when a summary definition changes: old cache files are then ignored (2: + fields, 3: + q_perp)

# The simulations the viewer offers: (label, directory under base, default run, one-line description), from
# viewer.json's "sims". The shot and equilibrium time come from each run's own files.
SIMS = [tuple(s) for s in CONFIG.get('sims', [])]

PHYSICAL = 5                                   # gkyl_oriented_edge (core/zero/gkyl_block_topo.h)
LOWER_POS, LOWER_NEG, UPPER_POS, UPPER_NEG = 1, 2, 3, 4
MOMS = ('M0', 'M1', 'M2', 'M2par', 'M2perp', 'M3par', 'M3perp')
SPECIES_TAG = {'elc': ('electron', 'e'), 'ion': ('ion', 'i')}
TARGET_ORDER = ['lower outer target', 'upper outer target', 'lower inner target', 'upper inner target']
LAMBDA_FRACTION = 0.63


# ----------------------------------------------------------------------------------------------------------- reading
def cav(path):
    """Cell averages of a 2D p = 1 modal field (orthonormal basis: c0 / 2)."""
    return np.asarray(read_gkyl(path)['data'])[..., 0] / 2.0


def meta(path):
    return read_gkyl(path)['meta']


def read_dynvec(path):
    """(time, data[n, ncomp]) of a gkyl dynvector file (file_type 2): every appended chunk repeats the full header."""
    buf = open(path, 'rb').read()
    off, ts, ds = 0, [], []
    while off < len(buf):
        if buf[off:off + 5] != b'gkyl0':
            raise ValueError(f'{path}: bad chunk header at byte {off}')
        version, ftype, msize = struct.unpack_from('<QQQ', buf, off + 5)
        if version != 1 or ftype != 2:
            raise ValueError(f'{path}: not a v1 dynvector (version {version}, type {ftype})')
        off += 29 + msize
        real_type, esznc, size = struct.unpack_from('<QQQ', buf, off)
        if real_type != 2:
            raise ValueError(f'{path}: only float64 dynvectors are supported')
        off += 24
        ts.append(np.frombuffer(buf, '<f8', size, off)); off += 8 * size
        ds.append(np.frombuffer(buf, '<f8', size * esznc // 8, off).reshape(size, esznc // 8)); off += size * esznc
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
    return int(k[0]) if j_near is None else int(k[np.argmin(np.abs(k + offset - j_near))])


def mid_edges(nd, j_near=None):
    """Radial node positions where each psi row of nodes crosses Z = 0 (linear in theta between the bracketing nodes);
    j_near: which crossing, for a block crossing Z = 0 more than once (default the first)."""
    out = []
    for R, Z in zip(nd[..., 0], nd[..., 1]):
        k = np.where(np.diff(np.sign(Z)) != 0)[0]
        if len(k) == 0:
            out.append(R[np.argmin(np.abs(Z))]); continue
        k = _pick(k, j_near, 0.5); w = Z[k] / (Z[k] - Z[k + 1])
        out.append(R[k] + w * (R[k + 1] - R[k]))
    return np.array(out)


def cell_centres(nd):
    return 0.25 * (nd[:-1, :-1] + nd[1:, :-1] + nd[:-1, 1:] + nd[1:, 1:])


def midplane_weights(nd, j_near=None):
    """Per cell row (j0, j1, w): the value at Z = 0 is (1 - w) v[i, j0] + w v[i, j1] (j_near: as mid_edges)."""
    Zc = cell_centres(nd)[..., 1]
    out = []
    for z in Zc:
        k = np.where(np.diff(np.sign(z)) != 0)[0]
        if len(k) == 0:   # Z = 0 is the block's edge (reflecting midplane): the cell next to it
            j = int(np.argmin(np.abs(z))); out.append((j, j, 0.0)); continue
        k = _pick(k, j_near, 1.0); out.append((k, k + 1, float(z[k] / (z[k] - z[k + 1]))))
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
    name: str          # 'lower outer target', ...
    leg: int           # SOL block holding the plate
    pf: object         # the block under it (private flux), or None
    j: int             # theta index of the plate cell (0 or -1)
    mid: object        # SOL block of this chain at the midplane (the surfaces' midplane radii), or None: no block of
                       # the chain reaches Z = 0, the surfaces are placed along the plate instead
    updown: str = ''   # 'lower' or 'upper'
    side: str = ''     # 'outer' or 'inner'
    mid_j: object = None   # theta node position of that block's midplane crossing, where it crosses Z = 0 twice


@dataclass
class Stack:
    side: str          # 'outboard' or 'inboard'
    core: object       # the core block under the SOL block at the midplane, or None
    sol: int
    mirror: bool       # the midplane is the reflecting boundary of a half domain
    j: object = None   # theta node position of the crossing, where the block crosses Z = 0 more than once


# strip bands, y in 0..8: (y0, height). A chain's SOL blocks over the blocks under them, or one band for a chain
# with nothing under it (a single-block run, a core ring)
BANDS = {'sol': (4.0, 4.0), 'in': (0.0, 4.0), 'all': (0.0, 8.0)}
TWO_BAND_LABELS = (('core edge', 0.4), ('core / PF', 2.2), ('separatrix', 4.0), ('SOL', 6.0))


@dataclass
class Strip:
    title: str
    items: list        # (block, x0, band: 'sol' | 'in' | 'all')
    segments: list     # (x0, x1, label)
    mirror_x: object   # x of the reflecting midplane, or None
    width: int
    ylabels: tuple = TWO_BAND_LABELS   # (label, y) down the strip's left side
    sep: object = 4.0                  # y of the separatrix line, or None (one band)


@dataclass
class Layout:
    region: dict       # block -> 'sol' | 'core' | 'pf'
    strips: list
    targets: list
    stacks: dict       # side -> Stack
    half: bool
    flip: bool = False   # the files' psi index grows inward: rows were reversed on reading (Run.rows)
    under: dict = field(default_factory=dict)   # block -> the aligned block under it (psi-lower partner), or None


def theta_chains(conn):
    """Blocks joined along theta, each in theta order: dict(blocks, ring) (ring: it closes on itself, a core)."""
    nb = len(conn)

    def link(b, e):   # the block across theta face e (0 lower, 1 upper) continuing the chain, or None
        if conn[b, 1, e, 2] == PHYSICAL:
            return None
        p, d, edge = (int(v) for v in conn[b, 1, e])
        return p if d == 1 and edge == (LOWER_POS if e == 1 else UPPER_POS) else None

    seen, out = set(), []
    for b0 in range(nb):
        if b0 in seen:
            continue
        b, path = b0, [b0]
        while (p := link(b, 0)) is not None and p not in path:
            b = p; path.append(b)
        ring = link(b, 0) is not None and link(b, 0) in path
        start = b0 if ring else b
        chain, b = [start], start
        while (p := link(b, 1)) is not None and p != start and p not in chain:
            b = p; chain.append(b)
        seen.update(chain)
        out.append(dict(blocks=chain, ring=ring))
    return out


def flip_conn(conn):
    """The topology with the psi direction reversed: each block's psi faces swapped, and every connection into a psi
    face pointed at the other one (lower <-> upper, orientation kept: the theta direction does not change)."""
    c = conn.copy()
    c[:, 0, 0], c[:, 0, 1] = conn[:, 0, 1], conn[:, 0, 0]
    swap = {LOWER_POS: UPPER_POS, UPPER_POS: LOWER_POS, LOWER_NEG: UPPER_NEG, UPPER_NEG: LOWER_NEG}
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
        closed = c['ring'] or all(conn[b, 1, e, 2] == PHYSICAL and np.abs(nodes[b][:, 0 if e == 0 else -1, 1]).max()
                                  <= 1e-9 * np.ptp(nodes[b][..., 1]) for b, e in ((c['blocks'][0], 0), (c['blocks'][-1], 1)))
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


def derive_layout(conn, nodes, r_axis=None):
    """The block layout from the topology (psi growing outward: flip_conn first where it does not) and the nodes.
    r_axis: R of the magnetic axis, for the inner/outer side where the domain has no core block."""
    nb = len(conn)
    allZ = np.concatenate([nodes[b][..., 1].ravel() for b in range(nb)])
    ztol = 1e-9 * np.ptp(allZ)

    def phys(b, d, e):
        return conn[b, d, e, 2] == PHYSICAL

    def face_kind(b, e):   # a physical theta face: 'mirror' on Z = 0, else 'target'
        col = nodes[b][:, 0 if e == 0 else -1, 1]
        return 'mirror' if np.abs(col).max() <= ztol else 'target'

    chains = theta_chains(conn)
    for c in chains:
        for which, b, e in (('start', c['blocks'][0], 0), ('end', c['blocks'][-1], 1)):
            c[which] = 'ring' if c['ring'] else face_kind(b, e) if phys(b, 1, e) else 'open'
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
        else:   # both psi faces physical (a single-block run) or neither: closed field lines are core
            region[b] = 'core' if chain_of[b]['closed'] else 'sol'

    def inner_of(b):   # the aligned block under b (its psi-lower partner), or None
        if phys(b, 0, 0):
            return None
        p, d, e = (int(v) for v in conn[b, 0, 0])
        return p if d == 0 and e == UPPER_POS and nodes[p].shape[1] == nodes[b].shape[1] else None

    cores = [b for b in range(nb) if region[b] == 'core']
    if cores:
        core_nodes = np.concatenate([nodes[b][..., 0].ravel() for b in cores])
        r_ref = 0.5 * (core_nodes.min() + core_nodes.max())
    else:
        r_ref = float(r_axis) if r_axis is not None else float(np.mean(np.concatenate(
            [nodes[b][..., 0].ravel() for b in range(nb)])))
    side_of = lambda R: 'outboard' if R > r_ref else 'inboard'  # noqa: E731

    # SOL chains (open, every block SOL), in the order of their first block; their midplane crossings (row 0)
    sol_chains = sorted([c for c in chains if not c['closed'] and all(region[b] == 'sol' for b in c['blocks'])],
                        key=lambda c: c['blocks'][0])
    span = np.ptp(np.concatenate([nodes[b][..., 0].ravel() for b in range(nb)]))
    stacks, targets = {}, []
    for c in sol_chains:
        cross = []   # (block, theta node position, R), in chain order; a crossing on a shared node once
        for b in c['blocks']:
            R0, Z0 = nodes[b][0, :, 0], nodes[b][0, :, 1]
            for p in row_crossings(Z0, ztol):
                k = min(int(p), len(R0) - 2)
                R = R0[k] + (p - k) * (R0[k + 1] - R0[k])
                if not cross or abs(R - cross[-1][2]) > 1e-9 * span:
                    cross.append((b, p, R))
        nby = {b: sum(1 for x in cross if x[0] == b) for b in c['blocks']}
        c['cross'] = [(b, p if nby[b] > 1 else None, side_of(R)) for b, p, R in cross]
        sides = list(dict.fromkeys(s for _, _, s in c['cross']))
        c['side'] = sides[0] if len(sides) == 1 else 'SOL'
        for b, j, side in c['cross']:
            if side in stacks:
                continue
            core = inner_of(b)
            core = core if core is not None and region[core] == 'core' else None
            at_mirror = 'mirror' in (c['start'], c['end'])
            if len(c['cross']) > 1:   # at the mirror only if this crossing is the chain's end on it
                at_mirror = ((c['start'] == 'mirror' and b == c['blocks'][0]) or
                             (c['end'] == 'mirror' and b == c['blocks'][-1]))
            stacks[side] = Stack(side, core, b, at_mirror, j)
        lab, last = [], len(c['blocks']) - 1
        for k, b in enumerate(c['blocks']):
            ends = [e for e, first in ((0, k == 0), (1, k == last)) if first and c['start' if e == 0 else 'end'] == 'target']
            for e in ends:
                zt = nodes[b][:, 0 if e == 0 else -1, 1].mean()
                ud = 'upper' if zt > 0 else 'lower'
                near = (c['cross'][0] if e == 0 else c['cross'][-1]) if c['cross'] else None
                if near is not None:
                    io = 'outer' if near[2] == 'outboard' else 'inner'
                else:
                    io = 'outer' if nodes[b][:, 0 if e == 0 else -1, 0].mean() > r_ref else 'inner'
                name = f'{ud} {io} target'
                if any(t.name == name for t in targets):
                    name += f' (block {b})'
                targets.append(Target(name, b, inner_of(b), 0 if e == 0 else -1, near[0] if near else None, ud, io,
                                      near[1] if near else None))
            own = list(dict.fromkeys(s for bb, _, s in c['cross'] if bb == b))
            if len(ends) == 1:
                lab.append(f"{targets[-1].updown} {targets[-1].side} leg")
            else:
                lab.append(own[0] if len(own) == 1 else 'main SOL' if len(own) > 1 else
                           c['side'] if c['side'] != 'SOL' else 'SOL')
        c['labels'] = lab

    # strips: a target-to-target chain alone; chains meeting at the mirror joined there (target -> mirror first)
    strips = []
    into = [c for c in sol_chains if c['end'] == 'mirror']
    outof = [c for c in sol_chains if c['start'] == 'mirror']
    pairs = list(zip(into, outof))
    paired = {id(c) for p in pairs for c in p}
    groups = ([[c] for c in sol_chains if 'mirror' not in (c['start'], c['end'])] + [list(p) for p in pairs] +
              [[c] for c in sol_chains if 'mirror' in (c['start'], c['end']) and id(c) not in paired])
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
        title = ('every cell, unrolled (midplane: reflecting boundary)' if mx is not None and len(g) == 2
                 else f'every cell, unrolled: {side} strip')
        if any(band == 'in' for _, _, band in items):
            strips.append(Strip(title, items, segs, mx, x))
        else:
            strips.append(Strip(title, [(b, x0, 'all') for b, x0, _ in items], segs, mx, x,
                                ylabels=(('inner edge', 0.4), ('SOL', 4.0), ('outer edge', 7.6)), sep=None))
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
        strips.append(Strip(f'every cell, unrolled: {regs}' + (' (closes on itself)' if c['ring'] else ''), items, segs,
                            mx, x, ylabels=(('inner edge', 0.4), (regs, 4.0), ('outer edge', 7.6)), sep=None))
    targets.sort(key=lambda t: (TARGET_ORDER.index(t.name) if t.name in TARGET_ORDER else len(TARGET_ORDER), t.name))
    half = any(c['start'] == 'mirror' or c['end'] == 'mirror' for c in chains)
    return Layout(region, strips, targets, stacks, half=half, under={b: inner_of(b) for b in range(nb)})


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
    layout: Layout = None
    options: dict = field(default_factory=dict)
    shot: int = 0
    eq_ms: int = 0
    half_meta: int = -1
    single: bool = False   # a single-block run (the gyrokinetic app's own files: no _b<k> in the names, no topology)
    flip: bool = False     # its psi index grows inward: every block's rows are reversed on reading (rows())

    @property
    def name(self):
        return os.path.basename(self.path.rstrip('/'))

    def f(self, b, what, k=None):
        tail = f'_{k}' if k is not None else ''
        return f'{self.path}/{self.stem}-{what}{tail}.gkyl' if self.single else \
            f'{self.path}/{self.stem}_b{b}-{what}{tail}.gkyl'

    def frames(self):
        """Frame numbers written so far (from block 0's potential files)."""
        out = []
        for p in glob.glob(self.f(0, 'field_*')):
            m = re.search(r'-field_(\d+)\.gkyl$', p)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    def time(self, k):
        return float(meta(self.f(0, f'{self.species[0]}_M0', k)).get('time', np.nan))

    def stamp(self, k):
        """Changes when frame k is rewritten (a restart from an earlier frame rewrites the later ones)."""
        st = os.stat(self.f(0, f'{self.species[0]}_M0', k))
        return st.st_mtime_ns ^ st.st_size

    def rows(self, a):
        """A block's array (psi rows first) in the viewer's order, row 0 innermost: the file's, or reversed where the
        psi index grows inward. Only for cell averages and node values (a modal coefficient odd in psi changes sign)."""
        return a[::-1] if self.flip else a

    def file_row(self, b, i):
        """The file's psi index of the viewer's row i of block b, and back (the same map both ways)."""
        return self.nodes[b].shape[0] - 2 - i if self.flip else i

    @property
    def title(self):
        """'NSTX-U #203899 @ 325 ms' when the run's geqdsk names a shot, else its directory."""
        return f'NSTX-U #{self.shot} @ {self.eq_ms} ms' if self.shot else self.name


def find_stem(path):
    """(stem, single) of the run in a directory: the multiblock app's topology file, else the single-block app's
    geometry nodes (<stem>-geo_corn_nodes.gkyl, the stem not ending in _b<k>)."""
    topo = glob.glob(f'{path}/*-block_topo.gkyl')
    if len(topo) == 1:
        return os.path.basename(topo[0])[:-len('-block_topo.gkyl')], False
    if len(topo) > 1:
        raise FileNotFoundError(f'{path}: {len(topo)} block-topology files')
    stems = [os.path.basename(p)[:-len('-geo_corn_nodes.gkyl')] for p in glob.glob(f'{path}/*-geo_corn_nodes.gkyl')]
    single = [s for s in stems if not re.search(r'_b\d+$', s)]
    if len(single) == 1:
        return single[0], True
    raise FileNotFoundError(f'{path}: no block-topology file' + (' (block files without one: copy it in)' if stems
                                                                 else ' and no single-block geometry'))


def single_block_topology(nd):
    """A single-block run's connections (the app writes none): every face physical, except theta when the block
    closes on itself (its first and last theta node columns coincide: a core ring, periodic in theta)."""
    conn = np.zeros((1, 2, 2, 3), dtype=int)
    conn[..., 2] = PHYSICAL
    if np.allclose(nd[:, 0], nd[:, -1], rtol=0, atol=1e-9 * np.ptp(nd[..., :2])):
        conn[0, 1, 0], conn[0, 1, 1] = (0, 1, UPPER_POS), (0, 1, LOWER_POS)
    return conn


def magnetic_axis_R(run):
    """R of the magnetic axis from the run's geqdsk header (rmaxis on line 4: its line-3 copy is transposed in some
    files), or None."""
    path = geqdsk_path(run)
    if path is None:
        return None
    fre = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?')
    floats = [float(x) for line in open(path).read().split('\n')[1:6] for x in fre.findall(line)]
    return floats[13] if len(floats) >= 14 else None


def open_run(path, sim=''):
    path = os.path.abspath(path)
    r = Run(path, sim)
    r.stem, r.single = find_stem(path)
    frames = r.frames()
    if not frames:
        raise FileNotFoundError(f'{path}: no frames')
    k0 = frames[0]
    pre = f'{r.stem}-' if r.single else f'{r.stem}_b0-'
    r.species = sorted({re.search(rf'{re.escape(pre)}(\w+?)_M0_{k0}\.gkyl$', p).group(1)
                        for p in glob.glob(f'{path}/{pre}*_M0_{k0}.gkyl')})
    if not r.species:
        raise FileNotFoundError(f'{path}: no moments at frame {k0}')
    for s in r.species:
        m = meta(r.f(0, f'{s}_M0', k0))
        r.mass[s], r.charge[s] = float(m['mass']), float(m['charge'])
    r.half_meta = int(meta(r.f(0, f'{r.species[0]}_M0', k0)).get('half_domain', -1))
    r.e = min(abs(q) for q in r.charge.values())   # the elementary charge the run used (for eV)
    r.options = read_options(path)
    if r.single:
        r.nb = 1
        nodes = {0: np.asarray(read_gkyl(r.f(0, 'geo_corn_nodes'))['data'])[..., :2]}
        conn = single_block_topology(nodes[0])
    else:
        tm = read_gkyl(f'{path}/{r.stem}-block_topo.gkyl')['meta']
        r.nb = int(tm['num_blocks'])
        conn = np.array(tm['connections']).reshape(r.nb, int(tm['ndim']), 2, 3)
        nodes = {b: np.asarray(read_gkyl(r.f(b, 'geo_corn_nodes'))['data'])[..., :2] for b in range(r.nb)}
    r_axis = magnetic_axis_R(r)
    r.flip = not psi_grows_outward(conn, nodes, r_axis)
    if r.flip:
        conn = flip_conn(conn)
    r.nodes = {b: r.rows(nd) for b, nd in nodes.items()}
    r.layout = derive_layout(conn, r.nodes, r_axis)
    r.layout.flip = r.flip
    # shot and equilibrium time: the header of the geqdsk the run read, else its file stem
    g = geqdsk_path(r)
    m = re.search(r'#\s*(\d+)\s+(\d+)\s*ms', open(g).readline()) if g else None
    m = m or re.search(r'shot(\d+)_ms(\d+)', r.stem)
    if m:
        r.shot, r.eq_ms = int(m.group(1)), int(m.group(2))
    return r


def is_run_dir(p):
    """A directory holding a run with at least one frame (multiblock or single-block)."""
    if not os.path.isdir(p):
        return False
    try:
        stem, single = find_stem(p)
    except FileNotFoundError:
        return False
    return bool(glob.glob(f'{p}/{stem}-field_*.gkyl' if single else f'{p}/{stem}_b0-field_*.gkyl'))


def runs_root(simdir):
    """Where a simulation's runs are: <simdir>/runs (simdir under research/nstxu or absolute), else simdir itself."""
    base = os.path.join(NSTXU, simdir)
    return os.path.join(base, 'runs') if os.path.isdir(os.path.join(base, 'runs')) else base


def runs_in(root):
    """The run directories in a directory: those with a topology file (or a single-block run's files) and a frame."""
    return [d for d in sorted(os.listdir(root)) if is_run_dir(os.path.join(root, d))]


def sim_runs(simdir):
    """Run directories of a simulation (runs_in its runs_root)."""
    return runs_in(runs_root(simdir))


def geqdsk_path(run):
    """The geqdsk the run read: options_used.txt, else the -o geqdsk= of command.txt (relative to the run), else None."""
    path = run.options.get('geqdsk')
    cmd = os.path.join(run.path, 'command.txt')
    if not path and os.path.exists(cmd):
        m = re.search(r'geqdsk=(\S+)', open(cmd).read())
        path = m.group(1) if m else None
    if path and not os.path.isabs(path):
        path = os.path.join(run.path, path)
    return path if path and os.path.exists(path) else None


def wall(run):
    """Limiter outline (R, Z) from the run's geqdsk, or None."""
    path = geqdsk_path(run)
    if path is None:
        return None
    lines = open(path).read().split('\n')
    fre = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?')
    floats, intline = [], None
    for line in lines[1:]:
        if intline is None and len(floats) > 20 and re.fullmatch(r'\s*\d+\s+\d+\s*', line):
            intline = (len(floats), [int(x) for x in line.split()]); continue
        floats += [float(x) for x in fre.findall(line)]
    if intline is None:
        return None
    pos, (nbdy, nlim) = intline
    lim = np.array(floats[pos + 2 * nbdy:pos + 2 * nbdy + 2 * nlim]).reshape(-1, 2)
    return np.vstack([lim, lim[:1]])


# ----------------------------------------------------------------------------------------------------------- frames
def species_tag(s):
    return SPECIES_TAG.get(s, (s, s))


def quantities(run):
    """Quantity key -> (label, symbol, unit, scale to the unit, kind: 'pos' | 'signed')."""
    q = {}
    for s in run.species:
        full, t = species_tag(s)
        q[f'n_{s}'] = (f'{full} density', f'n_{t}', '1e19 m^-3', 1e-19, 'pos')
        q[f'T_{s}'] = (f'{full} temperature', f'T_{t}', 'eV', 1.0, 'pos')
        q[f'Tpar_{s}'] = (f'{full} parallel temperature', f'T_par,{t}', 'eV', 1.0, 'pos')
        q[f'Tperp_{s}'] = (f'{full} perpendicular temperature', f'T_perp,{t}', 'eV', 1.0, 'pos')
        q[f'upar_{s}'] = (f'{full} parallel flow', f'u_par,{t}', 'km/s', 1e-3, 'signed')
    if 'elc' in run.species and 'ion' in run.species:
        q['mach'] = ('ion parallel Mach number', 'u_par,i / c_s', '', 1.0, 'signed')
    q['phi'] = ('electrostatic potential', 'phi', 'V', 1.0, 'signed')
    q['q_par'] = ('total parallel energy flux', 'q_par', 'MW m^-2', 1e-6, 'signed')
    q['q_perp'] = ('parallel flux of perpendicular energy', 'q_perp', 'MW m^-2', 1e-6, 'signed')
    return q


def _block_fields(run, b, k):
    d = {}
    qpar, qperp = 0.0, 0.0
    for s in run.species:
        M = {m: run.rows(cav(run.f(b, f'{s}_{m}', k))) for m in MOMS}
        n = M['M0']
        ok = n > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.where(ok, M['M1'] / n, np.nan)
            m_e = run.mass[s] / run.e
            d[f'n_{s}'] = n
            d[f'upar_{s}'] = u
            d[f'T_{s}'] = np.where(ok, m_e * (M['M2'] - M['M1'] * u) / (3 * n), np.nan)
            d[f'Tpar_{s}'] = np.where(ok, m_e * (M['M2par'] - M['M1'] * u) / n, np.nan)
            d[f'Tperp_{s}'] = np.where(ok, m_e * M['M2perp'] / (2 * n), np.nan)
        qpar = qpar + run.mass[s] / 2 * (M['M3par'] + M['M3perp'])
        qperp = qperp + run.mass[s] / 2 * M['M3perp']
    if 'mach' in quantities(run):
        with np.errstate(invalid='ignore'):
            cs = np.sqrt((d['T_elc'] + d['T_ion']) * run.e / run.mass['ion'])
            d['mach'] = d['upar_ion'] / cs
    d['phi'] = run.rows(cav(run.f(b, 'field', k)))
    d['q_par'] = qpar
    d['q_perp'] = qperp
    return d


_FRAMES = {}


def frame(run, k):
    """block -> quantity -> (nr, nt) cell averages at frame k (the last 24 frames kept in memory)."""
    key = (run.path, k, run.stamp(k))
    if key not in _FRAMES:
        with ThreadPoolExecutor(4) as ex:   # I/O bound; 4 threads keep a login node responsive
            blocks = list(ex.map(lambda b: _block_fields(run, b, k), range(run.nb)))
        _FRAMES[key] = dict(enumerate(blocks))
        while len(_FRAMES) > 24:
            _FRAMES.pop(next(iter(_FRAMES)))
    return _FRAMES[key]


# ------------------------------------------------------------------------------------------- profiles and targets
def stack_blocks(st):
    """The blocks of a midplane stack, inside out (the core block under the SOL block, when there is one)."""
    return [b for b in (st.core, st.sol) if b is not None]


def midplane_profile(run, fr, side, qty):
    """(R at Z = 0 per cell row, value) through the core block then the SOL block on one side; R_sep."""
    st = run.layout.stacks[side]
    Rs, vs = [], []
    for b in stack_blocks(st):
        e = mid_edges(run.nodes[b], st.j)
        Rs.append(0.5 * (e[1:] + e[:-1]))
        v = fr[b][qty]
        vs.append(np.array([(1 - w) * v[i, j0] + w * v[i, j1] for i, (j0, j1, w) in
                            enumerate(midplane_weights(run.nodes[b], st.j))]))
    return np.concatenate(Rs), np.concatenate(vs), mid_edges(run.nodes[st.sol], st.j)[0]


def stack_rows(run, side, blocks=None):
    """Which rows of a midplane profile (midplane_profile's order) belong to the chosen blocks (None: all)."""
    st = run.layout.stacks[side]
    return np.concatenate([np.full(run.nodes[b].shape[0] - 1, blocks is None or b in blocks) for b in stack_blocks(st)])


def target_heat_flux(run, fr, tgt):
    """|q_par| per SOL surface at the plate cell, the surfaces' midplane distance from the separatrix (mm), lambda_q.
    Where no block of the leg's chain reaches Z = 0 (tgt.mid None) the surfaces sit at their distance along the plate
    from its inner corner instead, and there is no lambda_q."""
    q = np.abs(fr[tgt.leg]['q_par'][:, tgt.j])
    if tgt.mid is None:
        face = run.nodes[tgt.leg][:, tgt.j]
        dist = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(face, axis=0).T))])
        return q, 0.5 * (dist[1:] + dist[:-1]) * 1e3, np.nan
    edges = mid_edges(run.nodes[tgt.mid], tgt.mid_j)
    dist = np.abs(edges - edges[0])
    C = np.concatenate([[0.0], np.cumsum(q * np.diff(dist))])
    lam = (np.interp(LAMBDA_FRACTION * C[-1], C, dist)) * 1e3 if C[-1] > 0 else np.nan
    return q, 0.5 * (dist[1:] + dist[:-1]) * 1e3, lam


# ---------------------------------------------------------------------------------------- per-frame summary cache
def cell_offsets(run):
    """block -> index of its first cell in the flat cell order of the cached fields (blocks in turn, rows of psi)."""
    off, n = {}, 0
    for b in range(run.nb):
        off[b] = n
        n += (run.nodes[b].shape[0] - 1) * (run.nodes[b].shape[1] - 1)
    return off


def _summary_one(run, k, fr0):
    fr = frame(run, k)
    out = {'t': run.time(k)}
    for tgt in run.layout.targets:
        q, _, lam = target_heat_flux(run, fr, tgt)
        out[f'q|{tgt.name}'] = q
        out[f'lam|{tgt.name}'] = lam
    for side in run.layout.stacks:
        for qty in quantities(run):
            out[f'mid|{side}|{qty}'] = midplane_profile(run, fr, side, qty)[1]
    for qty in quantities(run):
        v = np.concatenate([fr[b][qty].ravel() for b in range(run.nb)])
        out[f'fld|{qty}'] = v.astype(np.float32)   # every cell (cell_offsets order): the browser-side movie
        v = v[np.isfinite(v)]
        out[f'pct|{qty}'] = np.percentile(v, [0, 1, 50, 99, 100]) if len(v) else np.full(5, np.nan)
    for reg in ('core', 'sol', 'pf'):
        bs = [b for b in range(run.nb) if run.layout.region[b] == reg]
        for s in run.species:
            for base in ('n', 'T'):
                qty = f'{base}_{s}'
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = np.concatenate([(fr[b][qty] / fr0[b][qty] - 1).ravel() for b in bs] or [np.zeros(0)])
                rel = rel[np.isfinite(rel)]
                out[f'med|{reg}|{qty}'] = np.median(rel) if len(rel) else np.nan
    for s in run.species:
        out[f'neg|{s}'] = float(sum((fr[b][f'n_{s}'] <= 0).sum() for b in range(run.nb)))
    return out


def _cache_path(run):
    rel = os.path.relpath(run.path, NSTXU).replace(os.sep, '__')
    return os.path.join(CACHE_DIR, f'{rel}__v{CACHE_VERSION}.npz')


_SUMMARY = {}


def summary(run, progress=None):
    """Per-frame summary arrays (key -> [n_frames, ...]) over every frame written, cached on disk in runs/cache (and
    in memory); only new or rewritten frames are read. progress(iterable, total) wraps the loop over them (a progress
    bar). 'fld|<qty>' holds every cell's value (float32, cell_offsets order)."""
    frames = run.frames()
    stamps = np.array([run.stamp(k) for k in frames], dtype=np.int64)
    memo = _SUMMARY.get(run.path)
    if memo is not None and np.array_equal(memo['stamp'], stamps):
        return memo
    path = _cache_path(run)
    old = {}
    if os.path.exists(path):
        with np.load(path) as z:
            old = {key: z[key] for key in z.files}
    have = {}
    if old:
        for i, (k, s) in enumerate(zip(old['frames'], old['stamp'])):
            have[int(k)] = (int(s), i)
    todo = [k for k, s in zip(frames, stamps) if have.get(k, (None,))[0] != s]
    fr0 = frame(run, frames[0])
    new = {}
    it = progress(todo, len(todo)) if (progress and todo) else todo
    for k in it:
        new[k] = _summary_one(run, k, fr0)
    keys = sorted(set(new[todo[0]]) if todo else set(old) - {'frames', 'stamp'})
    out = {'frames': np.array(frames), 'stamp': stamps}
    for key in keys:
        rows = []
        for k in frames:
            if k in new:
                rows.append(np.asarray(new[k][key], dtype=np.float32 if key.startswith('fld|') else float))
            else:
                rows.append(old[key][have[k][1]])
        out[key] = np.array(rows)
    if todo:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + '.tmp.npz'
        np.savez(tmp, **out)
        os.replace(tmp, path)
    _SUMMARY[run.path] = out
    return out


# ------------------------------------------------------------------------------------------------------ time traces
def dt_history(run):
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
                tot = None; break
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
    return t0, tot


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
            raise ValueError(f'frame {f} counts back past the first of the {len(available)} frames')
        return available[f]

    def bound(v, default):
        return default if v is None else (v if v >= 0 else available[max(len(available) + v, 0)])

    text = str(spec).strip()
    if not text:
        return list(available)
    try:
        if ':' not in text:
            return [position(int(f)) for f in text.split(',')]
        parts = text.split(':')
        if len(parts) > 3:
            raise ValueError
        start, stop, step = [int(p) if p.strip() else None for p in parts] + [None] * (3 - len(parts))
    except ValueError:
        raise ValueError(f"frames {spec!r}: not a frame number, a comma-separated list or a 'start:stop[:step]' "
                         'range') from None
    step = 1 if step is None else step
    if step <= 0:
        raise ValueError(f'frames {spec!r}: the step must be a positive integer')
    if not available:
        return []
    lo, hi = bound(start, available[0]), bound(stop, available[-1] + 1)
    return [f for f in available if lo <= f < hi and (f - lo) % step == 0]


# ------------------------------------------------------------------------------------- volumes and region means
def cell_volumes(run):
    """Volume of every cell (cell_offsets order): the Jacobian's cell average times the logical cell size, per radian
    of toroidal angle like every integrated diagnostic (sum(n V) matches the integrated M0 to ~0.2 %)."""
    if getattr(run, '_volumes', None) is None:
        out = []
        for b in range(run.nb):
            g = read_gkyl(run.f(b, 'geo_int_jacobgeo'))
            cell = np.prod((np.asarray(g['upper']) - np.asarray(g['lower'])) / np.asarray(g['cells']))
            out.append(run.rows(np.asarray(g['data'])[..., 0] / 2.0 * cell).ravel())
        run._volumes = np.concatenate(out)
    return run._volumes


def region_cells(run, region):
    """Cell indices (cell_offsets order) of a region ('core', 'sol', 'pf'), of every cell ('all'), or of a list of
    blocks."""
    off = cell_offsets(run)
    pick = (lambda b: b in region) if isinstance(region, (list, tuple, set, frozenset)) else \
        (lambda b: region == 'all' or run.layout.region[b] == region)
    idx = [np.arange(off[b], off[b] + (run.nodes[b].shape[0] - 1) * (run.nodes[b].shape[1] - 1))
           for b in range(run.nb) if pick(b)]
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
    """(t, sum over blocks of one component) of an integrated time trace, the blocks that wrote it."""
    found = [read_dynvec(run.f(b, what)) for b in (range(run.nb) if blocks is None else blocks)
             if os.path.exists(run.f(b, what))]
    if not found:
        return None, None
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
            _, xN = _dyn_sum(run, f'{s}_bflux_{face}_integrated_HamiltonianMoments', 0)
            _, xE = _dyn_sum(run, f'{s}_bflux_{face}_integrated_HamiltonianMoments', 2)
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
    n = min(len(E_src), len(E_out), len(E_dot), len(fd) if fd is not None else len(E_dot))
    out['t'] = out['t'][:n]
    out['E'] = dict(src=E_src[:n], out=E_out[:n], dot=E_dot[:n] - (fd[:n] if fd is not None else 0))
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


# ------------------------------------------------------------------------------------------------------ diffusivity
def _ser_p1_2d():
    """The 2D p = 1 serendipity basis on [-1, 1]^2 (orthonormal: 1/2, sqrt(3) x/2, sqrt(3) y/2, 3 x y/2): the
    triple-product tensor T[k, i, j] = int phi_k phi_i phi_j, and the basis at the four corners (-,-) (+,-) (-,+) (+,+),
    x along psi, y along theta."""
    phi = lambda x, y: np.array([0.5, np.sqrt(3) / 2 * x, np.sqrt(3) / 2 * y, 1.5 * x * y])  # noqa: E731
    g, w = np.polynomial.legendre.leggauss(2)   # exact for the product of three p = 1 functions
    T = sum(wx * wy * np.einsum('k,i,j->kij', phi(x, y), phi(x, y), phi(x, y))
            for x, wx in zip(g, w) for y, wy in zip(g, w))
    return T, np.array([phi(x, y) for x, y in ((-1, -1), (1, -1), (-1, 1), (1, 1))])


SER_T, SER_CORNERS = _ser_p1_2d()
CORNER_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1))   # node (i + di, j + dj) of cell (i, j) for each corner


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
        gap = max(gap, float(np.nanmax(np.abs(np.where(np.isnan(cur), Dc[..., c], cur) - Dc[..., c]))))
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
    pats = ('slurm-*.out', '*.log', 'stdout*')
    found = {p for pat in pats for p in glob.glob(os.path.join(run.path, pat))
             if os.path.isfile(p) and os.path.basename(p) != 'restarts.log'}   # restarts.log: the job list
    return sorted(found, key=os.path.getmtime, reverse=True)


def _log_head(path, nlines=600):
    with open(path, errors='replace') as f:
        return [line for _, line in zip(range(nlines), f)]


# a uniform D in the run's own printout (each driver words it differently), the most specific first
D_LOG_LINES = [
    (r'Anomalous diffusion: D_perp \S+ m\^2/s x D_perp_mult \S+ = (\S+) m\^2/s', False),   # nstxu_multib, mult != 1
    (r'Physics preset/overrides:.*\bD=(\S+) m\^2/s', False),                               # the 2025 driver
    (r'Applied overrides:.*D_const=(\S+) \(m\^2/s\)', False),                              # the 2025 driver, csv
    (r'\bn0 \S+ m\^-3, Te = Ti = Tsource \S+ eV, D_perp (\S+) m\^2/s', True),             # nstxu_multib csv_init
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
    if 'D_perp' in opts and not opts.get('csv_init', 'false').lower() in ('1', 'true', 'yes'):
        return float(opts['D_perp']) * mult, f"its options: D_perp={opts['D_perp']}" + (f', D_perp_mult={mult:g}' if mult != 1 else '')
    if 'D_const' in opts:
        return float(opts['D_const']), f"its options: D_const={opts['D_const']}"
    return None, ('not recorded: no anom_diff file, no D option and no D line in its logs, so the driver\'s default '
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
    if all(os.path.exists(run.f(b, f'{s}_anom_diff', 0)) for b in range(run.nb) for s in run.species):
        for s in run.species:
            nodes, cells = {}, {}
            for b in range(run.nb):
                A = np.asarray(read_gkyl(run.f(b, f'{s}_anom_diff', 0))['data'])
                G = np.asarray(read_gkyl(run.f(b, 'geo_int_gxxj'))['data'])
                D = weak_divide(A, G)
                nv, gap = node_values(D)
                nodes[b], cells[b] = run.rows(nv), run.rows(D[..., 0] / 2.0)
                out['gap'] = max(out['gap'], gap / max(np.nanmax(np.abs(nv)), 1e-300))
            out['species'][s] = dict(nodes=nodes, cells=cells)
        out['kind'], out['source'] = 'file', (f'the run\'s {run.stem}{"" if run.single else "_b<k>"}-<species>_anom_diff_0 '
                                              'files (D × gˣˣJ as the solver uses it), divided by geo_int_gxxj per cell')
    else:
        D, why = recorded_uniform_D(run)
        out['source'] = why
        if D is not None:
            out['kind'] = 'uniform'
            for s in run.species:
                out['species'][s] = dict(nodes={b: np.full(run.nodes[b].shape[:2], D) for b in range(run.nb)},
                                         cells={b: np.full((run.nodes[b].shape[0] - 1, run.nodes[b].shape[1] - 1), D)
                                                for b in range(run.nb)})
    sp = list(out['species'])
    out['same'] = all(all(np.array_equal(out['species'][sp[0]]['nodes'][b], out['species'][s]['nodes'][b])
                          for b in range(run.nb)) for s in sp[1:])
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
                Dm.append(v[i, int(np.argmin(np.abs(Z)))]); continue
            k = _pick(k, st.j, 0.5); w = Z[k] / (Z[k] - Z[k + 1])
            Dm.append(v[i, k] + w * (v[i, k + 1] - v[i, k]))
        if Rs and abs(R[0] - Rs[-1][-1]) <= 1e-12 * max(abs(R[0]), 1.0):   # the shared separatrix row
            R, Dm = R[1:], Dm[1:]
        Rs.append(np.asarray(R)); Ds.append(np.asarray(Dm)); Bs.append(np.full(len(R), b))
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
            out = np.full((len(self.idx),) + B.shape[1:], np.nan)
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
            if pass_region and run.layout.region[b] != pass_region:
                continue
            c = cell_centres(run.nodes[b])
            d = np.hypot(c[..., 0] - R, c[..., 1] - Z)
            i, j = np.unravel_index(int(np.argmin(d)), d.shape)
            if d[i, j] < best[0]:
                best = (float(d[i, j]), off[b] + i * d.shape[1] + j)
        if best[1] >= 0:
            break
    return best[1], best[0]
