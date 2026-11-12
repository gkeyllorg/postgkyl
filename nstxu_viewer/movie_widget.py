"""Browser-side movie of one run (nstxu_viewer.py). Every frame of one quantity is sent to the page once, as 8-bit colour
indices on a colour scale fixed over the whole run, and the page animates it on canvases: playback makes no kernel
round trips. (Stepping the notebook's own frame re-runs its cells, a PNG every ~0.3 s, and marimo dims the stale
outputs meanwhile: the fade between frames.)

Panels: the poloidal plane; every cell unrolled (the posters' strips); one flux surface of each strip against time
(space-time: a disturbance running along the field is a slanted streak, its slope the speed); the outboard and
inboard midplane profiles. The page writes its frame back to the notebook when playback pauses or the frame is
scrubbed or stepped, so the tiles and tabs follow; the notebook's frame slider moves the movie too.

The data: the per-frame cache (gkview.summary, 'fld|<qty>' = every cell's cell average, 'mid|<side>|<qty>' = the
midplane profiles). 'Interpolate' blends neighbouring frames for display only (it is labelled on the page).
"""
import base64

import anywidget
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import traitlets

import gkview as gv
import pgkyl_cmd as pc
import viewer_plots as vp

NAN = 255   # colour index of a cell with no value (a temperature where n <= 0, 0/0 in a relative change)
# flux surfaces offered for the space-time panel: (label, region of the strip, position across it: 0 = its inner edge)
RINGS = [('SOL, next to the separatrix', 'sol', 0.0), ('SOL, middle', 'sol', 0.5), ('SOL, outermost', 'sol', 1.0),
         ('core / PF, next to the separatrix', 'in', 1.0), ('core / PF, innermost', 'in', 0.0)]


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


# The pgkyl box (pgkyl_cmd.py): under a chart or the movie, the commands for what was clicked, each with a Copy
# button; shared by Movie and LivePlot. Colours from marimo's own variables, so it follows the page's theme.
CMD_CSS = """
.nstxu-cmd { display: none; margin: 8px 0 2px; border: 1px solid var(--border, #8888); border-radius: 6px;
  padding: 6px 10px 8px; font-size: 12px; background: var(--card, transparent); color: var(--foreground, inherit); }
.nstxu-cmd.on { display: block; }
.nstxu-cmd .hd { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
.nstxu-cmd .hd .what { flex: 1; opacity: 0.85; }
.nstxu-cmd button { cursor: pointer; border-radius: 4px; border: 1px solid var(--border, #8888); padding: 1px 8px;
  background: transparent; color: inherit; font: inherit; font-size: 11px; }
.nstxu-cmd pre { margin: 3px 0 2px; padding: 6px 8px; overflow-x: auto; white-space: pre; font-size: 11.5px;
  line-height: 1.35; border-radius: 4px; background: var(--muted, #8881); user-select: all; }
"""
CMD_JS = r"""
function cmdBox(parent) {
  const box = document.createElement('div');
  box.className = 'nstxu-cmd';
  parent.append(box);
  async function copy(text, btn) {
    try { await navigator.clipboard.writeText(text); }
    catch (e) {   // no clipboard API (an http page that is not localhost): select and copy
      const ta = document.createElement('textarea'); ta.value = text; document.body.append(ta); ta.select();
      try { document.execCommand('copy'); } catch (e2) {} ta.remove();
    }
    btn.textContent = 'copied'; setTimeout(() => { btn.textContent = 'copy'; }, 1200);
  }
  return {
    // items: [{name, text}]; commands that are the same are shown once, their names joined
    show(items, title) {
      const merged = [];
      for (const it of items) { const m = merged.find(x => x.text === it.text);
        if (m) { if (!m.names.includes(it.name)) m.names.push(it.name); } else merged.push({ names: [it.name], text: it.text }); }
      box.innerHTML = '';
      const top = document.createElement('div'); top.className = 'hd';
      top.innerHTML = `<span class="what"><b>pgkyl</b> ${title || ''}</span><button class="x" title="close">close</button>`;
      top.querySelector('.x').onclick = () => box.classList.remove('on');
      box.append(top);
      for (const m of merged) {
        const hd = document.createElement('div'); hd.className = 'hd';
        const what = document.createElement('span'); what.className = 'what'; what.textContent = m.names.filter(Boolean).join(', ');
        const b = document.createElement('button'); b.textContent = 'copy'; b.onclick = () => copy(m.text, b);
        const pre = document.createElement('pre'); pre.textContent = m.text;
        hd.append(what, b); box.append(hd, pre);
      }
      box.classList.toggle('on', merged.length > 0);
    },
    hide() { box.classList.remove('on'); },
    el: box,
  };
}
"""


def _hexes(cmap, n=255):
    return ['#%02x%02x%02x' % tuple(int(round(255 * c)) for c in cmap(i / (n - 1))[:3]) for i in range(n)]


def movie_look(pal):
    """Colours and font of the page for one look (light / dark): inks, accent, both colour tables."""
    div = LinearSegmentedColormap.from_list(f'div_{pal.name}', pal.div)
    return dict(surf=pal.surf, ink=pal.ink, ink2=pal.ink2, muted=pal.muted, grid=pal.grid, axis=pal.axis,
                accent=pal.accent or vp.tone('hlink'), elc=pal.elc, ion=pal.ion, font=vp.CSS_FONT,
                lut_seq=_hexes(colormaps[vp.SEQ]), lut_div=_hexes(div), nan=pal.surf)


def movie_data(run, summ, qty, mode='value', log=False, x_mode='distance', frames=None, blocks=None):
    """Everything the page needs for one quantity and mode (sent once; JSON with base64 arrays). frames: the frame
    numbers to play (default: all); changes stay relative to the run's first / the previous written frame. blocks:
    draw only these (None: all): the colour scale, the poloidal view and the strips are theirs."""
    label, sym, unit, scale, kind = gv.quantities(run)[qty]
    u = f' ({unit})' if unit else ''
    allf = [int(k) for k in summ['frames']]
    have = set(allf)
    frames = allf if frames is None else [int(k) for k in frames if int(k) in have]
    if not frames:
        raise ValueError('no frame to play')
    rows = np.array([allf.index(k) for k in frames])
    V = vp.mode_values(summ[f'fld|{qty}'].astype(np.float64), rows, mode, kind, scale)
    shown = vp.shown_blocks(run, blocks)
    sel = None if len(shown) == run.nb else set(shown)
    keep = gv.region_cells(run, shown)   # the drawn cells
    # one colour scale over the frames played (a per-frame scale would make colours jump between frames), and over
    # the drawn cells
    sc = vp.fixed_scale(run, V[:, keep] if sel else V, qty, mode, log, step_first=(mode == 'step' and rows[0] == 0))
    norm, ticks, fmt, cblabel, cmap, rel = sc['norm'], sc['ticks'], sc['fmt'], sc['label'], sc['cmap'], sc['rel']
    with np.errstate(invalid='ignore'):
        tt = np.ma.filled(norm(np.ma.masked_invalid(V)), np.nan)
    idx = np.where(np.isfinite(tt), np.clip(np.round(tt * 254), 0, 254), NAN).astype(np.uint8)
    extend = sc['extend']

    # geometry: every cell's four corners (cell_offsets order), lines, views
    off = gv.cell_offsets(run)
    quads = np.concatenate([np.stack([nd[:-1, :-1], nd[1:, :-1], nd[1:, 1:], nd[:-1, 1:]], axis=2).reshape(-1, 8)
                            for nd in (run.nodes[b] for b in range(run.nb))]).astype(np.float32)
    rnd = lambda a: [round(float(x), 5) for x in np.asarray(a).ravel()]  # noqa: E731
    lines = []
    for b in shown:
        nd = run.nodes[b]
        lines.append(dict(kind='outline', pts=rnd(np.concatenate([nd[0, :], nd[:, -1], nd[-1, ::-1], nd[::-1, 0]]))))
    for b in shown:
        if run.layout.region[b] == 'sol' and run.layout.under.get(b) is not None:   # its inner edge: the separatrix
            lines.append(dict(kind='sep', pts=rnd(run.nodes[b][0, :])))
    w = gv.wall(run)
    if w is not None:
        lines.append(dict(kind='wall', pts=rnd(w)))

    # strips: one rectangle per cell (x in cells, y in 0..8: core/PF 0-4, SOL 4-8), and the space-time rings
    strips, rings = [], [dict(label=lab, strips=[]) for lab, _, _ in RINGS]
    for st in gv.select_strips(run.layout, sel):
        rects, cells = [], []
        cols = [np.full(st.width, -1, np.int32) for _ in RINGS]
        for b, x0, band in st.items:
            nr, nt = run.nodes[b].shape[0] - 1, run.nodes[b].shape[1] - 1
            y0, h = gv.BANDS[band]
            for i in range(nr):
                for j in range(nt):
                    rects += [x0 + j, y0 + h * i / nr, h / nr, 1.0 if nr <= 8 else 0.0]
                    cells.append(off[b] + i * nt + j)
            for r, (_, rreg, frac) in enumerate(RINGS):
                # a ring's band: SOL rings on the SOL band (and on a one-band strip), core/PF rings on the band under it
                if (band in ('sol', 'all')) == (rreg == 'sol'):
                    row = int(round(frac * (nr - 1)))
                    cols[r][x0:x0 + nt] = off[b] + row * nt + np.arange(nt)
                    rings[r].setdefault('rows', {})[f'{len(strips)}:{x0}'] = [y0 + h * row / nr, h / nr]
        strips.append(dict(title=st.title, width=st.width, mirror_x=st.mirror_x, segments=st.segments,
                           ylabels=[[lab, y] for lab, y in st.ylabels], sep=st.sep,
                           rects=_b64(np.array(rects, np.float32)), cells=_b64(np.array(cells, np.int32))))
        for r in range(len(RINGS)):
            rings[r]['strips'].append(cols[r].tolist())

    # midplane profiles of the same quantity and mode, from the cache
    fr0 = gv.frame(run, allf[0])
    sides = []
    for side in [s for s in ('outboard', 'inboard') if s in run.layout.stacks and gv.stack_rows(run, s, sel).any()]:
        R, _, rsep = gv.midplane_profile(run, fr0, side, qty)
        x, xsep, xlab = vp.side_coordinate(R, rsep, side, x_mode)
        P = vp.mode_values(summ[f'mid|{side}|{qty}'].astype(np.float64), rows, mode, kind, scale) * (100 if rel else 1)
        o = np.argsort(x)
        o = o[gv.stack_rows(run, side, sel)[o]]   # the chosen blocks' rows
        pg = pc.midplane(run, side, qty, '{frame}', sel)
        sides.append(dict(name=f'{side} midplane', x=rnd(x[o]), sep=float(xsep), xlabel=xlab,
                          v=_b64(P[:, o].astype(np.float32)),
                          pg=pg if mode == 'value' else '# the chart shows the change; this, the values\n' + pg))
    pv = np.concatenate([np.frombuffer(base64.b64decode(s['v']), np.float32) for s in sides]) if sides else np.zeros(1)
    pv = pv[np.isfinite(pv)]
    plo, phi = (float(pv.min()), float(pv.max())) if len(pv) else (0.0, 1.0)
    pad = 0.06 * (phi - plo or abs(phi) or 1.0)
    ylab = (f'{sym} change (%)' if rel else (f'{sym}' + (' change' if mode != 'value' else '') + u))

    # cells with n <= 0 (any species), per frame: x marks
    bad = np.zeros((len(rows), summ[f'fld|{qty}'].shape[1]), bool)
    for s in run.species:
        bad |= summ[f'fld|n_{s}'][rows] <= 0
    if sel:
        drawn = np.zeros(bad.shape[1], bool)
        drawn[keep] = True
        bad &= drawn
    ts, tu = vp.time_unit(float(summ['t'][-1]))
    species = next((s for s in run.species if qty.endswith(f'_{s}')), 'ion' if qty == 'mach' else None)
    # pgkyl commands (pgkyl_cmd) for a click on a cell: the page fills in the block, the file's indices, the frame
    # and the frame a change is taken from ({ref}: the run's first frame, or the previous written one)
    pg = dict(blocks=[[int(off[b]), int(run.nodes[b].shape[0] - 1), int(run.nodes[b].shape[1] - 1)] for b in range(run.nb)],
              flip=bool(run.flip), ref=None if mode == 'value' else
              [allf[0] if mode == 'change' else allf[max(r - 1, 0)] for r in rows.tolist()],
              cell=pc.cell_value(run, '{b}', '{i}', '{j}', qty, '{frame}'),
              map=pc.block_map(run, '{b}', qty, '{frame}', mode, None if mode == 'value' else '{ref}'),
              hist=pc.cell_history(run, '{b}', '{i}', '{j}', qty))
    return dict(
        frames=frames, tnum=[round(float(t) * ts, 6) for t in summ['t'][rows]], tunit=tu, ncell=int(V.shape[1]),
        idx=_b64(idx), cmap=cmap, cbar=dict(label=cblabel, ticks=[[float(norm(t)), fmt(t)] for t in ticks],
                                            extend=extend),
        quads=_b64(quads), lines=lines, boxes={k: list(v) for k, v in vp.zoom_boxes(run, sel).items()},
        rz=_b64(keep.astype(np.int32)), rz_note='' if sel is None else f' (blocks {vp.block_list(shown)})',
        st_caption=('cell along the strip: target → X point → … → target' if run.layout.targets
                    else 'cell along the strip (θ)'),
        strips=strips, rings=rings, bad=[np.nonzero(r)[0].tolist() for r in bad],
        prof=dict(sides=sides, ylim=[plo - pad, phi + pad], ylabel=ylab, first=mode == 'value',
                  zero=mode != 'value' or kind != 'pos'),
        species=species, title=f'{label}' + ('' if mode == 'value' else f', change since {vp.SINCE[mode]}'), pg=pg)


class Movie(anywidget.AnyWidget):
    """The page-side player: data (movie_data), look (movie_look), zoom (a key of data['boxes']), frame (the frame
    number shown; the page sets it on pause / scrub / step, the notebook sets it from its slider), probe (the cell a
    click picked; a click on the space-time strip also moves to that time)."""
    data = traitlets.Dict().tag(sync=True)
    look = traitlets.Dict().tag(sync=True)
    zoom = traitlets.Unicode('whole domain').tag(sync=True)
    frame = traitlets.Int(0).tag(sync=True)
    probe = traitlets.Int(-1).tag(sync=True)   # the probed cell (cell_offsets order), set by a click on the page

    _css = CMD_CSS + """
.nstxu-movie { font-family: var(--nm-font); color: var(--nm-ink); background: var(--nm-surf); padding: 10px 12px 12px;
  border-radius: 8px; outline: none; box-sizing: border-box; width: 100%; }
.nstxu-movie .bar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; font-size: 13px; }
.nstxu-movie button { display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 28px;
  border-radius: 6px; border: 1px solid var(--nm-axis); background: var(--nm-grid); color: var(--nm-ink); cursor: pointer;
  padding: 0; }
.nstxu-movie button.play { width: 40px; background: var(--nm-accent); border-color: var(--nm-accent); color: #fff; }
.nstxu-movie input[type=range] { flex: 1 1 200px; accent-color: var(--nm-accent); min-width: 120px; }
.nstxu-movie select { background: var(--nm-grid); color: var(--nm-ink); border: 1px solid var(--nm-axis);
  border-radius: 6px; padding: 3px 6px; font: inherit; }
.nstxu-movie .time { font-variant-numeric: tabular-nums; min-width: 17em; color: var(--nm-ink); }
.nstxu-movie label { display: inline-flex; gap: 4px; align-items: center; color: var(--nm-ink2); }
.nstxu-movie .grid { display: flex; gap: 12px; align-items: flex-start; }
.nstxu-movie canvas { display: block; cursor: crosshair; }
.nstxu-movie .hint { color: var(--nm-ink2); font-size: 12px; margin-top: 6px; }
"""

    _esm = CMD_JS + r"""
const ICON = {
  play: '<svg viewBox="0 0 16 16" width="15" height="15"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>',
  pause: '<svg viewBox="0 0 16 16" width="15" height="15"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/></svg>',
  prev: '<svg viewBox="0 0 16 16" width="14" height="14"><path d="M3 2.5h2v11H3zM13 2.5v11L6 8z" fill="currentColor"/></svg>',
  next: '<svg viewBox="0 0 16 16" width="14" height="14"><path d="M11 2.5h2v11h-2zM3 2.5v11L10 8z" fill="currentColor"/></svg>',
};
function bytes(s) { const b = atob(s), a = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a; }
const f32 = s => new Float32Array(bytes(s).buffer);
const i32 = s => new Int32Array(bytes(s).buffer);
function rgb(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
function niceStep(span, n) { const raw = span / n, e = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * e) return m * e; return 10 * e; }
function fmt(v) { const a = Math.abs(v); return (a !== 0 && (a < 1e-3 || a >= 1e5)) ? v.toExponential(1) : +v.toPrecision(4) + ''; }
// round tick values over [lo, hi], about n of them
function ticks(lo, hi, n) { const step = niceStep((hi - lo) || 1, Math.max(1, n)), out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * step; v += step) out.push(+v.toPrecision(12));
  return out; }

function render({ model, el }) {
  const D = model.get('data');
  const n = D.frames.length, nc = D.ncell, idx = bytes(D.idx), quads = f32(D.quads);
  // the drawn cells (the chosen blocks'; every cell without a choice)
  const rzc = D.rz ? i32(D.rz) : Int32Array.from({ length: nc }, (_, c) => c), drawn = new Uint8Array(nc);
  rzc.forEach(c => { drawn[c] = 1; });
  const strips = D.strips.map(s => {
    const rects = f32(s.rects), cells = i32(s.cells), where = new Int32Array(nc).fill(-1);
    cells.forEach((c, r) => { where[c] = r; });
    return { ...s, rects, cells, where };
  });
  const sides = D.prof.sides.map(p => ({ ...p, v: f32(p.v), nrow: p.x.length }));
  let L = model.get('look'), lut = [], lutRGB = [];
  let pos = Math.max(0, D.frames.indexOf(model.get('frame'))); if (D.frames.indexOf(model.get('frame')) < 0) pos = n - 1;
  let playing = false, raf = 0, last = 0, ring = 0, lastSent = null;

  el.classList.add('nstxu-movie'); el.tabIndex = 0;
  el.innerHTML = `<div class="bar">
      <button class="play" title="play / pause (space)"></button>
      <button class="prev" title="previous frame (left arrow)">${ICON.prev}</button>
      <button class="next" title="next frame (right arrow)">${ICON.next}</button>
      <input class="scrub" type="range" min="0" max="${n - 1}" step="1">
      <span class="time"></span>
      <label>speed <select class="fps">${[2, 5, 10, 20, 30, 60].map(f => `<option value="${f}"${f === 10 ? ' selected' : ''}>${f} frames/s</option>`).join('')}</select></label>
      <label><input type="checkbox" class="loop"> loop</label>
      <label title="blend neighbouring frames for display: smoother motion, but the in-between images are not simulation output"><input type="checkbox" class="interp"> interpolate</label>
      <label>space-time along <select class="ring">${D.rings.map((r, i) => `<option value="${i}">${r.label}</option>`).join('')}</select></label>
    </div>
    <div class="grid"><div class="left"><canvas class="rz"></canvas><canvas class="prof"></canvas></div>
      <div class="right"><canvas class="strips"></canvas><canvas class="st"></canvas></div></div>
    <div class="hint"></div>`;
  const $ = s => el.querySelector(s);
  const cmd = cmdBox(el);
  // the click's pgkyl commands: a cell (its value, its block's map at the frame, its history), a midplane profile
  const fillT = (t, vals) => t.replace(/\{(\w+)\}/g, (m, k) => (k in vals ? String(vals[k]) : m));
  function cellOf(c) {
    const P = D.pg.blocks; let b = 0;
    for (let k = 0; k < P.length; k++) if (P[k][0] <= c) b = k;
    const [o, nr, nt] = P[b], iv = Math.floor((c - o) / nt);
    return { b, i: D.pg.flip ? nr - 1 - iv : iv, j: (c - o) % nt };
  }
  function showCell(c) {
    if (!D.pg || c < 0 || model.get('probe') !== c) { cmd.hide(); return; }
    const { b, i, j } = cellOf(c), f = Math.min(n - 1, Math.floor(pos + 1e-9));
    const v = { b, i, j, frame: D.frames[f], ref: D.pg.ref ? D.pg.ref[f] : '' };
    cmd.show([{ name: `the cell at frame ${v.frame}`, text: fillT(D.pg.cell, v) },
              { name: `block ${b} at frame ${v.frame}` + (D.pg.ref ? ` (change since frame ${v.ref})` : ''), text: fillT(D.pg.map, v) },
              { name: 'the cell at every frame', text: fillT(D.pg.hist, v) }],
             `· block ${b}, cell i = ${i} (psi, the file's), j = ${j} (theta)`);
  }
  const btn = $('.play'), scrub = $('.scrub'), tlab = $('.time'), fpsSel = $('.fps'), loop = $('.loop'),
        interp = $('.interp'), ringSel = $('.ring'), hint = $('.hint');
  const cv = { rz: $('.rz'), prof: $('.prof'), strips: $('.strips'), st: $('.st') };
  btn.innerHTML = ICON.play;
  let ctx = {}, geo = {};

  function applyLook() {
    L = model.get('look');
    for (const [k, v] of Object.entries({ surf: L.surf, ink: L.ink, ink2: L.ink2, axis: L.axis, grid: L.grid,
                                          accent: L.accent, font: L.font })) el.style.setProperty('--nm-' + k, v);
    lut = (D.cmap === 'seq' ? L.lut_seq : L.lut_div).concat([L.nan]);
    lutRGB = lut.map(rgb);
  }

  function setup(c, w, h) {
    const r = window.devicePixelRatio || 1;
    c.width = Math.round(w * r); c.height = Math.round(h * r); c.style.width = w + 'px'; c.style.height = h + 'px';
    const x = c.getContext('2d'); x.setTransform(r, 0, 0, r, 0, 0); return x;
  }

  // ------------------------------------------------------------------------------------------------- layout
  function layout() {
    // two columns (poloidal plane + profiles | strips + space-time); under 900 px one column, never wider than the page
    const W = Math.max(320, Math.floor(el.clientWidth - 24)), stack = W < 900;
    $('.grid').style.flexDirection = stack ? 'column' : 'row';
    const LW = stack ? W : Math.round(W * 0.28), RW = stack ? W : W - LW - 12, ns = strips.length;
    const SH = ns === 1 ? 170 : ns === 2 ? 128 : 100, ST_TOP = 42, ST_GAP = 16, CB = 58;
    const TH = ns === 1 ? 150 : ns === 2 ? 104 : 80, T_TOP = 34;
    const stripsH = ns * (ST_TOP + SH) + (ns - 1) * ST_GAP + CB;
    const stH = ns * (T_TOP + TH) + (ns - 1) * 10 + 30;
    const PH = sides.length ? 146 : 0, profH = sides.length * PH + (sides.length ? 24 : 0);
    const rzH = stack ? Math.max(300, Math.min(560, Math.round(0.8 * W))) : Math.max(300, stripsH + stH - profH);
    ctx.strips = setup(cv.strips, RW, stripsH); ctx.st = setup(cv.st, RW, stH);
    ctx.rz = setup(cv.rz, LW, rzH); ctx.prof = setup(cv.prof, LW, Math.max(profH, 1));
    geo = { W, LW, RW, ns, SH, ST_TOP, ST_GAP, CB, TH, T_TOP, PH, rzH, stripsH, stH, ML: 82, MR: 10 };
    buildRZ(); buildST();
  }

  // the poloidal plane: one Path2D per cell in pixels, for the chosen view
  function buildRZ() {
    const b = D.boxes[model.get('zoom')] || D.boxes['whole domain'];
    const ml = 42, mr = 8, mt = 24, mb = 34, pw = geo.LW - ml - mr, ph = geo.rzH - mt - mb;
    const s = Math.min(pw / (b[1] - b[0]), ph / (b[3] - b[2]));
    const ox = ml + (pw - s * (b[1] - b[0])) / 2, oy = mt + (ph - s * (b[3] - b[2])) / 2;
    const X = R => ox + (R - b[0]) * s, Y = Z => oy + (b[3] - Z) * s;
    const paths = [], cen = new Float32Array(2 * nc);
    for (let c = 0; c < nc; c++) {
      const q = quads.subarray(8 * c, 8 * c + 8), p = new Path2D();
      p.moveTo(X(q[0]), Y(q[1])); p.lineTo(X(q[2]), Y(q[3])); p.lineTo(X(q[4]), Y(q[5])); p.lineTo(X(q[6]), Y(q[7]));
      p.closePath(); paths.push(p);
      cen[2 * c] = X((q[0] + q[2] + q[4] + q[6]) / 4); cen[2 * c + 1] = Y((q[1] + q[3] + q[5] + q[7]) / 4);
    }
    const lines = D.lines.map(l => { const p = new Path2D(); for (let i = 0; i < l.pts.length; i += 2)
      (i ? p.lineTo(X(l.pts[i]), Y(l.pts[i + 1])) : p.moveTo(X(l.pts[i]), Y(l.pts[i + 1]))); return { kind: l.kind, p }; });
    geo.rz = { b, X, Y, paths, cen, lines, ml, mt, mb, pw, ph, s, ox, oy, view: model.get('zoom') };
  }

  // space-time: one offscreen image per strip (x = cell along the strip, y = frame, time upward)
  function buildST() {
    const r = D.rings[ring];
    geo.st = strips.map((s, k) => {
      const off = document.createElement('canvas'); off.width = s.width; off.height = n;
      const o = off.getContext('2d'), im = o.createImageData(s.width, n), cols = r.strips[k];
      for (let f = 0; f < n; f++) for (let x = 0; x < s.width; x++) {
        const c = cols[x], ci = c < 0 ? 255 : idx[f * nc + c], px = 4 * ((n - 1 - f) * s.width + x), v = lutRGB[ci];
        im.data[px] = v[0]; im.data[px + 1] = v[1]; im.data[px + 2] = v[2]; im.data[px + 3] = 255;
      }
      o.putImageData(im, 0, 0); return off;
    });
  }

  // ----------------------------------------------------------------------------------------------- drawing
  function ci(c, i0, i1, a) {
    const v0 = idx[i0 * nc + c]; if (!a) return v0;
    const v1 = idx[i1 * nc + c]; if (v0 === 255 || v1 === 255) return a < 0.5 ? v0 : v1;
    return Math.round(v0 + (v1 - v0) * a);
  }
  function text(x, s, X, Y, o = {}) {
    x.font = `${o.bold ? 'bold ' : ''}${o.size || 11}px ${L.font}`; x.fillStyle = o.color || L.ink2;
    x.textAlign = o.align || 'left'; x.textBaseline = o.base || 'alphabetic'; x.fillText(s, X, Y);
  }
  // the first of the candidate texts that fits maxW at that size; else the last one shortened with an ellipsis
  // ('' if not even one letter fits)
  function fitText(x, candidates, maxW, size = 11) {
    x.font = `${size}px ${L.font}`;
    for (const c of candidates) if (x.measureText(c).width <= maxW) return c;
    let c = candidates[candidates.length - 1];
    while (c.length > 1 && x.measureText(c + '…').width > maxW) c = c.slice(0, -1);
    return c.length > 1 ? c + '…' : '';
  }
  function cross(x, X, Y) {
    x.lineWidth = 3; x.strokeStyle = L.surf; x.beginPath(); x.moveTo(X - 3, Y - 3); x.lineTo(X + 3, Y + 3);
    x.moveTo(X + 3, Y - 3); x.lineTo(X - 3, Y + 3); x.stroke(); x.lineWidth = 1.2; x.strokeStyle = L.ink; x.stroke();
  }

  function drawRZ(i0, i1, a) {
    const x = ctx.rz, g = geo.rz;
    x.fillStyle = L.surf; x.fillRect(0, 0, geo.LW, geo.rzH);
    for (const c of rzc) { x.fillStyle = lut[ci(c, i0, i1, a)]; x.fill(g.paths[c]); }
    for (const l of g.lines) {
      x.setLineDash(l.kind === 'sep' ? [4, 2] : []); x.globalAlpha = l.kind === 'outline' ? 0.6 : 1;
      x.lineWidth = l.kind === 'outline' ? 0.4 : l.kind === 'sep' ? 0.9 : 1.1;
      x.strokeStyle = l.kind === 'outline' ? L.muted : l.kind === 'sep' ? L.ink : L.ink2; x.stroke(l.p);
    }
    x.setLineDash([]); x.globalAlpha = 1;
    for (const c of D.bad[i0]) cross(x, g.cen[2 * c], g.cen[2 * c + 1]);
    const pr = model.get('probe');
    if (pr >= 0 && pr < nc && drawn[pr]) { x.lineWidth = 2.5; x.strokeStyle = L.surf; x.stroke(g.paths[pr]);
      x.lineWidth = 1.5; x.strokeStyle = L.accent; x.stroke(g.paths[pr]); }
    // fitted to the canvas (a narrow column, a zoomed view and a block choice ran past its edge)
    const view = g.view === 'whole domain' ? '' : g.view, note = D.rz_note || '';
    const tt = fitText(x, ['poloidal plane' + (view ? ': ' + view : '') + note, (view || 'poloidal plane') + note,
                           view ? 'poloidal plane: ' + view : 'poloidal plane', view || 'poloidal plane'],
                       geo.LW - g.ml - 4, 13);
    text(x, tt, g.ml, 16, { size: 13, color: L.ink });
    // axes: a few round R and Z values
    const b = g.b, sr = niceStep(b[1] - b[0], 4), sz = niceStep(b[3] - b[2], 6);
    for (let R = Math.ceil(b[0] / sr) * sr; R <= b[1] + 1e-9; R += sr)
      text(x, fmt(+R.toFixed(6)), g.X(R), geo.rzH - g.mb + 16, { align: 'center' });
    for (let Z = Math.ceil(b[2] / sz) * sz; Z <= b[3] + 1e-9; Z += sz)
      text(x, fmt(+Z.toFixed(6)), g.ml - 6, g.Y(Z), { align: 'right', base: 'middle' });
    text(x, 'R (m)', g.ox + g.s * (b[1] - b[0]) / 2, geo.rzH - 6, { align: 'center' });
    x.save(); x.translate(11, g.mt + g.ph / 2); x.rotate(-Math.PI / 2); text(x, 'Z (m)', 0, 0, { align: 'center' }); x.restore();
  }

  function stripArea(k) { const top = k * (geo.ST_TOP + geo.SH + geo.ST_GAP) + geo.ST_TOP;
    return { top, pw: geo.RW - geo.ML - geo.MR }; }

  function drawStrips(i0, i1, a) {
    const x = ctx.strips; x.fillStyle = L.surf; x.fillRect(0, 0, geo.RW, geo.stripsH);
    const rr = D.rings[ring].rows || {};
    strips.forEach((s, k) => {
      // reflected left-right (the strip's first cell at the right): a quarter turn lays it as the poloidal plane does
      const { top, pw } = stripArea(k), X = v => geo.ML + (1 - v / s.width) * pw, Y = v => top + (8 - v) / 8 * geo.SH;
      text(x, fitText(x, [s.title, s.title.replace('every cell, ', '')], geo.RW - geo.ML - 4, 13), geo.ML, top - 26,
           { size: 13, color: L.ink });
      for (const [xa, xb, lab] of s.segments) {   // shorter forms ('inner leg', 'leg') where the segment is narrow
        const words = lab.replace('\n', ' ').split(' '), w = Math.abs(X(xb) - X(xa)) - 4;
        const t = fitText(x, words.map((_, i) => words.slice(i).join(' ')), w);
        if (!t) continue;
        const half = x.measureText(t).width / 2, cx = Math.min(Math.max(X((xa + xb) / 2), half + 2), geo.RW - half - 2);
        text(x, t, cx, top - 7, { align: 'center' });
      }
      const R = s.rects;
      for (let r = 0; r < s.cells.length; r++) {
        const xa_ = X(R[4 * r]), xb_ = X(R[4 * r] + 1), x0 = Math.min(xa_, xb_), x1 = Math.max(xa_, xb_);
        const y1 = Y(R[4 * r + 1]), y0 = Y(R[4 * r + 1] + R[4 * r + 2]);
        x.fillStyle = lut[ci(s.cells[r], i0, i1, a)];
        if (R[4 * r + 3]) x.fillRect(x0 + 0.4, y0 + 0.4, x1 - x0 - 0.8, y1 - y0 - 0.8);
        else x.fillRect(Math.floor(x0), Math.floor(y0), Math.ceil(x1) - Math.floor(x0), Math.ceil(y1) - Math.floor(y0));
      }
      if (s.sep !== null && s.sep !== undefined) {   // the separatrix, between a strip's two bands
        x.strokeStyle = L.ink; x.lineWidth = 0.9; x.setLineDash([4, 2]);
        x.beginPath(); x.moveTo(X(0), Y(s.sep)); x.lineTo(X(s.width), Y(s.sep)); x.stroke(); x.setLineDash([]); }
      x.strokeStyle = L.muted; x.lineWidth = 0.6;
      for (const [xa] of s.segments) if (xa > 0) { x.beginPath(); x.moveTo(X(xa), Y(8)); x.lineTo(X(xa), Y(0)); x.stroke(); }
      if (s.mirror_x !== null) { x.strokeStyle = L.ink; x.lineWidth = 1.4; x.beginPath(); x.moveTo(X(s.mirror_x), Y(8));
        x.lineTo(X(s.mirror_x), Y(0)); x.stroke(); }
      // the space-time ring: its row outlined in every segment
      x.strokeStyle = L.accent; x.lineWidth = 1.2;
      for (const [xa, xb] of s.segments) { const q = rr[`${k}:${xa}`]; if (!q) continue;
        x.strokeRect(Math.min(X(xa), X(xb)), Y(q[0] + q[1]), Math.abs(X(xb) - X(xa)), Y(q[0]) - Y(q[0] + q[1])); }
      for (const [lab, v] of (s.ylabels || [['core edge', 0.4], ['core / PF', 2.2], ['separatrix', 4], ['SOL', 6]]))
        text(x, lab, geo.ML - 8, Y(v), { align: 'right', base: 'middle' });
      for (const c of D.bad[i0]) { const r = s.where[c]; if (r < 0) continue;
        cross(x, X(R[4 * r] + 0.5), Y(R[4 * r + 1] + R[4 * r + 2] / 2)); }
      const pr = model.get('probe'), q = pr >= 0 ? s.where[pr] : -1;
      if (q >= 0) { x.lineWidth = 2; x.strokeStyle = L.ink;
        x.strokeRect(Math.min(X(R[4 * q]), X(R[4 * q] + 1)) - 1, Y(R[4 * q + 1] + R[4 * q + 2]) - 1,
                     Math.abs(X(R[4 * q] + 1) - X(R[4 * q])) + 2, Y(R[4 * q + 1]) - Y(R[4 * q + 1] + R[4 * q + 2]) + 2); }
    });
    // colour bar
    const top = geo.stripsH - geo.CB + 8, pw = geo.RW - geo.ML - geo.MR, h = 12;
    for (let p = 0; p < pw; p++) { x.fillStyle = lut[Math.min(254, Math.floor(p / pw * 255))]; x.fillRect(geo.ML + p, top, 1.5, h); }
    x.fillStyle = L.ink2;
    if (D.cbar.extend[0]) { x.beginPath(); x.moveTo(geo.ML - 2, top); x.lineTo(geo.ML - 10, top + h / 2); x.lineTo(geo.ML - 2, top + h); x.fill(); }
    if (D.cbar.extend[1]) { const e = geo.ML + pw; x.beginPath(); x.moveTo(e + 2, top); x.lineTo(e + 10, top + h / 2); x.lineTo(e + 2, top + h); x.fill(); }
    for (const [f, lab] of D.cbar.ticks) text(x, lab, geo.ML + f * pw, top + h + 14, { align: 'center' });
    text(x, D.cbar.label + (D.bad.some(b => b.length) ? '   (x: n ≤ 0)' : ''), geo.ML + pw / 2, top + h + 30, { align: 'center' });
  }

  function drawST(pos) {
    const x = ctx.st; x.fillStyle = L.surf; x.fillRect(0, 0, geo.RW, geo.stH);
    const pw = geo.RW - geo.ML - geo.MR;
    strips.forEach((s, k) => {
      const top = k * (geo.T_TOP + geo.TH + 10) + geo.T_TOP, X = v => geo.ML + (1 - v / s.width) * pw;   // as the strips
      const ttl = fitText(x, [`${s.title.replace('every cell, unrolled', 'along the field')}: ${D.rings[ring].label}, ` +
                                `time upward`, `along the field: ${D.rings[ring].label}, time upward`,
                                `${D.rings[ring].label}, time upward`], geo.RW - geo.ML - 4, 12);
      text(x, ttl, geo.ML, top - 10, { size: 12, color: L.ink });
      x.imageSmoothingEnabled = false;   // the image's columns in strip order, drawn reflected as the strips
      x.save(); x.translate(geo.ML + pw, top); x.scale(-1, 1); x.drawImage(geo.st[k], 0, 0, pw, geo.TH); x.restore();
      x.strokeStyle = L.muted; x.lineWidth = 0.6;
      for (const [xa] of s.segments) if (xa > 0) { x.beginPath(); x.moveTo(X(xa), top); x.lineTo(X(xa), top + geo.TH); x.stroke(); }
      if (s.mirror_x !== null) { x.strokeStyle = L.ink; x.lineWidth = 1.4; x.beginPath(); x.moveTo(X(s.mirror_x), top);
        x.lineTo(X(s.mirror_x), top + geo.TH); x.stroke(); }
      const pc = D.rings[ring].strips[k].indexOf(model.get('probe'));
      if (pc >= 0) { x.fillStyle = L.ink; const px = X(pc + 0.5);
        x.beginPath(); x.moveTo(px - 5, top - 2); x.lineTo(px + 5, top - 2); x.lineTo(px, top + 5); x.fill(); }
      const y = top + geo.TH * (1 - (n > 1 ? pos / (n - 1) : 0));
      x.strokeStyle = L.ink; x.lineWidth = 1.2; x.beginPath(); x.moveTo(geo.ML - 4, y); x.lineTo(geo.ML + pw + 2, y); x.stroke();
      text(x, `${fmt(D.tnum[0])} ${D.tunit}`, geo.ML - 8, top + geo.TH, { align: 'right', base: 'bottom' });
      text(x, `${fmt(D.tnum[n - 1])} ${D.tunit}`, geo.ML - 8, top, { align: 'right', base: 'top' });
    });
    text(x, D.st_caption || 'cell along the strip', geo.ML + pw / 2, geo.stH - 8, { align: 'center' });
  }

  function drawProf(i0, i1, a) {
    // per side: the plot box; round y ticks (gridlines) at the left, round x ticks under it (a label that would
    // touch its neighbour or leave the box is dropped), the x title on its own row below them
    const x = ctx.prof, H = cv.prof.height / (window.devicePixelRatio || 1);
    x.fillStyle = L.surf; x.fillRect(0, 0, geo.LW, H);
    const [ylo, yhi] = D.prof.ylim, col = D.species ? L[D.species] : L.ink2, mr = 12;
    x.font = `11px ${L.font}`;
    const yt = ticks(ylo, yhi, 3), ml = Math.max(36, ...yt.map(v => x.measureText(fmt(v)).width)) + 10;
    const pw = geo.LW - ml - mr;
    sides.forEach((p, k) => {
      const top = k * geo.PH + 22, ph = geo.PH - 58, xs = p.x, x0 = xs[0], x1 = xs[xs.length - 1] || x0 + 1;
      const X = v => ml + (v - x0) / ((x1 - x0) || 1) * pw, Y = v => top + (yhi - v) / ((yhi - ylo) || 1) * ph;
      text(x, p.name, ml, top - 6, { size: 12, color: L.ink });
      x.lineWidth = 1;
      for (const v of yt) {
        x.strokeStyle = D.prof.zero && v === 0 ? L.muted : L.grid;
        x.beginPath(); x.moveTo(ml, Y(v)); x.lineTo(ml + pw, Y(v)); x.stroke();
        text(x, fmt(v), ml - 6, Y(v), { align: 'right', base: 'middle' });
      }
      x.strokeStyle = L.axis; x.strokeRect(ml, top, pw, ph);
      let right = -Infinity;
      x.font = `11px ${L.font}`;
      for (const v of ticks(x0, x1, Math.floor(pw / 70))) {
        const lab = fmt(v), w = x.measureText(lab).width, cx = X(v);
        x.strokeStyle = L.axis; x.beginPath(); x.moveTo(cx, top + ph); x.lineTo(cx, top + ph + 4); x.stroke();
        if (cx - w / 2 < right + 8 || cx - w / 2 < 0 || cx + w / 2 > geo.LW) continue;
        text(x, lab, cx, top + ph + 16, { align: 'center' });
        right = cx + w / 2;
      }
      text(x, p.xlabel, ml + pw / 2, top + ph + 32, { align: 'center' });
      if (p.sep >= x0 && p.sep <= x1) { x.strokeStyle = L.muted; x.setLineDash([4, 3]); x.beginPath();
        x.moveTo(X(p.sep), top); x.lineTo(X(p.sep), top + ph); x.stroke(); x.setLineDash([]); }
      const line = (f0, f1, w, dash, alpha) => {
        x.strokeStyle = col; x.lineWidth = w; x.setLineDash(dash); x.globalAlpha = alpha; x.beginPath(); let on = false;
        for (let r = 0; r < p.nrow; r++) { let v = p.v[f0 * p.nrow + r];
          if (f1 !== null && a) v = v + (p.v[f1 * p.nrow + r] - v) * a;
          if (!isFinite(v)) { on = false; continue; } const X_ = X(xs[r]), Y_ = Y(Math.max(ylo, Math.min(yhi, v)));
          on ? x.lineTo(X_, Y_) : x.moveTo(X_, Y_); on = true; }
        x.stroke(); x.setLineDash([]); x.globalAlpha = 1; };
      if (D.prof.first) line(0, null, 1.4, [2, 3], 0.65);
      line(i0, i1, 2, [], 1);
    });
    if (sides.length) text(x, D.prof.ylabel + (D.prof.first ? '   (dotted: first frame)' : ''), ml, H - 6, { size: 11 });
  }

  function draw() {
    const i0 = Math.min(n - 1, Math.floor(pos + 1e-9)), i1 = Math.min(n - 1, i0 + 1);
    const a = interp.checked && playing ? pos - i0 : 0;
    drawRZ(i0, i1, a); drawStrips(i0, i1, a); drawST(i0 + a); drawProf(i0, i1, a);
    scrub.value = i0;
    if (D.frames[i0] !== lastSent) {   // the tabs' live charts (LivePlot) follow, in the page
      lastSent = D.frames[i0];
      window.dispatchEvent(new CustomEvent('nstxu:frame', { detail: { frame: lastSent } }));
    }
    const t = D.tnum[i0] + (a ? (D.tnum[i1] - D.tnum[i0]) * a : 0);
    tlab.textContent = `t = ${fmt(t)} ${D.tunit}   frame ${D.frames[i0]}${a ? ' (interpolated)' : ''} of ${D.frames[0]}–${D.frames[n - 1]}`;
  }

  // ---------------------------------------------------------------------------------------------- playback
  function sync() { const k = D.frames[Math.min(n - 1, Math.floor(pos + 1e-9))];
    if (model.get('frame') !== k) { model.set('frame', k); model.save_changes(); } }
  function tick(ts) {
    if (!playing) return;
    const dt = last ? Math.min((ts - last) / 1000, 0.25) : 0; last = ts;
    pos += dt * (+fpsSel.value);
    if (pos >= n - 1) { if (loop.checked) pos -= n - 1; else { pos = n - 1; pause(); return; } }
    draw(); raf = requestAnimationFrame(tick);
  }
  function play() { if (n < 2) return; if (pos >= n - 1) pos = 0; playing = true; last = 0; btn.innerHTML = ICON.pause;
    raf = requestAnimationFrame(tick); }
  function pause() { playing = false; cancelAnimationFrame(raf); btn.innerHTML = ICON.play; pos = Math.floor(pos + 1e-9);
    draw(); sync(); }
  function step(d) { if (playing) { playing = false; cancelAnimationFrame(raf); btn.innerHTML = ICON.play; }
    pos = Math.max(0, Math.min(n - 1, Math.floor(pos + 1e-9) + d)); draw(); sync(); }

  // pick a cell; with toggle, picking the probed cell again lets it go (-1: none)
  function setProbe(c, toggle = true) {
    const cur = model.get('probe'), next = toggle && c === cur ? -1 : c;
    if (next !== cur) { model.set('probe', next); model.save_changes(); }
    draw(); }
  const dpr = () => window.devicePixelRatio || 1;
  cv.rz.onclick = e => { const g = geo.rz, x = e.offsetX * dpr(), y = e.offsetY * dpr();
    for (const c of rzc) if (ctx.rz.isPointInPath(g.paths[c], x, y)) { setProbe(c); showCell(c); return; }
    setProbe(-1, false); cmd.hide(); };   // the background: no probe
  cv.strips.onclick = e => {
    for (const [k, s] of strips.entries()) {
      const { top, pw } = stripArea(k), v = (1 - (e.offsetX - geo.ML) / pw) * s.width, y8 = 8 - (e.offsetY - top) / geo.SH * 8;
      if (y8 < 0 || y8 > 8 || v < 0 || v > s.width) continue;
      const R = s.rects;
      for (let r = 0; r < s.cells.length; r++)
        if (R[4 * r] <= v && v < R[4 * r] + 1 && R[4 * r + 1] <= y8 && y8 < R[4 * r + 1] + R[4 * r + 2]) {
          setProbe(s.cells[r]); showCell(s.cells[r]); return; }
    }
    setProbe(-1, false); cmd.hide(); };   // outside every cell: no probe
  cv.st.onclick = e => { const pw = geo.RW - geo.ML - geo.MR; strips.forEach((s, k) => {
    const top = k * (geo.T_TOP + geo.TH + 10) + geo.T_TOP,
          col = Math.min(s.width - 1, Math.floor((1 - (e.offsetX - geo.ML) / pw) * s.width));   // reflected
    if (e.offsetY < top || e.offsetY > top + geo.TH || col < 0 || col >= s.width) return;
    if (playing) { playing = false; cancelAnimationFrame(raf); btn.innerHTML = ICON.play; }
    pos = Math.round((1 - (e.offsetY - top) / geo.TH) * (n - 1)); sync();
    setProbe(D.rings[ring].strips[k][col], false);   // here a click picks a time too: it never lets the probe go
    showCell(D.rings[ring].strips[k][col]);
  }); };
  cv.prof.onclick = e => {   // a midplane panel: its profile's command at the shown frame
    const k = Math.floor((e.offsetY - 4) / geo.PH), p = sides[k];
    if (!p || !p.pg) return;
    const f = Math.min(n - 1, Math.floor(pos + 1e-9));
    cmd.show([{ name: `${p.name} at frame ${D.frames[f]}`, text: fillT(p.pg, { frame: D.frames[f] }) }], '· ' + p.name);
  };
  btn.onclick = () => (playing ? pause() : play());
  $('.prev').onclick = () => step(-1); $('.next').onclick = () => step(1);
  scrub.oninput = () => { if (playing) { playing = false; cancelAnimationFrame(raf); btn.innerHTML = ICON.play; }
    pos = +scrub.value; draw(); };
  scrub.onchange = () => sync();
  ringSel.onchange = () => { ring = +ringSel.value; buildST(); draw(); };
  el.addEventListener('keydown', e => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.key === ' ') { e.preventDefault(); playing ? pause() : play(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
    else if (e.key === 'Escape') setProbe(-1, false);
  });
  model.on('change:frame', () => { if (playing) return; const j = D.frames.indexOf(model.get('frame'));
    if (j >= 0 && j !== Math.floor(pos + 1e-9)) { pos = j; draw(); } });
  model.on('change:zoom', () => { buildRZ(); draw(); });
  model.on('change:probe', () => { if (!playing) draw(); if (model.get('probe') < 0) cmd.hide(); });
  model.on('change:look', () => { applyLook(); buildST(); draw(); });
  hint.textContent = 'Space: play / pause · arrows: step · click a cell: its time traces (the Probe tab) and its pgkyl ' +
                     'commands; click it again, the background or Esc: no probe; click the space-time strip: that cell ' +
                     'at that time; click a midplane profile: its pgkyl command · ' +
                     'one colour scale over the frames played · the notebook follows when playback pauses';

  let draws = 0; const _draw = draw;   // data-draws: frames drawn so far (the playback test reads it)
  draw = () => { _draw(); el.dataset.draws = ++draws; };
  applyLook(); layout(); draw();
  // re-lay out on a width change, on the next animation frame (not inside the observer's callback)
  let width = el.clientWidth, pending = 0;
  const ro = new ResizeObserver(() => {
    if (pending || Math.abs(el.clientWidth - width) <= 4) return;
    pending = requestAnimationFrame(() => { pending = 0; width = el.clientWidth; layout(); draw(); });
  });
  ro.observe(el);
  return () => { ro.disconnect(); cancelAnimationFrame(raf); cancelAnimationFrame(pending); playing = false; };
}
export default { render };
"""


class PlotlyLoader(anywidget.AnyWidget):
    """Puts plotly.js (the copy plotly.py ships, the version its figures are written for) into the page once, for
    the LivePlots: marimo bundles its own plotly but does not expose it. Hidden; ~5 MB, sent once per page load."""
    src = traitlets.Unicode('').tag(sync=True)
    _esm = r"""
function render({ model, el }) {
  el.style.display = 'none';
  if (window.Plotly || window.__nstxuPlotlyLoading) return;
  window.__nstxuPlotlyLoading = true;
  const s = document.createElement('script');
  s.textContent = model.get('src');   // plotly.min.js: a UMD bundle, sets window.Plotly
  document.head.appendChild(s);
  window.dispatchEvent(new Event('nstxu:plotly'));
}
export default { render };
"""


class LivePlot(anywidget.AnyWidget):
    """A plotly figure that follows the movie in the page: on each 'nstxu:frame' event (the movie draws a frame) its
    tagged traces take that frame's y and its tagged shapes that frame's position (viewer_plots.live_spec), without
    the kernel. While the movie plays it rewrites plotly's own SVG (the line paths, markers and shape lines, through
    plotly's axis maps: ~1 ms a frame); when playback stops it hands plotly the shown frame (Plotly.restyle /
    relayout), so hover and a resize agree with the drawing. Where plotly's SVG is not laid out as expected it uses
    those calls throughout, at most a third of the time. fig: the figure's JSON; frame: the notebook's frame."""
    fig = traitlets.Dict().tag(sync=True)
    live = traitlets.Dict().tag(sync=True)
    frame = traitlets.Int(-1).tag(sync=True)
    _css = CMD_CSS + """.nstxu-live { width: 100%; }
.nstxu-live .nstxu-live-note { min-height: 1.3em; font-size: 12px; opacity: 0.75; padding-left: 60px; }"""
    _esm = CMD_JS + r"""
function plotlyReady() {
  return new Promise((ok, fail) => {
    if (window.Plotly) return ok(window.Plotly);
    const t0 = Date.now(), poll = setInterval(() => {
      if (window.Plotly) { clearInterval(poll); ok(window.Plotly); }
      else if (Date.now() - t0 > 30000) { clearInterval(poll); fail(new Error('plotly.js did not load')); }
    }, 50);
  });
}
function bytes(s) { const b = atob(s), a = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a; }
const LINE = /^M([-\d.e+]+),([-\d.e+]+)L([-\d.e+]+),([-\d.e+]+)$/;

// What a frame changes in plotly's SVG, and the pixel maps: null when it is not laid out as expected
function bind(div, traces, shapes) {
  const fl = div._fullLayout;
  if (!fl) return null;
  const axis = ref => (/^[xy]\d*$/.test(ref || '') ? fl[ref[0] + 'axis' + ref.slice(1)] : null);   // 'y3' -> yaxis3
  const groups = new Map();
  for (const g of div.querySelectorAll('g.trace.scatter')) {
    const cd = g.__data__, t = cd && cd[0] && cd[0].trace;
    if (t) groups.set(t.index, [g, cd, t]);
  }
  const T = [], S = [];
  for (const tr of traces) {
    const [g, cd, t] = groups.get(tr.index) || [];
    if (!g || cd.length !== tr.n || (t.line && t.line.shape !== 'linear')) return null;
    const xa = axis(t.xaxis), ya = axis(t.yaxis);
    const line = g.querySelector('path.js-line');
    if (!xa || !ya || !xa.c2p || !ya.c2p || (!line && String(t.mode).includes('lines'))) return null;
    const xs = Float64Array.from(cd, c => xa.c2p(c.x));   // x is the same at every frame
    const pts = Array.from(g.querySelectorAll('.points path.point'), p => [p, p.__data__ && p.__data__.i]);
    if (pts.some(([, i]) => !(i >= 0 && i < tr.n))) return null;
    T.push({ tr, line, xs, ya, pts });
  }
  for (const sh of shapes) {
    const p = div.querySelector(`.shapelayer path[data-index="${sh.index}"]`), fs = fl.shapes && fl.shapes[sh.index];
    const m = p && LINE.exec(p.getAttribute('d') || ''), a = fs && axis(sh.axis === 'x' ? fs.xref : fs.yref);
    if (!m || !a || !a.c2p) return null;
    S.push({ sh, p, a, keep: sh.axis === 'x' ? [m[2], m[4]] : [m[1], m[3]] });   // the other axis's ends stay
  }
  return { T, S };
}
function drawSVG(F, j) {
  for (const { tr, line, xs, ya, pts } of F.T) {
    const y = tr.y, o = j * tr.n;
    if (line) {
      let d = '', pen = 'M';
      for (let k = 0; k < tr.n; k++) {
        const v = y[o + k];
        if (!isFinite(v)) { pen = 'M'; continue; }
        d += pen + xs[k].toFixed(2) + ',' + ya.c2p(v).toFixed(2); pen = 'L';
      }
      line.setAttribute('d', d);
    }
    for (const [p, i] of pts) {
      const v = y[o + i];
      if (isFinite(v)) { p.setAttribute('transform', `translate(${xs[i].toFixed(2)},${ya.c2p(v).toFixed(2)})`); p.style.display = ''; }
      else p.style.display = 'none';
    }
  }
  for (const { sh, p, a, keep } of F.S) {
    const v = sh.v[j];
    p.style.opacity = v === null ? 0 : sh.op;
    if (v === null) continue;
    const q = (a._offset + a.c2p(v)).toFixed(2);
    p.setAttribute('d', sh.axis === 'x' ? `M${q},${keep[0]}L${q},${keep[1]}` : `M${keep[0]},${q}L${keep[1]},${q}`);
  }
}

async function render({ model, el }) {
  // marimo renders a re-run cell's chart into the same element: the earlier one goes first
  if (el.__nstxuDone) el.__nstxuDone();
  for (const k of ['frame', 'updates', 'path', 'ms', 'error']) delete el.dataset[k];
  el.classList.add('nstxu-live');
  const div = document.createElement('div'), note = document.createElement('div');
  note.className = 'nstxu-live-note';
  el.append(div, note);
  const cmd = cmdBox(el);
  let gone = false, stop = () => {};
  const done = () => {
    if (gone) return;
    gone = true; stop();
    if (el.__nstxuDone === done) el.__nstxuDone = null;
    if (window.Plotly) window.Plotly.purge(div);
    div.remove(); note.remove(); cmd.el.remove();
  };
  el.__nstxuDone = done;
  let Plotly;
  try { Plotly = await plotlyReady(); }
  catch (e) { div.textContent = 'This chart needs plotly.js, which did not load: ' + e.message; return done; }
  if (gone) return done;
  const Fg = model.get('fig'), L = model.get('live');
  const traces = L.traces.map(t => ({ index: t.index, n: t.n, y: new Float32Array(bytes(t.y).buffer) }));
  const idxs = traces.map(t => t.index);
  const tell = j => {   // a frame running past the y axes says so (they span the movie's frames, far-out ones left out)
    note.textContent = L.over && L.over[j] ? 'This frame runs past the y axes: they span the movie\'s frames, leaving out '
      + 'the few far from the rest (pause here to see all of it).' : '';
  };
  // plotted: the frame plotly's own data holds; shown: the frame drawn; last: the frame last asked for.
  // data-frame / data-updates / data-path (svg | plotly) / data-ms / data-error: for the tests
  let plotted = model.get('frame'), shown = plotted, last = plotted, want = null;
  let F = null, sched = false, busy = false, updates = 0, nextOK = 0, idle = null;
  const within = (p, ms) => Promise.race([p, new Promise((_, no) => setTimeout(() => no(new Error('update timed out')), ms))]);
  async function plotlyTo(j, frame) {   // plotly's own data and drawing to frame j
    plotted = frame;
    if (traces.length) {
      const ys = traces.map(t => Array.from(t.y.subarray(j * t.n, (j + 1) * t.n), v => (isFinite(v) ? v : null)));
      await within(Plotly.restyle(div, { y: ys }, idxs), 3000);
    }
    const lay = {};
    for (const sh of L.shapes) {
      const v = sh.v[j], key = `shapes[${sh.index}]`;
      lay[`${key}.opacity`] = v === null ? 0 : sh.op;
      if (v !== null) { lay[`${key}.${sh.axis}0`] = v; lay[`${key}.${sh.axis}1`] = v; }
    }
    if (L.shapes.length) await within(Plotly.relayout(div, lay), 3000);
  }
  const schedule = () => { if (!sched && !busy && !gone) { sched = true; requestAnimationFrame(apply); } };
  function request(frame) { want = last = frame; schedule(); }
  async function apply() {   // on the page's next animation frame, after the movie has drawn
    sched = false;
    if (want === null || want === shown) return;
    const wait = nextOK - performance.now();
    if (!F && wait > 0) { sched = true; setTimeout(() => requestAnimationFrame(apply), wait); return; }
    const frame = want, j = L.frames.indexOf(frame);
    want = null;
    if (j < 0) return;
    const t0 = performance.now();
    try {
      if (F) drawSVG(F, j);
      else { busy = true; await plotlyTo(j, frame); }
      tell(j);
      shown = frame; el.dataset.frame = frame; el.dataset.updates = ++updates;
    } catch (e) { el.dataset.error = String((e && e.message) || e); }
    finally {
      busy = false;
      const ms = performance.now() - t0;
      el.dataset.ms = ms.toFixed(1); nextOK = performance.now() + 2 * ms;   // the plotly path: a third of the time
      if (want !== null && want !== shown) schedule();
      clearTimeout(idle); idle = setTimeout(settle, 600);
    }
  }
  async function settle() {   // playback stopped: plotly's data to the shown frame (hover, resize)
    if (gone || busy || shown === plotted) return;
    const j = L.frames.indexOf(shown);
    if (j < 0) return;
    busy = true;
    try { await plotlyTo(j, shown); } catch (e) { el.dataset.error = String((e && e.message) || e); }
    finally { busy = false; if (want !== null && want !== shown) schedule(); }
  }
  await Plotly.newPlot(div, Fg.data, Fg.layout, { responsive: true, displaylogo: false });
  if (gone) return done;
  const rebind = () => {   // every redraw by plotly (a resize, settle) draws plotly's data: bind again, redraw ours
    F = bind(div, traces, L.shapes);
    el.dataset.path = F ? 'svg' : 'plotly';
    shown = plotted;
    if (last !== shown) request(last);
  };
  rebind();
  div.on('plotly_afterplot', rebind);
  // a click on a chart: the pgkyl commands of the curves under it (traces' meta.pgkyl, pgkyl_cmd.py), at the frame
  // the chart shows; '{cmp_frame}': the overlay's frame matched to it; '{point_frame}': the clicked point's own frame
  div.on('plotly_click', ev => {
    const j = L.frames.indexOf(shown), items = [], seen = new Set();
    for (const p of (ev && ev.points) || []) {
      const tr = Fg.data[p.curveNumber], m = tr && tr.meta;
      if (!m || typeof m !== 'object' || !m.pgkyl || seen.has(p.curveNumber)) continue;
      seen.add(p.curveNumber);
      let t = String(m.pgkyl).replaceAll('{frame}', String(shown));
      if (t.includes('{cmp_frame}')) {
        const cf = L.cmp_frames && j >= 0 ? L.cmp_frames[j] : -1;
        t = cf >= 0 ? t.replaceAll('{cmp_frame}', String(cf)) : '# the overlay has no frame near this time';
      }
      if (t.includes('{point_frame}')) {
        const fl = m.pgkyl_frames, pf = fl ? fl[p.pointIndex] : null;
        t = pf != null ? t.replaceAll('{point_frame}', String(pf)) : t;
      }
      items.push({ name: String(tr.name || '').replace(/<[^>]+>/g, ''), text: t });
    }
    if (items.length) cmd.show(items, `· frame ${shown}`);
  });
  const onFrame = e => request(e.detail.frame);
  window.addEventListener('nstxu:frame', onFrame);
  model.on('change:frame', () => request(model.get('frame')));
  el.dataset.frame = shown;
  tell(L.frames.indexOf(shown));
  stop = () => { clearTimeout(idle); window.removeEventListener('nstxu:frame', onFrame); };
  return done;
}
export default { render };
"""


def live_plot(fig, summ, frame, frames=None):
    """A LivePlot of a plotly figure built by viewer_plots: its tagged parts follow the movie in the page (the y
    axes they move on fixed to their span over frames, the movie's: viewer_plots.live_spec, on a copy of fig)."""
    import json
    import plotly.graph_objects as go
    fig = go.Figure(fig)
    live = vp.live_spec(fig, summ, frames, shown=frame)
    return LivePlot(fig=json.loads(fig.to_json()), live=live, frame=int(frame))
