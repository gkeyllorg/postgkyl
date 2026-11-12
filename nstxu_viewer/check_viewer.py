#!/usr/bin/env python3
"""Checks of the viewer itself (the data layer is check_gkview.py's), with the marimo venv:

  NSTXU_VIEWER_DATA=<data dir> <viewer venv>/bin/python check_viewer.py                    headless + exports
  NSTXU_VIEWER_DATA=<data dir> <viewer venv>/bin/python check_viewer.py --browser CHROME   also the page in a browser
(the data directory's viewer.json must offer 'Sim 1', 'Sim 2' and '2025 re-run'; the checks read those runs)

1. headless: the notebook runs with every tab built (mo.ui.tabs made eager), for each simulation; a launch view
   (mo.cli_args) opens where it says; one that does not fit falls back with a note per key;
2. thomson: the reader equals the sim-init workflow's 203899 channel export; the beam flattop equals the scan's own
   loop on random traces; Sim 2's profiles carry the Thomson points on each side;
   exports: a state file round-trips and a foreign one is refused; the figure saves as png / pdf / svg (the svg names
   the theme font first); a 10-frame movie is H.264 with 10 frames;
3. --browser: ./view.sh with launch keys (OPEN=0, a fixed port) driven over the DevTools protocol: the launch frame
   and quantity, the view in the address, dark by default and the switch, the movie playing with the notebook
   untouched and the open tab's charts following it at full speed, the frame synced on pause, the page drawing a
   frame as plotly does, a click on the movie probing a cell, Save figure writing its file, a reload coming back to
   the view; then a LivePlot alone in a page (no marimo): a lambda_q line hiding where lambda_q is undefined, and
   the plotly fallback following the frames without stalling the page. Server and browser are stopped by process
   group whatever happens.
4. (2026-11-10) labels: the map figures' texts measured exactly (none over another, none off the figure); the plotly
   charts hold no text above the plots but subplot titles and the legend, the legend clear of them; pgkyl: every
   curve of every chart has its command (pgkyl_cmd.py), two of them run and give the viewer's numbers; themes:
   --theme opens a theme's look, its page CSS, the overlay's side-by-side options tinted by row.
5. (2026-11-12) the colour schemes kept in viewer_plots equal their PowerPoint files, where viewer.json's
   "theme_files" names files that are there.
Exit status: the number of failures.
"""
import argparse
import asyncio
import base64
import glob
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
fails = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        fails.append(what)


# --------------------------------------------------------------------------------------------------- headless
def headless(cli, sims_first=None):
    """Run the notebook with launch keys `cli`; returns (defs, {tab: 'ok' | 'failed'})."""
    import marimo as mo
    from marimo._runtime.params import CLIArgs
    import gkview as gv
    if sims_first is not None:
        gv.SIMS.insert(0, gv.SIMS.pop(sims_first))
    saved = mo.cli_args, mo.ui.tabs
    built = {}

    def eager(tabs, value=None, lazy=False, **kw):
        out = {}
        for name, f in tabs.items():
            r = f() if callable(f) else f
            txt = str(getattr(r, 'text', ''))
            built[name] = 'failed' if ('traceback' in txt and 'callout' in txt) else 'ok'
            out[name] = r
        return saved[1](out, value=value, **kw)

    mo.cli_args, mo.ui.tabs = (lambda: CLIArgs(cli)), eager
    try:
        spec = importlib.util.spec_from_file_location('nstxu_viewer', os.path.join(HERE, 'nstxu_viewer.py'))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _, defs = m.app.run()
    finally:
        mo.cli_args, mo.ui.tabs = saved
        if sims_first is not None:
            gv.SIMS.insert(sims_first, gv.SIMS.pop(0))
    return defs, built


def part_headless():
    import gkview as gv
    for i, (lab, _, dflt, _) in enumerate(gv.SIMS):
        t = time.time()
        d, built = headless({}, sims_first=i)
        bad = [n for n, v in built.items() if v != 'ok']
        check(d['run'].sim == lab and d['run'].name == dflt and not bad and len(built) == len(d['TABS']),
              f'{lab}: opens on {dflt}, its last frame {d["k"]}, every tab builds ({time.time() - t:.1f} s){bad or ""}')
    want = {'shot': 203899, 'frame': 15, 'qty': 'T_elc', 'mode': 'value', 'probe': '6,200,7', 'frames': '0:20',
            'tab': 'Balance & power'}
    d, built = headless(want)
    v = d['view']
    check((v['sim'], v['frame'], v['qty'], v['mode'], v['probe'], v['frames'], v['tab']) ==
          ('Sim 2', 15, 'T_elc', 'value', '6,200,7', '0:20', 'Balance & power') and len(d['movie_frames']) == 20
          and not d['boot']['notes'], f'launch keys {want} open that view (movie: {len(d["movie_frames"])} frames)')
    d, _ = headless({'sim': 'Sim 1', 'run': 'nonesuch', 'frame': 9999, 'qty': 'bogus', 'probe': '99,0,0'})
    notes = ' | '.join(d['boot']['notes'])
    check(d['run'].name == 'prod_D1_coll' and d['k'] == d['frames'][-1] and d['qty_pick'].value == 'n_elc'
          and all(w in notes for w in ('nonesuch', 'bogus', '9999', '99,0,0')),
          f'a launch view that does not fit falls back, a note per key: {notes}')
    # 2026-11-10: blocks, the overlay, the Diffusivity and Options tabs, a single-block run opened with --path
    d, built = headless({'sim': 'Sim 2', 'blocks': '1-3', 'cmp': '2025 re-run/prod', 'tab': 'Diffusivity'})
    bad = [n for n, v in built.items() if v != 'ok']
    check(d['blocks'] == {1, 2, 3} and d['cmp'] is not None and d['cmp'].label == '2025 re-run · prod' and
          d['cmp'].k == 15 and not bad and not d['boot']['notes'] and d['view']['blocks'] == '1-3' and
          d['view']['cmp'] == '2025 re-run/prod' and {'Diffusivity', 'Options'} <= set(built),
          f'blocks 1-3 with the 2025 re-run over Sim 2: the overlay at its frame {d["cmp"].k if d["cmp"] else None} '
          f'(300 us), every tab builds{bad or ""}, the view keeps both')
    d, _ = headless({'sim': 'Sim 1', 'blocks': '99', 'cmp': 'nonesuch/run'})
    notes = ' | '.join(d['boot']['notes'])
    check(d['blocks'] is None and d['cmp'] is None and 'blocks' in notes and 'nonesuch/run' in notes,
          f'blocks and an overlay that do not fit fall back to none, with notes: {notes}')
    sim2 = gv.open_run(os.path.join(gv.NSTXU, 'sims_20261022/sim2_dense_rho09/runs/prod_th02q4_coll'))
    with tempfile.TemporaryDirectory() as tmp:
        sb = os.path.join(tmp, 'single_b2')
        os.makedirs(sb)
        for f in glob.glob(f'{sim2.path}/{sim2.stem}_b2-*.gkyl') + [f'{sim2.path}/{sim2.stem}-dt.gkyl']:
            os.symlink(f, os.path.join(sb, os.path.basename(f).replace('_b2-', '-')))
        saved, gv.CACHE_DIR = gv.CACHE_DIR, os.path.join(tmp, 'cache')
        try:
            d, built = headless({'path': sb, 'cmp': 'Sim 2/prod_th02q4_coll', 'probe': '0,10,3'})
        finally:
            gv.CACHE_DIR = saved
        bad = [n for n, v in built.items() if v != 'ok']
        check(d['run'].single and d['run'].path == sb and d['cmp'] is not None and d['cmp'].k == d['k'] and not bad
              and not d['boot']['notes'], f'--path: a single-block run (Sim 2 block 2 under single-block names) opens '
              f'with Sim 2 over it, every tab builds{bad or ""}')


# ---------------------------------------------------------------------------------------------------- thomson
def part_thomson():
    import csv
    import gkview as gv
    import thomson
    import viewer_plots as vp
    # the reader against the sim-init workflow's own export of 203899's channels at its TS frame nearest 325 ms
    th = thomson.read(203899)
    ref = os.path.join(gv.NSTXU, 'nstxu_sim_init_workflow', 'reference_run', 'shot203899_ms325',
                       'mpts_fit_channels_exact_geqdsk_rho.csv')
    rows = list(csv.DictReader(open(ref)))
    i = int(np.argmin(np.abs(th.t - 0.325)))
    R = np.array([float(r['RADIUS_M']) for r in rows])
    ne = np.array([float(r['FIT_NE_CM3']) if float(r['VALID']) == 1 else np.nan for r in rows])
    rel = np.nanmax(np.abs(th.ne[:, i] * 1e-6 / ne - 1))
    check(th.branch == 'TS3' and np.array_equal(R, th.R) and np.array_equal(np.isnan(ne), np.isnan(th.ne[:, i]))
          and rel < 1e-14, f'Thomson reader == the workflow\'s 203899 channel export ({th.branch}, frame {i}, '
          f'max rel {rel:.1e})')
    # the flattop rule against the scan's own loop (upstream/elong_tri_range_scan.py), on beam traces with steps,
    # notches and noise
    rng = np.random.default_rng(0)
    agree = 0
    for _ in range(200):
        t = np.linspace(0, 1, 400)
        p = np.clip(rng.uniform(1, 6) * ((t > rng.uniform(0.05, 0.3)) & (t < rng.uniform(0.5, 0.95))), 0, None)
        p[rng.integers(0, 400, 3)] -= rng.uniform(0, 1, 3)
        p += rng.normal(0, rng.uniform(0, 0.12), 400)
        sel = [i for i in range(len(p)) if p[i] >= np.max(p) - 0.2 and 15 <= i < len(p) - 15
               and all(abs(p[j] - p[i]) <= 0.2 for j in range(i - 15, i + 16))]
        want = (t[sel].min(), t[sel].max()) if sel else None
        agree += thomson.flattop(t, p) == want
    check(agree == 200, f'beam flattop == the scan\'s loop on {agree}/200 random traces')
    # Sim 2 (203899) draws the Thomson points on its density row, inside the run's span on each side
    lab, d, dflt, _ = gv.SIMS[[s[0] for s in gv.SIMS].index('Sim 2')]
    run = gv.open_run(os.path.join(gv.NSTXU, d, 'runs', dflt), lab)
    fs = run.frames()
    fr = gv.frame(run, fs[-1])
    t0, t1 = thomson.default_window(th, run.eq_ms / 1e3)
    fig = vp.profiles_figure(run, fr, gv.frame(run, fs[0]), '', 'distance', measured=(th, t0, t1))
    pts = {tr.xaxis: len(tr.x) for tr in fig.data if tr.legendgroup == 'thomson'}
    check(pts.get('x', 0) > 0 and len(pts) == len(run.layout.stacks),
          f'Sim 2 profiles: Thomson points on each side ({pts})')


# ---------------------------------------------------------------------------------------------------- exports
def part_exports():
    import imageio_ffmpeg
    import gkview as gv
    import viewer_plots as vp
    import viewer_state as vs
    tmp = tempfile.mkdtemp(prefix='check_viewer_')
    view = {'sim': 'Sim 2', 'run': 'prod_th02q4_coll', 'frame': 15, 'qty': 'T_elc', 'frames': '0:20'}
    check(vs.load_state(vs.save_state(f'{tmp}/v.json', view)) == view, 'a state file round-trips')
    json.dump({'other': 1}, open(f'{tmp}/bad.json', 'w'))
    try:
        vs.load_state(f'{tmp}/bad.json')
        check(False, 'a foreign json is refused as a state file')
    except ValueError:
        check(True, 'a foreign json is refused as a state file')
    r = gv.open_run(f'{gv.NSTXU}/sims_20261022/sim2_dense_rho09/runs/prod_th02q4_coll', 'Sim 2')
    summ = gv.summary(r)
    for ext in ('png', 'pdf', 'svg'):
        fig = vp.map_figure(r, gv.frame(r, 30), gv.frame(r, 0), 'T_elc', 'value', pal=vp.LOOKS['light'])
        path = vp.save_figure(fig, f'{tmp}/fig.{ext}')
        vp.plt.close(fig)
        font = ext != 'svg' or vp.RENDER_FONT == vp.FONT or f"'{vp.FONT}', '{vp.RENDER_FONT}'" in open(path).read()
        check(os.path.getsize(path) > 10000 and font, f'figure saves as {ext} ({os.path.getsize(path)} bytes)'
              + (', the theme font named first' if ext == 'svg' else ''))
    path = vp.render_movie(r, summ, list(range(10)), 'n_elc', 'step', False, 'whole domain', vp.LOOKS['dark'],
                           f'{tmp}/m.mp4', fps=10)
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-i', path, '-map', '0:v', '-f', 'null', '-'],
                          capture_output=True, text=True).stderr
    nfr = re.findall(r'frame=\s*(\d+)', info)
    check('h264' in info and nfr and int(nfr[-1]) == 10, f'a 10-frame movie is H.264 with {nfr[-1] if nfr else "?"} '
          'frames')
    # 2026-11-10: a block choice and the diffusivity map save too; the movie's data for a block choice draws only
    # those blocks' cells, on their own colour scale; a state file keeps the new keys
    import movie_widget as mm
    fig = vp.map_figure(r, gv.frame(r, 30), gv.frame(r, 0), 'n_elc', 'value', pal=vp.LOOKS['dark'], blocks={1, 2, 3})
    p1 = vp.save_figure(fig, f'{tmp}/blocks.png'); vp.plt.close(fig)
    fig = vp.diffusivity_map(r, gv.diffusivity(r), pal=vp.LOOKS['dark'])
    p2 = vp.save_figure(fig, f'{tmp}/D.png'); vp.plt.close(fig)
    D = mm.movie_data(r, summ, 'n_elc', 'value', blocks={1, 2, 3})
    rz = np.frombuffer(base64.b64decode(D['rz']), np.int32)
    want = gv.region_cells(r, [1, 2, 3])
    in_strips = np.concatenate([np.frombuffer(base64.b64decode(st['cells']), np.int32) for st in D['strips']])
    check(os.path.getsize(p1) > 10000 and os.path.getsize(p2) > 10000 and np.array_equal(rz, want) and
          set(in_strips) == set(want) and all(st['sep'] == 4.0 for st in D['strips']),
          f'blocks 1-3: the figure and the D map save; the movie draws their {len(want)} cells (plane and strips) only')
    view2 = dict(view, blocks='1-3', cmp='2025 re-run/prod', path='')
    check(vs.load_state(vs.save_state(f'{tmp}/v2.json', view2)) == view2 and
          vs.parse_blocks('1-3,7', 8) == {1, 2, 3, 7} and vs.blocks_text({1, 2, 3, 7}) == '1-3,7',
          'a state file keeps blocks and the overlay; blocks text round-trips (1-3,7)')


# ----------------------------------------------------------------------------------- labels, pgkyl, themes (11-10)
def _charts(run, summ, cmp=None):
    """Every plotly chart the tabs draw for a run, at its last frame (with an overlay where given)."""
    import gkview as gv
    import thomson
    import viewer_plots as vp
    fs = run.frames()
    k = fs[-1]
    fr, fr0, t = gv.frame(run, k), gv.frame(run, fs[0]), run.time(k)
    out = {'profiles': vp.profiles_figure(run, fr, fr0, '', cmp=cmp),
           'history': vp.history_figure(run, summ, f'n_{run.species[0]}', 'change', t),
           'targets': vp.targets_figure(run, fr, summ, t, cmp=cmp),
           'time traces': vp.timetraces_figure(run, summ, gv.dt_history(run), gv.integrated(run), gv.field_energy(run),
                                               t, cmp=cmp, blocks={0, 1}),
           'probe': vp.probe_figure(run, summ, gv.cell_offsets(run)[1] + 2, t, cmp=cmp)}
    B = gv.balance(run)
    if B is not None:
        out['balance'] = vp.balance_figure(run, B, t, cmp=cmp, B_cmp=gv.balance(cmp.run) if cmp else None)[0]
    D = gv.diffusivity(run)
    if D['kind']:
        out['diffusivity'] = vp.diffusivity_figure(run, D, cmp=cmp, D_cmp=gv.diffusivity(cmp.run) if cmp else None)
    try:
        th = thomson.read(run.shot)
        t0, t1 = thomson.default_window(th, run.eq_ms / 1e3)
        out['thomson'] = vp.thomson_timeline(th, t0, t1, run.eq_ms / 1e3)
    except (OSError, KeyError, ValueError):
        pass
    return out


def part_labels():
    """No text overlaps: the matplotlib maps measured exactly (every drawn text's box, Agg); the plotly charts by
    construction (no text but the subplot titles and the legend above the plots, the legend clear of the titles)."""
    import itertools
    from matplotlib.text import Text
    import gkview as gv
    import viewer_plots as vp
    bad, n = [], 0
    for lab, d, dflt, _ in gv.SIMS:
        run = gv.open_run(os.path.join(gv.runs_root(d), dflt), lab)
        fs = run.frames()
        fr, fr0 = gv.frame(run, fs[-1]), gv.frame(run, fs[0])
        sol = [b for b in range(run.nb) if run.layout.region[b] == 'sol'][:2]
        figs = [('map', dict()), ('titled', dict(title=f'{run.title} · {run.sim} · {run.name} · a quantity   t = 1.234 '
                                                       'ms   (frame 123)')), ('blocks', dict(blocks=set(sol)))]
        figs += [(z, dict(zoom=z)) for z in vp.zoom_boxes(run) if z != 'whole domain']
        for name, kw in figs:
            fig = vp.map_figure(run, fr, fr0, 'q_par', 'value', **kw)
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            skip = {id(lb) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)
                    for tk in axis.get_major_ticks() + axis.get_minor_ticks() for lb in (tk.label1, tk.label2)
                    if not min(axis.get_view_interval()) - 1e-9 <= tk.get_loc() <= max(axis.get_view_interval()) + 1e-9}
            boxes = [(t.get_text(), t.get_window_extent(r)) for t in fig.findobj(Text)
                     if t.get_visible() and t.get_text().strip() and id(t) not in skip]
            W, H = fig.bbox.width, fig.bbox.height
            for (ta, a), (tb, b) in itertools.combinations(boxes, 2):
                if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                    bad.append(f'{lab} {name}: {ta[:30]!r} over {tb[:30]!r}')
            bad += [f'{lab} {name}: {t[:30]!r} off the figure' for t, b in boxes
                    if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1]
            n += len(boxes)
            vp.plt.close(fig)
    check(not bad, f'map figures: {n} texts over 3 simulations (whole, zoomed, titled, blocks), none over another '
          f'or off the figure{bad[:6] or ""}')
    bad, nfig = [], 0
    for lab, d, dflt, _ in gv.SIMS:
        run = gv.open_run(os.path.join(gv.runs_root(d), dflt), lab)
        for name, fig in _charts(run, gv.summary(run)).items():
            nfig += 1
            h = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
            for a in fig.layout.annotations:
                if a.yref == 'paper' and not (a.yanchor == 'bottom' and a.font.size == vp.SUBTITLE):
                    bad.append(f'{lab} {name}: a free note {a.text[:30]!r}')
            titled = any(a.yref == 'paper' and a.font.size == vp.SUBTITLE for a in fig.layout.annotations)
            if titled and (fig.layout.legend.y - 1) * h < vp.SUBTITLE * 1.3 + 6:
                bad.append(f'{lab} {name}: legend {(fig.layout.legend.y - 1) * h:.0f} px above the plots')
    check(not bad, f'{nfig} charts: above the plots only subplot titles and the legend, the legend clear of the '
          f'titles{bad[:6] or ""}')


def part_pgkyl():
    """Every curve has its pgkyl command; two of them run and give the viewer's numbers."""
    import gkview as gv
    import movie_widget as mm
    import pgkyl_cmd as pc
    bad, ncurve = [], 0
    runs = [gv.open_run(os.path.join(gv.runs_root(d), dflt), lab) for lab, d, dflt, _ in gv.SIMS]
    import viewer_plots as vp
    for n, run in enumerate(runs):
        summ = gv.summary(run)
        other = runs[(n + 1) % len(runs)]
        s2 = gv.summary(other)
        idx = gv.match_frames(summ['t'], s2['t'])
        j = int(idx[-1])
        cmp = vp.Cmp(other, s2, other.name, fr=gv.frame(other, int(s2['frames'][j])) if j >= 0 else None,
                     fr0=gv.frame(other, int(s2['frames'][0])), k=int(s2['frames'][j]) if j >= 0 else -1,
                     hist=(gv.dt_history(other), gv.integrated(other), gv.field_energy(other)), idx=idx)
        for name, fig in _charts(run, summ, cmp).items():
            for tr in fig.data:
                if tr.legendgroup == 'key' or tr.x is None or all(v is None for v in tr.x):   # legend entries only
                    continue
                ncurve += 1
                m = tr.meta if isinstance(tr.meta, dict) else {}
                cmd = m.get('pgkyl')
                if not cmd or not isinstance(cmd, str):
                    bad.append(f'{run.sim} {name}: {tr.name!r} has no command')
                    continue
                left = set(re.findall(r'\{(\w+)\}', cmd)) - {'frame', 'cmp_frame', 'point_frame'}
                if left:
                    bad.append(f'{run.sim} {name}: {tr.name!r} leaves {left}')
                if '{point_frame}' in cmd and len(m.get('pgkyl_frames') or []) != len(tr.x):
                    bad.append(f'{run.sim} {name}: {tr.name!r} has no frame for each point')
        D = mm.movie_data(run, summ, 'T_elc', 'change')
        if not all(x in D['pg']['map'] for x in ('{b}', '{frame}', '{ref}')) or \
                not all(x in D['pg']['cell'] for x in ('{b}', '{i}', '{j}', '{frame}')):
            bad.append(f'{run.sim}: the movie\'s cell templates')
    check(not bad, f'{ncurve} curves over every chart of 3 simulations (overlays too): each has its pgkyl command, '
          f'every placeholder known{bad[:6] or ""}')
    # commands run against the viewer's numbers (each result saved as .npy): a cell's T_e, a midplane profile, the
    # cell's T_e at every frame, the electrons' number; every other kind of command runs to a saved figure
    run = runs[1]
    fr = gv.frame(run, 20)
    b = [bb for bb in range(run.nb) if run.layout.region[bb] == 'sol'][0]
    i, j = min(1, run.nodes[b].shape[0] - 2), min(3, run.nodes[b].shape[1] - 2)
    side = next(iter(run.layout.stacks))
    st = run.layout.stacks[side]
    env = dict(os.environ, MPLBACKEND='Agg')

    def bare(cmd):
        return '\n'.join(x for x in cmd.split('\n') if not x.startswith('#'))

    def results(cmd, tmp, nm):
        """The command's results in order, its plot / print step replaced by saving each (pgkyl's save writes one
        file a call); None when pgkyl fails."""
        steps = bare(cmd).split(' \\\n  ')
        while steps[-1].startswith(('plot', 'print', 'activate')):
            steps.pop()
        outs = re.findall(r'evaluate -t (\S+)', bare(cmd))
        steps += [x for o in outs for x in (f'activate -t {o}', f'save -e npy -o {tmp}/{nm}_{o}')]
        p = subprocess.run(['bash', '-c', ' \\\n  '.join(steps)], capture_output=True, text=True, env=env)
        return [np.load(f'{tmp}/{nm}_{o}.npy') for o in outs] if not p.returncode else None

    with tempfile.TemporaryDirectory() as tmp:
        got = [results(pc.cell_value(run, b, run.file_row(b, i), j, 'T_elc', 20), tmp, 'cell'),
               results(pc.midplane(run, side, 'n_ion', 20), tmp, 'mid'),
               results(pc.cell_history(run, b, run.file_row(b, i), j, 'T_elc'), tmp, 'hist'),
               results(pc.particles(run, 'elc'), tmp, 'parts')]
        want_mid = np.concatenate([run.rows(fr[bb]['n_ion'])[:, pc._mid_column(run, bb, st.j)] * 1e-19
                                   for bb in gv.stack_blocks(st)])
        want_hist = [gv.frame(run, kk)[b]['T_elc'][i, j] for kk in run.frames()]
        want_parts = sum(gv.read_dynvec(run.f(bb, 'elc_integrated_moms'))[1][:, 0] for bb in range(run.nb)
                         if os.path.exists(run.f(bb, 'elc_integrated_moms')))
        ok = (None not in got and
              np.isclose(float(np.ravel(got[0][0])[0]), fr[b]['T_elc'][i, j], rtol=1e-12) and
              np.allclose(np.concatenate([np.ravel(g) for g in got[1]]), want_mid, rtol=1e-12) and
              np.allclose(np.ravel(got[2][0]), want_hist, rtol=1e-12) and
              np.allclose(np.ravel(got[3][0]), want_parts, rtol=1e-12))
        check(ok, f'pgkyl runs four of them (Sim 2): a cell\'s T_e at frame 20 and at every frame, the {side} midplane '
              'n_i, the electrons\' number: the viewer\'s numbers (rtol 1e-12)')
        tg = run.layout.targets[0]
        figs = {'block map': pc.block_map(run, b, 'n_elc', 20), 'change map': pc.block_map(run, b, 'T_elc', 20, 'change', 0),
                'midplane history': pc.midplane(run, side, 'n_ion', history=True), 'plate': pc.plate(run, tg, 20),
                'dt': pc.dt(run), 'field energy': pc.field_energy(run), 'target power': pc.target_power(run, tg),
                'particle balance': pc.balance(run), 'energy balance': pc.balance(run, True),
                'D map': pc.diffusivity(run, b, 'elc'), 'midplane D': pc.diffusivity(run, None, 'elc', side)}
        failed = []
        for nm, cmd in figs.items():
            png = f'{tmp}/{nm.replace(" ", "_")}.png'
            s = re.sub(r'(\b(?:plot|gk_\w+_balance)\b[^\\\n]*)', lambda m: m.group(1) + (
                ' --no_show' if m.group(1).startswith('plot') else '') + f' --saveas {png}', bare(cmd or ''))
            p = subprocess.run(['bash', '-c', s], capture_output=True, text=True, env=env)
            if not cmd or p.returncode or not os.path.exists(png):
                failed.append(f'{nm}: {p.stderr.strip()[-200:] if cmd else "no command"}')
        check(not failed, f'every other kind of pgkyl command runs to a figure ({len(figs)} kinds){failed or ""}')


def part_themes():
    """Each theme opens from its launch key, in the mode asked; its page CSS has its fonts and surfaces; the overlay's
    side-by-side options are tinted by row, whatever the row order."""
    import marimo as mo
    import gkview as gv
    import viewer_plots as vp
    import viewer_state as vs
    got = {}
    for theme, dark in (('PPPL', 'false'), ('diii-d', 'true')):
        d, built = headless({'sim': 'Sim 2', 'theme': theme, 'dark': dark})
        got[theme] = (d['pal'].name, d['view']['theme'], all(v == 'ok' for v in built.values()))
    check(got == {'PPPL': ('pppl-light', 'PPPL', True), 'diii-d': ('diii-d-dark', 'DIII-D', True)} and
          'theme' in vs.KEYS, f'--theme PPPL --dark false / --theme diii-d open those looks, every tab builds: {got}')
    files = {t: f for t, f in (gv.CONFIG.get('theme_files') or {}).items()}
    read = {t: vp.read_theme(os.path.join(gv.NSTXU, f)) for t, f in files.items() if os.path.exists(os.path.join(gv.NSTXU, f))}
    navy = dict(vp.SCHEMES['DIII-D'][0])
    navy.pop('navy')
    kept = {t: (navy, vp.SCHEMES[t][1]) if t == 'DIII-D' else vp.SCHEMES[t] for t in read}
    check(set(files) <= set(vp.SCHEMES) and kept == read and
          all(os.path.basename(f) == vp.THEME_SOURCES[t] for t, f in files.items()),
          f'the themes kept in viewer_plots == their PowerPoint files ({", ".join(sorted(read)) or "none here"})')
    css = {t: vp.page_css(t) for t in vp.THEME_LOOKS}
    check(set(css) == {'NSTX-U', 'PPPL', 'DIII-D'} and 'Roboto Condensed' in css['PPPL'] and
          'Century Gothic' in css['DIII-D'] and vp.LOOKS['pppl-dark'].surf in css['PPPL'] and
          vp.LOOKS['diii-d-dark'].surf in css['DIII-D'] and all('--background' in c for c in css.values()),
          'each theme\'s page CSS: its fonts (PPPL Roboto Condensed titles, DIII-D Century Gothic) and dark surface')
    pal = vp.LOOKS['pppl-dark']
    rows = [dict(compare=c, option=f'o{i}', this_run='1', overlay='1' if c == 'same' else '2', set_by='')
            for i, c in enumerate(['differs', 'same', 'same', 'differs', 'same'])]
    tint = vp._compare_tints(pal)
    ok = True
    for order in (rows, rows[::-1]):
        t = mo.ui.table(order, selection=None, style_cell=vp.compare_style(pal))
        styles = t._style_cells(0, len(order), len(order))
        ok &= all(styles[str(i)][col].get('backgroundColor') == tint[r['compare']]
                  for i, r in enumerate(order) for col in r)
    check(ok, 'options side by side: every cell of a row tinted by its compare value (differs / same), either order')


# ---------------------------------------------------------------------------------------------------- browser
DEEP = """window.__deep = s => { const o = []; const w = r => { r.querySelectorAll(s).forEach(e => o.push(e));
  r.querySelectorAll('*').forEach(e => { if (e.shadowRoot) w(e.shadowRoot); }); }; w(document); return o; };
window.__frame = () => { const m = (document.body.innerText || '').match(/frame (\\d+) of/); if (m) return +m[1];
  for (const e of document.querySelectorAll('*')) if (e.shadowRoot) { const k = (e.shadowRoot.textContent || '')
  .match(/frame (\\d+) of/); if (k) return +k[1]; } return null; };
window.__switch = t => __deep('button[role=switch]').find(s => { let n = s; for (let i = 0; i < 4 && n; i++) {
  n = n.parentElement || (n.getRootNode() && n.getRootNode().host); if (n && (n.textContent || '').includes(t))
  return true; } return false; });
window.__button = t => __deep('button').find(b => (b.textContent || '').trim() === t);
window.__tab = t => __deep('[role=tab]').find(b => (b.textContent || '').trim() === t);
'ok'"""


# every live chart's drawn curves (vertices), markers and shape lines; NEW: only charts drawn since MARK_OLD (null
# until they are all up), else every chart on the page now (marked old)
DRAWN = r"""(new_ => {
  const plot = e => e.querySelector('.js-plotly-plot');   // a redraw: the same element, a new plot in it
  const els = __deep('.nstxu-live').filter(e => plot(e) && (new_ ? !plot(e).__old : true));
  if (new_ && !(els.length && els.every(e => plot(e).querySelector('svg.main-svg')))) return null;
  const nums = d => (d || '').match(/-?[\d.]+(e[-+]?\d+)?/g) || [];
  const out = {frames: els.map(e => +e.dataset.frame), charts: [], n: 0};
  for (const e of els) {
    const div = plot(e), lines = {}, points = {}, shapes = {};
    if (!new_) div.__old = true;
    for (const g of div.querySelectorAll('.scatterlayer g.trace.scatter')) {
      const t = g.__data__ && g.__data__[0] && g.__data__[0].trace, p = g.querySelector('path.js-line');
      if (!t) continue;
      if (p) { lines[t.index] = nums(p.getAttribute('d')).map(Number); out.n++; }
      points[t.index] = Array.from(g.querySelectorAll('.points path.point'), q => nums(q.getAttribute('transform')).map(Number));
    }
    for (const p of div.querySelectorAll('.shapelayer path[data-index]'))
      shapes[p.getAttribute('data-index')] = [nums(p.getAttribute('d')).map(Number), +(p.style.opacity || getComputedStyle(p).opacity)];
    out.charts.push({lines, points, shapes});
  }
  return out;
})(NEW)"""


MARK_OLD = "__deep('.nstxu-live .js-plotly-plot').forEach(d => { d.__old = true; })"


def drawn_diff(a, b):
    """The largest pixel distance between two DRAWN readings of the same frame: each of b's curve vertices to a's
    (plotly drops collinear vertices), markers and visible shape lines point for point. None: they do not pair up."""
    if not (a and b and len(a['charts']) == len(b['charts'])):
        return None
    worst = 0.0
    for ca, cb in zip(a['charts'], b['charts']):
        if set(ca['lines']) != set(cb['lines']) or set(ca['shapes']) != set(cb['shapes']):
            return None
        for k, vb in cb['lines'].items():
            pa, pb = np.reshape(ca['lines'][k], (-1, 2)), np.reshape(vb, (-1, 2))
            if len(pb) and not len(pa):
                return None
            for x, y in pb:   # the nearest of a's vertices
                worst = max(worst, float(np.min(np.hypot(pa[:, 0] - x, pa[:, 1] - y))) if len(pa) else 0.0)
        for k, qb in cb['points'].items():
            qa = ca['points'].get(k, [])
            if len(qa) != len(qb):
                return None
            for u, v in zip(qa, qb):
                worst = max([worst] + [abs(p - q) for p, q in zip(u, v)])
        for k, (sb, ob) in cb['shapes'].items():
            sa, oa = ca['shapes'][k]
            if (oa > 0) != (ob > 0) or (ob > 0 and len(sa) != len(sb)):
                return None
            if ob > 0:
                worst = max([worst] + [abs(p - q) for p, q in zip(sa, sb)])
    return worst


class CDP:
    def __init__(self, ws):
        self.ws, self.i = ws, 0

    async def call(self, method, **params):
        self.i += 1
        mid = self.i
        await self.ws.send(json.dumps({'id': mid, 'method': method, 'params': params}))

        async def wait():
            while True:
                msg = json.loads(await self.ws.recv())
                if msg.get('id') == mid:
                    return msg.get('result', msg)
        try:
            return await asyncio.wait_for(wait(), 30)
        except asyncio.TimeoutError:
            return {}

    async def js(self, expr):
        r = await self.call('Runtime.evaluate', expression=expr, returnByValue=True, awaitPromise=True)
        return r.get('result', {}).get('value')

    async def until(self, expr, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = await self.js(expr)
            if v not in (None, False, '', 0):
                return v
            await asyncio.sleep(0.5)
        return None

    async def press(self, expr):
        """A real mouse click (press + release) at the centre of the element expr finds: Radix tabs activate on the
        pointer press, which a scripted element.click() does not send. False when there is no such element."""
        if not await self.js(f"(e => {{ if (!e) return false; e.scrollIntoView({{block: 'center'}}); return true; }})({expr})"):
            return False
        await asyncio.sleep(0.5)
        box = await self.js(f"(e => e ? (r => [r.left + r.width / 2, r.top + r.height / 2])(e.getBoundingClientRect()) "
                            f": null)({expr})")
        if not box:
            return False
        for kind in ('mousePressed', 'mouseReleased'):
            await self.call('Input.dispatchMouseEvent', type=kind, x=box[0], y=box[1], button='left', clickCount=1)
        return True

    async def shot(self, path):
        r = await self.call('Page.captureScreenshot', format='png')
        if 'data' in r:
            open(path, 'wb').write(base64.b64decode(r['data']))


async def drive(url, dbg, shots):
    import websockets
    info = None
    for _ in range(100):
        try:
            info = json.load(urllib.request.urlopen(f'http://127.0.0.1:{dbg}/json/list'))
            break
        except OSError:
            await asyncio.sleep(0.2)
    ws_url = [t for t in info if t['type'] == 'page'][0]['webSocketDebuggerUrl']
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        c = CDP(ws)
        await c.call('Page.enable')
        await c.call('Page.navigate', url=url)
        await asyncio.sleep(3)
        await c.js(DEEP)
        k = await c.until('window.__frame && __frame() !== null && __frame()', 120)
        await c.until("__deep('.nstxu-movie')[0] && +(__deep('.nstxu-movie')[0].dataset.draws || 0) > 0", 120)
        await asyncio.sleep(3)
        await c.js(DEEP)
        qs = await c.js('location.search')
        check(k == 20 and 'qty=T_elc' in qs and 'frame=20' in qs, f'launch keys open frame {k}, T_e; the address '
              f'holds the view ({qs[:120]}...)')
        dark = await c.js("document.body.classList.contains('dark') && getComputedStyle(document.body).backgroundColor")
        check(bool(dark) and '0.105882' in str(dark), f'dark by default on the theme slate ({dark})')
        await c.js("__switch('dark mode').click()")
        light = await c.until("!document.body.classList.contains('dark')", 15)
        await c.js("__switch('dark mode').click()")
        back = await c.until("document.body.classList.contains('dark')", 15)
        check(bool(light) and bool(back), 'the dark-mode switch goes light and back, live')
        W = "__deep('.nstxu-movie')[0]"
        await c.js(f"(s => {{ s.value = '30'; s.dispatchEvent(new Event('change')); }})({W}.querySelector('.fps'))")
        # the tab's charts render once scrolled into view (marimo); the movie keeps playing off screen
        await c.js("(__deep('[role=tablist]')[0] || document.body).scrollIntoView({block: 'start'})")
        await c.until("!!window.Plotly && __deep('.nstxu-live svg.main-svg').length > 0", 120)
        await asyncio.sleep(2)
        LIVE = "__deep('.nstxu-live').map(e => +e.dataset.frame)"
        lv0, nplots = await c.js(LIVE), await c.js("__deep('.nstxu-live svg.main-svg').length")
        d0, f0 = await c.js(f'+{W}.dataset.draws'), await c.js('__frame()')
        await c.js(f"{W}.querySelector('button.play').click()")
        t0 = time.time()
        await asyncio.sleep(3)
        d1, f1, lab = await c.js(f'+{W}.dataset.draws'), await c.js('__frame()'), await c.js(f"{W}.querySelector('.time').textContent")
        elapsed = time.time() - t0
        lv1, how = await c.js(LIVE), await c.js("__deep('.nstxu-live').map(e => [e.dataset.path, +e.dataset.ms, e.dataset.error || ''])")
        rate = (d1 - d0) / elapsed
        if shots:
            await c.shot(os.path.join(shots, 'playing.png'))
        await c.js(MARK_OLD)   # the charts drawn so far (the notebook redraws the tab once playback pauses)
        await c.js(f"{W}.querySelector('button.play').click()")
        stop = await c.js(f"{W}.querySelector('.time').textContent")
        frame_of = lambda text: int(re.search(r'frame (\d+)', text).group(1)) if text and re.search(r'frame (\d+)', text) else None
        mf, ms = frame_of(lab), frame_of(stop)
        synced = await c.until(f"__frame() === {ms}", 30)
        await asyncio.sleep(2)
        paused, still = await c.js('__frame()'), frame_of(await c.js(f"{W}.querySelector('.time').textContent"))
        check(bool(lv0) and nplots > 0 and all(b != a for a, b in zip(lv0, lv1)) and mf is not None
              and all(abs(b - mf) <= 3 for b in lv1) and all(h[0] == 'svg' and not h[2] for h in how),
              f'the open tab\'s {len(lv1 or [])} live charts follow the movie in the page (frames {lv0} -> {lv1}; '
              f'movie at {mf}; path, ms a frame: {[h[:2] for h in how]})')
        frames_played = (mf - f0) if mf is not None and f0 is not None else 0
        check(rate > 15 and frames_played >= 0.5 * 30 * elapsed and f1 == f0 and mf != f0 and bool(synced)
              and paused == ms == still,
              f'the movie plays in the page beside them ({rate:.0f} draws/s, {frames_played} frames in {elapsed:.1f} s '
              f'at 30 frames/s) with the notebook untouched (header {f0} -> {f1}); on pause at frame {ms} the notebook '
              f'follows (frame {paused}) and the movie stays (frame {still})')
        # the page draws a frame as plotly does: the notebook redraws the tab at the paused frame P, then (a scrub to
        # the movie's last frame) at F: plotly's own drawings; the page then draws P and F itself on the last charts
        # (two frames inside the axes: a frame the axes leave out widens them when shown)
        NEW = DRAWN.replace('NEW', 'true')
        theirs_p = await c.until(f"(r => r && r.frames.every(f => f === {paused}) ? r : null)({NEW})", 120)
        await c.js(MARK_OLD)
        await c.js(f"(s => {{ s.value = s.max; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); }})"
                   f"({W}.querySelector('.scrub'))")
        theirs_f = await c.until(f"(r => r && r.frames.length && r.frames.every(f => f === r.frames[0]) && "
                                 f"r.frames[0] !== {paused} ? r : null)({NEW})", 120)
        f_scrub = theirs_f and theirs_f['frames'][0]
        await c.until("__deep('.nstxu-live').every(e => e.dataset.path)", 30)
        send = lambda f: c.js(f"window.dispatchEvent(new CustomEvent('nstxu:frame', {{detail: {{frame: {f}}}}}))")
        await send(paused); await asyncio.sleep(0.25)
        mine_p = await c.js(NEW)
        await send(f_scrub); await asyncio.sleep(0.25)
        mine_f = await c.js(NEW)
        same_p, same_f, moved = drawn_diff(mine_p, theirs_p), drawn_diff(mine_f, theirs_f), drawn_diff(mine_p, theirs_f)
        r3 = lambda v: v if v is None else round(v, 3)
        check(None not in (same_p, same_f, moved) and same_p < 0.05 and same_f < 0.05 and moved > 1
              and mine_p['frames'] == [paused] * len(mine_p['frames']),
              f'the page draws a frame as plotly does (frames {paused} and {f_scrub}: every curve vertex, marker and '
              f'line within {r3(same_p)} / {r3(same_f)} px; {mine_p and mine_p["n"]} curves; the two frames '
              f'{r3(moved)} px apart)')
        # a click on the strips (first strip, its leftmost column: the strip's last cell, reflected; top SOL row) probes it
        box = await c.js(f"(r => [r.left, r.top, r.width])({W}.querySelector('canvas.strips').getBoundingClientRect())")
        x, y = box[0] + 82 + 2, box[1] + 42 + 3
        for kind in ('mousePressed', 'mouseReleased'):
            await c.call('Input.dispatchMouseEvent', type=kind, x=x, y=y, button='left', clickCount=1)
        probe = await c.until("new URLSearchParams(location.search).get('probe')", 20)
        check(bool(probe), f'a click on the movie probes a cell (block,i,j = {probe})')
        # the same cell again lets it go; so does Esc; then probe it once more (the reload check below wants one)
        async def click():
            for kind in ('mousePressed', 'mouseReleased'):
                await c.call('Input.dispatchMouseEvent', type=kind, x=x, y=y, button='left', clickCount=1)
        none = "!new URLSearchParams(location.search).get('probe')"
        await click()
        again = await c.until(none, 20)
        await click()
        await c.until("new URLSearchParams(location.search).get('probe')", 20)
        await c.js(f"{W}.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Escape', bubbles: true}}))")
        esc = await c.until(none, 20)
        await click()
        back = await c.until("new URLSearchParams(location.search).get('probe')", 20)
        check(bool(again) and bool(esc) and back == probe,
              'the probed cell lets go on a second click and on Esc, and comes back on a third click')
        # Save figure from the Save & export tab writes its file
        import gkview as gv
        before = set(glob.glob(os.path.join(gv.RUNS, 'exports', '*.png')))
        await c.press("__tab('Save & export')")
        await c.until("!!__button('Save figure')", 30)
        await c.press("__button('Save figure')")
        new = None
        for _ in range(60):
            new = set(glob.glob(os.path.join(gv.RUNS, 'exports', '*.png'))) - before
            if new:
                break
            await asyncio.sleep(0.5)
        check(bool(new), f'Save figure writes {", ".join(os.path.basename(p) for p in new or [])}')
        for p in new or []:
            os.remove(p)   # this check's own file
        # a reload comes back to the view in the address
        here = await c.js('location.search')
        await c.call('Page.reload')
        await asyncio.sleep(3)
        await c.js(DEEP)
        k2 = await c.until('window.__frame && __frame() !== null && __frame()', 120)
        check(k2 == f_scrub and 'probe=' in here and 'tab=Save' in here,
              f'a reload comes back to the view in the address (frame {k2}, the probe, the open tab)')
        if shots:
            await c.shot(os.path.join(shots, 'reloaded.png'))


def part_browser(chrome, shots):
    port, dbg = str(27300 + os.getpid() % 200), str(9500 + os.getpid() % 400)
    profile = tempfile.mkdtemp(prefix='check_viewer_chrome_')
    log = tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False)
    srv = subprocess.Popen(['./view.sh', '--sim', 'Sim 1', '--frame', '20', '--qty', 'T_elc'], cwd=HERE,
                           env=dict(os.environ, OPEN='0', PORT=port), stdout=log, stderr=subprocess.STDOUT,
                           start_new_session=True)
    chr_ = None
    try:
        url = None
        for _ in range(600):
            m = re.search(r'http://\S+access_token=[\w-]+', open(log.name).read())
            if m:
                url = m.group(0).replace('localhost', '127.0.0.1')
                break
            time.sleep(0.1)
        check(url is not None, 'view.sh serves the page')
        if url is None:
            return
        chr_ = subprocess.Popen([chrome, '--headless=new', '--no-sandbox', '--disable-gpu', f'--remote-debugging-port={dbg}',
                                 '--window-size=1680,1400', f'--user-data-dir={profile}', 'about:blank'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        asyncio.run(drive(url, dbg, shots))
    finally:
        for p in (chr_, srv):
            if p is not None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=20)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass


# ---------------------------------------------------------------------------------------------------- LivePlot alone
async def drive_liveplot(page, dbg, n_ev=60):
    import websockets
    info = None
    for _ in range(100):
        try:
            info = json.load(urllib.request.urlopen(f'http://127.0.0.1:{dbg}/json/list'))
            break
        except OSError:
            await asyncio.sleep(0.2)
    ws_url = [t for t in info if t['type'] == 'page'][0]['webSocketDebuggerUrl']
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        c = CDP(ws)
        await c.call('Page.enable')
        await c.call('Page.navigate', url='file://' + page)
        ready = await c.until('window.__ready', 60)
        send = lambda f: c.js(f"window.dispatchEvent(new CustomEvent('nstxu:frame', {{detail: {{frame: {f}}}}}))")
        L = await c.js('__L')
        op = lambda: c.js("+getComputedStyle(document.getElementById('a').querySelector("
                          "`.shapelayer path[data-index='${__lam}']`)).opacity")
        hole, K = L['hole'], L['K']
        drawn = lambda f: c.until(f"document.getElementById('a').dataset.frame === '{f}'", 10)
        await send(hole); await drawn(hole)   # the chart draws on the page's next frame: a busy page delays it
        svg_hidden = await op()
        await send(K); await drawn(K)
        svg_shown = await op()
        await send(hole); await asyncio.sleep(1.5)   # stopped there: plotly's own data and drawing (settle)
        plotly_hidden = await op()
        paths = await c.js("['a', 'b'].map(i => document.getElementById(i).dataset.path)")
        out = [f for f, o in zip(L['frames'], L['over']) if o]
        note = lambda: c.js("document.getElementById('a').querySelector('.nstxu-live-note').textContent")
        notes = []
        for f in (out[:1] + [K]):
            await send(f); await drawn(f)
            notes.append(await note())
        check(len(out) > 0 and K not in out and bool(notes[0]) and not notes[-1],
              f'a frame past the y axes says so under the chart (frames {out} of Sim 2\'s heat burst, shown at frame '
              f'{K}: "{notes[0][:60]}..."), the frame shown fits')
        check(bool(ready) and paths == ['svg', 'plotly'] and svg_hidden == 0 and svg_shown == 1 and plotly_hidden == 0,
              f'a lambda_q line hides where lambda_q is undefined (frame {hole}: opacity {svg_hidden}, frame {K}: '
              f'{svg_shown}; after settling there, plotly\'s own drawing: {plotly_hidden})')
        # the fallback (chart b): a burst of frames at 30 per second, the page's frame clock counted meanwhile
        ticks = await c.js(f"""(async () => {{ let n = 0, on = true; const tick = () => {{ n++; if (on) requestAnimationFrame(tick); }};
          requestAnimationFrame(tick); const t0 = performance.now();
          for (let i = 0; i < {n_ev}; i++) {{ window.dispatchEvent(new CustomEvent('nstxu:frame', {{detail: {{frame: __L.frames[i % __L.frames.length]}}}}));
            await new Promise(r => setTimeout(r, 33)); }}
          on = false; return [n, performance.now() - t0]; }})()""")
        await asyncio.sleep(1.5)
        b = await c.js("(e => [+e.dataset.frame, +e.dataset.updates, +e.dataset.ms, e.dataset.error || ''])(document.getElementById('b'))")
        want = L['frames'][(n_ev - 1) % len(L['frames'])]
        fps = ticks[0] / ticks[1] * 1e3 if ticks else 0
        check(b[0] == want and b[1] >= 3 and not b[3] and fps >= 30,
              f'the plotly fallback follows the frames (at frame {b[0]} of {want}, {b[1]} updates of {b[2]:.0f} ms '
              f'for {n_ev} frames sent) and the page keeps {fps:.0f} frames/s')


def part_liveplot(chrome):
    """A LivePlot alone in a page (its esm under a stand-in model): Sim 2's targets figure with lambda_q undefined
    at one frame (chart a); the same figure with spline curves, which plotly's SVG path does not draw (chart b)."""
    import plotly.offline
    import gkview as gv
    import movie_widget as mm
    import viewer_plots as vp
    lab, d, dflt, _ = gv.SIMS[1]
    run = gv.open_run(os.path.join(gv.NSTXU, d, 'runs', dflt), lab)
    summ = dict(gv.summary(run))
    frames = [int(f) for f in summ['frames']]
    K, hole, tg = frames[-1], frames[len(frames) // 2], run.layout.targets[0]
    lam = np.array(summ[f'lam|{tg.name}'], dtype=float)
    lam[frames.index(hole)] = np.nan
    summ[f'lam|{tg.name}'] = lam
    w = mm.live_plot(vp.targets_figure(run, gv.frame(run, K), summ, run.time(K)), summ, K)
    spline = json.loads(json.dumps(w.fig))
    for t in w.live['traces']:
        spline['data'][t['index']].setdefault('line', {})['shape'] = 'spline'
    i_lam = [i for i, sh in enumerate(w.fig['layout']['shapes']) if sh.get('name') == f'lam|{tg.name}'][0]
    tmp = tempfile.mkdtemp(prefix='check_viewer_liveplot_')
    page = os.path.join(tmp, 'liveplot.html')
    with open(page, 'w') as f:
        f.write(f"""<!doctype html><meta charset="utf-8"><body style="width: 1400px"><div id="a"></div><div id="b"></div>
<script>{plotly.offline.get_plotlyjs()}</script>
<script type="module">
window.__L = {json.dumps(dict(w.live, K=K, hole=hole))}; window.__lam = {i_lam};
const mod = await import(URL.createObjectURL(new Blob([{json.dumps(mm.LivePlot._esm)}], {{type: 'text/javascript'}})));
const model = fig => {{ const v = {{fig, live: window.__L, frame: {K}}};
  return {{get: k => v[k], on() {{}}, off() {{}}, set() {{}}, save_changes() {{}}}}; }};
await mod.default.render({{model: model({json.dumps(w.fig)}), el: document.getElementById('a')}});
await mod.default.render({{model: model({json.dumps(spline)}), el: document.getElementById('b')}});
window.__ready = true;
</script>""")
    dbg = str(9100 + os.getpid() % 300)
    chr_ = subprocess.Popen([chrome, '--headless=new', '--no-sandbox', '--disable-gpu', f'--remote-debugging-port={dbg}',
                             '--window-size=1500,1400', f'--user-data-dir={tempfile.mkdtemp(prefix="check_viewer_chrome_")}',
                             '--allow-file-access-from-files', 'about:blank'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        asyncio.run(drive_liveplot(page, dbg))
    finally:
        try:
            os.killpg(chr_.pid, signal.SIGTERM)
            chr_.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--browser', help='a Chrome / Chromium binary: also drive the page in it')
    ap.add_argument('--shots', help='a directory for screenshots of the browser check')
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    part_headless()
    part_thomson()
    part_exports()
    part_labels()
    part_pgkyl()
    part_themes()
    if a.browser:
        part_browser(a.browser, a.shots)
        part_liveplot(a.browser)
    print(f'\n{len(fails)} failed' if fails else '\nall passed')
    sys.exit(len(fails))
