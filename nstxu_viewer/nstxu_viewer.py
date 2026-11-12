import marimo

__generated_with = "0.25.2"
# runs/viewer.css: the poster theme's font and dark-mode surfaces, written by view.sh from viewer_plots.page_css()
app = marimo.App(width="full", app_title="NSTX-U run viewer", css_file="runs/viewer.css")


@app.cell
def _():
    import glob
    import html
    import json
    import os
    import subprocess
    import sys
    import traceback

    import marimo as mo
    import numpy as np

    HERE = str(mo.notebook_dir())
    sys.path.insert(0, HERE)
    import gkview as gv
    import movie_widget as mm
    import pgkyl_cmd as pc
    import run_options as ro
    import thomson
    import viewer_plots as vp
    import viewer_state as vs
    return HERE, glob, gv, html, json, mm, mo, np, os, pc, ro, subprocess, thomson, traceback, vp, vs


@app.cell
def _(mm, mo):
    # plotly.js for the tabs' live charts, put into the page once (hidden): marimo does not expose its own copy
    import plotly.offline as _po
    _loader = mo.ui.anywidget(mm.PlotlyLoader(src=_po.get_plotlyjs()))
    _loader
    return


@app.cell
def _(mo, vs):
    # where the viewer opens (viewer_state.py): a saved view (--state), then the launch command line, then the page's
    # address. prefs: the view keys, then the current choices (the widgets write them back: plain values, re-run
    # nothing). boot: notes about values that no longer fit, and the keys already resolved.
    boot = {'notes': [], 'done': set()}
    _state = {}
    _file = mo.cli_args().get('state')
    if _file:
        try:
            _state = vs.load_state(str(_file))
        except (OSError, ValueError) as _e:
            boot['notes'].append(f'state not restored: {_e}')
    _cli = {_k: mo.cli_args().get(_k) for _k in vs.KEYS}
    _qp = mo.query_params()   # read once into plain values: writing the address later re-runs no cell
    _url = {_k: _qp.get(_k) for _k in vs.KEYS}
    prefs, _unknown = vs.merge(_state, _cli, _url)
    boot['notes'] += _unknown
    return boot, prefs


@app.cell
def _(mo, prefs, vp, vs):
    dark_pick = mo.ui.switch(value=vs.as_bool(prefs.get('dark', True)), label='dark mode',
                             on_change=lambda v: prefs.update(dark=v))
    # the look: a theme (viewer_plots.THEME_LOOKS: each a PowerPoint theme) in its light or dark mode
    _themes = list(vp.THEME_LOOKS)
    _t = next((x for x in _themes if vs.norm_label(x) == vs.norm_label(prefs.get('theme') or '')), _themes[0])
    theme_pick = mo.ui.radio(_themes, value=_t, label='theme', inline=True, on_change=lambda v: prefs.update(theme=v))
    return dark_pick, theme_pick


@app.cell
def _(dark_pick, json, mo, theme_pick, vp):
    pal = vp.look(theme_pick.value, dark_pick.value)
    # marimo takes its theme from <body data-vscode-theme-kind> before its own setting (useTheme.js, watched by a
    # MutationObserver), so setting it from this same-origin iframe re-themes the page live, without a reload; the
    # theme's CSS (fonts, surfaces) goes into one <style> of the page's head, replaced on every change
    _kind = 'vscode-dark' if dark_pick.value else 'vscode-light'
    _css = json.dumps(vp.page_css(theme_pick.value))
    mo.iframe("<script>const d = parent.document; d.body.dataset.vscodeThemeKind = '" + _kind + "'; "
              "let s = d.getElementById('nstxu-theme'); if (!s) { s = d.createElement('style'); s.id = 'nstxu-theme'; "
              "d.head.appendChild(s); } s.textContent = " + _css + ";</script>", height='0px')
    return (pal,)


@app.cell
def _(boot, gv, mo, os, prefs, vs):
    # each simulation's default run is opened once at start-up: its own geqdsk names the shot and time
    # a run directory given with --path (or a directory of runs): one more simulation, its directory's runs offered
    sim_info = {}
    _sims = [(_lab, gv.runs_root(_dir), _dflt, _desc) for _lab, _dir, _dflt, _desc in gv.SIMS]
    _path = prefs.get('path')
    if _path:
        _p = os.path.abspath(os.path.expanduser(str(_path)))
        if gv.is_run_dir(_p):
            _sims.insert(0, (f'{os.path.basename(os.path.dirname(_p))} (--path)', os.path.dirname(_p),
                             os.path.basename(_p), f'the run directory {_p}'))
        elif os.path.isdir(_p) and gv.runs_in(_p):
            _sims.insert(0, (f'{os.path.basename(_p)} (--path)', _p, gv.runs_in(_p)[-1], f'the runs in {_p}'))
        else:
            boot['notes'].append(f'--path {_path}: not a run directory, nor a directory of runs')
        if len(_sims) > len(gv.SIMS) and prefs.get('sim') is None:
            prefs['sim'] = _sims[0][0]
    for _lab, _root, _dflt, _desc in _sims:
        try:
            _r = gv.open_run(os.path.join(_root, _dflt), _lab)
        except (FileNotFoundError, ValueError) as _e:
            boot['notes'].append(f'{_lab}: skipped ({_e})')
            continue
        sim_info[f'{_r.title} · {_lab}'] = (_lab, _root, _dflt, _desc)
    _keys = list(sim_info)
    _pick = None
    if prefs.get('sim') is not None:
        _pick = next((_k for _k in _keys if vs.norm_label(sim_info[_k][0]) == vs.norm_label(prefs['sim'])), None)
        if _pick is None:
            boot['notes'].append(f"no simulation {prefs['sim']!r} (offered: {', '.join(v[0] for v in sim_info.values())})")
    if _pick is None and prefs.get('shot') is not None:
        _pick = next((_k for _k in _keys if f"#{prefs['shot']} " in _k), None)
        if _pick is None:
            boot['notes'].append(f"no simulation of shot {prefs['shot']}")
    mo.stop(not _keys, mo.callout(mo.md(
        f'No simulation to show: `{gv.CONFIG_FILE}` offers none{"" if os.path.exists(gv.CONFIG_FILE) else " (no such file)"}'
        ' and no `--path` was given. List simulations there (see `viewer.example.json`), or open a run directory with '
        '`./view.sh --path DIR`.' + ''.join(f'\n\n{_n}' for _n in boot['notes'])), kind='warn'))
    _pick = _pick or _keys[0]
    prefs['sim'] = sim_info[_pick][0]
    # another simulation: its default run, from its last frame
    sim_pick = mo.ui.dropdown(sim_info, value=_pick, searchable=True, label='shot (type to search)', full_width=True,
                              on_change=lambda v: prefs.update(sim=v[0], run=None, frame=None, probe=None, blocks=None))
    return (sim_info, sim_pick)


@app.cell
def _(boot, gv, mo, prefs, sim_pick):
    _lab, _root, _dflt, _ = sim_pick.value
    _runs = gv.runs_in(_root)
    _want = prefs.get('run')
    if _want is not None and _want not in _runs and 'run' not in boot['done']:
        boot['notes'].append(f'{_lab} has no run {_want!r}: opened {_dflt}')
    boot['done'].add('run')
    _value = _want if _want in _runs else (_dflt if _dflt in _runs else _runs[-1])
    prefs['run'] = _value
    run_pick = mo.ui.dropdown(_runs, value=_value, searchable=True, label='run', full_width=True,
                              on_change=lambda v: prefs.update(run=v, frame=None, probe=None, blocks=None))
    reload_btn = mo.ui.button(value=0, on_click=lambda v: v + 1, label='reload frames',
                              tooltip='pick up frames written since the run was opened')
    return reload_btn, run_pick


@app.cell
def _(gv, os, run_pick, sim_pick):
    _lab, _root, _, _ = sim_pick.value
    run = gv.open_run(os.path.join(_root, run_pick.value), _lab)
    return (run,)


@app.cell
def _(boot, mo, prefs, run, vp, vs):
    # the blocks drawn (the movie, the maps, the cells table, the profiles' and targets' blocks): none chosen = all
    _role = {}
    for _st in run.layout.strips:
        for _b, _x0, _band in _st.items:
            _seg = next((lab for xa, xb, lab in _st.segments if xa == _x0), '')
            _role.setdefault(_b, _seg if _band != 'in' else f'under the {_seg}')
    _names = {_b: f"{_b} · {vp.REGION_NAME[run.layout.region[_b]]}" + (f' · {_role[_b]}' if _role.get(_b) else '')
              for _b in range(run.nb)}
    _want = None
    try:
        _want = vs.parse_blocks(prefs.get('blocks'), run.nb)
    except ValueError as _e:
        if 'blocks' not in boot['done']:
            boot['notes'].append(str(_e))
    boot['done'].add('blocks')
    blocks_pick = mo.ui.multiselect({v: k for k, v in _names.items()}, value=[_names[b] for b in sorted(_want or [])],
                                    label='blocks drawn (none chosen: every block)', full_width=True,
                                    on_change=lambda v: prefs.update(blocks=vs.blocks_text(set(v))))
    return (blocks_pick,)


@app.cell
def _(blocks_pick, prefs, run, vs):
    # None: every block; else the chosen set (every block chosen is no choice)
    blocks = set(blocks_pick.value) or None
    if blocks is not None and len(blocks) == run.nb:
        blocks = None
    prefs['blocks'] = vs.blocks_text(blocks)
    return (blocks,)


@app.cell
def _(boot, gv, mo, os, prefs, sim_info):
    # a second run drawn over the first (the overlay): any run of any simulation offered, or a run directory
    _opts = {}
    for _lab, _root, _, _ in sim_info.values():
        for _r in gv.runs_in(_root):
            _opts[f'{_lab} / {_r}'] = (_lab, os.path.join(_root, _r))
    _want, _value = prefs.get('cmp'), None
    if _want:
        _w = str(_want)
        _value = next((k for k, (lab, path) in _opts.items() if _w in (f'{lab}/{os.path.basename(path)}', k, path)), None)
        if _value is None and gv.is_run_dir(os.path.abspath(os.path.expanduser(_w))):
            _pth = os.path.abspath(os.path.expanduser(_w))
            _value = f'{os.path.basename(os.path.dirname(_pth))} / {os.path.basename(_pth)}'
            _opts[_value] = (os.path.basename(os.path.dirname(_pth)), _pth)
        if _value is None and 'cmp' not in boot['done']:
            boot['notes'].append(f'no run {_want!r} to compare with')
    boot['done'].add('cmp')
    cmp_pick = mo.ui.dropdown({'none': None, **_opts}, value=_value or 'none', searchable=True,
                              label='draw another run over it (type to search)', full_width=True,
                              on_change=lambda v: prefs.update(cmp=f'{v[0]}/{os.path.basename(v[1])}' if v else None))
    return (cmp_pick,)


@app.cell
def _(cmp_pick, gv, mo):
    # the overlay run: opened, its frames summarised (cached like the first run's), its time histories
    cmp_run = cmp_summ = cmp_hist = cmp_err = None
    if cmp_pick.value:
        _lab, _path = cmp_pick.value
        try:
            cmp_run = gv.open_run(_path, _lab)
            cmp_summ = gv.summary(cmp_run, progress=lambda it, n: mo.status.progress_bar(
                it, total=n, title=f'Reading the frames of {cmp_run.name} (overlay)', subtitle=f'{n} frames not cached yet',
                remove_on_exit=True))
            cmp_hist = (gv.dt_history(cmp_run), gv.integrated(cmp_run), gv.field_energy(cmp_run))
        except Exception as _e:   # a run that cannot be read: said in the sidebar, no overlay
            cmp_run = cmp_summ = cmp_hist = None
            cmp_err = f'{type(_e).__name__}: {_e}'
    return cmp_err, cmp_hist, cmp_run, cmp_summ


@app.cell
def _(gv, reload_btn, run):
    reload_btn.value  # re-list the frames (and re-read the time histories) when clicked
    frames = run.frames()
    dt_hist = gv.dt_history(run)
    integ = gv.integrated(run)
    fenergy = gv.field_energy(run)
    return dt_hist, fenergy, frames, integ


@app.cell
def _(mo, os):
    # Flux through ./view.sh --flux's login (FLUX_CM, its control socket; fetch_flux.sh): the profiles tab's Fetch button
    flux_cm = os.environ.get('FLUX_CM', '')
    fetch_btn = mo.ui.run_button(label='Fetch from Flux', kind='neutral',
                                 tooltip="copy this shot's Thomson, EFIT01, CHERS and beam power to runs/flux")
    return fetch_btn, flux_cm


@app.cell
def _(HERE, fetch_btn, flux_cm, mo, os, run, subprocess):
    # one fetch per click, through the login view.sh holds (never a password prompt here: BatchMode in fetch_flux.sh)
    fetch_log = None
    if fetch_btn.value and flux_cm:
        with mo.status.spinner(title=f'Fetching shot {run.shot} from Flux'):
            try:
                _p = subprocess.run(['bash', os.path.join(HERE, 'fetch_flux.sh'), str(run.shot)], capture_output=True,
                                    text=True, timeout=900, env=dict(os.environ, FLUX_CM=flux_cm))
                fetch_log = (_p.returncode, _p.stdout + _p.stderr)
            except subprocess.TimeoutExpired as _e:
                fetch_log = (-1, f'timed out after {_e.timeout:.0f} s')
    return (fetch_log,)


@app.cell
def _(fetch_log, mo, reload_btn, run, thomson):
    # Thomson n_e of the run's shot (thomson.py) and the frames to average: the window snaps to the Thomson frames of
    # the beam flattop (every frame when the file has no beam power) and opens on thomson.default_window
    reload_btn.value, fetch_log  # re-read after a fetch
    th, th_note, th_pick, th_show = None, '', None, None
    try:
        th = thomson.read(run.shot)
    except (OSError, KeyError, ValueError) as _e:
        th_note = f'{type(_e).__name__}: {_e}'
    if th is not None:
        _fr = th.t if th.flat is None else th.t[(th.t >= th.flat[0]) & (th.t <= th.flat[1])]
        _ms = lambda t: round(float(t) * 1e3, 1)
        _t0, _t1 = thomson.default_window(th, run.eq_ms / 1e3)
        th_pick = mo.ui.range_slider(steps=[_ms(t) for t in _fr], value=[_ms(_t0), _ms(_t1)], show_value=True,
                                     label='Thomson frames to average (ms in the shot)', full_width=True)
        th_show = mo.ui.switch(value=True, label='Thomson n_e')
    return th, th_note, th_pick, th_show


@app.cell
def _(boot, gv, mo, prefs, run, vp, vs):
    # every choice starts from prefs (the launch view, then the last choice) and writes itself back there
    def _first(key, options, default):
        v = prefs.get(key)
        if v is not None and v not in options and str(v) not in options and key not in boot['done']:
            boot['notes'].append(f'{key} {v!r} is not offered here: {default!r} instead')
        boot['done'].add(key)
        return v if v in options else (str(v) if str(v) in options else default)

    _qs = gv.quantities(run)
    _qlab = {k: f'{v[0]} ({v[1]})' for k, v in _qs.items()}
    _q = _first('qty', _qs, f'n_{run.species[0]}')
    qty_pick = mo.ui.dropdown({lab: k for k, lab in _qlab.items()}, value=_qlab[_q], label='quantity',
                              full_width=True, on_change=lambda v: prefs.update(qty=v))
    _modes = {'value': 'value', 'change since the first frame': 'change', 'change since the previous frame': 'step'}
    _m = _first('mode', set(_modes.values()), 'change')
    mode_pick = mo.ui.radio(_modes, value=next(lab for lab, v in _modes.items() if v == _m), label='show',
                            on_change=lambda v: prefs.update(mode=v))
    scale_pick = mo.ui.radio(['this frame', 'whole run'], value='this frame', label='colour scale (the movie: '
                             'always over the frames it plays)')
    log_pick = mo.ui.switch(value=vs.as_bool(prefs.get('log', False)), label='log scale (positive values)',
                            on_change=lambda v: prefs.update(log=v))
    _views = list(vp.zoom_boxes(run))
    zoom_pick = mo.ui.dropdown(_views, value=_first('zoom', _views, 'whole domain'), label='poloidal view',
                               full_width=True, on_change=lambda v: prefs.update(zoom=v))
    _xs = {'distance from the separatrix': 'distance', 'R': 'R'}
    _x = _first('x', set(_xs.values()), 'distance')
    x_pick = mo.ui.radio(_xs, value=next(lab for lab, v in _xs.items() if v == _x), label='profile x axis',
                         on_change=lambda v: prefs.update(x=v))
    return log_pick, mode_pick, qty_pick, scale_pick, x_pick, zoom_pick


@app.cell
def _(boot, frames, gv, mo, prefs, run):
    # the shown frame and the probed cell, shared by the sidebar and the movie: the launch values on the launch run,
    # else the last frame and no probe. now: the frame and probe the notebook shows and the ones the push cell last
    # saw (plain values: reading them re-runs nothing)
    def cell_of(text):
        """The cell index of 'block,i,j' (the file's indices) in this run, or -1."""
        try:
            b, i, j = (int(x) for x in str(text).replace(' ', '').split(','))
        except ValueError:
            return -1
        if not (0 <= b < run.nb and 0 <= i < run.nodes[b].shape[0] - 1 and 0 <= j < run.nodes[b].shape[1] - 1):
            return -1
        return gv.cell_offsets(run)[b] + run.file_row(b, i) * (run.nodes[b].shape[1] - 1) + j

    _k = frames[-1]
    if prefs.get('frame') is not None and prefs.get('run') == run.name:
        try:
            _sel = gv.select_frames(str(prefs['frame']), frames)
            if _sel and _sel[0] in frames:
                _k = _sel[0]
            elif 'frame' not in boot['done']:
                boot['notes'].append(f"{run.name} has no frame {prefs['frame']}: its last, {_k}")
        except ValueError as _e:
            boot['notes'].append(str(_e))
    boot['done'].add('frame')
    _c = -1
    if prefs.get('probe'):
        _c = cell_of(prefs['probe'])
        if _c < 0 and 'probe' not in boot['done']:
            boot['notes'].append(f"no cell {prefs['probe']!r} in {run.name} (block,i,j)")
    boot['done'].add('probe')
    get_k, set_k = mo.state(_k)
    get_probe, set_probe = mo.state(_c)
    now = {'k': _k, 'seen': _k, 'probe': _c, 'seen_probe': _c}
    return cell_of, get_k, get_probe, now, set_k, set_probe


@app.cell
def _(frames, get_k, mo, set_k):
    frame_pick = mo.ui.slider(steps=frames, value=get_k() if get_k() in frames else frames[-1], show_value=True,
                              include_input=True, label='frame', full_width=True, on_change=set_k)
    return (frame_pick,)


@app.cell
def _(cell_of, get_probe, mo, prefs, run, set_probe, vp):
    # the probed cell (the Probe tab): a click on the movie, or block,i,j typed here
    def _typed(text):
        if not text.strip():   # cleared: no probe
            prefs['probe'] = None
            set_probe(-1)
            return
        c = cell_of(text)
        if c >= 0:
            prefs['probe'] = text
            set_probe(c)

    _p = get_probe()
    _txt = ','.join(map(str, vp.probe_cell(run, _p)[:3])) if _p >= 0 else ''
    probe_text = mo.ui.text(value=_txt, label='probe cell (block, i, j; or click the movie; clear: none)',
                            placeholder='10,0,15',
                            full_width=True, on_change=_typed)
    return (probe_text,)


@app.cell
def _(mo, prefs):
    frames_text = mo.ui.text(value=str(prefs.get('frames', '') or ''), label='movie frames',
                             placeholder=': | ::2 | -50: | 0:60 (blank: all)', full_width=True,
                             on_change=lambda v: prefs.update(frames=v))
    return (frames_text,)


@app.cell
def _(frames, frames_text, gv, mo):
    # the frames the movie plays and exports ('pgkyl load --frame' syntax)
    try:
        movie_frames = gv.select_frames(frames_text.value, frames)
        movie_frames = [f for f in movie_frames if f in set(frames)]
        frames_msg = None if movie_frames else mo.callout(mo.md(f'**{frames_text.value}** selects no frame: '
                                                                'playing all'), kind='warn')
    except ValueError as _e:
        movie_frames, frames_msg = [], mo.callout(mo.md(f'{_e}: playing all'), kind='warn')
    movie_frames = movie_frames or list(frames)
    return frames_msg, movie_frames


@app.cell
def _(
    blocks_pick,
    cmp_err,
    cmp_pick,
    dark_pick,
    frame_pick,
    frames_msg,
    frames_text,
    log_pick,
    mo,
    mode_pick,
    probe_text,
    qty_pick,
    reload_btn,
    run_pick,
    sim_pick,
    theme_pick,
    x_pick,
    zoom_pick,
):
    mo.sidebar(
        [mo.md('### NSTX-U run viewer'), sim_pick, run_pick, mo.hstack([reload_btn], justify='start'), frame_pick,
         mo.md('<small>Play: the movie\'s play button (it runs in the browser; the notebook follows when it '
               'pauses)</small>'),
         frames_text, frames_msg or mo.md(''), probe_text,
         mo.md('---'), qty_pick, mode_pick, log_pick, zoom_pick, blocks_pick, mo.md('---'), cmp_pick,
         mo.callout(mo.md(f'Not drawn: {cmp_err}'), kind='warn') if cmp_err else mo.md(''),
         mo.md('---'), x_pick, mo.md('---'), theme_pick, dark_pick],
        footer=mo.md('<small>gkview.py reads the frames; layout, λq and profiles follow the progress posters.</small>'),
        width='24rem',
    )
    return


@app.cell
def _(frame_pick, frames, gv, now, prefs, run):
    k = int(frame_pick.value)
    now['k'] = k
    prefs['frame'] = k
    fr = gv.frame(run, k)
    fr0 = gv.frame(run, frames[0])
    fr_prev = gv.frame(run, frames[max(frames.index(k) - 1, 0)])
    t_now = run.time(k)
    return fr, fr0, fr_prev, k, t_now


@app.cell
def _(cmp_hist, cmp_run, cmp_summ, gv, k, np, summ, vp):
    # the overlay at the shown frame: its frame nearest in time (none where it has no frame within half a frame
    # spacing), and the first run's summary with its entries next to it ('cmp|<key>') for the live charts
    cmp, live_summ = None, summ
    if cmp_run is not None:
        _idx = gv.match_frames(summ['t'], cmp_summ['t'])
        _row = [int(f) for f in summ['frames']].index(k)
        _j = int(_idx[_row])
        _kb = int(cmp_summ['frames'][_j]) if _j >= 0 else -1
        cmp = vp.Cmp(cmp_run, cmp_summ, f'{cmp_run.sim} · {cmp_run.name}',
                     fr=gv.frame(cmp_run, _kb) if _kb >= 0 else None, fr0=gv.frame(cmp_run, int(cmp_summ['frames'][0])),
                     k=_kb, dt=float(cmp_summ['t'][_j] - summ['t'][_row]) if _j >= 0 else np.nan, hist=cmp_hist, idx=_idx)
        live_summ = gv.Compared(summ, cmp_summ, _idx)
    return cmp, live_summ


@app.cell
def _(boot, frame_pick, mo, probe_text, qty_pick, run_pick, sim_pick):
    # values from the launch view that no longer fit (after the cells that resolve them)
    _ = (sim_pick, run_pick, frame_pick, qty_pick, probe_text)
    mo.callout(mo.md('**Opened with changes:** ' + '; '.join(dict.fromkeys(boot['notes']))), kind='warn') \
        if boot['notes'] else None
    return


@app.cell
def _(blocks, cmp, dt_hist, fr, frames, gv, k, mo, np, run, sim_pick, t_now, vp):
    _, _, _, _desc = sim_pick.value
    _ts, _tu = vp.time_unit(run.time(frames[-1]))
    _ncell = sum(v.size for v in (fr[b][f'n_{run.species[0]}'] for b in range(run.nb)))
    _opt = run.options
    _bits = [f'run **{run.name}**', 'half domain (reflecting midplane)' if run.layout.half else 'full domain',
             f'single-block run, {_ncell} cells' if run.single else f'{run.nb} blocks, {_ncell} cells',
             f'{len(frames)} frames, {run.time(frames[0]) * _ts:g}–{run.time(frames[-1]) * _ts:.4g} {_tu}']
    for _key in ('drifts', 'D_perp', 'D_profile_csv', 'D_edge', 'bgk_implicit'):
        if _key in _opt:
            _bits.append(f'{_key} = {_opt[_key].split("/")[-1]}')
    if run.flip:
        _bits.append('psi index grows inward in the files (rows shown inside out)')
    if blocks is not None:
        _bits.append(f'drawing blocks {vp.block_list(blocks)}')
    _td, _vd = dt_hist
    _dt_now = _vd[min(np.searchsorted(_td, t_now), len(_vd) - 1)] if len(_vd) else np.nan
    _neg = int(sum((fr[b][f'n_{s}'] <= 0).sum() for b in range(run.nb) for s in run.species))
    _tiles = [mo.stat(f'{t_now * _ts:.4g} {_tu}', label='time', caption=f'frame {k} of {frames[0]}–{frames[-1]}',
                      bordered=True),
              mo.stat(f'{_dt_now:.3g} s', label='time step', caption=f'latest {_vd[-1]:.3g} s', bordered=True)]
    _cmp_tg = {t.name: t for t in cmp.run.layout.targets} if cmp is not None and cmp.fr is not None else {}
    for _tg in run.layout.targets:
        _q, _, _lam = gv.target_heat_flux(run, fr, _tg)
        _cap = f'peak heat flux {_q.max() / 1e6:.3g} MW/m²'
        if _tg.name in _cmp_tg:
            _qb, _, _lb = gv.target_heat_flux(cmp.run, cmp.fr, _cmp_tg[_tg.name])
            _cap += f' · overlay: {_lb:.2f} mm, {_qb.max() / 1e6:.3g} MW/m²' if np.isfinite(_lb) else ' · overlay: –'
        _tiles.append(mo.stat(f'{_lam:.2f} mm' if np.isfinite(_lam) else '–', label=f'λq, {_tg.name}',
                              caption=_cap, bordered=True))
    _tiles.append(mo.stat(str(_neg), label='cells with n ≤ 0', caption='all blocks, both species', bordered=True))
    _over = []
    if cmp is not None:
        _ts2, _tu2 = vp.time_unit(run.time(frames[-1]))
        _over.append(mo.md(f'**Overlay** (long dashes in the tabs): {cmp.label}, ' +
                           (f'its frame {cmp.k} at {(t_now + cmp.dt) * _ts2:.4g} {_tu2} ({cmp.dt * _ts2:+.3g} {_tu2} from '
                            'the shown time)' if cmp.k >= 0 else
                            f'no frame near {t_now * _ts2:.4g} {_tu2} (it spans {cmp.summ["t"][0] * _ts2:.4g}–'
                            f'{cmp.summ["t"][-1] * _ts2:.4g} {_tu2}): nothing of it drawn at this frame')))
    _files = f'{run.stem}-<what>_<frame>.gkyl' if run.single else f'{run.stem}_b<block>-<what>_<frame>.gkyl'
    _where = [f'**Simulation output:** `{run.path}/` (files `{_files}`)']
    if cmp is not None:
        _where.append(f'**Overlay output:** `{cmp.run.path}/`')
    mo.vstack([
        mo.md(f'## {run.title} · {run.sim}: {_desc}\n\n' + ' · '.join(_bits)),
        mo.md('<small>' + '<br>'.join(_where) + '</small>'),
        *_over,
        mo.hstack(_tiles, justify='start', wrap=True, gap=0.75),
    ])
    return


@app.cell
def _(blocks, log_pick, mm, mo, mode_pick, movie_frames, now, pal, qty_pick, run, summ, x_pick, zoom_pick):
    # the movie: every frame it plays sent to the page once, animated there (movie_widget.py)
    movie = mo.ui.anywidget(mm.Movie(
        data=mm.movie_data(run, summ, qty_pick.value, mode_pick.value, log_pick.value, x_pick.value, movie_frames,
                           blocks=blocks),
        look=mm.movie_look(pal), zoom=zoom_pick.value, frame=now['k'], probe=now['probe']))
    movie
    return (movie,)


@app.cell
def _(movie, now, set_k, set_probe):
    # the page wrote its frame (playback paused, a scrub or a step) or picked a cell: the notebook follows
    if movie.widget.frame != now['k'] and movie.widget.frame in movie.widget.data['frames']:
        set_k(movie.widget.frame)
    if movie.widget.probe != now['probe']:
        now['probe'] = movie.widget.probe
        set_probe(movie.widget.probe)
    return


@app.cell
def _(get_probe, k, movie, now, prefs, run, vp):
    # the notebook's slider moved or a cell was typed: the movie follows. Only on a change of the notebook's value:
    # this cell also re-runs when the page writes, before the notebook has taken the page's value (k still the old
    # one: pushing it would send the movie back)
    if now['seen'] != k:
        now['seen'] = k
        if movie.widget.frame != k:
            movie.widget.frame = k
    _p = get_probe()
    prefs['probe'] = ','.join(map(str, vp.probe_cell(run, _p)[:3])) if _p >= 0 else None
    if now['seen_probe'] != _p:
        now['seen_probe'] = now['probe'] = _p
        if movie.widget.probe != _p:
            movie.widget.probe = _p
    return


@app.cell
def _(frames, gv, mo, run):
    _ = frames  # rebuilt (new frames only) on reload
    summ = gv.summary(run, progress=lambda it, n: mo.status.progress_bar(
        it, total=n, title='Reading frames for the movie and the time traces', subtitle=f'{n} frames not cached yet',
        remove_on_exit=True))
    return (summ,)


@app.cell
def _(mo, prefs):
    TABS = ['Midplane profiles', 'Targets', 'Time traces', 'Probe', 'Balance & power', 'Diffusivity', 'Options',
            'Cells', 'Static figure', 'Save & export', 'Run']
    get_tab, set_tab = mo.state(prefs['tab'] if prefs.get('tab') in TABS else TABS[0])
    return TABS, get_tab, set_tab


@app.cell
def _(
    blocks,
    cmp_pick,
    dark_pick,
    frames_text,
    get_probe,
    get_tab,
    k,
    log_pick,
    mo,
    mode_pick,
    os,
    prefs,
    qty_pick,
    run,
    run_pick,
    sim_pick,
    theme_pick,
    vp,
    vs,
    x_pick,
    zoom_pick,
):
    # the current view, also written to the page's address (a reload or a bookmark comes back to it)
    _p = get_probe()
    view = dict(sim=sim_pick.value[0], run=run_pick.value, frame=int(k), qty=qty_pick.value, mode=mode_pick.value,
                log=bool(log_pick.value), zoom=zoom_pick.value, x=x_pick.value, theme=theme_pick.value,
                dark=bool(dark_pick.value),
                tab=get_tab(), frames=frames_text.value.strip(),
                probe=','.join(map(str, vp.probe_cell(run, _p)[:3])) if _p >= 0 else '',
                blocks=vs.blocks_text(blocks),
                cmp=f'{cmp_pick.value[0]}/{os.path.basename(cmp_pick.value[1])}' if cmp_pick.value else '',
                path=str(prefs.get('path') or ''))
    _qp = mo.query_params()   # a local: writing it re-runs no cell
    for _key, _v in view.items():
        _s = '' if _v is None else str(_v).lower() if isinstance(_v, bool) else str(_v)
        if _s and _qp.get(_key) != _s:
            _qp[_key] = _s
        elif not _s and _key in _qp:
            del _qp[_key]
    return (view,)


@app.cell
def _(blocks, fr, fr0, fr_prev, log_pick, mode_pick, pal, qty_pick, run, scale_pick, summ, vp, zoom_pick):
    def static_figure(title=''):
        """The map of the shown frame as a matplotlib figure (the Static figure tab, the figure export)."""
        _whole = scale_pick.value == 'whole run'
        return vp.map_figure(run, fr, fr0, qty_pick.value, mode_pick.value, _whole, log_pick.value,
                             summ if _whole else None, zoom_pick.value, pal=pal, fr_prev=fr_prev, title=title,
                             blocks=blocks)
    return (static_figure,)


@app.cell
def _(mo):
    # Save & export: the view (a state file), the shown figure, a movie of the movie's frames
    state_name = mo.ui.text(label='save the view as', placeholder='my_view (runs/states/<name>.json)',
                            full_width=True)
    save_view_btn = mo.ui.run_button(label='Save view')
    fig_name = mo.ui.text(label='figure file', placeholder='automatic (runs/exports/<name>)', full_width=True)
    fig_fmt = mo.ui.dropdown(['png', 'pdf', 'svg'], value='png', label='format')
    save_fig_btn = mo.ui.run_button(label='Save figure')
    movie_name = mo.ui.text(label='movie file', placeholder='automatic (runs/exports/<name>.mp4)', full_width=True)
    movie_fps = mo.ui.number(start=1, stop=60, step=1, value=10, label='frames/s')
    make_movie_btn = mo.ui.run_button(label='Make movie')
    return (fig_fmt, fig_name, make_movie_btn, movie_fps, movie_name, save_fig_btn, save_view_btn, state_name)


@app.cell
def _(gv, k, mode_pick, os, qty_pick, run):
    EXPORTS = os.path.join(gv.RUNS, 'exports')

    def out_path(name, default, ext, folder=EXPORTS):
        """A file name typed on the page: blank = default; relative = under folder; the extension added if missing."""
        name = (name or '').strip() or default
        if not name.lower().endswith('.' + ext):
            name += '.' + ext
        path = os.path.expanduser(name)
        return path if os.path.isabs(path) else os.path.join(folder, path)

    base_name = f"{run.sim.replace(' ', '').lower()}_{run.name}_{qty_pick.value}_{mode_pick.value}_frame{k}"
    return EXPORTS, base_name, out_path


@app.cell
def _(gv, mo, os, out_path, save_view_btn, state_name, view, vs):
    _cmd = vs.command_line(view)
    view_msg = mo.md('')
    if save_view_btn.value:
        _path = out_path(state_name.value, 'view', 'json', folder=os.path.join(gv.RUNS, 'states'))
        try:
            vs.save_state(_path, view)
            view_msg = mo.callout(mo.md(f'Saved `{_path}`. Reopen with `./view.sh --state {_path}`'), kind='success')
        except OSError as _e:
            view_msg = mo.callout(mo.md(f'Not saved: {_e}'), kind='danger')
    view_cmd = mo.md(f'This view from the command line (in the study directory):\n\n```bash\n{_cmd}\n```\n\nThe '
                     "page's address holds it too (a reload or a bookmark reopens it; the address also carries the "
                     'access token, so do not share it while the server runs).')
    return view_cmd, view_msg


@app.cell
def _(base_name, fig_fmt, fig_name, mo, os, out_path, save_fig_btn, static_figure, traceback, vp):
    def _render(path=None):
        _fig = static_figure()
        try:
            if path:
                return vp.save_figure(_fig, path)
            import io
            _buf = io.BytesIO()
            _fig.savefig(_buf, format=fig_fmt.value, dpi=200 if fig_fmt.value == 'png' else None)
            return _buf.getvalue()
        finally:
            vp.plt.close(_fig)

    fig_msg = mo.md('')
    if save_fig_btn.value:
        _path = out_path(fig_name.value, base_name, fig_fmt.value)
        try:
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            fig_msg = mo.callout(mo.md(f'Saved `{_render(_path)}`'), kind='success')
        except Exception as _e:
            fig_msg = mo.callout(mo.md(f'**{type(_e).__name__}:** {_e}\n\n```\n{traceback.format_exc()}\n```'),
                                 kind='danger')
    fig_download = mo.download(data=lambda: _render(), filename=f'{base_name}.{fig_fmt.value}',
                               label=f'Download the figure ({fig_fmt.value})')
    return fig_download, fig_msg


@app.cell
def _(
    base_name,
    blocks,
    log_pick,
    make_movie_btn,
    mo,
    mode_pick,
    movie_fps,
    movie_frames,
    movie_name,
    os,
    out_path,
    pal,
    qty_pick,
    run,
    summ,
    traceback,
    vp,
    zoom_pick,
):
    movie_msg = mo.md(f'_{len(movie_frames)} frames (the sidebar\'s movie frames), {movie_fps.value} frames/s: '
                      f'{len(movie_frames) / max(movie_fps.value, 1):.1f} s; about '
                      f'{0.3 * len(movie_frames):.0f} s to render._')
    if make_movie_btn.value:
        _path = out_path(movie_name.value, base_name.rsplit('_frame', 1)[0] +
                         f'_{movie_frames[0]}-{movie_frames[-1]}', 'mp4')
        try:
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            _label, _sym = vp.gv.quantities(run)[qty_pick.value][:2]
            vp.render_movie(run, summ, movie_frames, qty_pick.value, mode_pick.value, log_pick.value,
                            zoom_pick.value, pal, _path, fps=int(movie_fps.value), blocks=blocks,
                            title=f'{run.title} · {run.sim} · {run.name} · {_label}',
                            progress=lambda it, n: mo.status.progress_bar(it, total=n, title='Rendering the movie',
                                                                          remove_on_exit=True))
            _size = os.path.getsize(_path)
            _parts = [mo.callout(mo.md(f'Saved `{_path}` ({_size / 1e6:.1f} MB)'), kind='success')]
            if _size < 40e6:
                _data = open(_path, 'rb').read()
                _parts += [mo.video(_data, controls=True, loop=True),
                           mo.download(data=_data, filename=os.path.basename(_path), mimetype='video/mp4',
                                       label='Download the movie')]
            movie_msg = mo.vstack(_parts)
        except Exception as _e:
            movie_msg = mo.callout(mo.md(f'**{type(_e).__name__}:** {_e}\n\n```\n{traceback.format_exc()}\n```'),
                                   kind='danger')
    return (movie_msg,)


@app.cell
def _(
    blocks,
    cmp,
    live_summ,
    ro,
    zoom_pick,
    dt_hist,
    fenergy,
    fetch_btn,
    fetch_log,
    flux_cm,
    subprocess,
    fig_download,
    fig_fmt,
    fig_msg,
    fig_name,
    fr,
    fr0,
    get_probe,
    get_tab,
    glob,
    gv,
    integ,
    k,
    make_movie_btn,
    mm,
    mo,
    mode_pick,
    movie_fps,
    movie_frames,
    movie_msg,
    movie_name,
    os,
    pal,
    qty_pick,
    run,
    save_fig_btn,
    save_view_btn,
    scale_pick,
    set_tab,
    state_name,
    static_figure,
    summ,
    t_now,
    TABS,
    th,
    th_note,
    th_pick,
    th_show,
    traceback,
    view_cmd,
    view_msg,
    vp,
    x_pick,
    HERE,
    frames,
    html,
    pc,
):
    def _safe(build):
        # a tab that fails says why, the traceback folded away, instead of breaking the page
        def _run_it():
            try:
                return build()
            except Exception as _e:
                return mo.vstack([mo.callout(mo.md(f'**{type(_e).__name__}:** {_e}'), kind='warn'),
                                  mo.accordion({'traceback': mo.md(f'```\n{traceback.format_exc()}\n```')})])
        return _run_it

    def _clickable(fig, cmd):
        # a matplotlib figure that, clicked, shows its pgkyl command under it (no script: a <details>, the image its
        # summary; one click on the command selects all of it)
        _img = mo.as_html(fig).text
        return mo.Html(
            '<details class="nstxu-pg"><summary style="list-style: none; cursor: pointer" title="click: the pgkyl '
            f'command for this figure">{_img}</summary><div style="margin: 6px 0; font-size: 12px; opacity: 0.85">'
            '<b>pgkyl</b> (click the figure again to hide; one click on the command selects it)</div><pre style="'
            'white-space: pre; overflow-x: auto; font-size: 11.5px; padding: 6px 8px; border-radius: 4px; border: 1px '
            f'solid var(--border, #8888); user-select: all">{html.escape(cmd)}</pre></details>')

    def _live(fig):
        # a chart that follows the movie in the page (its profiles, target curves and time cursors), the kernel idle;
        # its y axes span the movie's frames
        return mo.ui.anywidget(mm.live_plot(fig, live_summ, k, movie_frames))

    def _fetch():
        # the Fetch button where the shot has no Flux copy yet (no data, or the workflow's copy without beam power)
        _items = []
        if fetch_log is not None:
            _rc, _out = fetch_log
            _items.append(mo.accordion({f'fetch {"done" if _rc == 0 else f"FAILED (exit {_rc})"}: log':
                                        mo.plain_text(_out)}))
        if th is not None and th.pnb is not None:
            return _items
        _why = (f'No Thomson data for shot {run.shot} here{f" ({th_note})" if th_note else ""}.' if th is None
                else f'Shot {run.shot}: the workflow\'s copy, without beam power (no flattop).')
        if flux_cm and subprocess.run(['bash', os.path.join(HERE, 'fetch_flux.sh'), '--check', flux_cm]).returncode == 0:
            return _items + [mo.hstack([mo.md(_why), fetch_btn], justify='start', align='center')]
        _how = (f'The Flux login has closed (the laptop\'s connection dropped?): restart `./view.sh --flux`.' if flux_cm
                else 'Start the viewer with `./view.sh --flux` (one Flux login, password + Duo) to fetch it from '
                     f'here, or on the laptop: `NSTXU_VIEWER_DATA={gv.DATA} bash <(ssh perlmutter cat {HERE}/fetch_flux.sh) '
                     f'{run.shot}`.')
        return _items + [mo.callout(mo.md(f'{_why} {_how}'), kind='neutral')]

    def _profiles():
        _measured, _top, _notes = None, _fetch(), []
        if th is not None:
            _t0, _t1 = (th.frame_time(v / 1e3) for v in th_pick.value)
            if th_show.value:
                _measured = (th, _t0, _t1)
            _top += [mo.hstack([th_show, th_pick], align='center', widths=[1, 6]),
                     _live(vp.thomson_timeline(th, _t0, _t1, run.eq_ms / 1e3, pal=pal))]
            _notes.append(mo.md(f'<small>{vp.thomson_note(run, fr, th, _t0, _t1)}</small>'))
        return mo.vstack(_top + [
            _live(vp.profiles_figure(run, fr, fr0, f'frame {k}', x_pick.value, pal=pal, measured=_measured, cmp=cmp,
                                     blocks=blocks)),
            *_notes,
            _live(vp.history_figure(run, summ, qty_pick.value, mode_pick.value, t_now, x_pick.value, pal=pal)),
        ])

    def _targets():
        return mo.vstack([
            _live(vp.targets_figure(run, fr, summ, t_now, pal=pal, cmp=cmp, blocks=blocks)),
            mo.md('<small>λq: the midplane width holding 63 % of the parallel heat flux reaching the target, each SOL '
                  'flux surface placed at its midplane radius (DPP 2025 definition, as on the progress posters). '
                  'Heat flux: q_par = sum over species of m (M3par + M3perp) / 2, cell average in the plate cell. '
                  'The total power to each target: the Balance & power tab.' +
                  (' Where a leg\'s SOL never reaches Z = 0, its surfaces sit at their distance along the plate from '
                   'its inner corner (no λq there).' if vp.targets_along(run, blocks) else '') + '</small>'),
        ])

    def _traces():
        return mo.vstack([
            _live(vp.timetraces_figure(run, summ, dt_hist, integ, fenergy, t_now, pal=pal, cmp=cmp, blocks=blocks)),
            mo.md('<small>Region changes: the Jacobian-weighted (volume) mean over the region\'s cells, relative to '
                  'the first frame.</small>'),
        ])

    def _probe():
        _c = get_probe()
        if _c < 0:
            return mo.callout(mo.md('Click a cell in the movie (poloidal plane, strips or the space-time strip), or '
                                    'type block,i,j in the sidebar.'), kind='neutral')
        return mo.vstack([mo.md(vp.probe_note(run, _c, cmp)),
                          _live(vp.probe_figure(run, summ, _c, t_now, pal=pal, cmp=cmp))])

    def _balance():
        _B = gv.balance(run)
        if _B is None:
            return mo.callout(mo.md('This run wrote no integrated diagnostics (no fdot_integrated_moms).'),
                              kind='neutral')
        _Bb = gv.balance(cmp.run) if cmp is not None else None
        _fig, _notes = vp.balance_figure(run, _B, t_now, pal=pal, cmp=cmp, B_cmp=_Bb)
        _i = len(_B['t']) - 1
        _tg = ', '.join(f"{name} {v['power'][_i] * 2 * vp.np.pi / 1e6:.3g} MW"
                        for name, v in _B['targets'].items())
        _share = sum(v['power'][_i] for v in _B['targets'].values()) / _B['E']['out'][_i] * 100
        return mo.vstack([
            _live(_fig),
            mo.md(f"At {_B['t'][_i] * vp.time_unit(_B['t'][-1])[0]:.4g} {vp.time_unit(_B['t'][-1])[1]}: source "
                  f"{_B['E']['src'][_i] * gv.full_device(run) / 1e6:.3g} MW (whole device); targets: {_tg} "
                  f"({_share:.0f} % of all boundary energy losses; the rest leaves through the radial walls). "
                  f"Residuals (source − losses − rate of change, of the source): {', '.join(_notes)}."),
            mo.callout(mo.md(
                '**Reading the energy residual.** The particle balance closes (Sim 1: 0.01 %), so the boundary '
                'fluxes carry every transport channel. The energy balance does not: +9 % of the source in Sim 1\'s '
                'steady state, and more, either sign, during transients; with and without collisions alike in the '
                '60 µs Sim 1 pair. Not understood yet; do not read it as a physical loss. Definitions: source energy '
                '= Σ m M2 / 2 of the source (this driver integrates the source\'s M0M1M2 moments; postgkyl\'s '
                'energy_balance would take that M2 for the Hamiltonian, which is off by the mass for this driver); '
                'losses and rate of change = the Hamiltonian moments of the boundary fluxes and of ∂f/∂t, minus the '
                'field-energy rate. The diagnostics are per radian: ×2π for each target\'s ring, ×2π×2 for a half '
                'domain\'s whole device (checked on Sim 2: its source × 4π is its set particle rate).'),
                kind='neutral'),
        ])

    def _options():
        # the options the run was started with, read with the driver's own source (run_options.py)
        _v = ro.options_view(run)
        _items = []
        _bin = _v['binary']
        _src = ', '.join(f'`{os.path.basename(p)}`' for p in _v['sources'])
        _items.append(mo.md(
            (f'**Driver:** `{_bin}`' + (f' on {_v["ranks"].group(1)} ranks' if _v['ranks'] else '') +
             f' (built {ro.when(_bin)}). ' if _bin else '**Driver:** not found in the run\'s command file. ') +
            (f'What each option is and its default: the driver\'s source {_src} next to it (its struct comments and '
             'designated defaults)' + (', **edited after this binary was built** (its defaults may not be the run\'s; '
                                       'the values the run set are its own)' if _v['stale'] else '') + '.'
             if _v['sources'] else 'No driver source next to the binary: the options without descriptions.')))
        _set = [r for r in _v['rows'] if r['value']]
        _row = lambda r: dict(section=ro.short_section(r['section']), option=r['key'], value=ro.shown_value(r['value']),  # noqa: E731
                              default=ro.shown_value(r['default']) if r['default'] else '–',
                              meaning=r['what'] or '–')
        _set_rows = [dict(_row(r), note='= the default' if r['value'] == r['default'] else '') for r in _set]
        _set_rows += [dict(section='(not read by this source)', option=k, value=ro.shown_value(v), default='–',
                           meaning='set at launch; the driver source here does not read it', note='')
                      for k, v in _v['unknown'].items()]
        _items.append(mo.md(f'#### What this run set at launch ({len(_set_rows)} options)'))
        _items.append(mo.ui.table(_set_rows, selection=None, page_size=max(len(_set_rows), 1), show_column_summaries=False,
                                  wrapped_columns=['meaning'])
                      if _set_rows else mo.md('_No options recorded (no options_used.txt or command file)._'))
        _paths = [dict(option=k, path=v, exists='yes' if os.path.exists(v) else 'no')
                  for k, v in _v['launch'].items() if str(v).startswith('/')]
        if cmp is not None:   # side by side with the overlay run: what each used (set, else the default)
            _vb = ro.options_view(cmp.run)
            _side = vp.options_side_by_side(ro, _v, _vb)
            _nd = sum(r['compare'] == 'differs' for r in _side)
            _items.append(mo.md(f'#### Side by side with the overlay, {cmp.label}: {_nd} differ, '
                                f'{len(_side) - _nd} the same'))
            _items.append(mo.md(vp.compare_legend(pal)))
            _items.append(mo.ui.table(_side, selection=None, page_size=max(len(_side), 1), show_column_summaries=False,
                                      style_cell=vp.compare_style(pal)) if _side else
                          mo.md('_Neither run records an option._'))
        _jobs = ro.jobs(run)
        _acc = {}
        if _v['rows']:
            _secs = {}
            for r in _v['rows']:
                _secs.setdefault(ro.short_section(r['section']), []).append(
                    dict(option=r['key'], value=ro.shown_value(r['value']) if r['value'] else '(default)',
                         default=ro.shown_value(r['default']) if r['default'] else '–', type=r['kind'],
                         meaning=r['what'] or '–'))
            _acc[f'Every option the driver reads ({len(_v["rows"])}), by its own sections'] = mo.vstack(
                [_x for _sec, _rs in _secs.items() for _x in (mo.md(f'**{_sec}**'),
                                                             mo.ui.table(_rs, selection=None, page_size=len(_rs),
                                                                         wrapped_columns=['meaning']))])
        if _paths:
            _acc[f'Files and directories it was given ({len(_paths)})'] = mo.ui.table(_paths, selection=None,
                                                                                     page_size=len(_paths))
        if _jobs:
            _acc[f'Its jobs ({len(_jobs)}, restarts.log)'] = mo.ui.table(_jobs, selection=None, page_size=len(_jobs))
        _log, _rep, _tagged = ro.startup_report(run)
        if _rep:
            _acc[f'What the driver reported while setting up ({_log}, every line once' +
                 (f'; {_tagged} tagged library diagnostics left out' if _tagged else '') + ')'] = \
                mo.plain_text('\n'.join(_rep))
        if _acc:
            _items.append(mo.accordion(_acc, multiple=True))
        return mo.vstack(_items)

    def _cells():
        return mo.ui.table(vp.cell_rows(run, fr, fr0, qty_pick.value, blocks=blocks), page_size=20, selection=None)

    def _diffusivity():
        # the anomalous diffusivity the run used (gkview.diffusivity): at the midplane, and in every cell
        _D = gv.diffusivity(run)
        _Db = gv.diffusivity(cmp.run) if cmp is not None else None
        _items = [mo.md(f'**{run.name}:** {"D from " if _D["kind"] == "file" else "a uniform D, from " if _D["kind"] else ""}'
                        f'{_D["source"]}.')]
        if _Db is not None:
            _items.append(mo.md(f'**{cmp.label}** (overlay): {"D from " if _Db["kind"] == "file" else "a uniform D, from " if _Db["kind"] else ""}{_Db["source"]}.'))
        if not _D['kind'] and not (_Db and _Db['kind']):
            return mo.vstack(_items + [mo.callout(mo.md(
                'Nothing to draw. A driver that writes the diffusivity (`anomalous_diffusion.write_diagnostics = '
                'true`: one `<species>_anom_diff_0.gkyl` per block) puts the D it used in the run\'s own files.'),
                kind='neutral')])
        try:
            _items.append(_live(vp.diffusivity_figure(run, _D, x_pick.value, pal=pal, cmp=cmp, D_cmp=_Db,
                                                      blocks=blocks)))
        except ValueError as _e:
            _items.append(mo.callout(mo.md(str(_e)), kind='neutral'))
        if _D['kind']:
            for _s in (list(_D['species'])[:1] if _D['same'] else list(_D['species'])):
                _fig = vp.diffusivity_map(run, _D, zoom_pick.value, pal=pal, blocks=blocks, species=_s)
                _items.append(_clickable(_fig, pc.diffusivity_maps(run, _s, blocks) if _D['kind'] == 'file' else
                                         f'# no pgkyl equivalent: no anom_diff file, a uniform D ({_D["source"]})'))
                vp.plt.close(_fig)
        _items.append(mo.md(
            '<small>D is the driver\'s `D_profile` at each cell corner (a node), as the solver used it. Where the run '
            'wrote `anom_diff` (D × gˣˣJ, the library\'s product of the two p = 1 expansions), it is divided back per '
            'cell here; the corner values then agree between neighbouring cells ' +
            (f'(largest disagreement {_D["gap"]:.1e} of D)' if _D['kind'] == 'file' else '') +
            '. The midplane curve: those node values at Z = 0, core then SOL; the map: cell averages.</small>'))
        return mo.vstack(_items)

    def _static():
        _fig = static_figure()
        _m = mode_pick.value
        _ref = None if _m == 'value' else frames[0] if _m == 'change' else frames[max(frames.index(k) - 1, 0)]
        _out = _clickable(_fig, pc.blocks_maps(run, qty_pick.value, k, blocks, _m, _ref))
        vp.plt.close(_fig)
        return mo.vstack([scale_pick, mo.md('<small>Click the figure for its pgkyl command.</small>'), _out])

    def _run():
        _lay = run.layout
        _role = {}
        for _st in _lay.strips:   # each item's segment: a SOL block's own, the block under it 'under the ...'
            for _b, _x0, _band in _st.items:
                _seg = next((lab for xa, xb, lab in _st.segments if xa == _x0), '')
                _role.setdefault(_b, f'under the {_seg}' if _band == 'in' else _seg)
        _rows = [dict(block=b, region=vp.REGION_NAME[_lay.region[b]], role=_role.get(b, ''),
                      cells=f'{run.nodes[b].shape[0] - 1} psi × {run.nodes[b].shape[1] - 1} theta',
                      R=f'{run.nodes[b][..., 0].min():.3f}–{run.nodes[b][..., 0].max():.3f} m',
                      Z=f'{run.nodes[b][..., 1].min():.3f}–{run.nodes[b][..., 1].max():.3f} m') for b in range(run.nb)]
        _items = [mo.md(f'**{run.path}**'), mo.ui.table(_rows, selection=None, page_size=run.nb)]
        _items.append(mo.md('Targets: ' + (', '.join(
            f'{t.name} (block {t.leg}, plate cell j = {t.j}, ' +
            (f'midplane radii from block {t.mid})' if t.mid is not None else 'placed along the plate: its chain never '
             'reaches Z = 0)') for t in _lay.targets) or 'none (no physical theta face off Z = 0)')))
        _notes = []
        if run.single:
            _notes.append('Single-block run: the gyrokinetic app\'s own files (no _b<k>, no topology), every face '
                          'taken as physical' + (', theta closing on itself' if run.layout.region[0] == 'core' else '')
                          + '.')
        if run.flip:
            _notes.append('The files\' psi index grows inward: every block\'s rows are shown inside out (row 0 '
                          'innermost); the cell table and the probe give the files\' own i.')
        if _notes:
            _items.append(mo.md(' '.join(_notes)))
        if run.options:
            _items.append(mo.accordion({'options_used.txt': mo.ui.table(
                [dict(option=a, value=b) for a, b in run.options.items()], selection=None, page_size=40)}))
        _logs = {}
        for _name in ('command.txt', 'restarts.log'):
            _p = os.path.join(run.path, _name)
            if os.path.exists(_p):
                _logs[_name] = mo.plain_text(open(_p).read())
        _outs = sorted(glob.glob(os.path.join(run.path, 'slurm-*.out')), key=os.path.getmtime)
        if _outs:
            _tail = open(_outs[-1], errors='replace').read().splitlines()[-30:]
            _logs[f'{os.path.basename(_outs[-1])} (last 30 lines)'] = mo.plain_text('\n'.join(_tail))
        if _logs:
            _items.append(mo.accordion(_logs))
        return mo.vstack(_items)

    _export = mo.vstack([
        mo.md('#### The view'), view_cmd, mo.hstack([state_name, save_view_btn], align='end', justify='start'),
        view_msg,
        mo.md('#### The figure (the Static figure tab\'s map of the shown frame)'),
        mo.hstack([fig_name, fig_fmt, save_fig_btn], align='end', justify='start'), fig_msg, fig_download,
        mo.md('#### A movie (the map over the sidebar\'s movie frames, one colour scale)'),
        mo.hstack([movie_name, movie_fps, make_movie_btn], align='end', justify='start'), movie_msg,
    ], gap=0.6)
    _tabs = {'Midplane profiles': _profiles, 'Targets': _targets, 'Time traces': _traces, 'Probe': _probe,
             'Balance & power': _balance, 'Diffusivity': _diffusivity, 'Options': _options, 'Cells': _cells,
             'Static figure': _static,
             'Save & export': _export,
             'Run': _run}
    mo.vstack([
        mo.md('<small>Click a curve in any chart, a cell or a profile in the movie, or a map figure: the equivalent '
              'pgkyl command (postgkyl\'s command line, the files this viewer read), ready to paste.</small>'),
        mo.ui.tabs({name: (_safe(_tabs[name]) if callable(_tabs[name]) else _tabs[name]) for name in TABS},
                   value=get_tab(), lazy=True, on_change=set_tab),
    ], gap=0.3)
    return


if __name__ == "__main__":
    app.run()
