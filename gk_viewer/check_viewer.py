#!/usr/bin/env python3
"""Checks of the viewer itself (the data layer is check_gkview.py's), with the marimo venv:

  GK_VIEWER_DATA=<data dir> <viewer venv>/bin/python check_viewer.py                    headless + exports
  GK_VIEWER_DATA=<data dir> <viewer venv>/bin/python check_viewer.py --browser CHROME   also the page in a browser
(the data directory's viewer.json must offer 'Sim 1', 'Sim 2' and '2025 re-run'; the checks read those runs)

1. headless: the notebook runs with every tab built (mo.ui.tabs made eager), for each simulation; a launch view
   (mo.cli_args) opens where it says; one that does not fit falls back with a note per key;
2. thomson: the reader equals the sim-init workflow's 203899 channel export (n_e and T_e); the beam flattop equals
   the scan's own loop on random traces; Sim 2's profiles carry the Thomson points on each side, n and T rows;
   measured (2026-11-16): CHERS and TRANSP (part_measured's docstring), with a postgkyl venv's python (viewer.json's
   pgkyl) for scipy;
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
6. (2026-11-13) fonts: every symbol the sources write that a theme font lacks is in Viewer Symbols (fonts/), whose
   files equal a fresh build; the page gets it; matplotlib finds the tracked files before a system copy; no missing
   glyph in text_px or in any theme's notebook. sync: sync.py mirrors Sim 2's run into a stand-in laptop tree (a
   stand-in ssh) and the viewer opens the mirror with its own geqdsk, Thomson file and cache.
7. (2026-11-14) the poloidal plane's extras, on Sim 2: a device picture red inside its wall and blue outside lies
   under the map and (--browser) under the movie where its extent says; the opacity slider redraws it at its value
   without remaking the movie; the half domain shown whole puts every cell's colour at its mirror image (map and
   page); switched off, the picture leaves no trace.
8. (2026-11-14, --browser) the page's layout: the movie's poloidal plane stays in its plot area (every simulation
   in its views, mirrored too, with the device picture, light and dark: nothing but text in the canvas's margins);
   the Options tab's tables fit their panel.
9. (2026-11-16) the lines over the poloidal plane's cells (the separatrix, the wall, the block outlines), each on a
   switch: on Sim 2 each switch draws or takes away pixels along its own line only (mirrored halves too), in the map
   and (--browser) in the movie, which is not made again; all off leaves the map without lines; every map the viewer
   draws takes the switches.
10. (2026-11-16) the grid (every cell edge, curves through the Gauss nodes) and the quantity, each on a switch: on
   ASDEX Upgrade's device run (by path, beside base) and Sim 2 (chords: its surface nodes are zeros, said on the
   plane), the map, and (--browser) TCV's and DIII-D's movies: the grid draws along the cell edges only, the
   quantity inside the cells only, the grid passes through the Gauss nodes and not along the chords.
11. (2026-11-16) blank: with no viewer.json, with "sims": [] and with a simulation whose directory is missing, the
    notebook opens on the empty run with every tab built, its charts and movie empty but whole; with --browser the
    three served, every tab opened in the page, screenshots (--shots); the per-simulation and per-device settings of
    viewer.json (part_schema).
12. (2026-11-17) species: the curves shown, four switches (electrons, ions, simulation, experiment) on every line
    chart of the three simulations (an overlay and the measured profiles drawn): all on, each chart as drawn without
    them; each switch alone off takes exactly the curves the chart says are its own (legend names, hover text, panel
    titles), notes in the panels and in place of the charts it empties; a species by its charge's sign; the notebook
    keeps the switches in its view and hands the page (and its slide downloads) only the curves shown; with
    --browser, Ions clicked off in Sim 2's page and the page reloaded.
13. (2026-11-17) the whole machine view (part_machine): every simulation's poloidal plane offers 'whole machine'
    where the wall or the device picture reaches out of the whole domain's frame, framing all of both; the map at it
    shows them whole; the layout audit (--browser) opens it too.
--parts a,b: only those parts (headless, thomson, measured, exports, labels, pgkyl, themes, fonts, sync, machine,
poloidal, lines, grid, blank, species, browser, layout, liveplot).
Exit status: the number of failures.
"""
import argparse
import asyncio
import base64
import contextlib
import io
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
def headless(cli, sims_first=None, texts=None):
  """Run the notebook with launch keys `cli`; returns (defs, {tab: 'ok' | 'failed'}); texts: a dict to fill with each
  tab's HTML."""
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
      built[name] = 'failed' if ('traceback' in txt
                                 and 'callout' in txt) else 'ok'
      if texts is not None:
        texts[name] = txt
      out[name] = r
    return saved[1](out, value=value, **kw)

  mo.cli_args, mo.ui.tabs = (lambda: CLIArgs(cli)), eager
  try:
    spec = importlib.util.spec_from_file_location(
        'gk_viewer', os.path.join(HERE, 'gk_viewer.py'))
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
    check(
        d['run'].sim == lab and d['run'].name == dflt and not bad
        and len(built) == len(d['TABS']),
        f'{lab}: opens on {dflt}, its last frame {d["k"]}, every tab builds ({time.time() - t:.1f} s){bad or ""}'
    )
  want = {
      'shot': 203899,
      'frame': 15,
      'qty': 'T_elc',
      'mode': 'value',
      'probe': '6,200,7',
      'frames': '0:20',
      'tab': 'Balance & power'
  }
  d, built = headless(want)
  v = d['view']
  check(
      (v['sim'], v['frame'], v['qty'], v['mode'], v['probe'], v['frames'],
       v['tab'])
      == ('Sim 2', 15, 'T_elc', 'value', '6,200,7', '0:20', 'Balance & power')
      and len(d['movie_frames']) == 20 and not d['boot']['notes'],
      f'launch keys {want} open that view (movie: {len(d["movie_frames"])} frames)'
  )
  d, _ = headless({
      'sim': 'Sim 1',
      'run': 'nonesuch',
      'frame': 9999,
      'qty': 'bogus',
      'probe': '99,0,0'
  })
  notes = ' | '.join(d['boot']['notes'])
  check(
      d['run'].name == 'prod_D1_coll' and d['k'] == d['frames'][-1]
      and d['qty_pick'].value == 'n_elc'
      and all(w in notes for w in ('nonesuch', 'bogus', '9999', '99,0,0')),
      f'a launch view that does not fit falls back, a note per key: {notes}')
  # 2026-11-10: blocks, the overlay, the Diffusivity and Options tabs, a single-block run opened with --path
  d, built = headless({
      'sim': 'Sim 2',
      'blocks': '1-3',
      'cmp': '2025 re-run/prod',
      'tab': 'Diffusivity'
  })
  bad = [n for n, v in built.items() if v != 'ok']
  check(
      d['blocks'] == {1, 2, 3} and d['cmp'] is not None
      and d['cmp'].label == '2025 re-run · prod' and d['cmp'].k == 15
      and not bad and not d['boot']['notes'] and d['view']['blocks'] == '1-3'
      and d['view']['cmp'] == '2025 re-run/prod'
      and {'Diffusivity', 'Options'} <= set(built),
      f'blocks 1-3 with the 2025 re-run over Sim 2: the overlay at its frame {d["cmp"].k if d["cmp"] else None} '
      f'(300 us), every tab builds{bad or ""}, the view keeps both')
  d, _ = headless({'sim': 'Sim 1', 'blocks': '99', 'cmp': 'nonesuch/run'})
  notes = ' | '.join(d['boot']['notes'])
  check(
      d['blocks'] is None and d['cmp'] is None and 'blocks' in notes
      and 'nonesuch/run' in notes,
      f'blocks and an overlay that do not fit fall back to none, with notes: {notes}'
  )
  sim2 = gv.open_run(
      os.path.join(gv.BASE,
                   'sims_20261022/sim2_dense_rho09/runs/prod_th02q4_coll'))
  with tempfile.TemporaryDirectory() as tmp:
    sb = os.path.join(tmp, 'single_b2')
    os.makedirs(sb)
    for f in glob.glob(f'{sim2.path}/{sim2.stem}_b2-*.gkyl') + [
        f'{sim2.path}/{sim2.stem}-dt.gkyl'
    ]:
      os.symlink(f, os.path.join(sb, os.path.basename(f).replace('_b2-', '-')))
    saved, gv.CACHE_DIR = gv.CACHE_DIR, os.path.join(tmp, 'cache')
    try:
      d, built = headless({
          'path': sb,
          'cmp': 'Sim 2/prod_th02q4_coll',
          'probe': '0,10,3'
      })
    finally:
      gv.CACHE_DIR = saved
    bad = [n for n, v in built.items() if v != 'ok']
    check(
        d['run'].single and d['run'].path == sb and d['cmp'] is not None
        and d['cmp'].k == d['k'] and not bad and not d['boot']['notes'],
        f'--path: a single-block run (Sim 2 block 2 under single-block names) opens '
        f'with Sim 2 over it, every tab builds{bad or ""}')


# ---------------------------------------------------------------------------------------------------- thomson
def part_thomson():
  import csv
  import gkview as gv
  import thomson
  import viewer_plots as vp
  # the reader against the sim-init workflow's own export of 203899's channels at its TS frame nearest 325 ms
  th = thomson.read(203899)
  ref = os.path.join(gv.BASE, 'nstxu_sim_init_workflow', 'reference_run',
                     'shot203899_ms325',
                     'mpts_fit_channels_exact_geqdsk_rho.csv')
  rows = list(csv.DictReader(open(ref)))
  i = int(np.argmin(np.abs(th.t - 0.325)))
  R = np.array([float(r['RADIUS_M']) for r in rows])
  rel, same = 0.0, True
  for ch, col, unit in ((th.ne, 'FIT_NE_CM3', 1e6), (th.te, 'FIT_TE_KEV', 1e3)):
    want = np.array([
        float(r[col]) * unit if float(r['VALID']) == 1 else np.nan for r in rows
    ])
    rel = max(rel, np.nanmax(np.abs(ch.v[:, i] / want - 1)))
    same &= np.array_equal(R, ch.R) and np.array_equal(np.isnan(want),
                                                       np.isnan(ch.v[:, i]))
  check(
      th.branch == 'TS3' and same and rel < 1e-14,
      f'Thomson reader == the workflow\'s 203899 channel export, n_e and T_e ({th.branch}, frame {i}, '
      f'max rel {rel:.1e})')
  # the flattop rule against the scan's own loop (upstream/elong_tri_range_scan.py), on beam traces with steps,
  # notches and noise
  rng = np.random.default_rng(0)
  agree = 0
  for _ in range(200):
    t = np.linspace(0, 1, 400)
    p = np.clip(
        rng.uniform(1, 6) * ((t > rng.uniform(0.05, 0.3)) &
                             (t < rng.uniform(0.5, 0.95))), 0, None)
    p[rng.integers(0, 400, 3)] -= rng.uniform(0, 1, 3)
    p += rng.normal(0, rng.uniform(0, 0.12), 400)
    sel = [
        i for i in range(len(p))
        if p[i] >= np.max(p) - 0.2 and 15 <= i < len(p) - 15 and all(
            abs(p[j] - p[i]) <= 0.2 for j in range(i - 15, i + 16))
    ]
    want = (t[sel].min(), t[sel].max()) if sel else None
    agree += thomson.flattop(t, p) == want
  check(agree == 200,
        f'beam flattop == the scan\'s loop on {agree}/200 random traces')
  # Sim 2 (203899) draws the Thomson points on its density row, inside the run's span on each side
  lab, d, dflt, _ = gv.SIMS[[s[0] for s in gv.SIMS].index('Sim 2')]
  run = gv.open_run(os.path.join(gv.BASE, d, 'runs', dflt), lab)
  fs = run.frames()
  fr = gv.frame(run, fs[-1])
  t0, t1 = thomson.default_window(th, run.eq_ms / 1e3)
  fig = vp.profiles_figure(run,
                           fr,
                           gv.frame(run, fs[0]),
                           '',
                           'distance',
                           measured=(th, t0, t1))
  cell = _cells(fig)
  pts = {
      cell[tr.xaxis, tr.yaxis]: len(tr.x)
      for tr in fig.data if tr.legendgroup == 'thomson'
  }
  sides = range(1, len(run.layout.stacks) + 1)
  check(
      pts.get((1, 1), 0) > 0 and all(
          (r, c) in pts for r in (1, 2) for c in sides),
      f'Sim 2 profiles: Thomson points on the density and temperature rows, each side ({pts})'
  )


def _cells(fig):
  """{(xaxis, yaxis) of a trace: (row, column)} of a figure made by plotly's make_subplots."""
  grid = fig._grid_ref
  out = {}
  for r in range(1, len(grid) + 1):
    for c in range(1, len(grid[0]) + 1):
      sp = fig.get_subplot(r, c)
      out[sp.xaxis.plotly_name.replace('axis', ''),
          sp.yaxis.plotly_name.replace('axis', '')] = (r, c)
  return out


# ---------------------------------------------------------------------------------- measured profiles (11-16)
# A TRANSP run for part_measured, NOT TRANSP output, made with a postgkyl venv's python (scipy): the variables of a
# real slim TRANSP CDF (argv[1]) and, which the slim copies lack, a SYNTHETIC ND (0.8 NE) and RMAJM (midplane radii
# of the zone boundaries, moving in time), as netCDF-3 classic (argv[2]), leaving out the variables named after
# argv[4:]; then every variable of it as scipy reads it (argv[3], .npz; the units as 'units:<name>')
TRANSP_FIXTURE = r'''
import sys
import numpy as np
from scipy.io import netcdf_file
src, dst, ref, drop = sys.argv[1], sys.argv[2], sys.argv[3], set(sys.argv[4:])
with netcdf_file(src, 'r', mmap=False) as f:
  out = netcdf_file(dst, 'w', version=1)
  out.synthetic = b'gk_viewer check fixture: ND and RMAJM are made up, not TRANSP output'
  for k, n in f.dimensions.items():
    out.createDimension(k, n)
  nb = f.dimensions['XB']
  out.createDimension('RMJSYMB', 2 * nb + 1)
  for name, v in f.variables.items():
    if name not in drop:
      w = out.createVariable(name, v.data.dtype, v.dimensions)
      w[:] = v.data
      for a, val in v._attributes.items():
        setattr(w, a, val)
  t = f.variables['TIME3'].data.astype(float)
  xb = f.variables['XB'].data.astype(float)
  ne = f.variables['NE'].data
  if 'ND' not in drop:
    w = out.createVariable('ND', ne.dtype, ('TIME3', 'X'))
    w[:] = (0.8 * ne).astype(ne.dtype)
    w.units, w.long_name = b'N/CM**3', b'SYNTHETIC: 0.8 NE (a viewer check fixture)'
  if 'RMAJM' not in drop:
    axis = 100 + 3 * np.sin(4 * t)  # cm
    a_out, a_in = 47 + np.cos(3 * t), 68 + 0.5 * np.sin(5 * t)
    shape = 0.75 * xb + 0.25 * xb**2
    R = np.hstack([(axis[:, None] - a_in[:, None] * shape)[:, ::-1], axis[:, None],
                   axis[:, None] + a_out[:, None] * shape])
    w = out.createVariable('RMAJM', np.dtype('>f4'), ('TIME3', 'RMJSYMB'))
    w[:] = R.astype('>f4')
    w.units, w.long_name = b'CM', b'SYNTHETIC: midplane radii (a viewer check fixture)'
  out.close()
with netcdf_file(dst, 'r', mmap=False) as g:
  np.savez(ref, **{k: np.asarray(v.data) for k, v in g.variables.items()},
           **{f'units:{k}': np.array(v._attributes.get('units', b'').decode().strip())
              for k, v in g.variables.items()})
'''
NAMELIST = {  # TRANSP namelists made up for the stand-in archive
    'unset': '! made up for the viewer\'s checks: no lpredict set\nNZONES=40\n',
    'interpretive': 'lpredict_te = 0\nlpredict_ti = 0, lpredict_ne = 0\n',
    'predictive': 'lpredict_te = 0\nlpredict_ti = 0\nlpredict_ne = 0\n~update_time = 0.2\nlpredict_te = 1\n',
}


def _h5_datasets(path, skip=None):
  """{name: value} of every dataset of an HDF5 file, but those under `skip`."""
  import h5py
  out = {}
  with h5py.File(path, 'r') as f:
    f.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(
        o, h5py.Dataset) and not (skip and n.startswith(skip)) else None)
  return out


def _without_transp(src, dst, shot):
  """dst: a copy of the shot file src with no /<shot>/TRANSP group, as Flux's MDSplus dumps are and as a file fetched
  before TRANSP was (the data directory's own copy may hold TRANSP since a later fetch). Returns dst."""
  import shutil
  import h5py
  shutil.copy(src, dst)
  with h5py.File(dst, 'r+') as f:
    if f'{shot}/TRANSP' in f:
      del f[f'{shot}/TRANSP']
  return dst


def _same(a, b):
  try:
    return np.array_equal(a, b, equal_nan=True)
  except TypeError:  # strings
    return np.array_equal(a, b)


def _by_hand(nrow, idx, value, dist, R):
  """Per row with a value: (mean R, mean distance, mean value, count) over the samples idx, in plain loops (the
    definition, against thomson._average)."""
  out = []
  for j in range(nrow):
    got = [(R(j, i), dist(j, i), value(j, i)) for i in idx
           if np.isfinite(value(j, i)) and np.isfinite(dist(j, i))]
    if got:
      out.append([sum(g[q] for g in got) / len(got)
                  for q in range(3)] + [len(got)])
  return np.array(out).reshape(-1, 4)


def part_measured():
  """(2026-11-16) Thomson T_e, CHERS T_i and TRANSP's T_i and n_D (thomson.py, flux_transp.py, fetch_flux.sh): CHERS
    against the sim-init workflow's direct points, its VALID masking, wrong units and axes refused (Thomson left
    standing where CHERS or TRANSP fail); a TRANSP run made up for the check (TRANSP_FIXTURE: the real slim 203899S18
    CDF and a synthetic ND and RMAJM) read as scipy reads it; fetch_flux.sh through a stand-in Flux (part_sync's ssh,
    a banner on stdout; the MDSplus dumps and TRANSP's archive in a temporary directory, the archive with runs that
    each rule step passes over) choosing each shot's run and delivering the files; window means against plain loops;
    the profiles figure's rows, colours and both x modes; the notes for missing sources; the notebook on the files."""
  import csv
  import shutil
  import h5py
  import flux_transp as ft
  import gkview as gv
  import pgkyl_cmd as pc
  import thomson
  import viewer_plots as vp
  wf = os.path.join(gv.BASE, 'nstxu_sim_init_workflow')
  slim = os.path.join(wf, 'data', 'transp', 'NSTU', '16')
  with tempfile.TemporaryDirectory() as tmp:
    # the fixture, and the netCDF-3 reader against scipy on it
    py = os.path.join(os.path.dirname(shutil.which(pc.PGKYL) or pc.PGKYL),
                      'python')
    with open(f'{tmp}/fixture.py', 'w') as f:
      f.write(TRANSP_FIXTURE)
    made = [
        subprocess.run([
            py, f'{tmp}/fixture.py', f'{slim}/203899S18.CDF', f'{tmp}/{nm}.CDF',
            f'{tmp}/{nm}.npz', *drop
        ],
                       capture_output=True,
                       text=True,
                       timeout=300)
        for nm, drop in (('full', ()), ('noND', ('ND', )))
    ]
    if any(p.returncode for p in made):
      check(
          False,
          f'TRANSP fixture: made with {py} (a postgkyl venv: scipy): {[p.stderr[-300:] for p in made]}'
      )
      return
    bad, nvar = [], 0
    for nm in ('full', 'noND'):
      ref, head = np.load(f'{tmp}/{nm}.npz'), ft.netcdf3(f'{tmp}/{nm}.CDF')
      names = [k for k in ref.files if not k.startswith('units:')]
      nvar += len(names)
      bad += [
          f'{nm}:{n}' for n in names if n not in head or not np.array_equal(
              ft.netcdf3_data(f'{tmp}/{nm}.CDF', head[n]), ref[n])
          or str(head[n][2].get('units', '')).strip() != str(ref[f'units:{n}'])
      ] + ([f'{nm}: other names'] if set(names) != set(head) else [])
    check(
        not bad,
        f'TRANSP fixture (the slim 203899S18 CDF + a synthetic ND, RMAJM): flux_transp\'s netCDF-3 reader == '
        f'scipy, values and units, {nvar} variables in 2 files {bad[:4] or ""}')
    # a stand-in Flux: the MDSplus dumps, TRANSP's archive with a run for each step of the rule to pass over
    arch, troot, dest, data = (f'{tmp}/{d}' for d in ('archive', 'transparch',
                                                      'dest', 'data'))
    for d in (arch, f'{troot}/16', data, f'{tmp}/bin'):
      os.makedirs(d)
    # the archive's dumps: the data directory's copies without TRANSP (Flux's MDSplus dumps hold none)
    sources = {
        s: _without_transp(p, f'{arch}/{s}.h5', s)
        for s, p in ((203655, os.path.join(thomson.flux_dir(), '203655.h5')),
                     (203899,
                      os.path.join(thomson.fallback_dir(), '203899.h5')))
    }
    t_old = time.time() - 86400
    runs = {  # run: (its CDF, namelist, TR.MSG, age order)
        '203899S18': ('full', None, None, 0),
        '203899X01': ('full', 'predictive', '', 9),
        '203655A01': ('full', 'unset', ' MCINIT: fine\n', 0),
        '203655A02': ('noND', 'unset', '', 1),
        '203655A03': ('full', 'unset', ' ... TRANSP ABORT ...\n', 2),
        '203655S05': (None, 'interpretive', '', 3),
    }
    for run, (cdf, nl, msg, age) in runs.items():
      base = f'{troot}/16/{run}'
      if cdf:
        os.symlink(f'{tmp}/{cdf}.CDF', f'{base}.CDF')
      if nl is None:  # the real namelist and log
        shutil.copy(f'{slim}/{run}TR.DAT', f'{base}TR.DAT')
        shutil.copy(f'{slim}/{run}TR.MSG', f'{base}TR.MSG')
      else:
        with open(f'{base}TR.DAT', 'w') as f:
          f.write(NAMELIST[nl])
        with open(f'{base}TR.MSG', 'w') as f:
          f.write(msg)
      os.utime(f'{base}TR.DAT', (t_old + 60 * age, t_old + 60 * age))
    with open(f'{tmp}/bin/ssh', 'w') as f:
      f.write(
          STAND_IN_SSH.replace(
              'exec sh -c "$*"',
              'echo "Flux stand-in: welcome"; echo "module: loaded"\nexec sh -c "$*"'
          ))
    os.chmod(f'{tmp}/bin/ssh', 0o755)
    with open(f'{data}/flux.env', 'w') as f:
      f.write(f'FLUX_HOST=stand-in\nFLUX_PY={sys.executable}\n'
              f'FLUX_TOOL={wf}/tools/extract_slim_h5.py\n'
              f'FLUX_TRANSP_ROOT={troot}\nFLUX_H5_ROOT={arch}\n')
    open(f'{tmp}/cm',
         'w').close()  # view.sh --flux's login, as the stand-in keeps one
    env = {k: v for k, v in os.environ.items() if not k.startswith('FLUX_')}
    env.update(GK_VIEWER_DATA=data,
               PATH=f'{tmp}/bin:' + os.environ['PATH'],
               FLUX_CM=f'{tmp}/cm',
               FLUX_DEST=dest,
               TMPDIR=tmp)
    p = subprocess.run(
        ['bash',
         os.path.join(HERE, 'fetch_flux.sh'), '203655', '203899'],
        env=env,
        capture_output=True,
        text=True,
        timeout=900)
    got = {s: f'{dest}/{s}.h5' for s in sources}
    check(
        p.returncode == 0 and all(os.path.exists(g) for g in got.values())
        and 'Flux stand-in: welcome' not in p.stdout + p.stderr
        and '203899: TRANSP 203899S18' in p.stderr,
        f'fetch_flux.sh through a stand-in Flux (banner on stdout; the workflow\'s CSV found from FLUX_TOOL): '
        f'rc {p.returncode}, delivered {sorted(os.listdir(dest)) if os.path.isdir(dest) else None} '
        f'{"" if p.returncode == 0 else p.stderr[-400:]}')
    if p.returncode:
      return
    diff = []
    for s, src in sources.items():
      a, b = _h5_datasets(src), _h5_datasets(got[s], skip=f'{s}/TRANSP')
      diff += [f'{s}: {set(a) ^ set(b)}'] if set(a) != set(b) else []
      diff += [f'{s}: {n}' for n in a if n in b and not _same(a[n], b[n])]
    check(
        not diff,
        f'fetch: Thomson, EFIT01, CHERS (and beam power) == the archive\'s, dataset for dataset, both shots '
        f'{diff[:4] or ""}')
    # the run each shot got, and why
    att = {}
    for s in sources:
      with h5py.File(got[s], 'r') as f:
        g = f[f'{s}/TRANSP']
        att[s] = {k: g.attrs[k] for k in g.attrs}
    txt = lambda v: [thomson._text(x) for x in np.atleast_1d(v)]
    c55, c99 = ' | '.join(txt(att[203655]['considered'])), ' | '.join(
        txt(att[203899]['considered']))
    check(
        thomson._text(att[203899]['run']) == '203899S18'
        and 'workflow' in thomson._text(att[203899]['why'])
        and '203899X01: namelist predictive' in c99
        and txt(att[203899]['predicts']) == ['lpredict_te', 'lpredict_ti']
        and np.allclose(att[203899]['predicts_from'], 0.128287)
        and thomson._text(att[203655]['run']) == '203655A01'
        and 'set no lpredict' in thomson._text(att[203655]['why'])
        and '203655S05: namelist interpretive; no CDF' in c55
        and '203655A03: namelist unset; TR.MSG: TRANSP ABORT' in c55
        and '203655A02: namelist unset; no ND' in c55
        and '203655A01: namelist unset; CHOSEN' in c55,
        f'TRANSP run choice: 203899 -> {txt(att[203899]["run"])} (the workflow\'s CSV; predicts '
        f'{txt(att[203899]["predicts"])} from {att[203899]["predicts_from"]} s), 203655 -> '
        f'{txt(att[203655]["run"])} ({c55})')
    none = ft.choose(203899, troot)
    check(
        none[0] == '' and none[2] == 'no run qualifies'
        and ft.choose(203655, troot, '203655Z99')[3][-1].startswith(
            '203655Z99: the sim-init workflow\'s choice: not in the archive'),
        f'TRANSP run choice without the CSV: 203899 (both runs predictive) gets none ({none[2]!r}); a CSV run '
        'missing from the archive is said so')
    # the delivered TRANSP == the fixture; the reader's units and midplane geometry
    ref = np.load(f'{tmp}/full.npz')
    head = ft.netcdf3(f'{tmp}/full.CDF')
    with h5py.File(got[203899], 'r') as f:
      g = f['203899/TRANSP']
      bad = [
          n for n in ft.VARIABLES
          if not np.array_equal(g[n]['data'][()], ref[n])
          or thomson._text(g[n]['units'][()]) != str(ref[f'units:{n}'])
          or tuple(txt(g[n]['dims'][()])) != head[n][0]
      ]
    th = thomson.load(got[203899])
    tr, nb = th.transp, ref['XB'].shape[1]
    rm = ref['RMAJM'].astype(float) / 100
    mid = lambda b: 0.5 * (b[:, :-1] + b[:, 1:]
                           ).T  # the fixture's X: midway between its XB
    geo = (np.array_equal(tr.edge['outboard'], rm[:, -1])
           and np.array_equal(tr.edge['inboard'], rm[:, 0])
           and np.allclose(tr.R['outboard'], mid(rm[:, nb:]), rtol=0, atol=1e-6)
           and np.allclose(
               tr.R['inboard'], mid(rm[:, nb::-1]), rtol=0, atol=1e-6))
    check(
        not bad and geo and np.array_equal(tr.ti, ref['TI'].T.astype(float))
        and np.array_equal(tr.nd, ref['ND'].T.astype(float) * 1e6)
        and tr.predicts == {
            'lpredict_te': 0.128287,
            'lpredict_ti': 0.128287
        },
        f'TRANSP delivered == the fixture ({len(ft.VARIABLES)} variables, units, dims {bad or ""}); read: n_D = '
        f'1e6 ND (m^-3), T_i in eV, the boundary at RMAJM\'s ends, each zone centre midway between its boundaries '
        f'({geo}), predicts {tr.predicts}')
    # CHERS: against the workflow's direct points at 325 ms; VALID masking
    pts = [
        r for r in csv.DictReader(
            open(
                os.path.join(wf, 'reference_run',
                             'transp_chers_rhoi_direct_chers_points.csv')))
        if r['SHOT'] == '203899'
    ]
    by = {c.name: c for c in th.chers}
    rel = 0.0
    for r in pts:
      c = by[r['CHERS_SYSTEM']]
      i, j = int(np.argmin(np.abs(c.t - float(r['NEAREST_TIME_S'])))), int(
          r['CHANNEL_INDEX'])
      rel = max(rel, abs(c.v[j, i] / float(r['TI_EV']) - 1),
                abs(c.err[j, i] / float(r['DTI_EV']) - 1),
                abs(c.R[j] / float(r['R_M']) - 1))
    check(
        len(pts) > 0 and rel < 1e-14,
        f'CHERS reader == the workflow\'s direct CHERS points (203899 @ 325 ms, CT1 and CT2): {len(pts)} points, '
        f'R, T_i, dT_i to max rel {rel:.1e}')
    pre = thomson.CHERS[1]
    masks = []
    with h5py.File(got[203899], 'r') as f:
      for c in th.chers:
        raw = {
            n: f[f'203899/NSTX/{pre}{c.name}:{n}/data'][()]
            for n in ('TI', 'DTI', 'VALID')
        }
        want = (raw['VALID'] == 1) & (raw['TI'] > 0) & (raw['DTI'] >= 0)
        masks.append((c.name, int(want.sum()), want.size,
                      np.array_equal(want.T, np.isfinite(c.v))))
    check(
        len(masks) == 2 and all(m[3] for m in masks),
        f'CHERS: a point counts exactly where VALID == 1, T_i > 0, dT_i >= 0 (analysis, counted, of): {masks}'
    )

    # units and axes refused: CHERS or TRANSP give way with a reason, Thomson stays; a wrong MPTS unit refuses all
    def broken(name, edit):
      path = f'{tmp}/{name}.h5'
      shutil.copy(got[203899], path)
      with h5py.File(path, 'a') as f:
        edit(f['203899'])
      return path

    def relabel(g, units):
      del g['units']
      g['units'] = units

    def swap(g):
      a, b = g['dim0'][()], g['dim1'][()]
      del g['dim0'], g['dim1']
      g['dim0'], g['dim1'] = b, a

    cases = [
        ('chers_units', lambda s: relabel(s['NSTX'][f'{pre}CT2:TI'], 'eV'),
         'chers', 'units'),
        ('chers_axes', lambda s: swap(s['NSTX'][f'{pre}CT1:TI']), 'chers',
         '(sample, channel)'),
        ('transp_units', lambda s: relabel(s['TRANSP/TI'], 'KEV'), 'transp',
         'units'),
        ('transp_rmajm',
         lambda s: s['TRANSP/RMAJM/data'].write_direct(s['TRANSP/RMAJM/data'][
             ()][:, ::-1].copy()), 'transp', 'RMAJM'),
    ]
    out = []
    for name, edit, src, word in cases:
      b = thomson.load(broken(name, edit))
      why = b.chers_why if src == 'chers' else b.transp_why
      gone = b.chers == () if src == 'chers' else b.transp is None
      out.append((name, gone and 'unreadable' in why and word in why
                  and b.branch == 'TS3' and np.isfinite(b.te.v).any()))
    try:
      thomson.load(
          broken(
              'mpts_units', lambda s: relabel(
                  s['ACTIVESPEC'][f'{thomson.MPTS}TS3:FIT_TE'], 'eV')))
      out.append(('mpts_units', False))
    except ValueError:
      out.append(('mpts_units', True))
    check(
        all(o for _, o in out),
        f'wrong units / axes: CHERS (TI in eV; dims swapped) and TRANSP (TI in KEV; RMAJM reversed) give way with '
        f'the reason, Thomson still read; MPTS T_e in eV refused: {out}')
    # window means against plain loops: 3 Thomson frames around 325 ms, the band half a frame past them
    i = int(np.argmin(np.abs(th.t - 0.325)))
    t0, t1, t_ref = th.t[i - 1], th.t[i + 1], 0.325
    rsep_ref = th.separatrix('outboard', t_ref)
    h = 0.5 * np.median(np.diff(th.t))
    lo, hi = t0 - h, t1 + h
    ps = {
        (p.source, p.quantity): p
        for p in thomson.profiles(th, 'outboard', t0, t1, t_ref, rsep_ref)
    }
    sep = lambda t: np.interp(t, th.efit_t, th.r_out)
    on = lambda R, t: R > 0.5 * (np.interp(t, th.efit_t, th.r_in) + sep(t))
    dist = lambda ch: lambda j, i: (ch.R[j] -
                                    (rsep_ref + sep(ch.t[i]) - sep(t_ref))
                                    if on(ch.R[j], ch.t[i]) else np.nan)
    inb = lambda t: [i for i in range(len(t)) if lo <= t[i] <= hi]
    cts = {
        c.name:
        sum(np.isfinite(c.v[j, i]) for j in range(len(c.R)) for i in inb(c.t))
        for c in th.chers
    }
    ch = by[max(cts, key=lambda n: (cts[n], n == 'CT2'))]
    want = {
        ('Thomson', 'T_e'):
        (_by_hand(len(th.te.R),
                  [i for i in range(len(th.t)) if t0 <= th.t[i] <= t1],
                  lambda j, i: th.te.v[j, i], dist(th.te),
                  lambda j, i: th.te.R[j]), 3),
        ('CHERS', 'T_i'): (_by_hand(len(ch.R), inb(ch.t),
                                    lambda j, i: ch.v[j, i], dist(ch),
                                    lambda j, i: ch.R[j]), len(inb(ch.t))),
    }
    for q, v in (('T_i', tr.ti), ('n_D', tr.nd)):
      want['TRANSP', q] = (_by_hand(
          v.shape[0], inb(tr.t), lambda j, i: v[j, i],
          lambda j, i: tr.R['outboard'][j, i] - tr.edge['outboard'][i],
          lambda j, i: tr.R['outboard'][j, i]), len(inb(tr.t)))
    bad = []
    for k, (w, n) in want.items():
      p = ps.get(k)
      if p is None or p.samples != n or len(w) != len(p.d) or not np.allclose(
          np.c_[p.R, p.d, p.v, p.n], w, rtol=1e-12, atol=1e-12):
        bad.append(k)
    check(
        not bad and thomson.band(th, t0, t1) == (lo, hi)
        and ps['CHERS', 'T_i'].label == f'CHERS {ch.name}',
        f'window means == plain loops (203899, frames {t0 * 1e3:.1f}-{t1 * 1e3:.1f} ms, band {lo * 1e3:.1f}-'
        f'{hi * 1e3:.1f} ms): ' +
        ', '.join(f'{k[0]} {k[1]} {len(w)} rows of {n} samples'
                  for k, (w, n) in want.items()) + f' {bad or ""}')
    # the profiles figure: each source on its rows and species, both x modes the same points
    lab, d, dflt, _ = gv.SIMS[[s[0] for s in gv.SIMS].index('Sim 2')]
    run = gv.open_run(os.path.join(gv.BASE, d, 'runs', dflt), lab)
    fs = run.frames()
    fr, fr0 = gv.frame(run, fs[-1]), gv.frame(run, fs[0])
    t0, t1 = thomson.default_window(th, run.eq_ms / 1e3)
    pal = vp.LOOKS['dark']
    figs = {
        x: vp.profiles_figure(run, fr, fr0, '', x, measured=(th, t0, t1))
        for x in ('distance', 'R')
    }
    cell, seen, moved = _cells(figs['distance']), {}, []
    sides = [s for s in ('outboard', 'inboard') if s in run.layout.stacks]
    for a, b in zip(figs['distance'].data, figs['R'].data):
      if a.legendgroup not in ('thomson', 'chers', 'transp'):
        continue
      r, c = cell[a.xaxis, a.yaxis]
      line = a.mode == 'lines'
      seen.setdefault(a.legendgroup, set()).add(
          (r, a.line.color if line else a.marker.color,
           a.line.dash if line else None))
      rsep = gv.midplane_profile(run, fr, sides[c - 1], 'n_elc')[2]
      sign = 1.0 if sides[c - 1] == 'outboard' else -1.0
      moved.append(
          np.allclose(b.x, rsep + sign * np.asarray(a.x) / 1e3, atol=1e-9))
    want = {
        'thomson': {(1, pal.elc, None), (2, pal.elc, None)},
        'chers': {(2, pal.ion, None)},
        'transp': {(1, pal.ion, 'solid'), (2, pal.ion, 'dot')}
    }
    keys = [t.name for t in figs['distance'].data if t.legendgroup == 'key']
    note = vp.measured_note(run, fr, th, t0, t1)
    words = (
        'TRANSP run 203899S18', 'T_e from 128 ms, T_i from 128 ms',
        f'CHERS T_i ({thomson.chers_in(th, *thomson.band(th, t0, t1)).name}',
        'nothing in the SOL')
    check(
        seen == want and all(moved) and len(moved) >= 5
        and 'TRANSP, predicted by its model' in keys and all(w in note
                                                             for w in words),
        f'profiles figure (Sim 2 on the fetched 203899): rows / colours / dashes {seen == want} '
        f'({ {k: sorted(v) for k, v in seen.items()} }), {len(moved)} measured traces the same points in both x '
        f'modes ({sum(moved)}), the predicted key, the note names each source '
        f'({[w for w in words if w not in note] or "all"})')
    # the notes where a source is missing or has nothing in the window
    lab1, d1, dflt1, _ = gv.SIMS[[s[0] for s in gv.SIMS].index('Sim 1')]
    run1 = gv.open_run(os.path.join(gv.BASE, d1, 'runs', dflt1), lab1)
    fr1 = gv.frame(run1, run1.frames()[-1])

    def note_of(th1):
      w = thomson.default_window(th1, run1.eq_ms / 1e3)
      return vp.measured_note(run1, fr1, th1, *w), thomson.profiles(
          th1, 'outboard', *w, run1.eq_ms / 1e3,
          gv.midplane_profile(run1, fr1, 'outboard', 'n_elc')[2])

    empty = f'{tmp}/no_run_203655.h5'
    shutil.copy(got[203655], empty)
    ft.record(empty, 203655, '', 'no run qualifies', ['203655A01: made up'],
              troot)
    nochers = broken('no_chers', lambda s: s.__delitem__('NSTX'))
    cases = {
        'fetched 203655 (CHERS ends at 216 ms)':
        (thomson.load(got[203655]), 'CHERS T_i: no sample from', 'CHERS'),
        'the Flux copy fetched before TRANSP':
        (thomson.load(sources[203655]), 'TRANSP: this file has none', 'TRANSP'),
        'no run qualified':
        (thomson.load(empty), 'TRANSP: none for this shot (no run qualifies)',
         'TRANSP'),
        'no CHERS in the file':
        (thomson.load(nochers), 'CHERS T_i: none in this file.', 'CHERS'),
    }
    out = {}
    for name, (th1, words1, src) in cases.items():
      n1, p1 = note_of(th1)
      out[name] = words1 in n1 and src not in {p.source for p in p1}
    check(
        all(out.values()),
        f'a source missing or with nothing in the window: the note says so, nothing drawn for it: {out}'
    )
    # the notebook on the fetched files (Sim 2: TRANSP 203899S18) and on a copy fetched before TRANSP (Sim 1)
    keep = thomson.flux_dir
    thomson.flux_dir = lambda: dest
    try:
      d2, b2 = headless({'sim': 'Sim 2'})
    finally:
      thomson.flux_dir = keep
    before = f'{tmp}/before_transp'
    os.makedirs(before)
    _without_transp(sources[203655], f'{before}/203655.h5', 203655)
    thomson.flux_dir = lambda: before
    try:
      d1, b1 = headless({'sim': 'Sim 1'})
    finally:
      thomson.flux_dir = keep
    bad = [n for n, v in {**b1, **b2}.items() if v != 'ok']
    check(
        not bad and d2['th'].transp.run == '203899S18'
        and d2['th'].path == got[203899] and d1['th'].transp_why is None
        and len(b1) == len(d1['TABS']) and len(b2) == len(d2['TABS']),
        f'notebook: Sim 2 on the fetched 203899 (TRANSP {d2["th"].transp.run if d2["th"].transp else None}) and '
        f'Sim 1 on the copy without TRANSP, every tab builds ({len(b2)} + {len(b1)} tabs) {bad or ""}'
    )


# ---------------------------------------------------------------------------------------------------- exports
def part_exports():
  import imageio_ffmpeg
  import gkview as gv
  import viewer_plots as vp
  import viewer_state as vs
  tmp = tempfile.mkdtemp(prefix='check_viewer_')
  view = {
      'sim': 'Sim 2',
      'run': 'prod_th02q4_coll',
      'frame': 15,
      'qty': 'T_elc',
      'frames': '0:20'
  }
  check(
      vs.load_state(vs.save_state(f'{tmp}/v.json', view)) == view,
      'a state file round-trips')
  json.dump({'other': 1}, open(f'{tmp}/bad.json', 'w'))
  try:
    vs.load_state(f'{tmp}/bad.json')
    check(False, 'a foreign json is refused as a state file')
  except ValueError:
    check(True, 'a foreign json is refused as a state file')
  r = gv.open_run(
      f'{gv.BASE}/sims_20261022/sim2_dense_rho09/runs/prod_th02q4_coll',
      'Sim 2')
  summ = gv.summary(r)
  for ext in ('png', 'pdf', 'svg'):
    fig = vp.map_figure(r,
                        gv.frame(r, 30),
                        gv.frame(r, 0),
                        'T_elc',
                        'value',
                        pal=vp.LOOKS['light'])
    path = vp.save_figure(fig, f'{tmp}/fig.{ext}')
    vp.plt.close(fig)
    font = ext != 'svg' or all(
        f"'{real}', '{drawn}'" in open(path).read()
        for drawn, real in getattr(fig, 'gkv_fonts', {}).items())
    check(
        os.path.getsize(path) > 10000 and font,
        f'figure saves as {ext} ({os.path.getsize(path)} bytes)' +
        (', the theme font named first' if ext == 'svg' else ''))
  path = vp.render_movie(r,
                         summ,
                         list(range(10)),
                         'n_elc',
                         'step',
                         False,
                         'whole domain',
                         vp.LOOKS['dark'],
                         f'{tmp}/m.mp4',
                         fps=10)
  info = subprocess.run([
      imageio_ffmpeg.get_ffmpeg_exe(), '-i', path, '-map', '0:v', '-f', 'null',
      '-'
  ],
                        capture_output=True,
                        text=True).stderr
  nfr = re.findall(r'frame=\s*(\d+)', info)
  check('h264' in info and nfr and int(nfr[-1]) == 10,
        f'a 10-frame movie is H.264 with {nfr[-1] if nfr else "?"} '
        'frames')
  # 2026-11-10: a block choice and the diffusivity map save too; the movie's data for a block choice draws only
  # those blocks' cells, on their own colour scale; a state file keeps the new keys
  import movie_widget as mm
  fig = vp.map_figure(r,
                      gv.frame(r, 30),
                      gv.frame(r, 0),
                      'n_elc',
                      'value',
                      pal=vp.LOOKS['dark'],
                      blocks={1, 2, 3})
  p1 = vp.save_figure(fig, f'{tmp}/blocks.png')
  vp.plt.close(fig)
  fig = vp.diffusivity_map(r, gv.diffusivity(r), pal=vp.LOOKS['dark'])
  p2 = vp.save_figure(fig, f'{tmp}/D.png')
  vp.plt.close(fig)
  D = mm.movie_data(r, summ, 'n_elc', 'value', blocks={1, 2, 3})
  rz = np.frombuffer(base64.b64decode(D['rz']), np.int32)
  want = gv.region_cells(r, [1, 2, 3])
  in_strips = np.concatenate([
      np.frombuffer(base64.b64decode(st['cells']), np.int32)
      for st in D['strips']
  ])
  check(
      os.path.getsize(p1) > 10000 and os.path.getsize(p2) > 10000
      and np.array_equal(rz, want) and set(in_strips) == set(want)
      and all(st['sep'] == 4.0 for st in D['strips']),
      f'blocks 1-3: the figure and the D map save; the movie draws their {len(want)} cells (plane and strips) only'
  )
  view2 = dict(view, blocks='1-3', cmp='2025 re-run/prod', path='')
  check(
      vs.load_state(vs.save_state(f'{tmp}/v2.json', view2)) == view2
      and vs.parse_blocks('1-3,7', 8) == {1, 2, 3, 7}
      and vs.blocks_text({1, 2, 3, 7}) == '1-3,7',
      'a state file keeps blocks and the overlay; blocks text round-trips (1-3,7)'
  )


# ----------------------------------------------------------------------------------- labels, pgkyl, themes (11-10)
def _charts(run, summ, cmp=None):
  """Every plotly chart the tabs draw for a run, at its last frame (with an overlay where given)."""
  import gkview as gv
  import thomson
  import viewer_plots as vp
  fs = run.frames()
  k = fs[-1]
  fr, fr0, t = gv.frame(run, k), gv.frame(run, fs[0]), run.time(k)
  out = {
      'profiles':
      vp.profiles_figure(run, fr, fr0, '', cmp=cmp),
      'history':
      vp.history_figure(run, summ, f'n_{run.species[0]}', 'change', t),
      'targets':
      vp.targets_figure(run, fr, summ, t, cmp=cmp),
      'time traces':
      vp.timetraces_figure(run,
                           summ,
                           gv.dt_history(run),
                           gv.integrated(run),
                           gv.field_energy(run),
                           t,
                           cmp=cmp,
                           blocks={0, 1}),
      'probe':
      vp.probe_figure(run,
                      summ,
                      gv.cell_offsets(run)[min(1, run.nb - 1)] + 2,
                      t,
                      cmp=cmp)  # block 1, or a single block's
  }
  B = gv.balance(run)
  if B is not None:
    Bc = gv.balance(cmp.run) if cmp else None
    out['conservation'] = vp.conservation_figure(
        run,
        B,
        summ['t'],
        t,
        cmp=cmp,
        B_cmp=Bc,
        t_frames_cmp=cmp.summ['t'] if cmp else None)[0]
    out['balance'] = vp.balance_figure(run, B, t, cmp=cmp, B_cmp=Bc)[0]
  D = gv.diffusivity(run)
  if D['kind']:
    out['diffusivity'] = vp.diffusivity_figure(
        run, D, cmp=cmp, D_cmp=gv.diffusivity(cmp.run) if cmp else None)
  try:
    th = thomson.read(
        run.shot
    )  # None: no file holds the shot (a device run: shot 0), as the notebook reads it
    if th is not None:
      t0, t1 = thomson.default_window(th, run.eq_ms / 1e3)
      out['thomson'] = vp.thomson_timeline(th, t0, t1, run.eq_ms / 1e3)
  except (OSError, KeyError, ValueError):
    pass
  return out


def _overlay(summ, other):
  """A run drawn over the one of summary summ (a viewer_plots.Cmp, as the notebook makes it): at its frame matched to
  that run's last."""
  import gkview as gv
  import viewer_plots as vp
  s2 = gv.summary(other)
  idx = gv.match_frames(summ['t'], s2['t'])
  j = int(idx[-1])
  return vp.Cmp(other,
                s2,
                other.name,
                fr=gv.frame(other, int(s2['frames'][j])) if j >= 0 else None,
                fr0=gv.frame(other, int(s2['frames'][0])),
                k=int(s2['frames'][j]) if j >= 0 else -1,
                hist=(gv.dt_history(other), gv.integrated(other),
                      gv.field_energy(other)),
                idx=idx)


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
    figs = [
        ('map', dict()),
        ('titled',
         dict(title=
              f'{run.title} · {run.sim} · {run.name} · a quantity   t = 1.234 '
              'ms   (frame 123)')), ('blocks', dict(blocks=set(sol)))
    ]
    figs += [(z, dict(zoom=z)) for z in vp.zoom_boxes(run)
             if z != 'whole domain']
    for name, kw in figs:
      fig = vp.map_figure(run, fr, fr0, 'q_par', 'value', **kw)
      fig.canvas.draw()
      r = fig.canvas.get_renderer()
      skip = {
          id(lb)
          for ax in fig.axes
          for axis in (ax.xaxis, ax.yaxis)
          for tk in axis.get_major_ticks() + axis.get_minor_ticks()
          for lb in (tk.label1, tk.label2)
          if not min(axis.get_view_interval()) -
          1e-9 <= tk.get_loc() <= max(axis.get_view_interval()) + 1e-9
      }
      boxes = [(t.get_text(), t.get_window_extent(r)) for t in fig.findobj(Text)
               if t.get_visible() and t.get_text().strip() and id(t) not in skip
               ]
      W, H = fig.bbox.width, fig.bbox.height
      for (ta, a), (tb, b) in itertools.combinations(boxes, 2):
        if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(
            a.y0, b.y0) > 1:
          bad.append(f'{lab} {name}: {ta[:30]!r} over {tb[:30]!r}')
      bad += [
          f'{lab} {name}: {t[:30]!r} off the figure' for t, b in boxes
          if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1
      ]
      n += len(boxes)
      vp.plt.close(fig)
  check(
      not bad,
      f'map figures: {n} texts over 3 simulations (whole, zoomed, titled, blocks), none over another '
      f'or off the figure{bad[:6] or ""}')
  bad, nfig = [], 0
  for lab, d, dflt, _ in gv.SIMS:
    run = gv.open_run(os.path.join(gv.runs_root(d), dflt), lab)
    for name, fig in _charts(run, gv.summary(run)).items():
      nfig += 1
      h = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
      for a in fig.layout.annotations:
        if a.yref == 'paper' and not (a.yanchor == 'bottom'
                                      and a.font.size == vp.SUBTITLE):
          bad.append(f'{lab} {name}: a free note {a.text[:30]!r}')
      titled = any(a.yref == 'paper' and a.font.size == vp.SUBTITLE
                   for a in fig.layout.annotations)
      if titled and (fig.layout.legend.y - 1) * h < vp.SUBTITLE * 1.3 + 6:
        bad.append(
            f'{lab} {name}: legend {(fig.layout.legend.y - 1) * h:.0f} px above the plots'
        )
  check(
      not bad,
      f'{nfig} charts: above the plots only subplot titles and the legend, the legend clear of the '
      f'titles{bad[:6] or ""}')


def part_pgkyl():
  """Every curve has its pgkyl command; two of them run and give the viewer's numbers."""
  import gkview as gv
  import movie_widget as mm
  import pgkyl_cmd as pc
  bad, ncurve = [], 0
  runs = [
      gv.open_run(os.path.join(gv.runs_root(d), dflt), lab)
      for lab, d, dflt, _ in gv.SIMS
  ]
  for n, run in enumerate(runs):
    summ = gv.summary(run)
    cmp = _overlay(summ, runs[(n + 1) % len(runs)])
    for name, fig in _charts(run, summ, cmp).items():
      for tr in fig.data:
        if tr.legendgroup == 'key' or tr.x is None or all(
            v is None for v in tr.x):  # legend entries only
          continue
        ncurve += 1
        m = tr.meta if isinstance(tr.meta, dict) else {}
        cmd = m.get('pgkyl')
        if not cmd or not isinstance(cmd, str):
          bad.append(f'{run.sim} {name}: {tr.name!r} has no command')
          continue
        left = set(re.findall(r'\{(\w+)\}',
                              cmd)) - {'frame', 'cmp_frame', 'point_frame'}
        if left:
          bad.append(f'{run.sim} {name}: {tr.name!r} leaves {left}')
        if '{point_frame}' in cmd and len(m.get('pgkyl_frames') or []) != len(
            tr.x):
          bad.append(
              f'{run.sim} {name}: {tr.name!r} has no frame for each point')
    D = mm.movie_data(run, summ, 'T_elc', 'change')
    if not all(x in D['pg']['map'] for x in ('{b}', '{frame}', '{ref}')) or \
            not all(x in D['pg']['cell'] for x in ('{b}', '{i}', '{j}', '{frame}')):
      bad.append(f'{run.sim}: the movie\'s cell templates')
  check(
      not bad,
      f'{ncurve} curves over every chart of 3 simulations (overlays too): each has its pgkyl command, '
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
    steps += [
        x for o in outs
        for x in (f'activate -t {o}', f'save -e npy -o {tmp}/{nm}_{o}')
    ]
    p = subprocess.run(['bash', '-c', ' \\\n  '.join(steps)],
                       capture_output=True,
                       text=True,
                       env=env)
    return [np.load(f'{tmp}/{nm}_{o}.npy')
            for o in outs] if not p.returncode else None

  with tempfile.TemporaryDirectory() as tmp:
    got = [
        results(pc.cell_value(run, b, run.file_row(b, i), j, 'T_elc', 20), tmp,
                'cell'),
        results(pc.midplane(run, side, 'n_ion', 20), tmp, 'mid'),
        results(pc.cell_history(run, b, run.file_row(b, i), j, 'T_elc'), tmp,
                'hist'),
        results(pc.particles(run, 'elc'), tmp, 'parts')
    ]
    want_mid = np.concatenate([
        run.rows(bb, fr[bb]['n_ion'])[:, pc._mid_column(run, bb, st.j)] * 1e-19
        for bb in gv.stack_blocks(st)
    ])
    want_hist = [gv.frame(run, kk)[b]['T_elc'][i, j] for kk in run.frames()]
    want_parts = sum(
        gv.read_dynvec(run.f(bb, 'elc_integrated_moms'))[1][:, 0]
        for bb in range(run.nb)
        if os.path.exists(run.f(bb, 'elc_integrated_moms')))
    ok = (None not in got and np.isclose(
        float(np.ravel(got[0][0])[0]), fr[b]['T_elc'][i, j], rtol=1e-12)
          and np.allclose(np.concatenate([np.ravel(g) for g in got[1]]),
                          want_mid,
                          rtol=1e-12)
          and np.allclose(np.ravel(got[2][0]), want_hist, rtol=1e-12)
          and np.allclose(np.ravel(got[3][0]), want_parts, rtol=1e-12))
    check(
        ok,
        f'pgkyl runs four of them (Sim 2): a cell\'s T_e at frame 20 and at every frame, the {side} midplane '
        'n_i, the electrons\' number: the viewer\'s numbers (rtol 1e-12)')
    tg = run.layout.targets[0]
    figs = {
        'block map': pc.block_map(run, b, 'n_elc', 20),
        'change map': pc.block_map(run, b, 'T_elc', 20, 'change', 0),
        'midplane history': pc.midplane(run, side, 'n_ion', history=True),
        'plate': pc.plate(run, tg, 20),
        'dt': pc.dt(run),
        'field energy': pc.field_energy(run),
        'target power': pc.target_power(run, tg),
        'particle balance': pc.balance(run),
        'energy balance': pc.balance(run, True),
        'D map': pc.diffusivity(run, b, 'elc'),
        'midplane D': pc.diffusivity(run, None, 'elc', side)
    }
    failed = []
    for nm, cmd in figs.items():
      png = f'{tmp}/{nm.replace(" ", "_")}.png'
      s = re.sub(
          r'(\b(?:plot|gk_\w+_balance)\b[^\\\n]*)', lambda m: m.group(1) +
          (' --no_show'
           if m.group(1).startswith('plot') else '') + f' --saveas {png}',
          bare(cmd or ''))
      p = subprocess.run(['bash', '-c', s],
                         capture_output=True,
                         text=True,
                         env=env)
      if not cmd or p.returncode or not os.path.exists(png):
        failed.append(
            f'{nm}: {p.stderr.strip()[-200:] if cmd else "no command"}')
    check(
        not failed,
        f'every other kind of pgkyl command runs to a figure ({len(figs)} kinds){failed or ""}'
    )


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
    got[theme] = (d['pal'].name, d['view']['theme'],
                  all(v == 'ok' for v in built.values()))
  check(
      got == {
          'PPPL': ('pppl-light', 'PPPL', True),
          'diii-d': ('diii-d-dark', 'DIII-D', True)
      } and 'theme' in vs.KEYS,
      f'--theme PPPL --dark false / --theme diii-d open those looks, every tab builds: {got}'
  )
  files = {t: f for t, f in (gv.CONFIG.get('theme_files') or {}).items()}
  read = {
      t: vp.read_theme(os.path.join(gv.BASE, f))
      for t, f in files.items() if os.path.exists(os.path.join(gv.BASE, f))
  }
  navy = dict(vp.SCHEMES['DIII-D'][0])
  navy.pop('navy')
  kept = {
      t: (navy, vp.SCHEMES[t][1]) if t == 'DIII-D' else vp.SCHEMES[t]
      for t in read
  }
  check(
      set(files) <= set(vp.SCHEMES) and kept == read and all(
          os.path.basename(f) == vp.THEME_SOURCES[t] for t, f in files.items()),
      f'the themes kept in viewer_plots == their PowerPoint files ({", ".join(sorted(read)) or "none here"})'
  )
  css = {t: vp.page_css(t) for t in vp.THEME_LOOKS}
  check(
      set(css) == {'NSTX-U', 'PPPL', 'DIII-D'}
      and 'Roboto Condensed' in css['PPPL']
      and 'Century Gothic' in css['DIII-D']
      and vp.LOOKS['pppl-dark'].surf in css['PPPL']
      and vp.LOOKS['diii-d-dark'].surf in css['DIII-D']
      and all('--background' in c for c in css.values()),
      'each theme\'s page CSS: its fonts (PPPL Roboto Condensed titles, DIII-D Century Gothic) and dark surface'
  )
  pal = vp.LOOKS['pppl-dark']
  rows = [
      dict(compare=c,
           option=f'o{i}',
           this_run='1',
           overlay='1' if c == 'same' else '2',
           set_by='')
      for i, c in enumerate(['differs', 'same', 'same', 'differs', 'same'])
  ]
  tint = vp._compare_tints(pal)
  ok = True
  for order in (rows, rows[::-1]):
    t = mo.ui.table(order, selection=None, style_cell=vp.compare_style(pal))
    styles = t._style_cells(0, len(order), len(order))
    ok &= all(styles[str(i)][col].get('backgroundColor') == tint[r['compare']]
              for i, r in enumerate(order) for col in r)
  check(
      ok,
      'options side by side: every cell of a row tinted by its compare value (differs / same), either order'
  )


# ------------------------------------------------------------------------------------------------- fonts (11-13)
def part_fonts():
  """The symbols the sources write that a theme face lacks (∥, ⊥, ...) are all in Viewer Symbols, whose tracked
    files equal a fresh build (fonts/make_symbols.py); the page receives both faces; text_px measures each symbol, in
    every theme's fonts, without a missing glyph."""
  import warnings
  import movie_widget as mw
  import viewer_plots as vp
  spec = importlib.util.spec_from_file_location(
      'make_symbols', os.path.join(HERE, 'fonts', 'make_symbols.py'))
  ms = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(ms)
  need = ms.lacking(ms.source_chars())
  tracked = {out: os.path.join(ms.FONTS, out) for out, _, _ in ms.FACES}
  short = {
      out:
      ''.join(c for c in need
              if ord(c) not in ms.cmap(p)) if os.path.exists(p) else 'no file'
      for out, p in tracked.items()
  }
  check(
      not any(short.values()),
      f'Viewer Symbols has the {len(need)} symbols the theme fonts lack '
      f'(short: {short}; else run fonts/make_symbols.py)')
  with tempfile.TemporaryDirectory() as tmp:
    same = {}
    for out, face, style in ms.FACES:
      ms.build(face, style, need, os.path.join(tmp, out))
      same[out] = (os.path.exists(tracked[out])
                   and open(os.path.join(tmp, out), 'rb').read() == open(
                       tracked[out], 'rb').read())
  check(all(same.values()),
        f'the tracked Viewer Symbols files == a fresh build: {same}')
  sent = {(f['family'], f['weight']) for f in mw.font_loader().faces}
  check({(vp.SYMBOLS, 400), (vp.SYMBOLS, 700)} <= sent and all(
      vp.SYMBOLS in vp.css_font(f) for t in vp.THEME_LOOKS
      for f in vp.theme_fonts(t)
  ), 'the page gets both Viewer Symbols faces, and every theme font\'s CSS stack falls back to them'
        )
  from matplotlib import font_manager
  found = {
      f'{fam} {w}':
      font_manager.findfont(font_manager.FontProperties(family=fam, weight=w),
                            fallback_to_default=False)
      for fam in vp._TRACKED
      for w in ('normal', 'bold')
  }
  check(
      all(os.path.dirname(p) == vp.FONT_DIR for p in found.values()),
      f'matplotlib finds each tracked family in its tracked file, not a system copy: '
      f'{ {k: p for k, p in found.items() if os.path.dirname(p) != vp.FONT_DIR} }'
  )
  with warnings.catch_warnings(record=True) as got:
    warnings.simplefilter('always')
    for f in {f for t in vp.THEME_LOOKS for f in vp.theme_fonts(t)}:
      vp.text_px(''.join(need), 16, f)
    for theme in vp.THEME_LOOKS:
      headless({'sim': 'Sim 2', 'theme': theme})
  lost = sorted(
      {str(w.message)
       for w in got if 'missing from font' in str(w.message)})
  check(
      not lost,
      f'no missing glyph: text_px over every symbol in each theme\'s fonts, and each theme\'s notebook '
      f'(Sim 2, every tab) {lost[:3]}')


# ------------------------------------------------------------------------------------------------- sync (11-13)
# ssh for part_sync: a kept login (-M -S cm) is a file at cm, -O check / exit test / remove it; any other call drops
# the options and the host and runs the command here, as the remote's shell would
STAND_IN_SSH = r'''#!/bin/bash
cm= master= op=
while [ $# -gt 0 ]; do
  case "$1" in -M) master=1; shift ;; -S) cm=$2; shift 2 ;; -O) op=$2; shift 2 ;; -o|-l|-p) shift 2 ;;
               -*) shift ;; *) break ;; esac
done
shift
[ -n "$master" ] && { touch "$cm"; exit 0; }
[ -n "$op" ] && { [ -e "$cm" ] || exit 255; [ "$op" = exit ] && rm -f "$cm"; exit 0; }
[ -z "$cm" ] || [ -e "$cm" ] || exit 255
exec sh -c "$*"
'''


def part_sync():
  """sync.py mirrors Sim 2's default run into a stand-in laptop (a temporary directory: its base and, apart from
    it, its data directory; ssh a stand-in that runs the remote's side here): the run without its
    distribution-function frames, its geqdsk and its Thomson file each at its place under the base, its cache into
    the data directory; a second sync copies nothing; the viewer opens the mirror with every tab, the geqdsk, Thomson
    file and cache the mirror's own; where the remote base is this machine's, nothing to do."""
  import gkview as gv
  import sync
  lab, simdir, run, _ = next(s for s in gv.SIMS if s[0] == 'Sim 2')
  with tempfile.TemporaryDirectory() as tmp:
    base, data = os.path.join(tmp, 'base'), os.path.join(tmp, 'data')
    os.makedirs(data)
    remote = {'host': 'stand-in', 'base': gv.BASE, 'data': gv.DATA}
    json.dump(dict(gv.CONFIG, base=base, remote=remote),
              open(f'{data}/viewer.json', 'w'))
    os.makedirs(f'{tmp}/bin')
    with open(f'{tmp}/bin/ssh', 'w') as f:
      f.write(STAND_IN_SSH)
    os.chmod(f'{tmp}/bin/ssh', 0o755)
    env = dict(os.environ,
               GK_VIEWER_DATA=data,
               PATH=f'{tmp}/bin:' + os.environ['PATH'])
    env.pop('SYNC_CM', None)
    sync_cmd = [
        sys.executable,
        os.path.join(HERE, 'sync.py'), '--batch', '--sim', lab
    ]
    first, again = (subprocess.run(sync_cmd,
                                   env=env,
                                   capture_output=True,
                                   text=True,
                                   timeout=1200) for _ in range(2))
    src, dst = os.path.join(gv.BASE, simdir, 'runs',
                            run), os.path.join(base, simdir, 'runs', run)
    want = sorted(
        os.path.relpath(os.path.join(d, f), src) for d, _, fs in os.walk(src)
        for f in fs if not re.search(
            r'-(?!field_)[A-Za-z0-9]+_[0-9]+\.gkyl$|_source_[0-9]+\.gkyl$|restart',
            f))
    got = sorted(
        os.path.relpath(os.path.join(d, f), dst) for d, _, fs in os.walk(dst)
        for f in fs)
    check(
        first.returncode == 0 and got == want and len(want) > 0,
        f'sync: {lab} / {run} mirrored without distribution-function, source and restart files '
        f'({len(got)} of {len(want)} files; rc {first.returncode} {first.stderr.strip()[-200:]})'
    )
    copied = re.findall(r'copied (\S+) files', again.stdout)
    check(
        again.returncode == 0 and len(copied) >= 4 and set(copied) == {'0'},
        f'sync: a second sync copies nothing (runs, caches, geqdsks, Thomson files): {again.stdout!r}'
    )
    p = subprocess.run([
        sys.executable, '-c',
        'import json, os, check_viewer as c, gkview as gv, thomson\n'
        f'd, built = c.headless({{"sim": "{lab}"}})\n'
        'r = d["run"]\n'
        'print(json.dumps([gv.geqdsk_path(r), thomson.shot_file(r.shot), d["sync_from"], '
        'os.path.exists(gv.cache_path(r.path)), sorted(set(built.values()))]))'
    ],
                       env=env,
                       capture_output=True,
                       text=True,
                       timeout=900,
                       cwd=HERE)
    out = json.loads(
        p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None
    check(
        out is not None and out[0].startswith(base) and out[1]
        # the Thomson file: the mirror's sim-init copy (under base) or its own Flux copy (data/runs/flux), which
        # the viewer prefers when the shot has one
        and out[1].startswith(
            (base, data)) and out[2] == ['stand-in', gv.BASE, gv.DATA] and
        out[3] and out[4] == ['ok'],
        f'sync: the viewer opens the mirror, every tab, its geqdsk, Thomson file and cache the mirror\'s own, '
        f'and offers the Sync button: {out or p.stderr.strip()[-300:]}; the sync said {first.stdout!r}'
    )
  keep = gv.CONFIG.get('remote')
  try:
    gv.CONFIG['remote'] = {'host': 'stand-in', 'base': gv.BASE}
    own = sync.remote()
  finally:
    gv.CONFIG.pop('remote') if keep is None else gv.CONFIG.update(remote=keep)
  check(
      own is None,
      'sync: a remote whose base is this machine\'s own base: nothing to sync')


# --------------------------------------------------------------------------------------- device background (11-14)
def _wall_picture(w, path):
  """A picture of the wall polygon w (R, Z), red inside and blue outside, written to path; returns its extent (the
  wall with a margin), [R_left, R_right, Z_bottom, Z_top]."""
  from PIL import Image, ImageDraw
  r0, r1, z0, z1 = w[:, 0].min() - 0.2, w[:, 0].max() + 0.3, w[:, 1].min(
  ) - 0.2, w[:, 1].max() + 0.2
  h = 1600
  wd = int(round(h * (r1 - r0) / (z1 - z0)))
  im = Image.new('RGB', (wd, h), (0, 0, 255))
  ImageDraw.Draw(im).polygon([((r - r0) / (r1 - r0) * wd,
                               (z1 - z) / (z1 - z0) * h) for r, z in w],
                             fill=(255, 0, 0))
  im.save(path)
  return [float(r0), float(r1), float(z0), float(z1)]


def _side_points(wall, clip, off):
  """Points off pixels either side of each wall segment at a quarter, half and three quarters of its length (wall
  in pixels, x right and y down; clip: the drawn box x0, y0, x1, y1), with whether each is inside the wall polygon.
  Points nearer the wall than off - 1 px (a corner, the wall line) and outside clip are left out."""
  from matplotlib.path import Path
  wall = np.asarray(wall, float)
  poly, a, b = Path(wall), wall[:-1], wall[1:]
  seg = b - a
  out = []
  for p0, d in zip(a, seg):
    length = np.hypot(*d)
    if length < 3 * off:
      continue
    for f, sgn in [(f, sgn) for f in (0.25, 0.5, 0.75) for sgn in (1, -1)]:
      p = p0 + f * d + sgn * off * np.array([-d[1], d[0]]) / length
      if not (clip[0] + off <= p[0] <= clip[2] - off
              and clip[1] + off <= p[1] <= clip[3] - off):
        continue
      t = np.clip(
          np.einsum('ij,ij->i', p - a, seg) /
          np.maximum(np.einsum('ij,ij->i', seg, seg), 1e-12), 0, 1)
      if np.min(np.hypot(*(a + t[:, None] * seg - p).T)) < off - 1:
        continue
      out.append((p, poly.contains_point(p)))
  return out


def _side_errors(img, wall, clip, off):
  """Sample img (rows from the top) at _side_points: a point outside the wall polygon must be blue, one inside must
  not be (the picture, or the cells over it). Returns (points sampled, the wrong ones)."""
  wrong, pts = [], _side_points(wall, clip, off)
  for p, inside in pts:
    r, g, bl = img[int(round(p[1])), int(round(p[0]))][:3]
    if (bl > 180 and r < 80 and g < 80) == inside:
      wrong.append((round(p[0]), round(p[1]), 'inside' if inside else 'outside',
                    (r, g, bl)))
  return len(pts), wrong


def _blue(px):
  """Pixels of the test picture's blue (outside the wall), drawn at full opacity."""
  return (px[..., 2] > 180) & (px[..., 0] < 80) & (px[..., 1] < 80)


def _pair_colours(img, pairs, clip):
  """Of pairs of pixels ((x, y), (x', y')) that both lie inside one colour (each one's 3 x 3 neighbourhood: a
  cell's inside, not its edge or a line, nor a pixel's rounding onto one; a thin cell gives none) inside clip (the
  drawn box x0, y0, x1, y1, 2 px in), how many have the same colour at both (each channel within 6), and how many
  there were."""
  n = same = 0
  for (x0, y0), (x1, y1) in pairs:
    p0, p1 = (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))
    if not all(clip[0] + 2 <= q[0] <= clip[2] - 2 and clip[1] +
               2 <= q[1] <= clip[3] - 2 for q in (p0, p1)):
      continue
    if any(
        np.ptp(img[q[1] - 1:q[1] + 2,
                   q[0] - 1:q[0] + 2, :3].astype(int).reshape(-1, 3),
               axis=0).max() > 6 for q in (p0, p1)):
      continue
    n += 1
    same += int(
        np.abs(img[p0[1], p0[0], :3].astype(int) -
               img[p1[1], p1[0], :3].astype(int)).max() <= 6)
  return same, n


async def _drive_poloidal(url, dbg, wall, centres):
  """The movie's poloidal plane, step by step (launched --background true --opacity 1 on Sim 2, a half domain):
  {'on': the picture at full opacity, 'slider': after the opacity slider's arrow keys, 'mirror': the half domain shown
  whole, 'off': the picture switched off}, each (canvas RGBA, the wall in canvas px, the drawn box in canvas px, the
  device pixel ratio, cell-centre pairs (Z, -Z) in canvas px, the surface colour, the slider's shown value, the
  render number of the movie shown, movie_widget's)."""
  from PIL import Image
  latest = "__deep('.gkv-movie').reduce((a, e) => (+e.dataset.render || 0) > (+(a && a.dataset.render) || 0) ? e : a, null)"  # the movie the page shows now (marimo may render a new one into the same element)
  grab = (
      "(() => { const el = " + latest + ", cv = el.querySelector('canvas.rz'), "
      f"b = el.__rz.box(), w = {json.dumps(np.asarray(wall).tolist())}, "
      f"c = {json.dumps(np.asarray(centres).round(5).tolist())}; "
      "const p0 = el.__rz.px(b[0], b[3]), p1 = el.__rz.px(b[1], b[2]), x = cv.getContext('2d'), "
      "s = x.getImageData(1, 1, 1, 1).data, sl = __deep('[role=slider]').find(e => { let n = e; for (let i = 0; "
      "i < 6 && n; i++) { n = n.parentElement || (n.getRootNode() && n.getRootNode().host); if (n && "
      "(n.textContent || '').includes('background opacity')) return true; } return false; }); "
      # the value the slider shows beside it (marimo's show_value): its aria-valuenow is an index into steps=
      "let v = null; for (let n = sl, i = 0; n && v === null && i < 6; i++) { "
      "n = n.parentElement || (n.getRootNode() && n.getRootNode().host); "
      "const d = n && n.querySelector && n.querySelector('div.text-muted-foreground'); "
      "if (d && /^\\d*\\.?\\d+$/.test(d.textContent.trim())) v = +d.textContent.trim(); } "
      "return { png: cv.toDataURL('image/png'), wall: w.map(q => el.__rz.px(q[0], q[1])), "
      "clip: [p0[0], p0[1], p1[0], p1[1]], dpr: window.devicePixelRatio || 1, "
      "pairs: c.map(q => [el.__rz.px(q[0], q[1]), el.__rz.px(q[0], -q[1])]), surf: [s[0], s[1], s[2]], "
      "slider: v, render: +el.dataset.render }; })()")
  out = {}

  async def shot(name):
    try:
      g = await c.strict(grab)
    except RuntimeError as e:
      raise RuntimeError(f'step {name!r}: {e}') from None
    out[name] = (np.asarray(
        Image.open(io.BytesIO(base64.b64decode(g['png'].split(
            ',', 1)[1]))).convert('RGBA')), g['wall'], g['clip'], g['dpr'],
                 g['pairs'], g['surf'], g['slider'], g['render'])

  async def rebuilt(
      switch):  # a switch that remakes the movie: wait for the new one, drawn
    before = await c.strict(f"+(({latest}) || {{}}).dataset.render || 0")
    try:
      await c.strict(f"__switch({json.dumps(switch)}).click()")
    except RuntimeError as e:
      raise RuntimeError(f'the switch {switch!r}: {e}') from None
    if not await c.until(
        f"(e => e && +e.dataset.render > {before} && +(e.dataset.draws || 0) > 0)({latest})",
        120):
      raise RuntimeError(f'the switch {switch!r}: no new movie in 120 s')
    await asyncio.sleep(2)

  async with opened(url, dbg) as c:
    await asyncio.sleep(3)
    await c.js(DEEP)
    await c.until(f"(e => e && +(e.dataset.draws || 0) > 0)({latest})", 180)
    await c.until(
        "Object.values(window.__gkvBackground || {}).some(i => i.complete && i.naturalWidth)",
        60)
    await asyncio.sleep(2)  # the movie redraws when the picture arrives
    await shot('on')
    # the opacity slider: eight steps down with the keyboard (Radix: the focused thumb takes the arrow keys)
    await c.strict(
        "__deep('[role=slider]').find(e => { let n = e; for (let i = 0; i < 6 && n; i++) { n = n.parentElement "
        "|| (n.getRootNode() && n.getRootNode().host); if (n && (n.textContent || '').includes('background "
        "opacity')) return true; } return false; }).focus()")
    for _ in range(8):
      for kind in ('keyDown', 'keyUp'):
        await c.call('Input.dispatchKeyEvent',
                     type=kind,
                     key='ArrowLeft',
                     code='ArrowLeft',
                     windowsVirtualKeyCode=37)
      await asyncio.sleep(0.15)
    await asyncio.sleep(3)
    await shot('slider')
    await rebuilt('show the full domain')
    await shot('mirror')
    # the picture off: the same movie redraws without it (a property of the widget, not a new movie)
    await c.strict("__switch('include device background').click()")
    await asyncio.sleep(4)
    await shot('off')
  return out


def part_machine():
  """(2026-11-17) The whole machine view of the poloidal plane: on every simulation viewer.json offers (a half domain
  also shown whole), 'whole machine' follows 'whole domain' and frames the domain, the whole wall and the device
  picture (its extent); where it is not offered neither reaches out of the whole domain's frame; a run with neither
  has none; the map drawn at it has the wall and the picture inside its axes."""
  import matplotlib.pyplot as plt
  import gkview as gv
  import viewer_plots as vp

  def inside(inner, outer, tol=1e-9):
    return (inner[0] >= outer[0] - tol and inner[1] <= outer[1] + tol
            and inner[2] >= outer[2] - tol and inner[3] <= outer[3] + tol)

  def extents(run):
    """(name, (R0, R1, Z0, Z1)) of the machine's parts: the whole wall, the picture."""
    w, pic = gv.wall(run), gv.device_image(run.device)
    p = [] if w is None else [('wall', (w[:, 0].min(), w[:, 0].max(),
                                        w[:, 1].min(), w[:, 1].max()))]
    return p + ([] if pic is None else [('picture', tuple(pic[1]))])

  runs, seen, bad = {}, [], []
  for lab, d, dflt, _ in gv.SIMS:
    runs[lab] = run = gv.open_run(os.path.join(gv.runs_root(d), dflt), lab)
    for mirror in ((False, True) if run.layout.half else (False, )):
      b = vp.zoom_boxes(run, mirror=mirror)
      keys, m = list(b), b.get('whole machine')
      frame = m if m is not None else b['whole domain']
      ok = all(inside(e, frame)
               for _, e in extents(run)) and inside(b['whole domain'], frame)
      if m is not None:
        ok &= keys.index('whole machine') == keys.index('whole domain') + 1
        seen.append(lab + (' mirrored' if mirror else ''))
      if not ok:
        bad.append((lab, mirror, keys, m, extents(run)))
  check(
      not bad and len(seen) >= 1,
      f"whole machine: offered after 'whole domain' on {seen}; it (or, where not offered, the whole domain) frames "
      f"the whole wall and the device picture of every simulation; failures {bad}"
  )

  run = runs['Sim 2']
  keep = gv.wall, gv.device_image
  gv.wall, gv.device_image = (lambda r: None), (lambda dev: None)
  try:
    none = vp.zoom_boxes(run, mirror=True)
  finally:
    gv.wall, gv.device_image = keep
  check(
      'whole machine' not in none and 'whole domain' in none,
      f'whole machine: not offered for a run with no wall and no picture (Sim 2 with both taken away): {list(none)}'
  )

  got = {}
  for lab in [l for l in ('ASDEX Upgrade', 'Sim 2') if l in runs]:
    run = runs[lab]
    fs = run.frames()
    fr, fr0 = gv.frame(run, fs[-1]), gv.frame(run, fs[0])
    fig = vp.map_figure(run,
                        fr,
                        fr0,
                        'n_elc',
                        'value',
                        zoom='whole machine',
                        background=vp.device_background(run.device),
                        mirror=run.layout.half)
    fig.canvas.draw()
    ax = next(a for a in fig.axes if a.get_xlabel().startswith('R'))
    lim = (*sorted(ax.get_xlim()), *sorted(ax.get_ylim()))
    got[lab] = (all(inside(e, lim, 1e-6)
                    for _, e in extents(run)), tuple(round(v, 3) for v in lim))
    plt.close(fig)
  check(
      len(got) == 2 and all(v[0] for v in got.values()),
      f'whole machine, matplotlib: the axes hold the whole wall and the picture (ASDEX Upgrade, Sim 2 shown whole): '
      f'{got}')


def part_poloidal(chrome=None):
  """The poloidal plane's extras on Sim 2 (a half domain), with a test picture red inside its wall and blue outside
  (viewer.json "device_image"): the picture lies where its extent says, under the map (matplotlib: the whole domain
  and the lower divertor) and, with chrome, under the page's movie (a few pixels either side of every wall segment,
  outside is blue and inside is not); the opacity slider redraws the movie's picture at its value without remaking
  the movie; the half domain shown whole puts each cell's colour at its mirror image (the map and the page; not
  without the switch); switched off, the picture leaves no trace."""
  import matplotlib.pyplot as plt
  import gkview as gv
  import viewer_plots as vp
  lab, simdir, run_name, _ = next(s for s in gv.SIMS if s[0] == 'Sim 2')
  run = gv.open_run(os.path.join(gv.runs_root(simdir), run_name), lab)
  w = gv.wall(run)
  cen = np.concatenate(
      [gv.cell_centres(run.nodes[b]).reshape(-1, 2) for b in range(run.nb)])
  with tempfile.TemporaryDirectory() as tmp:
    pic = os.path.join(tmp, 'wall.png')
    image = {'file': pic, 'extent': _wall_picture(w, pic), 'opacity': 1.0}
    keep = gv.CONFIG.get('device_image')
    gv.CONFIG['device_image'] = image
    try:
      bg = vp.device_background(run.device)
    finally:
      gv.CONFIG.pop('device_image') if keep is None else gv.CONFIG.update(
          device_image=keep)
    fs = run.frames()
    fr, fr0 = gv.frame(run, fs[-1]), gv.frame(run, fs[0])
    got, pairs = {}, {}
    for zoom, mirror in (('whole domain', False), ('lower divertor', False),
                         ('whole domain', True), ('outboard midplane', True),
                         ('inboard midplane', True)):
      fig = vp.map_figure(run,
                          fr,
                          fr0,
                          'T_elc',
                          'value',
                          zoom=zoom,
                          background=bg,
                          mirror=mirror)
      fig.set_dpi(150)
      fig.canvas.draw()
      img = np.asarray(fig.canvas.buffer_rgba())
      ax = next(a for a in fig.axes if a.get_xlabel().startswith('R'))
      h = img.shape[0]
      to_px = lambda pts: np.c_[ax.transData.transform(pts)[:, 0], h - ax.
                                transData.transform(pts)[:, 1]]  # noqa: E731
      bb = ax.bbox
      if mirror:
        pairs[zoom] = _pair_colours(img, zip(to_px(cen), to_px(cen * [1, -1])),
                                    (bb.x0, h - bb.y1, bb.x1, h - bb.y0))
      else:
        got[zoom] = _side_errors(img, to_px(w),
                                 (bb.x0, h - bb.y1, bb.x1, h - bb.y0), 5)
      plt.close(fig)
    check(
        all(n >= 10 and not bad for n, bad in got.values()),
        'device background, matplotlib: the picture under the map is where its extent puts it (outside the wall '
        f'blue, inside not, 5 px either side of each wall segment): '
        f'{ {z: (n, bad[:3]) for z, (n, bad) in got.items()} }')
    same, n = (sum(v[k] for v in pairs.values()) for k in (0, 1))
    check(
        n >= 50 and same == n
        and 'upper divertor' in vp.zoom_boxes(run, mirror=True)
        and 'upper divertor' not in vp.zoom_boxes(run),
        f"half domain shown whole, matplotlib: every cell's colour also at its mirror image ({same} of {n} cell "
        f"insides; by view {pairs}); "
        "the mirror image's divertor offered as a view")
    if chrome is None:
      return
    data = os.path.join(tmp, 'data')
    os.makedirs(data)
    json.dump(dict(gv.CONFIG, base=gv.BASE, device_image=image),
              open(os.path.join(data, 'viewer.json'), 'w'))
    os.symlink(gv.RUNS,
               os.path.join(data,
                            'runs'))  # the caches: the movie opens at once
    args = [
        '--sim', 'Sim 2', '--background', 'true', '--opacity', '1', '--mirror',
        'false'
    ]
    with served(chrome, args, env={'GK_VIEWER_DATA': data}) as (url, dbg):
      check(
          url is not None,
          'poloidal extras: view.sh serves the page on a data directory with a picture'
      )
      if url is None:
        return
      st = asyncio.run(_drive_poloidal(url, dbg, w, cen))
  img, wall_px, clip, dpr, pr, surf, _, _ = st['on']
  n_on, bad_on = _side_errors(img, wall_px, clip, 6 * dpr)
  check(
      n_on >= 10 and not bad_on,
      f'device background, the page: the movie draws the picture where its extent puts it ({n_on} points, '
      f'wrong {bad_on[:3]})')
  img2, wall2, clip2, dpr2, _, surf2, val, render2 = st['slider']
  same_movie = render2 == st['on'][7]
  # outside the wall (no cells there), the picture's blue over the surface at the slider's opacity
  want = None if val is None else np.array(
      [0, 0, 255]) * val + np.array(surf2) * (1 - val)
  seen = [
      img2[int(round(p[1])), int(round(p[0])), :3].astype(int)
      for p, inside in _side_points(wall2, clip2, 6 * dpr2) if not inside
  ]
  off_blend = [
      c.tolist() for c in seen if want is None or np.abs(c - want).max() > 12
  ]
  check(
      val is not None and abs(val - 0.6) < 1e-6 and same_movie
      and len(seen) >= 5 and not off_blend,
      f'the opacity slider: eight arrow-key steps take it to {val}, the movie redraws the picture at that opacity '
      f'(outside the wall {len(seen) - len(off_blend)} of {len(seen)} pixels the blend '
      f'{None if want is None else want.round().tolist()}; others {off_blend[:4]}), the movie itself not remade '
      f'({same_movie})')
  same_off, n_off = _pair_colours(st['on'][0], st['on'][4], st['on'][2])
  same_m, n_m = _pair_colours(st['mirror'][0], st['mirror'][4], st['mirror'][2])
  check(
      n_m >= 15 and same_m == n_m and (n_off == 0 or same_off < 0.2 * n_off),
      f"half domain shown whole, the page: every sampled cell's colour also at its mirror image ({same_m} of "
      f"{n_m}); without the switch the view holds no mirror image ({n_off} pairs in the drawn box, {same_off} alike)"
  )
  check(
      int(_blue(st['off'][0]).sum()) == 0 and st['off'][7] == st['mirror'][7],
      f'device background switched off: no trace of the picture ({int(_blue(st["off"][0]).sum())} blue pixels), '
      f'the movie not made again (render {st["mirror"][7]} -> {st["off"][7]})')


# ------------------------------------------------------------------------------ the lines over the cells (11-16)
# viewer_plots.Layers' fields for the lines over the cells, the view keys, the switches
LINES = ('separatrix', 'wall', 'outlines')
LINE_LABELS = {
    'separatrix': 'show the separatrix',
    'wall': 'show the wall (limiter)',
    'outlines': 'show the block boundaries'
}


def _line_paths(run, mirror):
  """The lines the poloidal plane draws, by switch, as groups of (R, Z) polylines along the cell edges
  (gkview.cell_edges: curves through the Gauss nodes, or chords): the separatrix (the inner edge of each SOL block
  with an aligned block under it, the LCFS row of a limited block), every block's outline and the grid (every node
  row and column), one group each, and for a half domain shown whole a second one, their mirror image across Z = 0;
  the wall (the geqdsk's limiter, the whole vessel already), one group."""
  import gkview as gv
  lay, sz = run.layout, (1, -1) if mirror and run.layout.half else (1, )
  E = gv.cell_edges(run)
  rows, cols = E['rows'], E['cols']
  ring = lambda b: np.concatenate(  # noqa: E731
      [rows[b][0], cols[b][-1], rows[b][-1][::-1], cols[b][0][::-1]])
  sep = [(b, 0) for b in range(run.nb)
         if lay.region[b] == 'sol' and lay.under.get(b) is not None] + list(
             lay.lcfs.items())
  return {
      'separatrix': [[rows[b][i] * [1, s] for b, i in sep] for s in sz],
      'wall': [[gv.wall(run)]] if gv.wall(run) is not None else [[]],
      'outlines': [[ring(b) * [1, s] for b in range(run.nb)] for s in sz],
      'grid':
      [[p * [1, s] for b in range(run.nb) for p in (*rows[b], *cols[b])]
       for s in sz]
  }


def _near(shape, paths, r):
  """A mask [h, w] of the pixels within about r of the polylines paths (px: x right, y down)."""
  from PIL import Image, ImageDraw
  im = Image.new('L', (shape[1], shape[0]), 0)
  d = ImageDraw.Draw(im)
  for p in paths:
    p = np.asarray(p, float)
    d.line([tuple(q) for q in p], fill=255, width=int(2 * r + 1), joint='curve')
    for q in (p[0], p[-1]):
      d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=255)
  return np.asarray(im) > 0


def _along(paths, clip, step=3.0):
  """Points every step px along the polylines paths (px) that lie inside clip (x0, y0, x1, y1), 3 px in."""
  out = []
  for p in paths:
    p = np.asarray(p, float)
    for a, b in zip(p[:-1], p[1:]):
      n = max(int(np.hypot(*(b - a)) // step), 1)
      out += [a + (b - a) * t for t in np.arange(n) / n]
  out = np.array(out).reshape(-1, 2)
  return out[(out[:, 0] >= clip[0] + 3) & (out[:, 0] <= clip[2] - 3) &
             (out[:, 1] >= clip[1] + 3) & (out[:, 1] <= clip[3] - 3)]


def _switched(a, b, groups, clip, r):
  """What one switch changed between images a and b (rows from the top), judged on its line's groups of paths (px;
  _line_paths') inside the drawn box clip (x0, y0, x1, y1): (the least, over the groups, share of points along a
  group's paths, every 3 px, where a pixel within 1 px differs by more than 8 in a channel; the fewest points of a
  group; how many pixels of the whole image differ but further than r from the paths or outside clip: what else it
  changed)."""
  diff = np.abs(a[..., :3].astype(int) - b[..., :3].astype(int)).max(-1) > 8
  grown = diff.copy()
  for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
      grown |= np.roll(np.roll(diff, dy, 0), dx, 1)
  share, npts = [], []
  for paths in groups:
    pts = _along(paths, clip)
    hit = grown[np.round(pts[:, 1]).astype(int),
                np.round(pts[:, 0]).astype(int)]
    share.append(float(hit.mean()) if len(pts) else 0.0)
    npts.append(len(pts))
  x0, y0, x1, y1 = (int(round(v)) for v in clip)
  allowed = np.zeros_like(diff)
  allowed[max(y0 - 1, 0):y1 + 2, max(x0 - 1, 0):x1 + 2] = True
  allowed &= _near(diff.shape, [p for paths in groups for p in paths], r)
  return round(min(share), 3), min(npts), int((diff & ~allowed).sum())


def _line_verdicts(imgs, paths, clip, r):
  """imgs: {(separatrix, wall, outlines) switch states: image}, with all on, all off, each alone and each but one.
  For each line: (alone against all off: its switch draws it, and nothing else; all on against all but it: its
  switch takes it away, and nothing else), each _switched's."""
  out = {}
  for k, name in enumerate(LINES):
    alone = tuple(i == k for i in range(3))
    but = tuple(i != k for i in range(3))
    out[name] = (_switched(imgs[alone], imgs[(False, ) * 3], paths[name], clip,
                           r),
                 _switched(imgs[(True, ) * 3], imgs[but], paths[name], clip, r))
  return out


def _verdicts_ok(v, share=0.5):
  return all(n >= 10 and s >= share and not stray for pair in v.values()
             for s, n, stray in pair)


LINE_STATES = [(True, True, True), (False, True, True), (True, False, True),
               (True, True, False), (True, False, False), (False, True, False),
               (False, False, True), (False, False, False)]


async def _drive_lines(url, dbg, paths, shots):
  """The movie of Sim 2 (launched with its half domain shown whole and the device picture), its poloidal plane in
  each LINE_STATES state, reached by clicking the sidebar's switches: {state: (canvas RGBA, the drawn box in canvas
  px, the paths in canvas px by line, the render number of the movie shown, the page's address)}. A screenshot of
  the page in each state with one line off, in shots."""
  from PIL import Image
  latest = (
      "__deep('.gkv-movie').reduce((a, e) => (+e.dataset.render || 0) > (+(a && a.dataset.render) || 0) "
      "? e : a, null)")
  rz = {
      k: [[np.asarray(p).round(5).tolist() for p in g] for g in v]
      for k, v in paths.items()
  }
  grab = (
      "(() => { const el = " + latest + ", cv = el.querySelector('canvas.rz'), "
      f"P = {json.dumps(rz)}; "
      "const px = {}; for (const k in P) px[k] = P[k].map(g => g.map(p => p.map(q => el.__rz.px(q[0], q[1])))); "
      "return { png: cv.toDataURL('image/png'), area: el.__rz.area(), px, render: +el.dataset.render, "
      "search: location.search }; })()")
  out = {}
  async with opened(url, dbg) as c:
    await asyncio.sleep(3)
    await c.js(DEEP)
    await c.until(f"(e => e && +(e.dataset.draws || 0) > 0)({latest})", 180)
    await c.until(
        "Object.values(window.__gkvBackground || {}).some(i => i.complete && i.naturalWidth)",
        60)
    await asyncio.sleep(2)  # the movie redraws when the picture arrives
    now = (True, True, True)
    for state in LINE_STATES:
      # the movie shown, drawn again (or a new one: the check below then fails on its render number)
      render, draws = await c.strict(
          f"(e => [+e.dataset.render, +e.dataset.draws])({latest})")
      for name, a, b in zip(LINES, now, state):
        if a != b:
          await c.strict(f"__switch({json.dumps(LINE_LABELS[name])}).click()")
      if state != now:
        kinds = sorted(
            ['quantity'] +
            [k for k, on in zip(('sep', 'wall', 'outline'), state) if on])
        if not await c.until(
            f"(e => e && (+e.dataset.render !== {render} || +e.dataset.draws > {draws}) && "
            f"JSON.stringify([...e.__rz.lines()].sort()) === "
            f"{json.dumps(json.dumps(kinds, separators=(',', ':')))})({latest})",
            60):
          raise RuntimeError(
              f'the movie did not redraw with the lines {kinds} in 60 s')
        await asyncio.sleep(1.5)
      now = state
      g = await c.strict(grab)
      out[state] = (np.asarray(
          Image.open(io.BytesIO(base64.b64decode(g['png'].split(
              ',', 1)[1]))).convert('RGBA')), g['area'], g['px'], g['render'],
                    g['search'])
      if shots and sum(state) == 2:
        await c.shot(
            os.path.join(shots, f'lines_{LINES[state.index(False)]}_off.png'))
  return out


def part_lines(chrome=None, shots=None):
  """The lines over the poloidal plane's cells (2026-11-16), each on a switch: the separatrix, the wall, the block
  outlines. On Sim 2 (a half domain), the map (matplotlib: the whole domain shown whole, the lower divertor) and,
  with chrome, the page's movie (the whole domain shown whole, over the device picture): each line's switch alone
  draws pixels along that line and nowhere else, and taking it away from all three changes pixels along that line
  and nowhere else (the strips unchanged); every switch off leaves the map as it is with its lines removed; the
  movie redraws at each click, not made again. The Static figure and Diffusivity tabs (the notebook launched with
  switches off) and the movie export draw with the switches given."""
  import matplotlib.pyplot as plt
  import gkview as gv
  import viewer_plots as vp
  lab, simdir, run_name, _ = next(s for s in gv.SIMS if s[0] == 'Sim 2')
  run = gv.open_run(os.path.join(gv.runs_root(simdir), run_name), lab)
  fs = run.frames()
  fr, fr0 = gv.frame(run, fs[-1]), gv.frame(run, fs[0])
  for zoom, mirror in (('whole domain', True), ('lower divertor', False)):
    imgs, bare = {}, None
    for state in LINE_STATES:
      fig = vp.map_figure(run,
                          fr,
                          fr0,
                          'T_elc',
                          'value',
                          zoom=zoom,
                          mirror=mirror,
                          layers=vp.Layers(**dict(zip(LINES, state))))
      fig.set_dpi(150)
      fig.canvas.draw()
      imgs[state] = np.asarray(fig.canvas.buffer_rgba()).copy()
      ax = next(a for a in fig.axes if a.get_xlabel().startswith('R'))
      # all on: the paths in px, and the same map with every line taken off its axes (what all off must give)
      if all(state):
        h = imgs[state].shape[0]
        to_px = lambda pts: np.c_[ax.transData.transform(pts)[:, 0], h - ax.
                                  transData.transform(pts)[:, 1]]  # noqa: E731
        clip = (ax.bbox.x0, h - ax.bbox.y1, ax.bbox.x1, h - ax.bbox.y0)
        px = {
            k: [[to_px(p) for p in g] for g in v]
            for k, v in _line_paths(run, mirror).items()
        }
        for ln in list(ax.lines):
          ln.remove()
        fig.canvas.draw()
        bare = np.asarray(fig.canvas.buffer_rgba()).copy()
      plt.close(fig)
    v = _line_verdicts(imgs, px, clip, 4)
    left = int((np.abs(imgs[(False, ) * 3][..., :3].astype(int) -
                       bare[..., :3].astype(int)).max(-1) > 0).sum())
    check(
        _verdicts_ok(v) and left == 0,
        f'lines over the cells, matplotlib ({zoom}{", shown whole" if mirror else ""}): each switch alone draws '
        f'its line and nothing else, and taken away removes it and nothing else (share of points along the line '
        f'changed, points, pixels changed elsewhere: {v}); all off = the map with its lines removed ({left} '
        f'pixels differ)')
  # every map the viewer draws takes the switches: the notebook's Static figure and Diffusivity tabs (launched with
  # the separatrix and the outlines off), the movie export
  want = vp.Layers(separatrix=False, outlines=False)
  seen, orig = [], vp.map_figure

  def recorded(*a, **kw):
    seen.append((sys._getframe(1).f_code.co_name, kw.get('layers')))
    return orig(*a, **kw)

  vp.map_figure = recorded
  try:
    d, built = headless({
        'sim': 'Sim 2',
        'separatrix': 'false',
        'outlines': 'false'
    })
    plt.close(d['static_figure']())
    with tempfile.TemporaryDirectory() as tmp:
      vp.render_movie(run,
                      d['summ'],
                      fs[-2:],
                      'T_elc',
                      'value',
                      False,
                      'whole domain',
                      vp.LOOKS['dark'],
                      os.path.join(tmp, 'm.mp4'),
                      layers=want)
  finally:
    vp.map_figure = orig
  callers = {f for f, _ in seen}
  check(
      d['layers_shown'] == want
      and d['movie'].widget.layers == ['quantity', 'wall']
      and {'static_figure', 'diffusivity_map', 'render_movie'} <= callers
      and all(ln == want for _, ln in seen)
      and (d['view']['separatrix'], d['view']['wall'],
           d['view']['outlines']) == (False, True, False),
      f'launched with the separatrix and the block outlines off: the switches {d["layers_shown"]}, the movie draws '
      f'{d["movie"].widget.layers}, every map drawn takes them ({sorted(callers)}: '
      f'{sorted({str(ln) for _, ln in seen})}), the view keeps them')
  if chrome is None:
    return
  args = [
      '--sim', 'Sim 2', '--mirror', 'true', '--background', 'true', '--opacity',
      '0.6'
  ]
  with served(chrome, args) as (url, dbg):
    check(url is not None, 'lines over the cells: view.sh serves the page')
    if url is None:
      return
    try:
      st = asyncio.run(
          _drive_lines(url, dbg, {
              k: v
              for k, v in _line_paths(run, True).items() if k in LINES
          }, shots))
    except RuntimeError as e:
      check(False, f'lines over the cells, the page: {e}')
      return
  _, area, px, render0, _ = st[(True, ) * 3]
  v = _line_verdicts({k: s[0] for k, s in st.items()}, px, area, 3)
  renders = {s[3] for s in st.values()}
  search = st[(False, ) * 3][4]
  check(
      _verdicts_ok(v) and renders == {render0}
      and all(f'{k}=false' in search for k in LINES),
      f'lines over the cells, the page (the whole domain shown whole, over the device picture): each switch alone '
      f'draws its line and nothing else, and taken away removes it and nothing else ({v}); the movie redrawn at '
      f'each click, not made again (renders {sorted(renders)}); the address holds the switches ({search[-80:]})'
  )
  if shots:
    from PIL import Image
    for state, s in st.items():
      if sum(state) == 2:
        Image.fromarray(s[0]).save(
            os.path.join(shots,
                         f'lines_canvas_{LINES[state.index(False)]}_off.png'))


# ------------------------------------------------------------------------------ the grid and the quantity (11-16)
# the four device runs of research/tokamak/regr_devices_20261116 (beside base's research/nstxu): ASDEX Upgrade's
# coarse grid for the curves, TCV's and DIII-D's in the page
DEVICE_RUNS = os.path.join('..', 'tokamak', 'regr_devices_20261116', 'runs')


def _cells_mask(shape, polys, r):
  """A mask [h, w] of the pixels inside the polygons polys (px) or within about r of them."""
  from PIL import Image, ImageDraw
  im = Image.new('L', (shape[1], shape[0]), 0)
  d = ImageDraw.Draw(im)
  for q in polys:
    d.polygon([tuple(p) for p in q], fill=255)
  # grown by r: each outline drawn r wide on either side (a polygon's own outline width grows it inward)
  return (np.asarray(im) > 0) | _near(shape,
                                      [np.vstack([q, q[:1]]) for q in polys], r)


def _quantity_verdict(on, off, polys, centres, clip, r):
  """What the quantity switch changed between images on and off: (the share of the cell centres, px, inside clip,
  where a pixel differs; how many; how many pixels differ outside the cells, polys px, grown by r, or outside
  clip)."""
  diff = np.abs(on[..., :3].astype(int) - off[..., :3].astype(int)).max(-1) > 8
  c = np.array([
      p for p in centres if clip[0] + 3 <= p[0] <= clip[2] - 3 and clip[1] +
      3 <= p[1] <= clip[3] - 3
  ]).reshape(-1, 2)
  hit = diff[np.round(c[:, 1]).astype(int), np.round(c[:, 0]).astype(int)]
  x0, y0, x1, y1 = (int(round(v)) for v in clip)
  allowed = np.zeros_like(diff)
  allowed[max(y0 - 1, 0):y1 + 2, max(x0 - 1, 0):x1 + 2] = True
  allowed &= _cells_mask(diff.shape, polys, r)
  return (round(float(hit.mean()), 3) if len(c) else 0.0, len(c),
          int((diff & ~allowed).sum()))


def _cell_geometry(run, mirror):
  """(cell polygons [n, m, (R, Z)]: the regions their edges bound, gkview.cell_polygons; a point inside each [n, (R, Z)])
  of every block, mirrored too for a half domain shown whole; cells without a value at the first frame left out
  (they are not filled)."""
  import gkview as gv
  fr = gv.frame(run, run.frames()[0])
  quads, cen = [], []
  for s in ((1, -1) if mirror and run.layout.half else (1, )):
    for b in range(run.nb):
      q = gv.cell_polygons(run, b) * [1, s]
      ok = np.isfinite(fr[b]['T_elc'])
      quads.append(q[ok])
      k = q.shape[
          2] // 4  # points to an edge: a point inside each cell, between its psi edges' middles (the
      # corners' mean of a long curved cell lies outside it)
      cen.append(0.5 * (q[:, :, k // 2] + q[:, :, 2 * k + k // 2])[ok])
  return np.concatenate(quads), np.concatenate(cen)


def _gauss_vs_chords(img_grid, img_none, run, to_px, clip):
  """Curves, not chords: (the share of the Gauss nodes, px, where the grid image differs from the bare one; how
  many; of the points on the chords facing Gauss nodes 4 px or more off their chord, those further than 2 px from
  every curve: how many, and how many of them differ, i.e. a chord drawn)."""
  import gkview as gv
  from gkyl_read import read_gkyl
  diff = np.abs(img_grid[..., :3].astype(int) -
                img_none[..., :3].astype(int)).max(-1) > 8
  grown = diff.copy()
  for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
      grown |= np.roll(np.roll(diff, dy, 0), dx, 1)
  E = gv.cell_edges(run)
  near = _near(
      diff.shape,
      [to_px(p) for b in range(run.nb)
       for p in (*E['rows'][b], *E['cols'][b])], 2)
  gauss, chord = [], []
  for b in range(run.nb):
    g = read_gkyl(run.f(b, 'geo_corn_nodes'))
    gt = int(g['meta'].get('geometry_type', -1))
    C = to_px(gv.physical_rz(g['data'], gt).reshape(
        -1, 2)).reshape(np.shape(g['data'])[:2] + (2, ))
    for d, (A, B) in ((0, (C[:, :-1], C[:, 1:])), (1, (C[:-1], C[1:]))):
      s = gv.physical_rz(
          np.asarray(read_gkyl(run.f(
              b, f'geo_surf{d}_mc2p_nodal_fd'))['data'])[..., :3], gt)
      S = to_px(s.reshape(-1, 2)).reshape(s.shape)
      for G in ((S[:, 0::2], S[:, 1::2]) if d == 0 else (S[0::2], S[1::2])):
        AB = B - A
        t = np.clip(((G - A) * AB).sum(-1) / np.maximum(
            (AB * AB).sum(-1), 1e-9), 0, 1)
        P = A + t[..., None] * AB  # the chord's point facing the node
        off = np.hypot(*(G - P).transpose(2, 0, 1))
        gauss += list(G.reshape(-1, 2))
        chord += list(P[off >= 4].reshape(-1, 2))
  inside = lambda p: (
      clip[0] + 3 <= p[0] <= clip[2] - 3 and clip[1] + 3 <= p[  # noqa: E731
          1] <= clip[3] - 3)
  gauss = np.array([p for p in gauss if inside(p)]).reshape(-1, 2)
  chord = np.array([
      p for p in chord if inside(p) and not near[int(round(p[1])),
                                                 int(round(p[0]))]
  ]).reshape(-1, 2)
  at = lambda m, P: m[np.round(P[:, 1]).astype(int),
                      np.round(P[:, 0]).astype(int)]  # noqa: E731
  return (round(float(at(grown, gauss).mean()), 3) if len(gauss) else 0.0,
          len(gauss), len(chord),
          int(at(diff, chord).sum()) if len(chord) else 0)


async def _drive_grid(url, dbg, cases, shots):
  """The movie's poloidal plane of each case (name, view keys opening it with the grid on and the quantity off, the
  grid's paths, cell quadrilaterals, cell centres in (R, Z)), the switches clicked through (quantity, grid) =
  (off, on), (on, on), (on, off), (off, off): {name: {state: (canvas RGBA, drawn box, grid paths px, quads px,
  centres px, render number)}}; screenshots of the first two states (the page and the plane) in shots."""
  from urllib.parse import urlencode
  from PIL import Image
  latest = (
      "__deep('.gkv-movie').reduce((a, e) => (+e.dataset.render || 0) > (+(a && a.dataset.render) || 0) "
      "? e : a, null)")
  out = {}
  async with opened(url, dbg) as c:
    for name, state, paths, quads, cen in cases:
      await c.call('Page.navigate', url=url + '&' + urlencode(state))
      await asyncio.sleep(3)
      await c.js(DEEP)
      await c.until(f"(e => e && +(e.dataset.draws || 0) > 0)({latest})", 240)
      await asyncio.sleep(2)
      P = json.dumps({
          'g': [[np.asarray(p).round(5).tolist() for p in g] for g in paths],
          'q':
          np.asarray(quads).round(5).tolist(),
          'c':
          np.asarray(cen).round(5).tolist()
      })
      grab = (
          "(() => { const el = " + latest +
          ", cv = el.querySelector('canvas.rz'), P = " + P + ", "
          "f = q => el.__rz.px(q[0], q[1]); return { png: cv.toDataURL('image/png'), area: el.__rz.area(), "
          "g: P.g.map(g => g.map(p => p.map(f))), q: P.q.map(p => p.map(f)), c: P.c.map(f), "
          "render: +el.dataset.render, layers: el.__rz.lines() }; })()")
      got, now = {}, (False, True)
      for st in ((False, True), (True, True), (True, False), (False, False)):
        render, draws = await c.strict(
            f"(e => [+e.dataset.render, +e.dataset.draws])({latest})")
        for lab, a, b in zip(('show the plotted quantity', 'show the grid'),
                             now, st):
          if a != b:
            await c.strict(f"__switch({json.dumps(lab)}).click()")
        if st != now:  # the movie drawn again with the quantity and the grid as clicked (a line the run lacks: on)
          if not await c.until(
              f"(e => e && (+e.dataset.render !== {render} || +e.dataset.draws > {draws}) && "
              f"e.__rz.lines().includes('quantity') === {str(st[0]).lower()} && "
              f"e.__rz.lines().includes('grid') === {str(st[1]).lower()})({latest})",
              60):
            raise RuntimeError(
                f'{name}: the movie did not redraw with quantity, grid = {st} in 60 s'
            )
          await asyncio.sleep(1.5)
        now = st
        g = await c.strict(grab)
        got[st] = (np.asarray(
            Image.open(io.BytesIO(base64.b64decode(g['png'].split(
                ',', 1)[1]))).convert('RGBA')), g['area'], g['g'], g['q'],
                   g['c'], g['render'])
        if shots and st[1]:
          tag = f'grid_{name}_quantity_{"on" if st[0] else "off"}'
          await c.shot(os.path.join(shots, f'{tag}.png'))
          Image.fromarray(got[st][0]).save(
              os.path.join(shots, f'{tag}_plane.png'))
      out[name] = got
  return out


def part_grid(chrome=None, shots=None):
  """The grid (every cell edge, as curves) and the quantity on the poloidal plane, each on a switch (2026-11-16):
  on ASDEX Upgrade's device run (a coarse grid: curved edges, by path) and Sim 2 (shown whole; its surface node files hold zeros: chords and
  a note), the map (matplotlib) and, with chrome, the page's movie (TCV and DIII-D): the grid's switch draws or takes
  away pixels along the cell edges only (over the quantity too); the quantity's, inside the cells only, nearly every
  cell's centre; with both off and no other line the plane is bare; the grid passes through the Gauss nodes and not
  along the chords where they bend away; the plane alone (a half-width slide) loses its colour bar with the
  quantity; the movie is not made again."""
  import matplotlib.pyplot as plt
  import gkview as gv
  import viewer_plots as vp
  devices = os.path.normpath(os.path.join(gv.BASE, DEVICE_RUNS))
  runs = []
  if os.path.isdir(os.path.join(devices, 'asdex', 'rt_2x2v_p1')):
    runs.append(('ASDEX Upgrade',
                 gv.open_run(os.path.join(devices, 'asdex',
                                          'rt_2x2v_p1')), False))
  else:
    print(f'SKIP the grid on a device run: no {devices}/asdex/rt_2x2v_p1',
          flush=True)
  lab, simdir, run_name, _ = next(s for s in gv.SIMS if s[0] == 'Sim 2')
  runs.append(
      (lab, gv.open_run(os.path.join(gv.runs_root(simdir), run_name),
                        lab), True))
  off = dict(outlines=False, separatrix=False, wall=False)
  for name, run, mirror in runs:
    fr = fr0 = gv.frame(
        run,
        run.frames()[0])  # the first frame: no cell marked n <= 0 (its x)
    imgs = {}
    for q, g in ((False, False), (False, True), (True, False), (True, True)):
      fig = vp.map_figure(run,
                          fr,
                          fr0,
                          'T_elc',
                          'value',
                          mirror=mirror,
                          layers=vp.Layers(quantity=q, grid=g, **off))
      fig.set_dpi(150)
      ax = next(a for a in fig.axes if a.get_xlabel().startswith('R'))
      if (q, g) == (False, True):
        notes = [t.get_text() for t in ax.texts]
      for t in list(
          ax.texts):  # the chords' note (checked above): not a line's pixels
        t.remove()
      fig.canvas.draw()
      imgs[q, g] = np.asarray(fig.canvas.buffer_rgba()).copy()
      # (R, Z) -> px, frozen: a closed figure's transforms change
      h, tr = imgs[q, g].shape[0], ax.transData.frozen()
      to_px = lambda pts, tr=tr, h=h: tr.transform(
          pts) * [1, -1] + [0, h]  # noqa: E731
      clip = (ax.bbox.x0, h - ax.bbox.y1, ax.bbox.x1, h - ax.bbox.y0)
      plt.close(fig)
    E = gv.cell_edges(run)
    grid = [[to_px(p) for p in grp] for grp in _line_paths(run, mirror)['grid']]
    quads, cen = _cell_geometry(run, mirror)
    qpx = [to_px(qq) for qq in quads]
    v_alone = _switched(imgs[False, True], imgs[False, False], grid, clip, 3)
    v_over = _switched(imgs[True, True], imgs[True, False], grid, clip, 3)
    v_q = _quantity_verdict(imgs[True, False], imgs[False, False], qpx,
                            to_px(cen), clip, 2)
    x0, y0, x1, y1 = (int(round(v)) for v in clip)
    inner = imgs[False, False][y0 + 2:y1 - 2, x0 + 2:x1 - 2, :3].astype(int)
    surf = np.array(
        [int(vp.LOOKS['dark'].surf[i:i + 2], 16) for i in (1, 3, 5)])
    bare = int((np.abs(inner - surf).max(-1) > 2).sum())
    curves = _gauss_vs_chords(imgs[False, True], imgs[False, False], run, to_px,
                              clip) if E['curved'] else None
    figs = [
        vp.map_figure(run,
                      fr,
                      fr0,
                      'T_elc',
                      'value',
                      slide='half',
                      layers=vp.Layers(quantity=q)) for q in (True, False)
    ]
    nax = [len(f.axes) for f in figs]
    for f in figs:
      plt.close(f)
    note_ok = (not E['curved']) == any(
        t.startswith('cell edges drawn as chords') for t in notes)
    check(
        v_alone[0] >= 0.5 and v_alone[1] >= 10 and not v_alone[2]
        and v_over[0] >= 0.5 and not v_over[2] and v_q[0] >= 0.9
        and v_q[1] >= 10 and not v_q[2] and bare == 0 and nax == [2, 1]
        and note_ok
        and (curves is None or
             (curves[0] >= 0.9 and curves[2] >= 5 and curves[3] == 0)),
        f'the grid and the quantity, matplotlib ({name}{", shown whole" if mirror else ""}; edges '
        f'{"curved" if E["curved"] else "chords, said on the plane"}): the grid alone {v_alone}, over the quantity '
        f'{v_over} (share of points along the edges changed, points, pixels changed elsewhere); the quantity '
        f'{v_q} (share of cell centres changed, centres, pixels changed outside the cells); both off: {bare} '
        f'pixels of the plane not its surface; the plane alone has {nax} axes with / without the quantity'
        +
        ('' if curves is None else
         f'; through the Gauss nodes {curves[0]} of {curves[1]}, along the chords where they bend away '
         f'{curves[3]} of {curves[2]} points'))
  if chrome is None:
    return
  cases = []
  for dev in ('tcv', 'd3d'):
    p = os.path.join(devices, dev, 'rt_2x2v_p1')
    if not os.path.isdir(p):
      continue
    run = gv.open_run(p)
    quads, cen = _cell_geometry(run, False)
    cases.append(
        (dev,
         dict(path=p,
              frame='0',
              grid='true',
              quantity='false',
              **{k: 'false'
                 for k in off}), _line_paths(run, False)['grid'], quads, cen))
  if not cases:
    return
  with served(chrome, ['--path', cases[0][1]['path']]) as (url, dbg):
    check(url is not None, 'the grid and the quantity: view.sh serves the page')
    if url is None:
      return
    try:
      got = asyncio.run(_drive_grid(url, dbg, cases, shots))
    except RuntimeError as e:
      check(False, f'the grid and the quantity, the page: {e}')
      return
  for name, st in got.items():
    _, area, gpx, qpx, cpx, _ = st[False, True]
    v_alone = _switched(st[False, True][0], st[False, False][0], gpx, area, 3)
    v_over = _switched(st[True, True][0], st[True, False][0], gpx, area, 3)
    v_q = _quantity_verdict(st[True, False][0], st[False, False][0], qpx, cpx,
                            area, 2)
    renders = {s[5] for s in st.values()}
    check(
        v_alone[0] >= 0.5 and v_alone[1] >= 10 and not v_alone[2]
        and v_over[0] >= 0.5 and not v_over[2] and v_q[0] >= 0.9 and not v_q[2]
        and len(renders) == 1,
        f'the grid and the quantity, the page ({name}): the grid alone {v_alone}, over the quantity {v_over}; the '
        f'quantity {v_q}; the movie not made again (renders {sorted(renders)})')


# --------------------------------------------------------------------------------------- the movie's margins (11-14)
def _margin_foreign(img, area, surf, texts):
  """Pixels of img outside the plot area (x0, y0, x1, y1; 1 px slack) that are neither the surface nor text: off the
  colour line from surf to each text colour (antialiased glyphs lie on it) by more than 40. What the poloidal plane
  drew past its area."""
  h, w = img.shape[:2]
  x0, y0, x1, y1 = (int(round(v)) for v in area)
  mask = np.ones((h, w), bool)
  mask[max(y0 - 1, 0):y1 + 2, max(x0 - 1, 0):x1 + 2] = False
  px = img[mask][:, :3].astype(float)
  surf = np.asarray(surf, float)
  d = np.full(len(px), np.inf)
  for t in texts:
    v = np.asarray(t, float) - surf
    u = np.clip((px - surf) @ v / max(v @ v, 1e-9), 0, 1)
    d = np.minimum(d, np.linalg.norm(px - (surf + u[:, None] * v), axis=1))
  return int((d > 40).sum())


async def _drive_canvas(url, dbg, states, shots):
  """For each state (view keys), the page opened on it: (state, the poloidal canvas RGBA, its plot area in canvas
  px, the surface colour, the text colours, the view's title on the canvas); a screenshot of each in shots."""
  from urllib.parse import urlencode
  from PIL import Image
  latest = (
      "__deep('.gkv-movie').reduce((a, e) => (+e.dataset.render || 0) > (+(a && a.dataset.render) || 0) "
      "? e : a, null)")
  rgb = (
      "(c => { const k = document.createElement('canvas').getContext('2d'); k.fillStyle = c; k.fillRect(0, 0, 1, 1); "
      "return [...k.getImageData(0, 0, 1, 1).data].slice(0, 3); })")
  grab = (
      "(() => { const el = " + latest +
      ", cv = el.querySelector('canvas.rz'), st = el.style, c = " + rgb + "; "
      "return { png: cv.toDataURL('image/png'), area: el.__rz.area(), surf: c(st.getPropertyValue('--nm-surf')), "
      "texts: [c(st.getPropertyValue('--nm-ink')), c(st.getPropertyValue('--nm-ink2'))] }; })()"
  )
  out = []
  async with opened(url, dbg) as c:
    for k, state in enumerate(states):
      await c.call('Page.navigate', url=url + '&' + urlencode(state))
      await asyncio.sleep(3)
      await c.js(DEEP)
      await c.until(f"(e => e && +(e.dataset.draws || 0) > 0)({latest})", 180)
      if vs_bool(state.get('background')):
        await c.until(
            "Object.values(window.__gkvBackground || {}).some(i => i.complete && i.naturalWidth)",
            60)
      await asyncio.sleep(2.5)
      g = await c.strict(grab)
      out.append((state,
                  np.asarray(
                      Image.open(
                          io.BytesIO(base64.b64decode(g['png'].split(
                              ',', 1)[1]))).convert('RGBA')), g['area'],
                  g['surf'], g['texts']))
      if shots:
        await c.shot(os.path.join(shots, f'canvas_{k:02d}.png'))
  return out


def vs_bool(v):
  return str(v).lower() in ('1', 'true', 'yes', 'on')


def canvas_states(full=False):
  """Views of the movie's poloidal plane to open: each simulation in its views (full: every one; else the whole
  domain, the whole machine where it has one, and a divertor), a half domain mirrored too, with the device picture
  where viewer.json names one; and the light theme."""
  import gkview as gv
  import viewer_plots as vp
  out = []
  for lab, d, dflt, _ in gv.SIMS:
    run = gv.open_run(os.path.join(gv.runs_root(d), dflt), lab)
    pic = 'true' if gv.device_image(run.device) else 'false'
    for mirror in ((False, True) if run.layout.half else (False, )):
      views = list(vp.zoom_boxes(run, mirror=mirror))
      views = views if full else [
          v for v in views if v.startswith('whole ') or v.endswith('divertor')
      ][:3]
      out += [
          dict(sim=lab,
               zoom=v,
               mirror=str(mirror).lower(),
               background=pic,
               opacity='0.6') for v in views
      ]
  half = next((st for st in out
               if st['mirror'] == 'false' and st['zoom'] == 'lower divertor'),
              out[0])
  out.append(dict(half, dark='false'))
  return out


async def _drive_tables(url, dbg):
  """The Options tab of Sim 2 with the 2025 re-run over it (both its tables): [(content width, visible width)] of
  each table on the page."""
  from urllib.parse import urlencode
  async with opened(url, dbg) as c:
    await c.call(
        'Page.navigate',
        url=url + '&' +
        urlencode(dict(sim='Sim 2', cmp='2025 re-run/prod', tab='Options')))
    await asyncio.sleep(3)
    await c.js(DEEP)
    await c.until("__deep('table').length >= 2", 180)
    await asyncio.sleep(3)
    return await c.strict(
        "__deep('table').map(t => [Math.round(t.scrollWidth), "
        "Math.round((t.parentElement || t).clientWidth)])")


def part_layout(chrome, shots=None, full=False):
  """The page's layout (2026-11-14): the movie's poloidal plane stays in its plot area (in every state of
  canvas_states, nothing but text in the canvas's margins: a zoomed view's cells and wall covered the title and the
  axes); the Options tab's tables fit their panel (their last columns ran off its edge)."""
  states = canvas_states(full)
  with served(chrome, []) as (url, dbg):
    check(url is not None, "the page's layout: view.sh serves the page")
    if url is None:
      return
    got = asyncio.run(_drive_canvas(url, dbg, states, shots))
    tables = asyncio.run(_drive_tables(url, dbg))
  check(
      len(tables) >= 2 and all(w <= v + 1 for w, v in tables),
      f'the Options tab with an overlay: each table fits its panel ((content, visible) px: {tables})'
  )
  bad = [(st, n) for st, img, area, surf, texts in got
         if (n := _margin_foreign(img, area, surf, texts))]
  check(
      len(got) == len(states) and not bad,
      f"the movie's margins: nothing of the poloidal plane past its plot area in {len(got)} views "
      f"(simulations x views, mirrored, the device picture, light and dark); stray pixels: "
      f"{[(st['sim'], st['zoom'], st['mirror'], st.get('dark', 'true'), n) for st, n in bad][:6]}"
  )


# ------------------------------------------------------------------------------------ a blank start, devices (11-16)
@contextlib.contextmanager
def config_file(path):
  """gkview reading the viewer.json at path (gkview.reload, as a form that writes it would), its own afterwards."""
  import gkview as gv
  keep = gv.CONFIG_FILE
  gv.CONFIG_FILE = path
  gv.reload()
  try:
    yield gv
  finally:
    gv.CONFIG_FILE = keep
    gv.reload()


BLANK_CASES = {  # data directory -> its viewer.json (None: none)
    'no viewer.json': None,
    '"sims": []': {
        'sims': []
    },
    'a simulation whose directory is missing': {
        'device': 'NSTX-U',
        'sims': [['Gone', 'no_such_simulation', 'prod', 'its directory is missing']]
    },
}


def blank_data_dirs(tmp):
  """BLANK_CASES as data directories under tmp: {case: directory}."""
  out = {}
  for i, (case, cfg) in enumerate(BLANK_CASES.items()):
    d = os.path.join(tmp, f'blank{i}')
    os.makedirs(d)
    if cfg is not None:
      json.dump(cfg, open(os.path.join(d, 'viewer.json'), 'w'))
    out[case] = d
  return out


def _plotly_blank(fig):
  """(data points drawn, panel titles, axis titles, legend entries) of a plotly figure."""
  pts = 0
  for tr in fig.data:
    for a in ('x', 'y', 'z'):
      v = getattr(tr, a, None)
      if v is not None:
        pts = max(
            pts,
            sum(x is not None for x in np.ravel(np.asarray(v, dtype=object))))
  titles = [a.text for a in fig.layout.annotations or ()
            ] + ([fig.layout.title.text] if fig.layout.title.text else [])
  axes = [
      fig.layout[k].title.text for k in fig.layout
      if k.startswith(('xaxis', 'yaxis')) and fig.layout[k].title.text
  ]
  legend = [
      tr.name for tr in fig.data if tr.showlegend is not False and tr.name
  ]
  return pts, titles, axes, legend


def part_blank(chrome=None, shots=None):
  """A blank start (2026-11-16): on a data directory with no viewer.json, with "sims": [], and naming a simulation
  whose directory is missing, the notebook runs with every tab built on the empty run (gkview.empty_run); each chart
  of the empty run has its panel titles, axis titles and legend and draws no data point; the movie has no frame and
  every view of the poloidal plane none. With chrome, the same three served by view.sh: the note at the top, the
  movie's canvases with its controls idle, every tab opened with its charts drawn and no error text (screenshots of
  the blank page, dark and light, in shots). The per-device schema of viewer.json: an old file reads as before; a
  simulation's own device, picture, geqdsk and wall; entries that do not parse left out with a note."""
  import gkview as gv
  import movie_widget as mm
  import viewer_plots as vp
  with tempfile.TemporaryDirectory() as tmp:
    dirs = blank_data_dirs(tmp)
    tabs = None
    for case, d in dirs.items():
      t = time.time()
      with config_file(os.path.join(d, 'viewer.json')):
        defs, built = headless({})
      tabs = defs['TABS']
      bad = [n for n, v in built.items() if v != 'ok']
      notes = defs['boot']['notes']
      check(
          defs['run'].path == '' and not defs['frames'] and defs['k'] is None
          and defs['sim_pick'].value is None and defs['run_pick'].value is None
          and len(built) == len(tabs) and not bad
          and (any('Gone: skipped' in n
                   for n in notes) if 'missing' in case else not notes),
          f'a blank start, {case}: the notebook opens on the empty run, every tab builds ({len(built)} of '
          f'{len(tabs)}{", failed: " + str(bad) if bad else ""}), notes {notes} ({time.time() - t:.1f} s)'
      )
    run, pal = gv.empty_run(), vp.look('NSTX-U', True)
    summ = gv.summary(run)
    B = gv.balance(run)
    figs = {
        'profiles':
        vp.profiles_figure(run, {}, {}, '', pal=pal),
        'profiles (R)':
        vp.profiles_figure(run, {}, {}, '', 'R', pal=pal),
        'midplane history':
        vp.history_figure(run, summ, 'n_elc', 'change', np.nan, pal=pal),
        'targets':
        vp.targets_figure(run, {}, summ, np.nan, pal=pal),
        'time traces':
        vp.timetraces_figure(run,
                             summ,
                             gv.dt_history(run),
                             gv.integrated(run),
                             gv.field_energy(run),
                             np.nan,
                             pal=pal),
        'conservation':
        vp.conservation_figure(run, B, summ['t'], np.nan, pal=pal)[0],
        'balance':
        vp.balance_figure(run, B, np.nan, pal=pal)[0],
        'diffusivity':
        vp.diffusivity_figure(run, gv.diffusivity(run), pal=pal),
    }
    got = {name: _plotly_blank(f) for name, f in figs.items()}
    live = [
        mm.live_plot(f, summ, None, [], pal, name) for name, f in figs.items()
    ]
    # (the midplane history, a heat map, has no legend, data or not)
    check(
        all(p == 0 and ax and (leg or name == 'midplane history')
            for name, (p, _, ax, leg) in got.items()) and
        all(ti for name, (_, ti, _, _) in got.items() if name != 'conservation')
        and all(not w.live['frames'] and not w.live['shapes'] for w in live),
        'the empty run\'s charts: panels, axis titles and legend, no data point and nothing following a frame '
        f'(points, panel titles, axis titles, legend entries: '
        f'{ {n: (p, len(ti), len(ax), leg) for n, (p, ti, ax, leg) in got.items()} })'
    )
    data = mm.movie_data(run, summ, 'n_elc', 'change')
    fig = vp.map_figure(run, {}, {}, 'n_elc', 'change', pal=pal)
    # the poloidal plane's meshes (the colour bar has its own)
    meshes = len(fig.axes[0].collections)
    labels = [a.get_title(loc='left')
              for a in fig.axes] + [a.get_xlabel() for a in fig.axes]
    vp.plt.close(fig)
    check(
        data['frames'] == [] and data['ncell'] == 0 and data['strips'] == []
        and data['boxes'] == {'whole domain': None}
        and [len(s['x']) for s in data['prof']['sides']] == [0, 0]
        and meshes == 0 and 'poloidal plane' in labels and 'R (m)' in labels,
        f'the empty run\'s movie: no frame, no cell, no strip, no view to frame, both midplanes empty; its map: '
        f'{meshes} meshes, the axes {[x for x in labels if x]}')
    part_schema(tmp)
    if chrome is None:
      return
    # the charts each tab draws (the midplane history, a heat map, has no legend with data or without)
    want = {
        'Midplane profiles': 2,
        'Targets': 1,
        'Time traces': 1,
        'Balance & power': 2,
        'Diffusivity': 1
    }
    for k, (case, d) in enumerate(dirs.items()):
      with served(chrome, [], env={'GK_VIEWER_DATA': d}) as (url, dbg):
        check(url is not None,
              f'a blank start, {case}: view.sh serves the page')
        if url is None:
          continue
        st = asyncio.run(
            _drive_blank(url, dbg, tabs, want, shots if k == 0 else None))
      charts = {t: v['charts'] for t, v in st['tabs'].items()}
      not_blank = [(t, c['title'], c['lines'], c['images'], len(c['axes']),
                    c['legend']) for t, v in charts.items() for c in v
                   if c['lines'] or c['images'] or not c['axes']
                   or not (c['legend'] or 'over time' in c['title'])]
      check(
          st['note'] and st['movie']
          and st['movie']['time'] == 'no frames (no run)'
          and st['movie']['canvases'] == 4 and st['movie']['idle']
          and all(len(charts.get(t, [])) == n
                  for t, n in want.items()) and not not_blank
          and st['tabs']['Static figure']['images'] >= 1 and not st['errors'],
          f'a blank start in the browser, {case}: the note at the top {st["note"]}, the movie {st["movie"]}, every '
          f'tab opened ({len(st["tabs"])}): charts drawn {({t: len(c) for t, c in charts.items() if c})} of {want}, '
          f'each with axis titles and a legend and no curve or map in it (but {not_blank}), the static figure an '
          f'image ({st["tabs"]["Static figure"]["images"]}); error text: {st["errors"][:3]}'
      )


async def _drive_blank(url, dbg, tabs, want, shots):
  """The blank page: the note at the top, the movie, each tab opened (want: tab -> the charts it should draw, waited
  for; each chart scrolled into view and drawn: its title, axis titles, legend, drawn lines and images), the page's
  error text; screenshots (dark, then light) in shots."""
  panel = (
      "__deep('[role=tabpanel]').filter(p => p.getAttribute('data-state') === 'active' && "
      "p.getBoundingClientRect().height > 0)")
  inside = (
      "(root, s) => { const o = []; const w = r => { r.querySelectorAll(s).forEach(e => o.push(e)); "
      "r.querySelectorAll('*').forEach(e => { if (e.shadowRoot) w(e.shadowRoot); }); }; w(root); return o; }"
  )
  text = (
      "(() => { const o = []; const w = r => { for (const n of r.childNodes) { if (n.nodeType === 3) "
      "o.push(n.textContent); else if (n.nodeType === 1) { if (n.shadowRoot) w(n.shadowRoot); w(n); } } }; "
      "w(document.body); return o.join(' '); })()")
  out = {'tabs': {}}
  async with opened(url, dbg) as c:
    await asyncio.sleep(3)
    await c.js(DEEP)
    await c.until("__deep('.gkv-movie').some(e => +(e.dataset.draws || 0) > 0)",
                  180)
    await asyncio.sleep(2)
    out['movie'] = await c.js(
        "(e => e && {time: e.querySelector('.time').textContent, canvases: e.querySelectorAll('canvas').length, "
        "idle: ['.play', '.prev', '.next', '.scrub'].every(s => e.querySelector(s).disabled)})"
        "(__deep('.gkv-movie')[0])")
    out['note'] = bool(await c.js(f"{text}.includes('No simulation data yet')"))
    if shots:
      os.makedirs(shots, exist_ok=True)
      await c.shot(os.path.join(shots, 'blank_dark_top.png'))
    for name in tabs:
      await c.press(f"__tab({json.dumps(name)})")
      await asyncio.sleep(2)
      # the panel scrolled to (its outputs mount as they come into view) until the charts it should draw are there
      count = (
          f"(p => {{ if (!p) return 0; p.scrollIntoView({{block: 'end'}}); "
          f"return ({inside})(p, '.gkv-live').length; }})(({panel})[0])")
      if want.get(name):
        await c.until(f"{count} >= {want[name]}", 90)
      await asyncio.sleep(2)
      n = await c.js(count) or 0
      charts = []
      for i in range(n):
        await c.js(
            f"({inside})({panel}[0], '.gkv-live')[{i}].scrollIntoView({{block: 'center'}})"
        )
        await c.until(
            f"!!({inside})({panel}[0], '.gkv-live')[{i}].querySelector('svg.main-svg .xtitle, "
            f"svg.main-svg .g-xtitle text')", 60)
        await asyncio.sleep(0.5)
        charts.append(await c.js(
            f"(e => ({{title: [...e.querySelectorAll('.gtitle, .annotation-text')].map(t => t.textContent)"
            f".join(' / '), axes: [...e.querySelectorAll('g[class^=g-x][class$=title] text, g[class^=g-y][class$=title] text')]"
            f".map(t => t.textContent)"
            f".filter(Boolean), legend: [...e.querySelectorAll('.legendtext')].map(t => t.textContent), "
            f"lines: [...e.querySelectorAll('.scatterlayer path.js-line')].filter(p => /L/.test(p.getAttribute('d') "
            f"|| '')).length, images: e.querySelectorAll('.heatmaplayer image, .hm image').length}}))"
            f"(({inside})({panel}[0], '.gkv-live')[{i}])"))
      images = await c.js(
          f"({panel}).length ? ({inside})({panel}[0], 'img').length : 0") or 0
      out['tabs'][name] = {'charts': charts, 'images': images}
      if shots and name == 'Midplane profiles':
        await c.shot(os.path.join(shots, 'blank_dark_profiles.png'))
    page = await c.js(text) or ''
    out['errors'] = re.findall(
        r'\b[A-Z][A-Za-z]*Error\b[^.]{0,60}|Traceback|An internal error occurred',
        page)
    if shots:  # the light look: the top, then the profiles tab's charts
      await c.js("__switch('dark mode').click()")
      await c.until("!document.body.classList.contains('dark')", 15)
      await asyncio.sleep(3)
      await c.shot(os.path.join(shots, 'blank_light_top.png'))
      await c.press("__tab('Midplane profiles')")
      await asyncio.sleep(2)
      await c.until(
          f"(p => p && ({inside})(p, '.gkv-live').length >= 2)(({panel})[0])",
          60)
      await c.js(
          f"({inside})(({panel})[0], '.gkv-live')[0].scrollIntoView({{block: 'start'}})"
      )
      await asyncio.sleep(3)
      await c.shot(os.path.join(shots, 'blank_light_profiles.png'))
  return out


def part_schema(tmp):
  """viewer.json's per-simulation and per-device settings (gkview.load_config): this data directory's own file reads
  as it did before them (the old reading, written out here); a simulation's own device names its runs and picks
  its picture over the top-level one, which stays the default device's; its geqdsk serves a run whose files name
  none (shot, time, wall), the run's own winning where both name one (the disagreement said); a device's wall where
  the geqdsk's limiter has fewer than 3 points, never its shot; entries that do not parse left out with a note."""
  import gkview as gv
  cfg = json.load(open(gv.CONFIG_FILE))
  old_base = os.path.normpath(os.path.join(gv.DATA, cfg.get('base', '.')))
  d = cfg.get('device_image')
  old_image = d and (os.path.normpath(os.path.join(old_base, d['file'])),
                     tuple(float(v)
                           for v in d['extent']), float(d.get('opacity', 1.0)))
  lab, simdir, dflt, _ = next(s for s in gv.SIMS if s[0] == 'Sim 2')
  sim2 = gv.open_run(os.path.join(gv.runs_root(simdir), dflt), lab)
  # its entries without settings of their own (the NSTX-U simulations) read as before; those with their own (the
  # device runs, since 2026-11-17) name their device
  plain = [e for e in cfg['sims'] if len(e) == 4]
  own = [e for e in cfg['sims'] if len(e) == 5]
  check(
      len(plain) >= 3 and gv.SIMS == [tuple(e[:4]) for e in cfg['sims']]
      and gv.BASE == old_base and gv.DEVICE == cfg.get('device', '')
      and not gv.CONFIG_PROBLEMS
      and all(gv.SIM_SETTINGS[e[0]] == gv.SimSettings(gv.DEVICE, None)
              for e in plain)
      and all(gv.SIM_SETTINGS[e[0]].device == (e[4].get('device') or gv.DEVICE)
              for e in own) and gv.device_image(gv.DEVICE) == old_image
      and sim2.device == gv.DEVICE
      and sim2.title == f'{gv.DEVICE} #203899 @ 325 ms'
      and gv.geqdsk_disagreement(sim2) is None,
      f'this data directory\'s viewer.json reads as before for its {len(plain)} simulations without settings of '
      f'their own (the {len(own)} with their own name their device): the simulations, base, device '
      f'{gv.DEVICE!r}, its picture, Sim 2\'s title {sim2.title!r}')
  geq = gv.geqdsk_path(sim2)
  # a run that names no geqdsk: Sim 2's files but its options and command, under a stem naming no shot
  farm = os.path.join(tmp, 'farm', 'runs', 'run')
  os.makedirs(farm)
  k0 = sim2.frames()[0]
  for pat in (f'{sim2.stem}-block_topo.gkyl',
              f'{sim2.stem}_b*-geo_corn_nodes.gkyl',
              f'{sim2.stem}_b0-field_[0-9]*.gkyl',
              f'{sim2.stem}_b0-*_M0_{k0}.gkyl'):
    for f in glob.glob(os.path.join(sim2.path, pat)):
      os.symlink(
          f, os.path.join(farm,
                          os.path.basename(f).replace(sim2.stem, 'farm')))
  # a geqdsk with no shot in its header and a 1-point limiter (as STEP's)
  lines = open(geq).read().split('\n')
  i = next(i for i, ln in enumerate(lines)
           if i > 5 and re.fullmatch(r'\s*\d+\s+\d+\s*', ln))
  lines[0] = re.sub(r'#\s*\d+\s+\d+\s*ms', 'no shot here', lines[0])
  lines[i] = f'{lines[i].split()[0]:>5} {1:>4}'
  one = os.path.join(tmp, 'one_point_limiter.geqdsk')
  open(one, 'w').write('\n'.join(lines))
  pic = os.path.join(tmp, 'test_picture.png')
  from PIL import Image
  Image.new('RGB', (40, 80), (0, 0, 255)).save(pic)
  image = {'file': pic, 'extent': [0.1, 2.1, -2.0, 2.0], 'opacity': 0.5}
  test = dict(cfg,
              base=old_base,
              sims=[[
                  'Sim 2 as Test', simdir, dflt, 'Sim 2, of device Test', {
                      'device': 'Test',
                      'geqdsk': one
                  }
              ],
                    [
                        'Farm',
                        os.path.join(tmp, 'farm'), 'run',
                        'Sim 2 naming no geqdsk', {
                            'geqdsk': geq
                        }
                    ],
                    [
                        'Farm, one point',
                        os.path.join(tmp, 'farm'), 'run', 'the same', {
                            'device': 'Walled',
                            'geqdsk': one
                        }
                    ],
                    [
                        'Other', simdir, dflt,
                        'a device with nothing of its own', {
                            'device': 'Other'
                        }
                    ], ['bad', 'x', 'y'],
                    ['bad too', 'x', 'y', 'z', {
                        'geqdks': 'typo'
                    }]],
              devices={
                  'Test': {
                      'device_image': image
                  },
                  'Walled': {
                      'wall': geq
                  }
              })
  path = os.path.join(tmp, 'schema', 'viewer.json')
  os.makedirs(os.path.dirname(path))
  json.dump(test, open(path, 'w'))
  with config_file(path):
    a = gv.open_run(os.path.join(gv.runs_root(simdir), dflt), 'Sim 2 as Test')
    b = gv.open_run(farm, 'Farm')
    c = gv.open_run(farm, 'Farm, one point')
    o = gv.open_run(os.path.join(gv.runs_root(simdir), dflt), 'Other')
    got = dict(labels=[s[0] for s in gv.SIMS],
               problems=len(gv.CONFIG_PROBLEMS),
               test_image=gv.device_image('Test'),
               default_image=gv.device_image(gv.DEVICE),
               other_image=gv.device_image('Other'),
               a=(a.device, a.title, gv.geqdsk_path(a) == geq,
                  gv.geqdsk_disagreement(a) == (geq, one)),
               b=(b.device, b.title, gv.geqdsk_path(b) == geq,
                  np.array_equal(gv.wall(b), gv.limiter(geq))),
               c=(c.title == c.name, gv.limiter(one)
                  is None, np.array_equal(gv.wall(c), gv.limiter(geq))),
               o=(o.device, o.title))
  check(
      got['labels'] == ['Sim 2 as Test', 'Farm', 'Farm, one point', 'Other']
      and got['problems'] == 2
      and got['test_image'] == (pic, (0.1, 2.1, -2.0, 2.0), 0.5)
      and got['default_image'] == old_image and got['other_image'] is None
      and got['a'] == ('Test', 'Test #203899 @ 325 ms', True, True)
      and got['b'] == (cfg.get('device', ''),
                       f'{cfg.get("device", "")} #203899 @ 325 ms', True, True)
      and got['c'] == (True, True, True)
      and got['o'] == ('Other', 'Other #203899 @ 325 ms'),
      'viewer.json per simulation and per device: a simulation\'s own device names its runs and gets its own picture '
      '(the top-level one stays the default device\'s; a device with none gets none); a run naming its geqdsk keeps it '
      'over the simulation\'s (the disagreement said); one naming none takes the simulation\'s (shot, time, wall); a '
      '1-point limiter gives way to the device\'s wall, which gives no shot; 2 malformed entries left out with notes: '
      f'{ {k: v for k, v in got.items() if k not in ("test_image", "default_image")} }'
  )


# ----------------------------------------------------------------------------------------- the curves shown (11-17)
MEASURED_GROUPS = {'thomson', 'chers',
                   'transp'}  # measured_trace's legend groups, one per source
SAID = {  # a species named in what the page shows of a curve (its words, a subscript, a measured quantity)
    'electrons': re.compile(r'\belectrons?\b|\belc\b|<sub>e</sub>|\b[nT]_e\b'),
    'ions': re.compile(r'\bions?\b|<sub>i</sub>|\bT_i\b|\bn_D\b')
}


def _said(fig):
  """What a chart says each of its traces is, read from what the page shows of it, not from its tag: [(kind,
  species)]. kind: 'experiment' for a measured source's trace (its legend group: thomson, chers, transp) and TRANSP's
  legend key, None for a reference line's key (the separatrix), else 'simulation'; species: electrons or ions where
  exactly one of them is named by the trace's legend name or group, its hover text or (not a legend key) its panel's
  title and y axis title, else None; TRANSP's key: ions (the profile it marks, T_i)."""
  import viewer_plots as vp
  panel, out = {}, []
  for tr in fig.data:
    if tr.legendgroup == 'key':
      name = tr.name or ''
      out.append((None, None) if 'separatrix' in name else (
          'experiment', 'ions') if name.startswith('TRANSP') else ('simulation',
                                                                   None))
      continue
    x = tr.xaxis or 'x'
    if x not in panel:
      pf = vp.panel_figure(fig, x)[0]
      panel[
          x] = f'{pf.layout.title.text or ""} | {pf.layout.yaxis.title.text or ""}'
    text = ' | '.join(
        str(v or '')
        for v in (tr.name, tr.legendgroup, tr.hovertemplate, panel[x]))
    named = [k for k, rx in SAID.items() if rx.search(text)]
    out.append(
        ('experiment' if tr.legendgroup in MEASURED_GROUPS else 'simulation',
         named[0] if len(named) == 1 else None))
  return out


def _numbered(fig):
  """A copy of a chart with each trace's uid its index (shown_curves keeps them: what it removed can be read)."""
  import plotly.graph_objects as go
  f = go.Figure(fig)
  for i, tr in enumerate(f.data):
    tr.uid = str(i)
  return f


def _removed(fig, curves):
  """The traces of a _numbered chart that the switches take away; None: the chart is left with no curve."""
  import viewer_plots as vp
  shown = vp.shown_curves(fig, curves)
  return None if shown is None else set(range(len(
      fig.data))) - {int(tr.uid)
                     for tr in shown.data}


def _one_off(f, k):
  """Switch k alone off on a _numbered chart: (the problems, the traces it should take away, whether it empties the
  chart). It should take away exactly the traces _said makes its own; leave the rest as they were (a legend entry may
  move to a trace of its group: name and showlegend aside), each group left with its legend entry; put the note
  '<k> switched off' in each panel it empties and nowhere else; leave no label (an annotation in a panel's data
  coordinates) in such a panel; and where it takes every curve, leave none and give hidden_note's words."""
  from plotly.utils import PlotlyJSONEncoder
  import viewer_plots as vp
  curves = vp.Curves(**{k: False})
  shown = vp.shown_curves(f, curves)
  gone = {i for i, said in enumerate(_said(f)) if k in said}
  data = [i for i, tr in enumerate(f.data) if tr.legendgroup != 'key']
  if all(i in gone for i in data):
    return ([] if shown is None
            and vp.hidden_note(f, curves) == f'{k} switched off' else
            ['not emptied, or a wrong note']), gone, True
  if shown is None:
    return ['emptied'], gone, False
  bad, kept = [], {int(tr.uid): tr for tr in shown.data}
  if set(kept) != set(range(len(f.data))) - gone:
    bad.append(
        f'took {sorted(set(range(len(f.data))) - set(kept))}, its own: {sorted(gone)}'
    )
  for i, tr in kept.items():
    a, b = (json.loads(json.dumps(t.to_plotly_json(), cls=PlotlyJSONEncoder))
            for t in (tr, f.data[i]))
    for d in (a, b):
      d.pop('showlegend', None)
      d.pop('name', None)
    if a != b:
      bad.append(f'trace {i} ({f.data[i].name}) changed')
  for g in {tr.legendgroup for tr in f.data} - {None, '', 'key'}:
    had = any(tr.showlegend is not False for tr in f.data
              if tr.legendgroup == g)
    left = [tr for tr in shown.data if tr.legendgroup == g]
    if had and left and all(tr.showlegend is False for tr in left):
      bad.append(f'legend group {g} lost its entry')
  where = lambda tr: (tr.xaxis or 'x', tr.yaxis or 'y')  # noqa: E731
  emptied = {where(tr)
             for tr in f.data if tr.legendgroup != 'key'
             } - {where(tr)
                  for tr in shown.data if tr.legendgroup != 'key'}
  notes = {
      (a.xref.split(' ')[0], a.yref.split(' ')[0])
      for a in shown.layout.annotations
      if (a.xref or '').endswith(' domain') and a.text == f'{k} switched off'
  }
  if notes != emptied:
    bad.append(f'notes in panels {sorted(notes)}, emptied {sorted(emptied)}')
  stray = [
      a.text for a in shown.layout.annotations if (a.xref, a.yref) in emptied
  ]
  if stray:
    bad.append(f'labels left in emptied panels: {stray}')
  return bad, gone, False


KEPT_CHARTS = (
    'history', 'thomson'
)  # the midplane over time (a map of one quantity), the shot's timeline
SWITCH_IS = """window.__switchIs = t => __deep('button[role=switch]').find(s => { let n = s;
  for (let i = 0; i < 4 && n; i++) { n = n.parentElement || (n.getRootNode() && n.getRootNode().host);
  if (n && (n.textContent || '').trim() === t) return true; } return false; }); 'ok'"""
CURVE_LABELS = ['Electrons', 'Ions', 'Simulation', 'Experiment']
# the Midplane profiles chart drawn (its plotly div), and what it draws: its traces, those tagged ions, the name its
# slide downloads go by
PROFILES_CHART = (
    "__deep('.gkv-live .js-plotly-plot').find(d => d.querySelector('svg.main-svg') && (d.data || []).some(t => "
    "String((t.meta || {}).live || '').startsWith('mid|')))")
PROFILES_READ = (
    "(d => d && { n: d.data.length, ions: d.data.filter(t => ((t.meta || {}).curve || []).includes('ions')).length, "
    "name: Object.keys(window.__gkvSlides || {}).find(k => k.includes('_midplane_profiles_')) })("
    + PROFILES_CHART + ")")


async def _drive_species(url, dbg, shots):
  """Sim 2's page: the four switches' states, the profiles chart, Ions clicked off (the chart drawn again, its slide
  figure, the address), then the page reloaded. Returns those readings."""
  out = {}

  async def ready():
    await asyncio.sleep(3)
    await c.js(DEEP)
    await c.js(SWITCH_IS)
    await c.until("__deep('[role=tablist]').length > 0", 180)
    await c.js(
        "(__deep('[role=tablist]')[0] || document.body).scrollIntoView({block: 'start'})"
    )
    return await c.until(PROFILES_READ, 180)

  states = ("[" + ', '.join(
      f"(s => s ? s.getAttribute('aria-checked') : null)(__switchIs({json.dumps(t)}))"
      for t in CURVE_LABELS) + "]")
  async with opened(url, dbg) as c:
    out['before'] = await ready()
    out['states'] = await c.js(states)
    await c.strict("__switchIs('Ions').click()")
    out['off'] = await c.until(
        f"(r => r && r.ions === 0 ? r : null)({PROFILES_READ})", 120)
    out['search'] = await c.js('location.search')
    if out['off'] and out['off']['name']:
      out['slide'] = await c.js(
          f"(f => ({{n: f.data.length, ions: f.data.filter(t => ((t.meta || {{}}).curve || [])"
          f".includes('ions')).length}}))(window.__gkvSlides[{json.dumps(out['off']['name'])}]('full', true))"
      )
    if shots:  # the control, then the profiles chart without the ions
      for what, el in (('control', "__switchIs('Ions')"), ('profiles',
                                                           PROFILES_CHART)):
        await c.js(f"(e => e && e.scrollIntoView({{block: 'center'}}))({el})")
        await asyncio.sleep(1)
        await c.shot(os.path.join(shots, f'species_ions_off_{what}.png'))
    await c.call('Page.reload')
    out['reloaded'] = await ready()
    out['states_reloaded'] = await c.js(states)
  return out


def part_species(chrome=None, shots=None):
  """The curves shown (2026-11-17): the sidebar's four switches (electrons, ions, simulation, experiment) on every
  line chart of the three NSTX-U simulations, each with another drawn over it and its shot's measured profiles. All on,
  each chart is the one drawn without the switches (shown_curves gives it back). Each switch alone off takes away
  exactly the curves the chart says are its own (_said: legend names and groups, hover text, panel titles; not the
  tags) and their labels, leaves the rest as they were with a legend entry for each group left, puts a note in each
  panel it empties, and leaves a note in place of a chart it empties (_one_off). A species is told by its charge's
  sign: the charges swapped, electrons and ions swap; a charge of 0 answers to neither. The notebook, launched with
  switches off: the view keeps them (the ./view.sh line, a state file), every chart it hands the page and its slide
  downloads hold only the curves shown, and a tab whose chart is emptied shows the note. With chrome, Sim 2's page:
  the four switches, on by default; Ions clicked off, the profiles chart drawn again without its ion curves (its
  slide figure too) and the address saying so; reloaded, still so."""
  import dataclasses
  import gkview as gv
  import movie_widget as mm
  import thomson
  import viewer_plots as vp
  import viewer_state as vs
  runs = [
      gv.open_run(os.path.join(gv.runs_root(d), dflt), lab)
      for lab, d, dflt, _ in gv.SIMS if lab in ('Sim 1', 'Sim 2', '2025 re-run')
  ]

  def charts(run, summ, cmp=None):
    # every line chart of a run (_charts), its profiles with the shot's measured ones over them
    out = _charts(run, summ, cmp)
    fs = run.frames()
    th = thomson.read(run.shot)
    t0, t1 = thomson.default_window(th, run.eq_ms / 1e3)
    out['profiles'] = vp.profiles_figure(run,
                                         gv.frame(run, fs[-1]),
                                         gv.frame(run, fs[0]),
                                         '',
                                         measured=(th, t0, t1),
                                         cmp=cmp)
    return out

  same, bad, n, ntr, empty, kept = [], [], 0, 0, set(), []
  took = dict.fromkeys(vp.CURVE_SWITCHES, 0)
  for i, run in enumerate(runs):
    summ = gv.summary(run)
    cmp = _overlay(summ, runs[(i + 1) % len(runs)])
    for name, fig in charts(run, summ, cmp).items():
      if name in KEPT_CHARTS:  # whatever the switches
        kept += [
            vp.shown_curves(fig, vp.Curves(**{k: False})) is fig
            for k in vp.CURVE_SWITCHES
        ]
        continue
      n, ntr = n + 1, ntr + len(fig.data)
      same.append(vp.shown_curves(fig, vp.Curves()) is fig)
      f = _numbered(fig)
      for k in vp.CURVE_SWITCHES:
        problems, gone, emptied = _one_off(f, k)
        took[k] += len(gone)
        bad += [f'{run.sim} {name}, {k} off: {p}' for p in problems]
        if emptied:
          empty.add(f'{name} ({k})')
  check(
      all(same) and not bad and all(took.values()) and all(kept),
      f'{n} line charts of 3 simulations with an overlay and the measured profiles ({ntr} traces): all on, each as '
      f'drawn without the switches; each switch alone off takes exactly the curves the chart says are its own '
      f'({took}) and their labels, the rest unchanged with their legend entries, a note in each panel it empties, a '
      f'note in place of the chart it empties ({sorted(empty)}); the midplane over time (a map) and the shot\'s '
      f'timeline as they are whatever the switches ({sum(kept)} of {len(kept)}){bad[:6] or ""}'
  )
  # a species by its charge's sign, not its name: swapped charges swap the switches; a neutral answers to neither
  run = runs[1]
  summ = gv.summary(run)
  e = next(s for s, q in run.charge.items() if q < 0)
  swapped = dataclasses.replace(run,
                                charge={
                                    s: -q
                                    for s, q in run.charge.items()
                                })
  neutral = dataclasses.replace(run, charge=dict(run.charge, **{e: 0.0}))
  off = {k: vp.Curves(**{k: False}) for k in ('electrons', 'ions')}
  wrong, nsw = [], 0
  for (name, a), b, c in zip(
      charts(run, summ).items(),
      charts(swapped, summ).values(),
      charts(neutral, summ).values()):
    a, b, c = _numbered(a), _numbered(b), _numbered(c)
    ra = {k: _removed(a, v) for k, v in off.items()}
    nsw += len(ra['electrons'] or ()) + len(ra['ions'] or ())
    if (_removed(b, off['electrons']),
        _removed(b, off['ions'])) != (ra['ions'], ra['electrons']):
      wrong.append(f'{name}: swapped charges')
    if (_removed(c, off['electrons']),
        _removed(c, off['ions'])) != (set(), ra['ions']):
      wrong.append(f'{name}: {e} neutral')
  check(
      not wrong and nsw > 0,
      f'a species by its charge\'s sign (Sim 2, {nsw} species curves): the charges swapped, the electrons\' switch '
      f'takes what the ions\' took and back; {e} given a charge of 0, its switch takes nothing, the ions\' the same as '
      f'before{wrong or ""}')
  # the notebook: launched with ions and the experiment off (as a reload of an address saying so would be)
  handed, orig = [], mm.live_plot

  def recorded(fig, *a, **kw):
    w = orig(fig, *a, **kw)
    handed.append(w)
    return w

  mm.live_plot = recorded
  try:
    d, built = headless({
        'sim': 'Sim 2',
        'cmp': '2025 re-run/prod',
        'probe': '6,200,7',
        'ions': 'false',
        'experiment': 'false'
    })
  finally:
    mm.live_plot = orig
  want = vp.Curves(ions=False, experiment=False)
  v = d['view']
  cmd = vs.command_line(v)
  with tempfile.TemporaryDirectory() as tmp:
    back, _ = vs.merge(
        vs.load_state(vs.save_state(os.path.join(tmp, 's.json'), v)))
  keys = {k: vs.as_bool(back[k]) for k in vp.CURVE_SWITCHES}
  tags = lambda w: [((t.get('meta') or {}).get('curve') or [])
                    for t in w.fig['data']]  # noqa: E731
  leaks = [
      w.name for w in handed
      if any('ions' in t or 'experiment' in t for t in tags(w))
  ]
  slides = [
      w.name for w in handed
      if w.slides['full']['indices'] != list(range(len(w.fig['data'])))
  ]
  check(
      d['curves_shown'] == want and keys == dataclasses.asdict(want)
      and '--ions false --experiment false' in cmd and '--electrons' not in cmd
      and '--simulation' not in cmd and not leaks and not slides
      and len(handed) >= 8 and all(x == 'ok' for x in built.values()),
      f'the notebook launched with ions and the experiment off: switches {d["curves_shown"]}, the view keeps them '
      f'({cmd[cmd.find("--ions"):][:40]}...; a state file: {keys}); the {len(handed)} charts it hands the page, and '
      f'their slide downloads, hold only the curves shown (leaks {leaks}, slides {slides}); every tab builds'
  )
  texts = {}
  d, built = headless(
      {
          'sim': 'Sim 2',
          'probe': '6,200,7',
          'simulation': 'false'
      }, texts=texts)
  note = 'Nothing to draw here: simulation switched off in the sidebar.'
  count = {t: texts[t].count(note) for t in texts if note in texts[t]}
  check(
      count == {
          'Targets': 1,
          'Time traces': 1,
          'Probe': 1,
          'Balance & power': 2,
          'Diffusivity': 1
      } and all(x == 'ok' for x in built.values()),
      f'the simulation switched off (Sim 2): each tab\'s simulated chart is the note "{note}" ({count}); the '
      'midplane profiles keep the measured ones')
  if chrome is None:
    return
  with served(chrome, ['--sim', 'Sim 2']) as (url, dbg):
    check(url is not None, 'curves shown: view.sh serves the page')
    if url is None:
      return
    r = asyncio.run(_drive_species(url, dbg, shots))
  b, o, a = r.get('before'), r.get('off'), r.get('reloaded')
  check(
      r.get('states') == ['true'] * 4 and b and b['ions'] > 0 and o
      and o['n'] == b['n'] - b['ions'] and 'ions=false' in r.get('search', '')
      and r.get('slide') == {
          'n': o['n'],
          'ions': 0
      } and r.get('states_reloaded') == ['true', 'false', 'true', 'true'] and a
      and a['ions'] == 0 and a['n'] == o['n'],
      f'the page (Sim 2): the switches {CURVE_LABELS} on by default ({r.get("states")}); Ions clicked off, the '
      f'profiles chart drawn again without its {b and b["ions"]} ion curves ({b and b["n"]} -> {o and o["n"]} '
      f'traces; its slide figure {r.get("slide")}), the address saying so; reloaded, still off '
      f'({r.get("states_reloaded")}, {a and a["n"]} traces)')


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
  const els = __deep('.gkv-live').filter(e => plot(e) && (new_ ? !plot(e).__old : true));
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

MARK_OLD = "__deep('.gkv-live .js-plotly-plot').forEach(d => { d.__old = true; })"


def drawn_diff(a, b):
  """The largest pixel distance between two DRAWN readings of the same frame: each of b's curve vertices to a's
    (plotly drops collinear vertices), markers and visible shape lines point for point. None: they do not pair up."""
  if not (a and b and len(a['charts']) == len(b['charts'])):
    return None
  worst = 0.0
  for ca, cb in zip(a['charts'], b['charts']):
    if set(ca['lines']) != set(cb['lines']) or set(ca['shapes']) != set(
        cb['shapes']):
      return None
    for k, vb in cb['lines'].items():
      pa, pb = np.reshape(ca['lines'][k], (-1, 2)), np.reshape(vb, (-1, 2))
      if len(pb) and not len(pa):
        return None
      for x, y in pb:  # the nearest of a's vertices
        worst = max(
            worst,
            float(np.min(np.hypot(pa[:, 0] - x, pa[:, 1] -
                                  y))) if len(pa) else 0.0)
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
    await self.ws.send(
        json.dumps({
            'id': mid,
            'method': method,
            'params': params
        }))

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
    r = await self.call('Runtime.evaluate',
                        expression=expr,
                        returnByValue=True,
                        awaitPromise=True)
    return r.get('result', {}).get('value')

  async def strict(self, expr):
    """js, but an exception in the page raises here (js gives None: the waits use that)."""
    r = await self.call('Runtime.evaluate',
                        expression=expr,
                        returnByValue=True,
                        awaitPromise=True)
    if r.get('exceptionDetails'):
      d = r['exceptionDetails']
      raise RuntimeError(
          f"in the page: {d.get('exception', {}).get('description') or d.get('text')}"
      )
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
    if not await self.js(
        f"(e => {{ if (!e) return false; e.scrollIntoView({{block: 'center'}}); return true; }})({expr})"
    ):
      return False
    await asyncio.sleep(0.5)
    box = await self.js(
        f"(e => e ? (r => [r.left + r.width / 2, r.top + r.height / 2])(e.getBoundingClientRect()) "
        f": null)({expr})")
    if not box:
      return False
    for kind in ('mousePressed', 'mouseReleased'):
      await self.call('Input.dispatchMouseEvent',
                      type=kind,
                      x=box[0],
                      y=box[1],
                      button='left',
                      clickCount=1)
    return True

  async def shot(self, path):
    r = await self.call('Page.captureScreenshot', format='png')
    if 'data' in r:
      open(path, 'wb').write(base64.b64decode(r['data']))


@contextlib.asynccontextmanager
async def opened(url, dbg):
  """The page at url in the Chrome listening on DevTools port dbg (served()): yields a CDP on it, navigated."""
  import websockets
  info = None
  for _ in range(100):
    try:
      info = json.load(
          urllib.request.urlopen(f'http://127.0.0.1:{dbg}/json/list'))
      break
    except OSError:
      await asyncio.sleep(0.2)
  ws_url = [t for t in info if t['type'] == 'page'][0]['webSocketDebuggerUrl']
  async with websockets.connect(ws_url, max_size=50_000_000) as ws:
    c = CDP(ws)
    await c.call('Page.enable')
    await c.call('Page.navigate', url=url)
    yield c


async def drive(url, dbg, shots):
  async with opened(url, dbg) as c:
    await asyncio.sleep(3)
    await c.js(DEEP)
    k = await c.until('window.__frame && __frame() !== null && __frame()', 120)
    await c.until(
        "__deep('.gkv-movie')[0] && +(__deep('.gkv-movie')[0].dataset.draws || 0) > 0",
        120)
    await asyncio.sleep(3)
    await c.js(DEEP)
    qs = await c.js('location.search')
    check(
        k == 20 and 'qty=T_elc' in qs and 'frame=20' in qs,
        f'launch keys open frame {k}, T_e; the address '
        f'holds the view ({qs[:120]}...)')
    dark = await c.js(
        "document.body.classList.contains('dark') && getComputedStyle(document.body).backgroundColor"
    )
    check(
        bool(dark) and '0.105882' in str(dark),
        f'dark by default on the theme slate ({dark})')
    await c.js("__switch('dark mode').click()")
    light = await c.until("!document.body.classList.contains('dark')", 15)
    await c.js("__switch('dark mode').click()")
    back = await c.until("document.body.classList.contains('dark')", 15)
    check(
        bool(light) and bool(back),
        'the dark-mode switch goes light and back, live')
    W = "__deep('.gkv-movie')[0]"
    await c.js(
        f"(s => {{ s.value = '30'; s.dispatchEvent(new Event('change')); }})({W}.querySelector('.fps'))"
    )
    # the tab's charts render once scrolled into view (marimo); the movie keeps playing off screen
    await c.js(
        "(__deep('[role=tablist]')[0] || document.body).scrollIntoView({block: 'start'})"
    )
    await c.until(
        "!!window.Plotly && __deep('.gkv-live svg.main-svg').length > 0", 120)
    await asyncio.sleep(2)
    LIVE = "__deep('.gkv-live').map(e => +e.dataset.frame)"
    lv0, nplots = await c.js(LIVE), await c.js(
        "__deep('.gkv-live svg.main-svg').length")
    d0, f0 = await c.js(f'+{W}.dataset.draws'), await c.js('__frame()')
    await c.js(f"{W}.querySelector('button.play').click()")
    t0 = time.time()
    await asyncio.sleep(3)
    d1, f1, lab = await c.js(f'+{W}.dataset.draws'), await c.js(
        '__frame()'), await c.js(f"{W}.querySelector('.time').textContent")
    elapsed = time.time() - t0
    lv1, how = await c.js(LIVE), await c.js(
        "__deep('.gkv-live').map(e => [e.dataset.path, +e.dataset.ms, e.dataset.error || ''])"
    )
    rate = (d1 - d0) / elapsed
    if shots:
      await c.shot(os.path.join(shots, 'playing.png'))
    await c.js(
        MARK_OLD
    )  # the charts drawn so far (the notebook redraws the tab once playback pauses)
    await c.js(f"{W}.querySelector('button.play').click()")
    stop = await c.js(f"{W}.querySelector('.time').textContent")
    frame_of = lambda text: int(re.search(r'frame (\d+)', text).group(
        1)) if text and re.search(r'frame (\d+)', text) else None
    mf, ms = frame_of(lab), frame_of(stop)
    synced = await c.until(f"__frame() === {ms}", 30)
    await asyncio.sleep(2)
    paused, still = await c.js('__frame()'), frame_of(
        await c.js(f"{W}.querySelector('.time').textContent"))
    check(
        bool(lv0) and nplots > 0 and all(b != a for a, b in zip(lv0, lv1))
        and mf is not None and all(abs(b - mf) <= 3 for b in lv1)
        and all(h[0] == 'svg' and not h[2] for h in how),
        f'the open tab\'s {len(lv1 or [])} live charts follow the movie in the page (frames {lv0} -> {lv1}; '
        f'movie at {mf}; path, ms a frame: {[h[:2] for h in how]})')
    frames_played = (mf - f0) if mf is not None and f0 is not None else 0
    check(
        rate > 15 and frames_played >= 0.5 * 30 * elapsed and f1 == f0
        and mf != f0 and bool(synced) and paused == ms == still,
        f'the movie plays in the page beside them ({rate:.0f} draws/s, {frames_played} frames in {elapsed:.1f} s '
        f'at 30 frames/s) with the notebook untouched (header {f0} -> {f1}); on pause at frame {ms} the notebook '
        f'follows (frame {paused}) and the movie stays (frame {still})')
    # the page draws a frame as plotly does: the notebook redraws the tab at the paused frame P, then (a scrub to
    # the movie's last frame) at F: plotly's own drawings; the page then draws P and F itself on the last charts
    # (two frames inside the axes: a frame the axes leave out widens them when shown)
    NEW = DRAWN.replace('NEW', 'true')
    theirs_p = await c.until(
        f"(r => r && r.frames.every(f => f === {paused}) ? r : null)({NEW})",
        120)
    await c.js(MARK_OLD)
    await c.js(
        f"(s => {{ s.value = s.max; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); }})"
        f"({W}.querySelector('.scrub'))")
    theirs_f = await c.until(
        f"(r => r && r.frames.length && r.frames.every(f => f === r.frames[0]) && "
        f"r.frames[0] !== {paused} ? r : null)({NEW})", 120)
    f_scrub = theirs_f and theirs_f['frames'][0]
    await c.until("__deep('.gkv-live').every(e => e.dataset.path)", 30)
    send = lambda f: c.js(
        f"window.dispatchEvent(new CustomEvent('gkv:frame', {{detail: {{frame: {f}}}}}))"
    )
    await send(paused)
    await asyncio.sleep(0.25)
    mine_p = await c.js(NEW)
    await send(f_scrub)
    await asyncio.sleep(0.25)
    mine_f = await c.js(NEW)
    same_p, same_f, moved = drawn_diff(mine_p, theirs_p), drawn_diff(
        mine_f, theirs_f), drawn_diff(mine_p, theirs_f)
    r3 = lambda v: v if v is None else round(v, 3)
    check(
        None not in (same_p, same_f, moved) and same_p < 0.05 and same_f < 0.05
        and moved > 1 and mine_p['frames'] == [paused] * len(mine_p['frames']),
        f'the page draws a frame as plotly does (frames {paused} and {f_scrub}: every curve vertex, marker and '
        f'line within {r3(same_p)} / {r3(same_f)} px; {mine_p and mine_p["n"]} curves; the two frames '
        f'{r3(moved)} px apart)')
    # a click on the strips (first strip, its leftmost column: the strip's last cell, reflected; top SOL row) probes it
    # scrolled into view first: a click outside the viewport (the page scrolled past the strips) reaches nothing
    box = await c.js(
        f"(e => {{ e.scrollIntoView({{block: 'center'}}); const r = e.getBoundingClientRect(); "
        f"return [r.left, r.top, r.width]; }})({W}.querySelector('canvas.strips'))"
    )
    x, y = box[0] + 82 + 2, box[1] + 42 + 3
    for kind in ('mousePressed', 'mouseReleased'):
      await c.call('Input.dispatchMouseEvent',
                   type=kind,
                   x=x,
                   y=y,
                   button='left',
                   clickCount=1)
    probe = await c.until("new URLSearchParams(location.search).get('probe')",
                          20)
    check(bool(probe),
          f'a click on the movie probes a cell (block,i,j = {probe})')
    # the same cell again lets it go; so does Esc; then probe it once more (the reload check below wants one)
    boxes = []

    async def click(
    ):  # where the strips are now: a re-run may have moved or re-made the movie
      b = await c.js(
          f"(e => {{ e.scrollIntoView({{block: 'center'}}); const r = e.getBoundingClientRect(); "
          f"return [r.left, r.top]; }})({W}.querySelector('canvas.strips'))")
      boxes.append([round(v) for v in b])
      for kind in ('mousePressed', 'mouseReleased'):
        await c.call('Input.dispatchMouseEvent',
                     type=kind,
                     x=b[0] + 82 + 2,
                     y=b[1] + 42 + 3,
                     button='left',
                     clickCount=1)

    none = "!new URLSearchParams(location.search).get('probe')"
    await click()
    again = await c.until(none, 20)
    await click()
    await c.until("new URLSearchParams(location.search).get('probe')", 20)
    await c.js(
        f"{W}.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Escape', bubbles: true}}))"
    )
    esc = await c.until(none, 20)
    await click()
    back = await c.until("new URLSearchParams(location.search).get('probe')",
                         20)
    check(
        bool(again) and bool(esc) and back == probe,
        f'the probed cell lets go on a second click and on Esc, and comes back on a third click (second click: '
        f'{bool(again)}, Esc: {bool(esc)}, third click: {back} for {probe}; the strips at {box[:2]}, then {boxes})'
    )
    # an option changed re-runs the tabs' cell (the look: dark mode off, then on): the open tab stays. It used to
    # fall back to the first: the tabs element had no name, marimo dropped it, and a click never reached the
    # notebook. The tab: Balance & power, whose live charts each re-run draws anew (proof that one happened); the
    # conservation chart among them (its log y axis)
    BAR = "__deep('[role=tablist]').filter(b => b.textContent.includes('Midplane profiles'))"
    OPEN = (
        f"(t => t ? t.textContent.trim() : null)({BAR}.flatMap(b => [...b.querySelectorAll('[role=tab]')])"
        ".find(t => t.getAttribute('aria-selected') === 'true' || t.dataset.state === 'active'))"
    )
    FRESH = (
        "(p => p.length >= 2 && p.every(d => !d.__old && d.querySelector('svg.main-svg')))"
        "(__deep('.gkv-live .js-plotly-plot'))")
    await c.press("__tab('Balance & power')")
    await c.until(f"{OPEN} === 'Balance & power' && {FRESH}", 120)
    kept = []
    for _ in range(2):
      await c.js(MARK_OLD)
      await c.js("__switch('dark mode').click()")
      redrawn = await c.until(FRESH, 120)
      await asyncio.sleep(1)
      kept.append((bool(redrawn), await c.js(OPEN)))
    axes = await c.js(
        "__deep('.gkv-live .js-plotly-plot').map(d => (d.layout && d.layout.yaxis || {}).type || '')"
    )
    check(
        kept == [(True, 'Balance & power')] * 2
        and 'tab=Balance' in await c.js('location.search')
        and 'log' in (axes or []),
        f'an option changed (dark mode off, then on) redraws the tab\'s charts and the open tab stays: {kept}; '
        f'its charts\' y axes {axes} (the conservation chart: log)')
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
    check(
        bool(new),
        f'Save figure writes {", ".join(os.path.basename(p) for p in new or [])}'
    )
    for p in new or []:
      os.remove(p)  # this check's own file
    # a reload comes back to the view in the address
    here = await c.js('location.search')
    await c.call('Page.reload')
    await asyncio.sleep(3)
    await c.js(DEEP)
    k2 = await c.until('window.__frame && __frame() !== null && __frame()', 120)
    check(
        k2 == f_scrub and 'probe=' in here and 'tab=Save' in here,
        f'a reload comes back to the view in the address (frame {k2}, the probe, the open tab)'
    )
    if shots:
      await c.shot(os.path.join(shots, 'reloaded.png'))


@contextlib.contextmanager
def served(chrome, args, env=None):
  """./view.sh args (OPEN=0, a port of its own; env: more variables for it) and a headless Chrome to drive the page:
  yields (url, the DevTools port), url None when the server printed none in 60 s; both stopped by process group
  afterwards, whatever happens."""
  port, dbg = str(27300 + os.getpid() % 200), str(9500 + os.getpid() % 400)
  profile = tempfile.mkdtemp(prefix='check_viewer_chrome_')
  log = tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False)
  srv = subprocess.Popen(['./view.sh'] + args,
                         cwd=HERE,
                         env=dict(os.environ,
                                  OPEN='0',
                                  PORT=port,
                                  **(env or {})),
                         stdout=log,
                         stderr=subprocess.STDOUT,
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
    if url is not None:
      chr_ = subprocess.Popen([
          chrome, '--headless=new', '--no-sandbox', '--disable-gpu',
          f'--remote-debugging-port={dbg}', '--window-size=1680,1400',
          f'--user-data-dir={profile}', 'about:blank'
      ],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              start_new_session=True)
    yield url, dbg
  finally:
    for p in (chr_, srv):
      if p is not None:
        try:
          os.killpg(p.pid, signal.SIGTERM)
          p.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
          pass


def part_browser(chrome, shots):
  with served(chrome,
              ['--sim', 'Sim 1', '--frame', '20', '--qty', 'T_elc']) as (url,
                                                                         dbg):
    check(url is not None, 'view.sh serves the page')
    if url is not None:
      asyncio.run(drive(url, dbg, shots))


# ---------------------------------------------------------------------------------------------------- LivePlot alone
async def drive_liveplot(page, dbg, n_ev=60):
  import websockets
  info = None
  for _ in range(100):
    try:
      info = json.load(
          urllib.request.urlopen(f'http://127.0.0.1:{dbg}/json/list'))
      break
    except OSError:
      await asyncio.sleep(0.2)
  ws_url = [t for t in info if t['type'] == 'page'][0]['webSocketDebuggerUrl']
  async with websockets.connect(ws_url, max_size=50_000_000) as ws:
    c = CDP(ws)
    await c.call('Page.enable')
    await c.call('Page.navigate', url='file://' + page)
    ready = await c.until('window.__ready', 60)
    send = lambda f: c.js(
        f"window.dispatchEvent(new CustomEvent('gkv:frame', {{detail: {{frame: {f}}}}}))"
    )
    L = await c.js('__L')
    op = lambda: c.js(
        "+getComputedStyle(document.getElementById('a').querySelector("
        "`.shapelayer path[data-index='${__lam}']`)).opacity")
    hole, K = L['hole'], L['K']
    drawn = lambda f: c.until(
        f"document.getElementById('a').dataset.frame === '{f}'", 10)
    await send(hole)
    await drawn(
        hole)  # the chart draws on the page's next frame: a busy page delays it
    svg_hidden = await op()
    await send(K)
    await drawn(K)
    svg_shown = await op()
    await send(hole)
    await asyncio.sleep(
        1.5)  # stopped there: plotly's own data and drawing (settle)
    plotly_hidden = await op()
    paths = await c.js(
        "['a', 'b'].map(i => document.getElementById(i).dataset.path)")
    out = [f for f, o in zip(L['frames'], L['over']) if o]
    note = lambda: c.js(
        "document.getElementById('a').querySelector('.gkv-live-note').textContent"
    )
    notes = []
    for f in (out[:1] + [K]):
      await send(f)
      await drawn(f)
      notes.append(await note())
    check(
        len(out) > 0 and K not in out and bool(notes[0]) and not notes[-1],
        f'a frame past the y axes says so under the chart (frames {out} of Sim 2\'s heat burst, shown at frame '
        f'{K}: "{notes[0][:60]}..."), the frame shown fits')
    check(
        bool(ready) and paths == ['svg', 'plotly'] and svg_hidden == 0
        and svg_shown == 1 and plotly_hidden == 0,
        f'a lambda_q line hides where lambda_q is undefined (frame {hole}: opacity {svg_hidden}, frame {K}: '
        f'{svg_shown}; after settling there, plotly\'s own drawing: {plotly_hidden})'
    )
    # the fallback (chart b): a burst of frames at 30 per second, the page's frame clock counted meanwhile
    ticks = await c.js(
        f"""(async () => {{ let n = 0, on = true; const tick = () => {{ n++; if (on) requestAnimationFrame(tick); }};
          requestAnimationFrame(tick); const t0 = performance.now();
          for (let i = 0; i < {n_ev}; i++) {{ window.dispatchEvent(new CustomEvent('gkv:frame', {{detail: {{frame: __L.frames[i % __L.frames.length]}}}}));
            await new Promise(r => setTimeout(r, 33)); }}
          on = false; return [n, performance.now() - t0]; }})()""")
    await asyncio.sleep(1.5)
    b = await c.js(
        "(e => [+e.dataset.frame, +e.dataset.updates, +e.dataset.ms, e.dataset.error || ''])(document.getElementById('b'))"
    )
    want = L['frames'][(n_ev - 1) % len(L['frames'])]
    fps = ticks[0] / ticks[1] * 1e3 if ticks else 0
    check(
        b[0] == want and b[1] >= 3 and not b[3] and fps >= 30,
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
  run = gv.open_run(os.path.join(gv.BASE, d, 'runs', dflt), lab)
  summ = dict(gv.summary(run))
  frames = [int(f) for f in summ['frames']]
  K, hole, tg = frames[-1], frames[len(frames) // 2], run.layout.targets[0]
  lam = np.array(summ[f'lam|{tg.name}'], dtype=float)
  lam[frames.index(hole)] = np.nan
  summ[f'lam|{tg.name}'] = lam
  w = mm.live_plot(vp.targets_figure(run, gv.frame(run, K), summ, run.time(K)),
                   summ, K)
  spline = json.loads(json.dumps(w.fig))
  for t in w.live['traces']:
    spline['data'][t['index']].setdefault('line', {})['shape'] = 'spline'
  i_lam = [
      i for i, sh in enumerate(w.fig['layout']['shapes'])
      if sh.get('name') == f'lam|{tg.name}'
  ][0]
  tmp = tempfile.mkdtemp(prefix='check_viewer_liveplot_')
  page = os.path.join(tmp, 'liveplot.html')
  with open(page, 'w') as f:
    f.write(
        f"""<!doctype html><meta charset="utf-8"><body style="width: 1400px"><div id="a"></div><div id="b"></div>
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
  chr_ = subprocess.Popen([
      chrome, '--headless=new', '--no-sandbox', '--disable-gpu',
      f'--remote-debugging-port={dbg}', '--window-size=1500,1400',
      f'--user-data-dir={tempfile.mkdtemp(prefix="check_viewer_chrome_")}',
      '--allow-file-access-from-files', 'about:blank'
  ],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL,
                          start_new_session=True)
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
  ap.add_argument('--browser',
                  help='a Chrome / Chromium binary: also drive the page in it')
  ap.add_argument('--shots',
                  help='a directory for screenshots of the browser check')
  ap.add_argument(
      '--parts',
      help=
      'only these parts, comma-separated (e.g. blank,poloidal; default: every part)'
  )
  a = ap.parse_args()
  signal.signal(signal.SIGTERM, lambda *_:
                (_ for _ in ()).throw(KeyboardInterrupt()))
  parts = {
      'headless': part_headless,
      'thomson': part_thomson,
      'measured': part_measured,
      'exports': part_exports,
      'labels': part_labels,
      'pgkyl': part_pgkyl,
      'themes': part_themes,
      'fonts': part_fonts,
      'sync': part_sync,
      'machine': part_machine,
      'poloidal': lambda: part_poloidal(a.browser),
      'lines': lambda: part_lines(a.browser, a.shots),
      'grid': lambda: part_grid(a.browser, a.shots),
      'blank': lambda: part_blank(a.browser, a.shots),
      'species': lambda: part_species(a.browser, a.shots),
      'browser': lambda: a.browser and part_browser(a.browser, a.shots),
      'layout': lambda: a.browser and part_layout(a.browser, a.shots),
      'liveplot': lambda: a.browser and part_liveplot(a.browser),
  }
  chosen = a.parts.split(',') if a.parts else list(parts)
  unknown = [p for p in chosen if p not in parts]
  if unknown:
    ap.error(f'no part {unknown}: the parts are {", ".join(parts)}')
  for p in chosen:
    parts[p]()
  print(f'\n{len(fails)} failed' if fails else '\nall passed')
  sys.exit(len(fails))
