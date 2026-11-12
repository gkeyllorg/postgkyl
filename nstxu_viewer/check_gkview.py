#!/usr/bin/env python3
"""Checks gkview.py against the tools that made the posters and movies (run with the postgkyl venv: it has scipy,
which poster_progress imports):

  tools/postgkyl/.venv/bin/python check_gkview.py

1. derived strips == movie_density's (hand-written for Sim 1 and the 2025 re-run, half_layout for Sim 2);
   reflecting midplane found <=> the files say half_domain = 1;
2. q_par and lambda_q at every target == poster_progress.heat_flux (Sim 1: both outer targets, Sim 2: lower outer), at
   the posters' frames and frame 1;
3. Sim 2 outboard midplane profile == the poster's (mirror-adjacent theta column at the Z = 0 radii);
4. temperatures == m (M2 - M1^2/M0) / (3 M0) computed directly from the files;
5. read_dynvec == postgkyl for dt and the integrated moments.
Prints one line per check and exits non-zero on any failure.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gkview as gv  # noqa: E402

SIMS = os.path.join(gv.NSTXU, 'sims_20261022')
sys.path.insert(0, SIMS)
import movie_density as md  # noqa: E402
import poster_progress as pp  # noqa: E402
import postgkyl as pg  # noqa: E402

fails = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what)
    if not ok:
        fails.append(what)


def strips_of(layout):
    return [[(b, x0, reg) for b, x0, reg in s.items] for s in layout.strips]


runs = {
    'sim1_prod': gv.open_run(os.path.join(SIMS, 'sim1_full_drifts/runs/prod_D1_coll')),
    'sim2_prod': gv.open_run(os.path.join(SIMS, 'sim2_dense_rho09/runs/prod_th02q4_coll')),
    'prod': gv.open_run(os.path.join(gv.NSTXU, 'sim203655_20261012/runs/prod')),
}

# 1. layout
for key, run in runs.items():
    cfg = md.resolve_layout(md.CONFIGS[key], run.path)
    want = [sorted(items) for _, items, _ in cfg['strips']]
    got = [sorted(s) for s in strips_of(run.layout)]
    check(want == got, f'{key}: strips {got} == movie_density')
    mx = [s.mirror_x for s in run.layout.strips]
    check(mx == [cfg['midplane']] * len(mx), f'{key}: reflecting midplane at {mx} == movie_density {cfg["midplane"]}')
    check(run.layout.half == (run.half_meta == 1), f'{key}: half domain {run.layout.half}, file half_domain {run.half_meta}')
    print('     ', key, 'regions', run.layout.region)
    print('     ', key, 'targets', [(t.name, t.leg, t.j, t.mid) for t in run.layout.targets])
    print('     ', key, 'stacks', run.layout.stacks)
    print('     ', key, 'segments', [s.segments for s in run.layout.strips])

# 2. heat flux and lambda_q
for key, run in runs.items():
    post = pp.POST[key]
    frames = run.frames()
    for k in sorted({1, frames[-1] if post['frame'] is None else post['frame']}):
        fr = gv.frame(run, k)
        for lab, leg, j, _ in post['targets']:
            tgt = next(t for t in run.layout.targets if t.name == lab)
            check((tgt.leg, tgt.j % run.nodes[leg].shape[1], tgt.mid) == (leg, j % run.nodes[leg].shape[1], post['mid']),
                  f'{key}: {lab} = leg b{tgt.leg} plate j {tgt.j}, midplane b{tgt.mid} (poster: b{leg} j {j} b{post["mid"]})')
            edges = md.mid_edges(run.nodes[post['mid']])
            q1, x1, lam1 = gv.target_heat_flux(run, fr, tgt)
            # the poster script's own masses (CODATA m_e, 2.014 amu), then the masses the run used (file metadata)
            _, _, lam_pp = pp.heat_flux(run.path, run.stem, k, leg, j, edges)
            me_pp, mi_pp = pp.ME, pp.MI
            pp.ME, pp.MI = run.mass['elc'], run.mass['ion']
            q0, x0, lam0 = pp.heat_flux(run.path, run.stem, k, leg, j, edges)
            pp.ME, pp.MI = me_pp, mi_pp
            err = max(np.abs(q1 - q0).max() / np.abs(q0).max(), np.abs(x1 - x0).max() / x0.max(), abs(lam1 - lam0) / lam0)
            check(err < 1e-12, f'{key} frame {k}: {lab} lambda_q {lam1:.4f} mm == poster formula with the run\'s masses '
                  f'(max rel diff {err:.1e}; poster masses give {lam_pp:.4f})')
    print(f'      {key} masses: elc {run.mass["elc"]:.9e} (poster {pp.ME:.9e}), ion {run.mass["ion"]:.9e} '
          f'(poster {pp.MI:.9e}, ratio {run.mass["ion"] / pp.MI:.6f})')

# 3. Sim 2 midplane profile
run = runs['sim2_prod']
k = run.frames()[-1]
fr = gv.frame(run, k)
cols = md.CONFIGS['sim2']['profile']
R0 = np.concatenate([0.5 * (md.mid_edges(run.nodes[b])[1:] + md.mid_edges(run.nodes[b])[:-1]) for b, _ in cols])
n0 = np.concatenate([md.cav(f'{run.path}/{run.stem}_b{b}-elc_M0_{k}.gkyl')[:, j] for b, j in cols])
R1, n1, rsep = gv.midplane_profile(run, fr, 'outboard', 'n_elc')
check(np.array_equal(R0, R1) and np.array_equal(n0, n1), f'sim2_prod frame {k}: outboard midplane n_e == poster profile '
      f'({len(R1)} rows, R_sep {rsep:.4f} m)')

# 4. temperature
for key in ('sim1_prod', 'sim2_prod'):
    run = runs[key]
    k = run.frames()[-1]
    fr = gv.frame(run, k)
    worst = 0.0
    for b in range(run.nb):
        for s in run.species:
            c = {m: md.cav(f'{run.path}/{run.stem}_b{b}-{s}_{m}_{k}.gkyl') for m in ('M0', 'M1', 'M2')}
            T = run.mass[s] * (c['M2'] - c['M1'] ** 2 / c['M0']) / (3 * c['M0']) / run.e
            ok = c['M0'] > 0
            worst = max(worst, np.abs(fr[b][f'T_{s}'][ok] / T[ok] - 1).max())
    check(worst < 1e-12, f'{key} frame {k}: T from gkview == direct formula (max rel diff {worst:.1e})')

# 5. dynvectors
for key in ('sim1_prod', 'sim2_prod'):
    run = runs[key]
    t, dt = gv.dt_history(run)
    d = pg.GData(f'{run.path}/{run.stem}-dt.gkyl')
    tp, vp = np.asarray(d.get_grid()[0]).ravel(), np.asarray(d.get_values()).ravel()
    check(len(tp) == len(t) and np.array_equal(tp, t) and np.array_equal(vp, dt), f'{key}: dt history ({len(t)} samples) == postgkyl')
    p = run.f(0, 'elc_integrated_moms')
    t2, v2 = gv.read_dynvec(p)
    d = pg.GData(p)
    check(np.array_equal(np.asarray(d.get_values()), v2), f'{key}: b0 elc integrated moments {v2.shape} == postgkyl')

# 6. the per-frame cache the movie draws from: every cell of every quantity == the frame read directly (float32)
for key in ('sim1_prod', 'sim2_prod'):
    run = runs[key]
    summ = gv.summary(run)
    frames = run.frames()
    worst, nq = 0.0, 0
    for k in (frames[0], frames[len(frames) // 2], frames[-1]):
        fr = gv.frame(run, k)
        i = list(summ['frames']).index(k)
        for qty in gv.quantities(run):
            direct = np.concatenate([fr[b][qty].ravel() for b in range(run.nb)]).astype(np.float32)
            same = np.array_equal(direct, summ[f'fld|{qty}'][i], equal_nan=True)
            worst = max(worst, 0.0 if same else 1.0); nq += 1
    check(worst == 0.0, f'{key}: cached fields == direct read ({nq} quantity-frames, frames first/middle/last)')

# 7. frame ranges: the semantics of postgkyl.io.naming.select_frames (its documented examples)
_f = list(range(31))
for _spec, _want in ((':', _f), ('::2', _f[::2]), ('-10:', _f[-10:]), ('-1', [30]), ('7', [7]), ('0,2', [0, 2]),
                     ('10:20', list(range(10, 20))), ('5::5', [5, 10, 15, 20, 25, 30])):
    check(gv.select_frames(_spec, _f) == _want, f'select_frames({_spec!r}) == {_want if len(_want) < 6 else "..."}')
for _bad in ('1:2:3:4', 'x', '::0', '-40'):
    try:
        gv.select_frames(_bad, _f)
        check(False, f'select_frames({_bad!r}) refused')
    except ValueError:
        check(True, f'select_frames({_bad!r}) refused')

# 8. cell volumes (the Jacobian-weighted region means): sum(n V) == the code's integrated M0, every frame
for key in ('sim1_prod', 'sim2_prod'):
    run = runs[key]
    summ = gv.summary(run)
    V = gv.cell_volumes(run)
    integ = gv.integrated(run)
    worst = 0.0
    for s in run.species:
        ti, d = integ[s]
        ratio = (summ[f'fld|n_{s}'].astype(float) * V).sum(1) / np.interp(summ['t'], ti, d[:, 0])
        worst = max(worst, np.abs(ratio - 1).max())
    check((V > 0).all() and worst < 5e-3, f'{key}: cell volumes > 0; sum(n V) / integrated M0 within {worst:.1e} '
          '(cell averages of n and J: not exact)')

# 9. balances: particles close in Sim 1 (every channel crosses a boundary-flux face; interior faces cancel); the
#    plate faces are physical theta faces of the leg and PF blocks
run = runs['sim1_prod']
B = gv.balance(run)
for s, d in B['species'].items():
    res = (d['src'] - d['out'] - d['dot'])[-1] / d['src'][-1]
    check(abs(res) < 1e-3, f'sim1_prod: {s} particle balance closes at the last sample ({res * 100:+.3f} % of the source)')
topo = gv.read_gkyl(f'{run.path}/{run.stem}-block_topo.gkyl')['meta']
conn = np.array(topo['connections']).reshape(run.nb, 2, 2, 3)
plates = all(conn[b, 1, 0 if t.j == 0 else 1, 2] == gv.PHYSICAL for t in run.layout.targets for b in (t.leg, t.pf))
check(plates, 'sim1_prod: every target plate face (leg and PF block) is a physical theta face')


# 10. the layout of other devices' grids (the device-agnostic driver's geometry-only runs: topology and nodes only):
#     STEP's psi index grows inward (read inside out), the single nulls' main SOL block crosses Z = 0 twice
import glob  # noqa: E402
import tempfile  # noqa: E402
TOK = os.path.join(os.path.dirname(gv.NSTXU), 'tokamak', 'tok_multib_inp_20261103', 'runs')


def layout_of(path):
    topo = glob.glob(f'{path}/*-block_topo.gkyl')[0]
    stem = os.path.basename(topo)[:-len('-block_topo.gkyl')]
    tm = gv.read_gkyl(topo)['meta']
    nb = int(tm['num_blocks'])
    conn = np.array(tm['connections']).reshape(nb, 2, 2, 3)
    nodes = {b: np.asarray(gv.read_gkyl(f'{path}/{stem}_b{b}-geo_corn_nodes.gkyl')['data'])[..., :2] for b in range(nb)}
    flip = not gv.psi_grows_outward(conn, nodes)
    if flip:
        conn, nodes = gv.flip_conn(conn), {b: nd[::-1] for b, nd in nodes.items()}
    return gv.derive_layout(conn, nodes), flip, nodes


for name, want_flip, want_targets in (('final_20261103/step_full', True, gv.TARGET_ORDER),
                                      ('final_20261103/asdex_lsn', False, ['lower outer target', 'lower inner target']),
                                      ('lsn_20261103/tcv', True, ['lower outer target', 'lower inner target'])):
    path = os.path.join(TOK, name)
    if not os.path.isdir(path):
        print(f'SKIP {name}: no such geometry run')
        continue
    L, flip, nodes = layout_of(path)
    st = L.stacks
    ok = (flip == want_flip and [t.name for t in L.targets] == want_targets and set(st) == {'outboard', 'inboard'}
          and all(s.core is not None and L.region[s.core] == 'core' for s in st.values())
          and all(L.region[b] in ('sol', 'core', 'pf') for b in L.region)
          and sorted(b for s in L.strips for b, _, _ in s.items) == list(range(len(L.region))))
    if len(L.strips) == 1:   # a single null: one main SOL block crossing Z = 0 on both sides, at two places
        ok &= st['outboard'].sol == st['inboard'].sol and st['outboard'].j is not None and \
            abs(st['outboard'].j - st['inboard'].j) > 1
        ok &= gv.mid_edges(nodes[st['outboard'].sol], st['outboard'].j)[0] > gv.mid_edges(nodes[st['inboard'].sol],
                                                                                          st['inboard'].j)[0]
    check(ok, f'{name}: psi inward {flip}, targets {[t.name for t in L.targets]}, stacks '
              f'{ {k: (v.core, v.sol, None if v.j is None else round(v.j, 1)) for k, v in st.items()} }, every block in a strip')

# 11. a single-block run (the gyrokinetic app's own file names, no topology): Sim 2's block 2 alone, linked into a
#     temporary directory under single-block names. It opens, its layout has the outboard midplane (no core under it)
#     and the plate, its fields equal block 2's, its summary builds (in a temporary cache)
sim2 = runs['sim2_prod']
with tempfile.TemporaryDirectory() as tmp:
    sb = os.path.join(tmp, 'single_b2')
    os.makedirs(sb)
    for f in glob.glob(f'{sim2.path}/{sim2.stem}_b2-*.gkyl') + [f'{sim2.path}/{sim2.stem}-dt.gkyl']:
        os.symlink(f, os.path.join(sb, os.path.basename(f).replace('_b2-', '-')))
    r1 = gv.open_run(sb)
    k = r1.frames()[-1]
    same = all(np.array_equal(gv.frame(r1, k)[0][q], gv.frame(sim2, k)[2][q], equal_nan=True) for q in gv.quantities(r1))
    saved, gv.CACHE_DIR = gv.CACHE_DIR, os.path.join(tmp, 'cache')
    try:
        S1 = gv.summary(r1)
    finally:
        gv.CACHE_DIR = saved
    check(r1.single and r1.nb == 1 and list(r1.layout.stacks) == ['outboard'] and r1.layout.stacks['outboard'].core is None
          and len(r1.layout.targets) == 1 and same and S1['fld|n_elc'].shape == (len(r1.frames()), r1.nodes[0][..., 0][:-1, :-1].size),
          f'a single-block run opens (layout {r1.layout.region}, stacks {list(r1.layout.stacks)}, targets '
          f'{[t.name for t in r1.layout.targets]}), its fields == Sim 2 block 2 at frame {k}, its summary builds')

# 12. the diffusivity: Sim 2's anom_diff files divided by g^xx J give one value per node (cells sharing a node agree),
#     constant along each psi row (the driver's D is a function of psi only), inside the range the driver printed;
#     the edge-diffusion run's 10 m^2/s on the core inner rows; the uniform runs' D from their options / log
D = gv.diffusivity(sim2)
nodes_D = D['species']['elc']['nodes']
rows_ok = max(float(np.max(np.ptp(v, axis=1) / np.abs(v).max())) for v in nodes_D.values())
lo, hi = min(float(v.min()) for v in nodes_D.values()), max(float(v.max()) for v in nodes_D.values())
check(D['kind'] == 'file' and D['same'] and D['gap'] < 1e-12 and rows_ok < 1e-12 and 0.487 - 5e-4 <= lo and hi <= 0.9938,
      f'sim2_prod: D from anom_diff, node agreement {D["gap"]:.1e}, along-row spread {rows_ok:.1e}, {lo:.4g}..{hi:.4g} '
      'm^2/s within the driver\'s printed 0.487..0.9938')
e10 = gv.open_run(os.path.join(SIMS, 'sim1_full_drifts/runs/ab_edgeD10_coll'))
De = gv.diffusivity(e10)
core = [b for b in range(e10.nb) if e10.layout.region[b] == 'core']
check(De['kind'] == 'file' and all(np.allclose(De['species']['elc']['nodes'][b][:3], 10.0, rtol=1e-12) for b in core),
      'ab_edgeD10_coll: D = 10 m^2/s on the core blocks\' first node rows (D_edge=10, D_edge_rows = 2)')
for key, run_, want in (('sim1_prod', runs['sim1_prod'], 1.0), ('prod', runs['prod'], 0.22)):
    Du = gv.diffusivity(run_)
    val = float(next(iter(Du['species']['elc']['nodes'].values())).flat[0]) if Du['kind'] else None
    check(Du['kind'] == 'uniform' and val == want, f'{key}: a uniform D = {val} m^2/s, from {Du["source"][:90]}')

# 13. the midplane D, node rows inside out and the shared separatrix row once
Rm, Dm, Bm = gv.midplane_D(sim2, D['species']['elc'], 'outboard')
st = sim2.layout.stacks['outboard']
check(np.all(np.diff(Rm) > 0) and len(Rm) == sim2.nodes[st.core].shape[0] + sim2.nodes[st.sol].shape[0] - 1,
      f'sim2_prod: the outboard midplane D on {len(Rm)} node rows, R increasing outward')

# 14. the overlay's time matching: the same times match one to one; a frame of a outside b's span (beyond half a frame
#     spacing) matches nothing
t = runs['sim2_prod'].time
T2 = np.array([t(k) for k in runs['sim2_prod'].frames()])
idx_same = gv.match_frames(T2, T2)
idx_short = gv.match_frames(T2, T2[:10])
dt_ = np.median(np.diff(T2))
idx_shift = gv.match_frames(T2 + 0.3 * dt_, T2)
check(np.array_equal(idx_same, np.arange(len(T2))) and np.array_equal(idx_short[:10], np.arange(10)) and
      np.all(idx_short[11:] == -1) and np.array_equal(idx_shift[:-1], np.arange(len(T2) - 1)),
      'match_frames: identity on the same times, -1 past the overlay\'s end, nearest within half a spacing')
S2 = gv.summary(sim2)
C = gv.Compared(S2, S2, idx_short)
qk = 'mid|outboard|n_elc'
check(np.array_equal(C[f'cmp|{qk}'][:10], S2[qk][:10]) and np.isnan(C[f'cmp|{qk}'][11:]).all() and C['t'] is S2['t'],
      'Compared: the overlay\'s rows at the matched frames, NaN where none, the first run\'s entries as they are')

# 15. the cache predates these changes (CACHE_VERSION unchanged): a frame recomputed now equals its cached row
k = runs['sim2_prod'].frames()[-1]
row = list(S2['frames']).index(k)
fresh = gv._summary_one(sim2, k, gv.frame(sim2, sim2.frames()[0]))
bad = [key for key in fresh if not np.array_equal(np.asarray(fresh[key], dtype=S2[key].dtype), S2[key][row], equal_nan=True)]
check(not bad, f'sim2_prod: frame {k} recomputed == its cached row in every entry ({len(fresh)} entries){bad[:5]}')

# 16. the options: every option the driver reads (its source next to the binary), the run's own values among them
import run_options as ro  # noqa: E402
v = ro.options_view(sim2)
launch = gv.launch_options(sim2)
got = {r['key']: r['value'] for r in v['rows'] if r['value']}
check(v['sources'] and len(v['rows']) >= 60 and got == {k_: v_ for k_, v_ in launch.items() if k_ in got} and
      set(launch) == set(got) | set(v['unknown']) and ro.jobs(sim2) and
      any('Anomalous diffusion' in line for line in ro.startup_report(sim2)[1]),
      f'sim2_prod options: {len(v["rows"])} read by {[os.path.basename(p) for p in v["sources"]]}, the {len(got)} set at '
      f'launch among them ({len(v["unknown"])} unknown), {len(ro.jobs(sim2))} job(s), the start-up report')

print(f'\n{len(fails)} failed' if fails else '\nall passed')
sys.exit(1 if fails else 0)
