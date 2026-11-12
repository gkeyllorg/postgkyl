"""The pgkyl command (postgkyl 2's command line) equivalent to what the viewer draws: click a chart, the movie or a
figure on the page to see one, ready to paste into a shell.

Each command cds into the run and runs pgkyl (viewer.json's "pgkyl", else the one on PATH) on the files the viewer
read, computing the numbers it shows:
- the viewer shows cell averages: component 0 of a p = 1 modal file over 2 (the 2D orthonormal basis' 1/2). Each
  file is loaded under a tag saying what it holds, `select -c 0` takes that stored component (pgkyl does no
  arithmetic on modal coefficients), and `evaluate` (pgkyl's reverse-Polish calculator) does the arithmetic of
  gkview._block_fields in the page's unit (checked: density bitwise, the rest to rounding). evaluate names its inputs
  by position, f0, f1, ... in load order, and replaces the whole working set with its result;
- one curve per block (a midplane across core and SOL blocks): each block's files are loaded and `activate`d, so
  evaluate sees only them while the earlier blocks' results wait aside; `activate` brings them back, and `plot
  --multiblock` draws them on one figure;
- every frame (the Probe tab): each file is loaded as a glob over its frames and `collect`ed into a time series on
  its own (collect stacks one quantity), then evaluated;
- pgkyl draws a block on its computational grid (psi, theta): the viewer's strips, block by block, not the poloidal
  plane (pgkyl 1 could not map these files' 12-component (R, Z, phi) geometry; pgkyl 2's mapping is untried here);
- a midplane profile takes each block's theta cell column nearest Z = 0 (the viewer interpolates each psi row to
  Z = 0 between two cells), against psi (the viewer: distance from the separatrix);
- the integrated diagnostics are per radian, as the files hold them; a time trace's component is `select -c` too.
Where a command depends on the frame it holds '{frame}' (the overlay's: '{cmp_frame}'), which the page fills in.
Lines starting with '#' say where a command differs from the chart."""
import os
import re
import shlex

import numpy as np

import gkview as gv

_EXE = gv.data_path('pgkyl')   # viewer.json: a pgkyl executable (e.g. a venv's), else pgkyl from PATH
PGKYL = _EXE if _EXE and os.path.exists(_EXE) else 'pgkyl'
SP = {'elc': 'e', 'ion': 'i'}   # tag prefixes (no tag starts with f: evaluate's own names are f0, f1, ...)
NO_DISPLAY = 'no display? add --no_show --saveas <file>.png to plot'


def q(s):
    return shlex.quote(str(s))


def _sp(s):
    return SP.get(s, re.sub(r'^f', 'sp_f', s))


def recipe(run, qty, suffix=''):
    """(needs, rpn, label): the files' c0 components a quantity needs [(tag, what)], the ev expression of its cell
    averages in the page's unit, its label. suffix: appended to every tag (a second frame)."""
    label, sym, unit, scale, kind = gv.quantities(run)[qty]

    def T(s, what='M2'):
        p, C = _sp(s), run.mass[s] / run.e
        t = lambda m: f'{p}{m}{suffix}'  # noqa: E731
        if what == 'M2':
            return [t('M0'), t('M1'), t('M2')], f'{t("M2")} {t("M1")} {t("M1")} * {t("M0")} / - {t("M0")} / 3 / {C!r} *'
        if what == 'M2par':
            return [t('M0'), t('M1'), t('M2par')], f'{t("M2par")} {t("M1")} {t("M1")} * {t("M0")} / - {t("M0")} / {C!r} *'
        return [t('M0'), t('M2perp')], f'{t("M2perp")} {t("M0")} / 2 / {C!r} *'

    m = re.match(r'^(n|T|Tpar|Tperp|upar)_(\w+)$', qty)
    if m:
        base, s = m.groups()
        p = _sp(s)
        if base == 'n':
            tags, e = [f'{p}M0{suffix}'], f'{p}M0{suffix} 2 /'
        elif base == 'upar':
            tags, e = [f'{p}M0{suffix}', f'{p}M1{suffix}'], f'{p}M1{suffix} {p}M0{suffix} /'
        else:
            tags, e = T(s, {'T': 'M2', 'Tpar': 'M2par', 'Tperp': 'M2perp'}[base])
    elif qty == 'phi':
        tags, e = [f'phi{suffix}'], f'phi{suffix} 2 /'
    elif qty == 'mach':
        (te, Te), (ti, Ti) = T('elc'), T('ion')
        tags = te + [x for x in ti if x not in te]
        e = f'iM1{suffix} iM0{suffix} / {Te} {Ti} + {run.e / run.mass["ion"]!r} * sqrt /'
    elif qty in ('q_par', 'q_perp'):
        tags, terms = [], []
        for s in run.species:
            p = _sp(s)
            ms = [f'{p}M3par{suffix}', f'{p}M3perp{suffix}'] if qty == 'q_par' else [f'{p}M3perp{suffix}']
            tags += ms
            terms.append(' '.join(ms) + ' +' * (len(ms) - 1) + f' {run.mass[s] / 4!r} *')   # m/2 x (c0/2)
        e = ' '.join(terms) + ' +' * (len(terms) - 1)
    else:
        raise KeyError(qty)
    if scale != 1:
        e += f' {scale!r} *'
    what = {}
    for tg in tags:
        t = tg[:-len(suffix)] if suffix else tg
        if t == 'phi':
            what[tg] = 'field'
        else:
            s = next(sp for sp in run.species if t.startswith(_sp(sp)) and t[len(_sp(sp)):].startswith('M'))
            what[tg] = f'{s}_{t[len(_sp(s)):]}'
    return [(tg, what[tg]) for tg in tags], e, f'{label}' + (f' ({unit})' if unit else '')


def _file(run, b, what, k):
    name = os.path.basename(run.f(b, what, k))
    return f'"{name}"' if '*' in name or '[' in name else name


def _loads(run, b, needs, k, sel=''):
    """[(tag, file, selection)]: the files a recipe needs, block b, frame k (a glob: every frame)."""
    return [(tag, _file(run, b, what, k), sel) for tag, what in needs]


def _group(loads, rpn, out='v', label='', first=True, comp=0):
    """The pgkyl steps evaluating rpn (written over the loads' tags) into one dataset tagged `out`: load each (tag,
    file, selection), take the stored component comp, evaluate, the tags replaced by evaluate's positional names.
    A file holding a glob over frames is collected into a time series on its own first; activate lists the newest
    first, so the files are collected last to first and come back in load order. A later group (first=False)
    activates its loads, so that evaluate, which takes and replaces the whole working set, sees only them."""
    tags = [t for t, _, _ in loads]
    pos = {t: f'f{n}' for n, t in enumerate(tags)}
    steps = []
    if any('*' in f for _, f, _ in loads):
        for t, f, sel in reversed(loads):
            steps += [f'load -f {f} -t {t}{sel}', f'activate -t {t}', f'select -c {comp}', f'collect -t c{t}']
        steps.append('activate ' + ' '.join(f'-t c{t}' for t in tags))
    else:
        steps += [f'load -f {f} -t {t}{sel}' for t, f, sel in loads]
        if not first:
            steps.append('activate ' + ' '.join(f'-t {t}' for t in tags))
        steps.append(f'select -c {comp}')
    lab = f' -l {q(label)}' if label else ''
    return steps + [f'evaluate -t {out}{lab} {q(" ".join(pos.get(x, x) for x in rpn.split()))}']


def _plot(outs, opts):
    """The plot of every group's result: activated back from aside (each later group set the earlier ones aside) and
    drawn on one figure (pgkyl 2 gives each dataset its own figure otherwise)."""
    if len(outs) == 1:
        return [f'plot {opts}']
    return ['activate ' + ' '.join(f'-t {o}' for o in outs), f'plot {opts} --multiblock']


def _cmd(run, parts, notes=()):
    """'# note' lines, then cd into the run and the pgkyl chain, one step per line."""
    head = [f'# {n}' for n in notes]
    return '\n'.join(head + [f'cd {q(run.path)} && \\\n  ' + ' \\\n  '.join([PGKYL] + parts)])


def _change(run, qty, mode, k, k_ref, load):
    """(loads, rpn, label) of a quantity or its change since another frame (mode 'change' / 'step'): relative (%) for
    a positive quantity, a difference for a signed one."""
    needs, e, lab = recipe(run, qty)
    loads = load(needs, k)
    if mode == 'value' or k_ref is None:
        return loads, e, lab
    n2, e2, _ = recipe(run, qty, suffix='r')
    loads += load(n2, k_ref)
    kind = gv.quantities(run)[qty][4]
    since = 'the first frame' if mode == 'change' else 'the previous frame'
    if kind == 'pos':
        return loads, f'{e} {e2} / 1 - 100 *', f'{lab.split(" (")[0]}, change since {since} (%)'
    return loads, f'{e} {e2} -', f'{lab}, change since {since}'


def _map_parts(run, b, qty, k, mode, k_ref):
    """The pgkyl steps of one block's map (b: a block number, or '$b' inside a shell loop)."""
    loads, e, lab = _change(run, qty, mode, k, k_ref, lambda needs, kk: _loads(run, b, needs, kk))
    signed = gv.quantities(run)[qty][4] != 'pos' or mode != 'value'
    title = f'frame {k}' if run.single else f'block {b}, frame {k}'   # the quantity: the label's box
    return _group(loads, e, label=lab) + ['plot --xlabel psi --ylabel theta --title '
                                          + (f'"{title}"' if '$' in title else q(title)) + (' --diverging' if signed else '')]


def block_map(run, b, qty, k='{frame}', mode='value', k_ref=None):
    """One block's quantity (or its change since frame k_ref) at a frame, on the block's (psi, theta) grid."""
    return _cmd(run, _map_parts(run, b, qty, k, mode, k_ref),
                [f'{run.name}: every cell of block {b} at frame {k}, on its (psi, theta) grid (the viewer\'s strip of '
                 'that block)', NO_DISPLAY])


def blocks_maps(run, qty, k, blocks=None, mode='value', k_ref=None):
    """Every shown block's map, one pgkyl window per block (a shell loop over the blocks)."""
    bs = [b for b in range(run.nb) if blocks is None or b in blocks] or list(range(run.nb))
    if run.single or len(bs) == 1:
        return block_map(run, bs[0], qty, k, mode, k_ref)
    steps = ' \\\n    '.join([PGKYL] + _map_parts(run, '$b', qty, k, mode, k_ref))
    return '\n'.join([f'# {run.name}: each block\'s cells at frame {k} on its (psi, theta) grid, one window per block '
                      '(pgkyl has no poloidal-plane view of these files)', f'# {NO_DISPLAY}',
                      f'cd {q(run.path)} && \\\nfor b in {" ".join(map(str, bs))}; do \\\n  {steps}; \\\ndone'])


def cell_value(run, b, i, j, qty, k='{frame}'):
    """The cell average the viewer shows for cell (i, j) of block b (i: the FILE's psi index)."""
    needs, e, lab = recipe(run, qty)
    parts = _group(_loads(run, b, needs, k, f' --z0 {i} --z1 {j}'), e) + ['print']
    return _cmd(run, parts, [f'{run.name}: {lab} in block {b}, cell i = {i} (psi), j = {j} (theta), frame {k}'])


def cell_history(run, b, i, j, qty):
    """One cell's quantity at every frame (the Probe tab)."""
    needs, e, lab = recipe(run, qty)
    parts = _group(_loads(run, b, needs, '[0-9]*', f' --z0 {i} --z1 {j}'), e, label=lab) + ['plot --xlabel "t (s)"']
    return _cmd(run, parts, [f'{run.name}: {lab} in block {b}, cell i = {i}, j = {j}, at every frame', NO_DISPLAY])


def _mid_column(run, b, j_near):
    """The theta cell nearest Z = 0 for most psi rows of block b (the viewer interpolates each row)."""
    js = [j0 if w < 0.5 else j1 for j0, j1, w in gv.midplane_weights(run.nodes[b], j_near)]
    return int(np.bincount(js).argmax())


def midplane(run, side, qty, k='{frame}', blocks=None, history=False):
    """A quantity across the core and SOL blocks at one midplane (history: at every frame, collected)."""
    st = run.layout.stacks[side]
    bs = [b for b in gv.stack_blocks(st) if blocks is None or b in blocks] or gv.stack_blocks(st)
    needs, e, lab = recipe(run, qty)
    parts, cols, outs = [], [], []
    for b in bs:
        j = _mid_column(run, b, st.j)
        cols.append(f'block {b} j = {j}')
        outs.append(f'v{b}')
        parts += _group(_loads(run, b, needs, '[0-9]*' if history else k, f' --z1 {j}'), e, outs[-1],
                        f'{lab}, block {b}', first=len(outs) == 1)
    parts += _plot(outs, '--xlabel "t (s)" --ylabel psi' if history else '--xlabel psi')
    return _cmd(run, parts, [f'{run.name}: {lab} at the {side} midplane' + (' at every frame' if history else
                                                                            f', frame {k}') +
                             f', against psi ({", ".join(cols)})',
                             'the viewer interpolates each psi row to Z = 0 between two theta cells and plots against '
                             'the distance from the separatrix; this is the theta cell nearest Z = 0', NO_DISPLAY])


def plate(run, tg, k='{frame}'):
    """|q_par| along a target plate (the plate cell of its leg block, across the SOL)."""
    nt = run.nodes[tg.leg].shape[1] - 1
    needs, e, lab = recipe(run, 'q_par')
    parts = _group(_loads(run, tg.leg, needs, k, f' --z1 {tg.j % nt}'), e + ' abs', label='|q_par| (MW m^-2)')
    return _cmd(run, parts + ['plot --xlabel psi'],
                [f'{run.name}: |q_par| at the {tg.name} (block {tg.leg}, plate cell j = {tg.j % nt}), frame {k}, '
                 'against psi (the viewer: the surfaces\' midplane distance)', NO_DISPLAY])


def _sum(tags):
    """The RPN summing datasets by tag."""
    return ' '.join(tags[:1] + [f'{t} +' for t in tags[1:]])


def dyn_sum(run, what, comp, blocks=None, scale=1.0, label='', logy=False, notes=()):
    """A time trace summed over blocks (each block's integrated diagnostic, one component), times scale."""
    bs = [b for b in (range(run.nb) if blocks is None else blocks) if os.path.exists(run.f(b, what))]
    if not bs:
        return None
    loads = [(f'b{b}', _file(run, b, what, None), '') for b in bs]
    e = _sum([t for t, _, _ in loads]) + (f' {scale!r} *' if scale != 1 else '')
    parts = _group(loads, e, label=label, comp=comp) + ['plot --xlabel "t (s)"' + (' --logy' if logy else '')]
    return _cmd(run, parts, list(notes) + [NO_DISPLAY])


def dt(run):
    return _cmd(run, [f'load -f {run.stem}-dt.gkyl', 'plot --logy --xlabel "t (s)" --ylabel "dt (s)"'], [NO_DISPLAY])


def particles(run, s):
    return dyn_sum(run, f'{s}_integrated_moms', 0, label=f'{gv.species_tag(s)[0]}s (per radian)',
                   notes=[f'{run.name}: the {gv.species_tag(s)[0]}s\' number, summed over blocks (the chart divides it '
                          'by its first sample)'])


def field_energy(run):
    return dyn_sum(run, 'field_energy', 0, label='field energy', logy=True,
                   notes=[f'{run.name}: the field energy, summed over blocks'])


def target_power(run, tg):
    """The power through a target's plate: its leg and PF blocks' boundary fluxes, both species, x 2 pi (MW)."""
    face = gv.plate_face(tg)
    plate_blocks = [b for b in (tg.leg, tg.pf) if b is not None]
    loads = []
    for s in run.species:
        for b in plate_blocks:
            what = f'{s}_bflux_{face}_integrated_HamiltonianMoments'
            if os.path.exists(run.f(b, what)):
                loads.append((f'{_sp(s)}b{b}', _file(run, b, what, None), ''))
    if not loads:
        return None
    e = _sum([t for t, _, _ in loads]) + f' {2 * np.pi!r} * 1e-06 *'
    parts = _group(loads, e, label=tg.name + ' (MW)', comp=2) + ['plot --xlabel "t (s)"']   # comp 2: the energy
    return _cmd(run, parts, [f'{run.name}: the power through the {tg.name}\'s plate (blocks '
                             f'{", ".join(map(str, plate_blocks))}, face {face}), its full toroidal ring', NO_DISPLAY])


def balance(run, energy=False):
    """postgkyl's own particle (each species) or energy balance of the run (per radian)."""
    mb = '' if run.single else ' -m -1'   # -1: every block
    if energy:
        return _cmd(run, [f'gk_energy_balance -n {run.stem} {" ".join(f"-s {s}" for s in run.species)} -p ./{mb}'],
                    [f'{run.name}: postgkyl\'s energy balance, per radian (the chart: x full device)',
                     'it takes the source file\'s M2 for its Hamiltonian; this driver writes M0M1M2 there, so its '
                     'source energy is off by the mass (the chart uses m M2 / 2)', NO_DISPLAY])
    return '\n\n'.join(_cmd(run, [f'gk_particle_balance -n {run.stem} -s {s} -p ./{mb}'],
                            [f'{run.name}: postgkyl\'s {gv.species_tag(s)[0]} particle balance, per radian '
                             '(the chart: x full device)', NO_DISPLAY]) for s in run.species)


def _d_loads(run, b, s, sel=''):
    """D x g^xx J and g^xx J of block b (b: a number, or '$b' in a shell loop)."""
    return [('Dg', _file(run, b, f'{s}_anom_diff', 0), sel), ('g', _file(run, b, 'geo_int_gxxj', None), sel)]


def diffusivity(run, b, s, side=None):
    """The anomalous diffusivity of block b (or, side given, across that midplane's blocks): the anom_diff file
    (D x g^xx J) over geo_int_gxxj, cell by cell."""
    st = run.layout.stacks.get(side) if side else None
    bs = gv.stack_blocks(st) if st else [b]
    parts, outs = [], []
    for bb in bs:
        outs.append(f'v{bb}')
        sel = f' --z1 {_mid_column(run, bb, st.j)}' if st else ''
        parts += _group(_d_loads(run, bb, s, sel), 'Dg g /', outs[-1], f'D (m^2/s), block {bb}', first=len(outs) == 1)
    parts += _plot(outs, '--xlabel psi' + ('' if st else ' --ylabel theta'))
    return _cmd(run, parts, [f'{run.name}: D of the {gv.species_tag(s)[0]}s' + (f' at the {side} midplane' if st else
                                                                              f', block {b}'),
                             'a ratio of cell averages; the viewer divides the DG expansions (weak division): the '
                             'same where D or g^xx J is constant across a cell', NO_DISPLAY])


def diffusivity_maps(run, s, blocks=None):
    """Every shown block's D (the diffusivity map), one pgkyl window per block (a shell loop)."""
    bs = [b for b in range(run.nb) if blocks is None or b in blocks] or list(range(run.nb))
    if run.single or len(bs) == 1:
        return diffusivity(run, bs[0], s)
    parts = _group(_d_loads(run, '$b', s), 'Dg g /', label='D (m^2/s)') + [
        'plot --xlabel psi --ylabel theta --title "block $b"']
    steps = ' \\\n    '.join([PGKYL] + parts)
    return '\n'.join([f'# {run.name}: D of the {gv.species_tag(s)[0]}s in each block on its (psi, theta) grid (a ratio of '
                      'cell averages: the viewer divides the DG expansions)', f'# {NO_DISPLAY}',
                      f'cd {q(run.path)} && \\\nfor b in {" ".join(map(str, bs))}; do \\\n  {steps}; \\\ndone'])


NONE = {
    'region': 'no pgkyl equivalent: a Jacobian-weighted mean over the region\'s cells (gkview.region_mean)',
    'neg': 'no pgkyl equivalent: a count of cells with n ≤ 0 over every block (gkview.summary)',
    'thomson': 'no pgkyl equivalent: Thomson n_e from Flux (thomson.py), not simulation output',
    'lambda': 'λq has no pgkyl command (gkview.target_heat_flux: 63 % of the plate\'s q_par); click a point for '
              'the plate\'s q_par at that frame',
}
