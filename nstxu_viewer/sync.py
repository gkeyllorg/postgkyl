#!/usr/bin/env python3
"""Mirror runs from the machine that holds them into the same layout here, over ssh, so the viewer can run anywhere
(a laptop, another cluster) and keep working while that machine is down.

viewer.json's "remote": {"host": <an ssh host name>, "base": <the path there of what "base" is here>}. For each run
<simulation>/runs/<run> it copies, to the same place under the base here:
- the run's files the viewer reads: all but distribution-function and source frames and restart files (most of a
  run's bytes);
- its summary cache (the data directory's runs/cache), so the viewer opens it without reading every frame;
- the geqdsk it read (the vessel outline, the shot), when that is under the remote base;
- its shot's Thomson files that are there: the Flux copy (the data directory's runs/flux) and the sim-init workflow's
  (viewer.json "thomson_fallback").
The viewer finds them where the run's files name the remote's paths (gkview.here). Symlinks there arrive as what they
point to. rsync copies what changed since the last time (new frames). Where the base here is the remote's own (the
viewer runs on that machine), there is nothing to do.

    python sync.py                         every simulation's default run
    python sync.py --sim 'Sim 2' --run prod_th02q4_coll --run cpu_dprof203899
    python sync.py --dry-run               what would be copied (the runs and caches: the geqdsks and Thomson files
                                           are found from the runs' own files, once they are here)

ssh: through the control socket $SYNC_CM when set (view.sh --sync keeps one login open: no prompt), else the user's
own ssh (its config and keys; a password and one-time code at the prompt, unless --batch: then it fails instead).
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gkview as gv  # noqa: E402
import thomson  # noqa: E402

# Run on the remote (its python3; nothing but the standard library). Of the paths given relative to the base (the
# first argument), the files there, relative to it: a file as it is, a directory's files but distribution-function
# and source frames and restart files. A path that is not there gives nothing.
LISTER = r'''
import os, re, sys
skip = re.compile(r'-(?!field_)[A-Za-z0-9]+_[0-9]+\.gkyl$|_source_[0-9]+\.gkyl$|restart')
base = sys.argv[1]
for rel in sys.argv[2:]:
    p = os.path.join(base, rel)
    if os.path.isfile(p):
        print(rel)
    for d, _, fs in os.walk(p):
        for f in fs:
            if not skip.search(f):
                print(os.path.relpath(os.path.join(d, f), base))
'''


def remote():
    """(host, base) from viewer.json, or None when it names no remote or this is it."""
    r = gv.CONFIG.get('remote') or {}
    if not r.get('host') or not r.get('base'):
        return None
    if os.path.isdir(r['base']) and os.path.samefile(r['base'], gv.NSTXU):
        return None
    return r['host'], r['base'].rstrip('/')


def ssh_cmd(batch):
    cm = os.environ.get('SYNC_CM')
    return ['ssh', '-x'] + (['-S', cm] if cm else []) + (['-o', 'BatchMode=yes'] if cm or batch else [])


def under(path, base):
    """path relative to base, or None when it is not under it."""
    rel = os.path.relpath(path, base)
    return None if rel == '..' or rel.startswith('..' + os.sep) else rel


def listing(host, rbase, rels, batch):
    """Of rels (relative to the remote base), the files there the viewer reads (LISTER), relative to it."""
    p = subprocess.run(ssh_cmd(batch) + [host, 'python3', '-', shlex.quote(rbase)] + [shlex.quote(r) for r in rels],
                       input=LISTER, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f'listing on {host}: {p.stderr.strip()[-400:]}')
    return [ln for ln in p.stdout.splitlines() if ln.strip()]


def pull(host, rbase, files, dry, batch):
    """rsync files (relative to the remote base) to the same place under the base here. Returns 'n files, size'."""
    if not files:
        return 'nothing'
    cmd = ['rsync', '-a', '--copy-links', '--stats', '-e', ' '.join(ssh_cmd(batch))] + (['--dry-run'] if dry else [])
    with tempfile.NamedTemporaryFile('w', suffix='.txt') as lst:
        lst.write('\n'.join(files) + '\n')
        lst.flush()
        p = subprocess.run(cmd + [f'--files-from={lst.name}', f'{host}:{rbase}/', gv.NSTXU + '/'],
                           capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f'rsync from {host}: {p.stderr.strip()[-400:]}')
    return stats(p.stdout)


def stats(out):
    """'n files, size' from rsync --stats (rsync 3: 'regular files'; 2.6: 'files')."""
    def get(*keys):
        return next((ln.split(':', 1)[1].strip().split()[0] for ln in out.splitlines()
                     for key in keys if ln.startswith(key)), '?')
    return (f"{get('Number of regular files transferred', 'Number of files transferred')} files, "
            f"{get('Total transferred file size')} bytes")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sim', action='append', help='a simulation label from viewer.json (default: all)')
    ap.add_argument('--run', action='append', help='a run of it (default: its default run)')
    ap.add_argument('--no-cache', action='store_true', help='leave the viewer\'s caches')
    ap.add_argument('--dry-run', action='store_true', help='say what would be copied')
    ap.add_argument('--batch', action='store_true', help='never prompt for a password (fail instead)')
    a = ap.parse_args(argv)
    r = remote()
    if r is None:
        print('nothing to sync: viewer.json names no remote, or this machine is it')
        return 0
    host, rbase = r
    sims = [s for s in gv.SIMS if not a.sim or s[0] in a.sim]
    if a.sim and len(sims) < len(a.sim):
        print(f'no simulation {set(a.sim) - {s[0] for s in sims}} in viewer.json', file=sys.stderr)
        return 2
    runs = []   # (label, run, its path relative to base)
    for lab, simdir, dflt, _ in sims:
        if os.path.isabs(simdir):
            print(f'{lab}: {simdir} is an absolute path, not under the base: not synced', file=sys.stderr)
            continue
        runs += [(lab, run, f'{simdir}/runs/{run}') for run in a.run or [dflt]]
    verb = 'would copy' if a.dry_run else 'copied'

    def copy(what, rels):
        files = listing(host, rbase, rels, a.batch)
        print(f'{what}: {len(files)} files there; {verb} {pull(host, rbase, files, a.dry_run, a.batch)}', flush=True)
        return files

    # the runs, and their caches
    caches = [] if a.no_cache else [under(gv.cache_path(os.path.join(gv.NSTXU, rel)), gv.NSTXU) for _, _, rel in runs]
    files = copy('runs' + ('' if a.no_cache else ' and caches'), [rel for _, _, rel in runs] + [c for c in caches if c])
    for lab, run, rel in runs:
        n = sum(f.startswith(rel + '/') for f in files)
        print(f'  {lab} / {run}: {n} files' + ('' if n else f' (no run {rbase}/{rel} on {host}?)'))
    if a.dry_run:
        return 0
    # the geqdsks they read (a relative name is under the run, so under the base)
    local = [(lab, run, os.path.join(gv.NSTXU, rel)) for lab, run, rel in runs
             if gv.is_run_dir(os.path.join(gv.NSTXU, rel))]
    geq = []
    for lab, run, path in local:
        g = gv.geqdsk_named(path, gv.read_options(path))
        g_rel = g and (under(g, rbase) or under(g, gv.NSTXU))
        if g and not g_rel:
            print(f'  {lab} / {run}: its geqdsk {g} is not under the base: not synced', file=sys.stderr)
        geq += [g_rel] if g_rel else []
    copy('geqdsks', sorted(set(geq)))
    # their shots' Thomson files
    dirs = [under(d, gv.NSTXU) for d in (thomson.CACHE, thomson.WORKFLOW) if d]
    shots = {gv.open_run(path, lab).shot for lab, _, path in local} - {0}
    copy('Thomson files', [f'{d}/{s}.h5' for d in dirs if d for s in sorted(shots)])
    return 0


if __name__ == '__main__':
    sys.exit(main())
