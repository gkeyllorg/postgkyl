#!/usr/bin/env python3
"""Mirror runs from the machine that holds them into the same layout here, over ssh, so the viewer can run anywhere
(a laptop, another cluster) and keep working while that machine is down.

viewer.json's "remote": {"host": <an ssh host name>, "base": <the path there of what "base" is here>, "data": <the
viewer's data directory there>}. For each run <simulation>/runs/<run> it copies, to the same place under the base
here:
- the run's files the viewer reads: all but distribution-function and source frames and restart files (most of a
  run's bytes);
- the geqdsk it read (the vessel outline, the shot), when that is under the remote base; and the geqdsk and the
  device wall viewer.json names for its simulation, when they are under the base;
- its shot's Thomson file from the sim-init workflow (viewer.json "thomson_fallback"), when there is one;
and from the data directory there to the one here (runs/): the run's summary cache (runs/cache), so the viewer opens
it without reading every frame, and its shot's Flux copy (runs/flux). Without "data", a data directory under the
base is taken to be at the same place there; one elsewhere gets neither.
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
  """(host, base, data) from viewer.json: the remote's base and data directory (None: not known), or None when it
    names no remote or this is it."""
  r = gv.CONFIG.get('remote') or {}
  if not r.get('host') or not r.get('base'):
    return None
  if os.path.isdir(r['base']) and os.path.isdir(gv.BASE) and os.path.samefile(
      r['base'], gv.BASE):
    return None  # (a first mirror: no base here yet)
  base = r['base'].rstrip('/')
  here = under(gv.DATA, gv.BASE)
  data = r['data'].rstrip('/') if r.get(
      'data') else f'{base}/{here}' if here else None
  return r['host'], base, data


def ssh_cmd(batch):
  cm = os.environ.get('SYNC_CM')
  return ['ssh', '-x'] + (['-S', cm] if cm else []) + (['-o', 'BatchMode=yes']
                                                       if cm or batch else [])


def under(path, base):
  """path relative to base, or None when it is not under it."""
  rel = os.path.relpath(path, base)
  return None if rel == '..' or rel.startswith('..' + os.sep) else rel


def listing(host, rbase, rels, batch):
  """Of rels (relative to the remote base), the files there the viewer reads (LISTER), relative to it."""
  p = subprocess.run(
      ssh_cmd(batch) +
      [host, 'python3', '-', shlex.quote(rbase)] +
      [shlex.quote(r) for r in rels],
      input=LISTER,
      capture_output=True,
      text=True)
  if p.returncode:
    raise RuntimeError(f'listing on {host}: {p.stderr.strip()[-400:]}')
  return [ln for ln in p.stdout.splitlines() if ln.strip()]


def pull(host, rroot, files, root, dry, batch):
  """rsync files (relative to rroot there) to the same place under root here. Returns 'n files, size'."""
  if not files:
    return 'nothing'
  cmd = [
      'rsync', '-a', '--copy-links', '--stats', '-e', ' '.join(ssh_cmd(batch))
  ] + (['--dry-run'] if dry else [])
  with tempfile.NamedTemporaryFile('w', suffix='.txt') as lst:
    lst.write('\n'.join(files) + '\n')
    lst.flush()
    p = subprocess.run(
        cmd + [f'--files-from={lst.name}', f'{host}:{rroot}/', root + '/'],
        capture_output=True,
        text=True)
  if p.returncode:
    raise RuntimeError(f'rsync from {host}: {p.stderr.strip()[-400:]}')
  return stats(p.stdout)


def stats(out):
  """'n files, size' from rsync --stats (rsync 3: 'regular files'; 2.6: 'files')."""

  def get(*keys):
    return next((ln.split(':', 1)[1].strip().split()[0]
                 for ln in out.splitlines()
                 for key in keys if ln.startswith(key)), '?')

  return (
      f"{get('Number of regular files transferred', 'Number of files transferred')} files, "
      f"{get('Total transferred file size')} bytes")


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('--sim',
                  action='append',
                  help='a simulation label from viewer.json (default: all)')
  ap.add_argument('--run',
                  action='append',
                  help='a run of it (default: its default run)')
  ap.add_argument('--no-cache',
                  action='store_true',
                  help='leave the viewer\'s caches')
  ap.add_argument('--dry-run',
                  action='store_true',
                  help='say what would be copied')
  ap.add_argument('--batch',
                  action='store_true',
                  help='never prompt for a password (fail instead)')
  a = ap.parse_args(argv)
  r = remote()
  if r is None:
    print('nothing to sync: viewer.json names no remote, or this machine is it')
    return 0
  host, rbase, rdata = r
  sims = [s for s in gv.SIMS if not a.sim or s[0] in a.sim]
  if a.sim and len(sims) < len(a.sim):
    print(f'no simulation {set(a.sim) - {s[0] for s in sims}} in viewer.json',
          file=sys.stderr)
    return 2
  runs = []  # (label, run, its path relative to base)
  for lab, simdir, dflt, _ in sims:
    if os.path.isabs(simdir):
      print(
          f'{lab}: {simdir} is an absolute path, not under the base: not synced',
          file=sys.stderr)
      continue
    runs += [(lab, run, f'{simdir}/runs/{run}') for run in a.run or [dflt]]
  verb = 'would copy' if a.dry_run else 'copied'

  def copy(what, rels, rroot=rbase, root=gv.BASE):
    files = listing(host, rroot, rels, a.batch)
    print(
        f'{what}: {len(files)} files there; {verb} {pull(host, rroot, files, root, a.dry_run, a.batch)}',
        flush=True)
    return files

  if rdata is None:
    print(
        'no "data" in viewer.json\'s "remote" (the viewer\'s data directory there): no caches or Flux copies',
        file=sys.stderr)
  # the runs, and their caches
  files = copy('runs', [rel for _, _, rel in runs])
  for lab, run, rel in runs:
    n = sum(f.startswith(rel + '/') for f in files)
    print(f'  {lab} / {run}: {n} files' +
          ('' if n else f' (no run {rbase}/{rel} on {host}?)'))
  if rdata is not None and not a.no_cache:
    copy('caches', [
        os.path.relpath(f, gv.DATA) for _, _, rel in runs
        for f in gv.cache_files(os.path.join(gv.BASE, rel))
    ], rdata, gv.DATA)
  if a.dry_run:
    return 0
  # the geqdsks they read (a relative name is under the run, so under the base)
  local = [(lab, run, os.path.join(gv.BASE, rel)) for lab, run, rel in runs
           if gv.is_run_dir(os.path.join(gv.BASE, rel))]
  geq = []
  for lab, run, path in local:
    g = gv.geqdsk_named(path, gv.read_options(path))
    g_rel = g and (under(g, rbase) or under(g, gv.BASE))
    if g and not g_rel:
      print(
          f'  {lab} / {run}: its geqdsk {g} is not under the base: not synced',
          file=sys.stderr)
    geq += [g_rel] if g_rel else []
  # and the ones viewer.json names (relative to the base: the same place there): a simulation's geqdsk, for runs
  # whose own files name none, and its device's wall
  for lab, _, _, _ in sims:
    own = gv.SIM_SETTINGS[lab]
    wall = gv.device_settings(own.device).get('wall')
    for g in (own.geqdsk, os.path.join(gv.BASE, wall) if wall else None):
      g_rel = g and under(os.path.normpath(g), gv.BASE)
      if g and not g_rel:
        print(f'  {lab}: {g} (viewer.json) is not under the base: not synced',
              file=sys.stderr)
      geq += [g_rel] if g_rel else []
  copy('geqdsks', sorted(set(geq)))
  # their shots' Thomson files: the sim-init workflow's (under the base), the Flux copies (the data directory)
  shots = sorted({gv.open_run(path, lab).shot for lab, _, path in local} - {0})
  wf = thomson.fallback_dir() and under(thomson.fallback_dir(), gv.BASE)
  if wf:
    copy('Thomson files (sim-init workflow)', [f'{wf}/{s}.h5' for s in shots])
  if rdata is not None:
    flux = os.path.relpath(thomson.flux_dir(), gv.DATA)
    copy('Flux copies', [f'{flux}/{s}.h5' for s in shots], rdata, gv.DATA)
  return 0


if __name__ == '__main__':
  sys.exit(main())
