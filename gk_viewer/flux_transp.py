#!/usr/bin/env python3
"""TRANSP for the run viewer, run on PPPL Flux inside fetch_flux.sh's remote script (after the sim-init workflow's
extract_slim_h5.py): for each shot, one TRANSP run's profiles the viewer reads, added to the shot's slim file
<out>/<shot>.h5 as /<shot>/TRANSP/<variable>/{data, units, dims}. The group's attributes say which run (run; '' when
none qualifies), why (why), every run of the shot and what became of it (considered), the CDF and its time, the
run's namelist (TR.DAT), and the profiles it predicts rather than fits to measurements, from when (predicts:
lpredict_te / _ti / _ne, predicts_from: s, -inf from the start). Needs numpy and h5py only: TRANSP writes its output
CDFs as netCDF-3, read here directly.

The run, among the shot's runs in TRANSP's archive (<root>/{16,15}/<shot><letter><two digits>{.CDF, TR.DAT, TR.MSG}):
1. the sim-init workflow's choice for the shot, where its candidate CSV (--picks; column TRANSP_RUN_ID) lists the
   shot (203899 -> 203899S18: the run the workflow took its edge T_i from; it predicts T_e and T_i from 128 ms, a
   ~update_time block, as the workflow's reference_run/transp_effective_mode_audit.csv records);
2. else the newest of its runs that are interpretive throughout: the namelist sets lpredict_te, lpredict_ti and
   lpredict_ne false wherever it sets them (the first block and every ~update_time block);
3. else the newest of its runs that set none of them, taken as TRANSP's default, interpretive (UNVERIFIED: an
   assumption, not checked against TRANSP's documentation). The workflow requires them set
   (code/transp_qualification.py) and so found no run for 203655, whose five runs set none.
Newest: by the TR.DAT's time, A01 first on a tie (the workflow's order). The first run that qualifies is taken: its CDF
holds every variable of VARIABLES on those dimensions in those units, and the last 200 kB of its TR.MSG holds no
fatal marker (the workflow's).

    python flux_transp.py 203655,203899 --out DIR [--root /p/transparch/result/NSTU] [--picks CSV]
"""
import argparse
import csv
import datetime
import glob
import os
import re

import numpy as np

ROOT = '/p/transparch/result/NSTU'
CAMPAIGNS = ('16', '15')
# what the viewer reads (thomson.py): name -> (its dimensions, None: any name; its units, as TRANSP writes them)
VARIABLES = {
    'TIME3': (('TIME3', ), 'SECONDS'),
    'X':
    (('TIME3', 'X'), ''),  # zone centres: sqrt of the normalised toroidal flux
    'XB': (('TIME3', 'XB'), ''),  # zone boundaries
    'TI': (('TIME3', 'X'), 'EV'),
    'ND': (('TIME3', 'X'), 'N/CM**3'),  # thermal deuterium
    'RMAJM':
    (('TIME3', None), 'CM'
     ),  # midplane major radius: inboard edge to outboard edge through the axis
}
PREDICT = ('lpredict_te', 'lpredict_ti', 'lpredict_ne')
FALSE = {'0', '0.0', '.false.', '.f.', 'false', 'f'}  # the workflow's
FATAL = ('FATAL ERROR', 'TRANSP ABORT', 'JOB ABORTED')  # the workflow's
MSG_TAIL = 200_000

# ------------------------------------------------------------------------------------------------- netCDF-3
NC_TYPES = {1: 'i1', 2: 'S1', 3: '>i2', 4: '>i4', 5: '>f4', 6: '>f8'}


def netcdf3(path):
  """The variables of a netCDF-3 file (classic or 64-bit offset: TRANSP's CDFs): {name: (dimension names, shape,
    attributes, dtype, file offset)}. The header as Unidata's "NetCDF Classic and 64-bit Offset Format" lays it out:
    magic, record count, then the dimension, global attribute and variable lists, each a tag and a count (absent:
    two zeros); names and values padded to 4 bytes; big-endian."""
  with open(path, 'rb') as f:
    magic = f.read(4)
    if magic[:3] != b'CDF' or magic[3] not in (1, 2):
      raise ValueError(
          f'{path}: not a netCDF-3 classic or 64-bit-offset file (it starts {magic!r})'
      )

    def i4():
      return int(np.frombuffer(f.read(4), '>i4')[0])

    def padded(n):
      b = f.read(n)
      f.read(-n % 4)
      return b

    def name():
      return padded(i4()).decode('latin-1')

    def count(tag):
      t, n = i4(), i4()
      if t not in (0, tag):
        raise ValueError(f'{path}: malformed netCDF-3 header')
      return n

    def attributes():
      out = {}
      for _ in range(count(12)):
        key, typ = name(), i4()
        dt = np.dtype(NC_TYPES[typ])
        raw = padded(dt.itemsize * i4())
        out[key] = raw.decode('latin-1').rstrip(
            '\0 ') if typ == 2 else np.frombuffer(raw, dt)
      return out

    i4()  # the record count
    dims = [(name(), i4()) for _ in range(count(10))]
    attributes()  # the global ones
    out = {}
    for _ in range(count(11)):
      var = name()
      ids = [i4() for _ in range(i4())]
      atts = attributes()
      dt = np.dtype(NC_TYPES[i4()])
      i4()  # its size
      begin = int(
          np.frombuffer(f.read(4 * magic[3]),
                        '>i4' if magic[3] == 1 else '>i8')[0])
      out[var] = (tuple(dims[i][0] for i in ids),
                  tuple(dims[i][1] for i in ids), atts, dt, begin)
  return out


def netcdf3_data(path, entry):
  """A variable's values (native byte order), from its netcdf3 entry; a record variable (on the unlimited dimension,
    stored interleaved) is refused: TRANSP's CDFs have none."""
  _, shape, _, dt, begin = entry
  if 0 in shape:
    raise ValueError(f'{path}: a record variable (not read here)')
  with open(path, 'rb') as f:
    f.seek(begin)
    a = np.fromfile(f, dt, int(np.prod(shape)))
  return a.reshape(shape).astype(dt.newbyteorder('='))


# ---------------------------------------------------------------------------------------------- the run
def settings(namelist):
  """Every lpredict_te / _ti / _ne assignment of a TRANSP namelist, in order, with the time from which it applies
    (-inf: the first block; then each ~update_time block's time, s): {flag: [(t, value), ...]}."""
  out, t = {p: [] for p in PREDICT}, -np.inf
  for line in namelist.splitlines():
    line = line.split('!', 1)[0]
    m = re.match(r'(?i)\s*~update_time\s*=\s*([^\s,/]+)', line)
    if m:
      t = float(m.group(1).lower().replace('d', 'e'))
      continue
    for p in PREDICT:
      out[p] += [(t, v.lower())
                 for v in re.findall(rf'(?i)\b{p}\s*=\s*([^\s,/]+)', line)]
  return out


def predicts_from(namelist):
  """{flag: the earliest time (s; -inf: from the start) from which the namelist sets it true}, for the flags it
    sets true anywhere."""
  return {
      p: min(t for t, v in vs if v not in FALSE)
      for p, vs in settings(namelist).items()
      if any(v not in FALSE for _, v in vs)
  }


def mode(namelist):
  """How a TRANSP namelist sets the profiles' prediction over all its blocks: 'interpretive' (lpredict_te, _ti and
    _ne each set, false wherever set), 'unset' (none of them set: TRANSP's default, interpretive), 'predictive' (one
    set true somewhere) or 'partly set'."""
  s = settings(namelist)
  if not any(s.values()):
    return 'unset'
  if predicts_from(namelist):
    return 'predictive'
  return 'interpretive' if all(s.values()) else 'partly set'


def problem(run_dir, run):
  """Why a run cannot be used ('' when it can): its CDF lacks a variable or has it on other dimensions or in other
    units; its TR.MSG ends with a fatal marker."""
  cdf = os.path.join(run_dir, f'{run}.CDF')
  if not os.path.exists(cdf):
    return 'no CDF'
  try:
    head = netcdf3(cdf)
  except (OSError, ValueError, KeyError) as e:
    return f'CDF unreadable ({e})'
  bad = []
  for nm, (dims, units) in VARIABLES.items():
    if nm not in head:
      bad.append(f'no {nm}')
      continue
    got, _, atts, _, _ = head[nm]
    have = str(atts.get('units', '')).strip()
    if len(got) != len(dims) or any(w not in (None, g)
                                    for w, g in zip(dims, got)):
      bad.append(f'{nm} on {got}')
    elif have != units:
      bad.append(f'{nm} in {have!r}')
  msg = os.path.join(run_dir, f'{run}TR.MSG')
  if os.path.exists(msg):
    with open(msg, 'rb') as f:
      f.seek(max(0, os.path.getsize(msg) - MSG_TAIL))
      tail = f.read().decode('latin-1').upper()
    bad += [f'TR.MSG: {m}' for m in FATAL if m in tail]
  return '; '.join(bad)


def choose(shot, root, pick=None):
  """(run, its directory, why, considered) for a shot by the rule of this module's docstring: run '' when none
    qualifies; considered: '<run>: <what became of it>' for every run of the shot. pick: the sim-init workflow's
    choice for it, if any."""
  decks = {}
  for c in CAMPAIGNS:
    for p in sorted(glob.glob(os.path.join(root, c, f'{shot}*TR.DAT'))):
      run = os.path.basename(p)[:-len('TR.DAT')]
      if re.fullmatch(rf'{shot}[A-Z]\d\d', run):
        decks.setdefault(run, p)
  if not decks:
    return '', None, f'no TRANSP run of {shot} in {root}/{{{",".join(CAMPAIGNS)}}}', []
  modes = {}
  for r, p in decks.items():
    with open(p, errors='replace') as f:
      modes[r] = mode(f.read())
  newest = lambda r: (-os.path.getmtime(decks[r]), not r.endswith('A01'), r)
  tiers = [
      ([pick] if pick in decks else [],
       'the sim-init workflow\'s choice for this shot (its candidate CSV)'),
      (sorted([r for r in decks if modes[r] == 'interpretive'], key=newest),
       'the newest of its runs interpretive throughout (lpredict_te, _ti, _ne false in every block)'
       ),
      (sorted([r for r in decks if modes[r] == 'unset'], key=newest),
       'the newest of its runs that set no lpredict_te, _ti, _ne (TRANSP\'s default, interpretive), '
       'none setting them false')
  ]
  status = {r: f'namelist {modes[r]}' for r in decks}
  if pick and pick not in decks:
    status[pick] = 'the sim-init workflow\'s choice: not in the archive'
  chosen, why, tried = '', 'no run qualifies', set()
  for runs, rule in tiers:
    for r in runs:
      if r in tried:  # the workflow's choice, tried first
        continue
      if chosen:
        status[r] = f'namelist {modes[r]}; not tried ({chosen} qualified first)'
        continue
      tried.add(r)
      bad = problem(os.path.dirname(decks[r]), r)
      status[r] = f'namelist {modes[r]}; ' + (bad or 'CHOSEN')
      if not bad:
        chosen, why = r, rule
  return (chosen, os.path.dirname(decks[chosen]) if chosen else None, why,
          [f'{r}: {s}' for r, s in sorted(status.items())])


# ---------------------------------------------------------------------------------------------- output
def record(path, shot, run, why, considered, root, cdf=None):
  """The shot file's /<shot>/TRANSP group, replaced: the attributes; with a CDF, its VARIABLES and the namelist."""
  import h5py
  text = h5py.string_dtype()
  with h5py.File(path, 'a') as f:
    s = f.require_group(str(shot))
    if 'TRANSP' in s:
      del s['TRANSP']
    g = s.create_group('TRANSP')
    g.attrs['run'] = run
    g.attrs['why'] = why
    g.attrs['considered'] = np.array(considered, dtype=text)
    g.attrs['root'] = root
    g.attrs['tool'] = 'gk_viewer/flux_transp.py'
    g.attrs['created'] = datetime.datetime.now().isoformat(timespec='seconds')
    if cdf is None:
      return
    head = netcdf3(cdf)
    for nm in VARIABLES:
      dims, _, atts, _, _ = head[nm]
      v = g.create_group(nm)
      v.create_dataset('data',
                       data=netcdf3_data(cdf, head[nm]),
                       compression='gzip')
      v['units'] = str(atts.get('units', '')).strip()
      v['dims'] = np.array(dims, dtype=text)
    g.attrs['source'] = cdf
    g.attrs['source_mtime'] = datetime.datetime.fromtimestamp(
        os.path.getmtime(cdf)).isoformat(timespec='seconds')
    with open(cdf[:-len('.CDF')] + 'TR.DAT', errors='replace') as nl:
      g.attrs['namelist'] = nl.read()
    on = predicts_from(g.attrs['namelist'])
    g.attrs['predicts'] = np.array(list(on), dtype=text)
    g.attrs['predicts_from'] = np.array(list(on.values()), dtype=float)


def picks_from(csv_path):
  """{shot: run} of the sim-init workflow's candidate CSV (SHOT, TRANSP_RUN_ID); {} without one."""
  if not csv_path or not os.path.exists(csv_path):
    return {}
  with open(csv_path, newline='') as f:
    return {
        int(r['SHOT']): r['TRANSP_RUN_ID']
        for r in csv.DictReader(f) if r.get('SHOT') and r.get('TRANSP_RUN_ID')
    }


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('shots', help='a comma list')
  ap.add_argument(
      '--out',
      required=True,
      help='where the shots\' slim files are (extract_slim_h5.py --out)')
  ap.add_argument('--root', default=ROOT, help=f'TRANSP\'s archive ({ROOT})')
  ap.add_argument(
      '--picks',
      help='the sim-init workflow\'s candidate CSV (SHOT, TRANSP_RUN_ID)')
  a = ap.parse_args()
  picks = picks_from(a.picks)
  for shot in sorted({int(s) for s in a.shots.split(',') if s.strip()}):
    path = os.path.join(a.out, f'{shot}.h5')
    if not os.path.exists(path):
      print(f'{shot}: no {path}: TRANSP not added', flush=True)
      continue
    try:
      run, rdir, why, considered = choose(shot, a.root, picks.get(shot))
      record(path, shot, run, why, considered, a.root,
             os.path.join(rdir, f'{run}.CDF') if run else None)
    except Exception as e:  # the file still goes, saying why it has no TRANSP
      run, why = '', f'failed: {type(e).__name__}: {e}'
      record(path, shot, run, why, [], a.root)
    print(f'{shot}: TRANSP {run or "none"} ({why})', flush=True)


if __name__ == '__main__':
  main()
