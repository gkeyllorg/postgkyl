"""Where the NSTX-U run viewer (nstxu_viewer.py) opens: a saved view (./view.sh --state <file>), then the launch
command line (./view.sh --shot 203899 --frame 30 ...), then the page's address (the viewer writes the view there, so
a reload or a bookmark comes back to it), each overriding the one before; the defaults under all three. One set of
keys everywhere. A state file is versioned JSON (runs/states/*.json). A value that no longer fits (a run removed, a
quantity the run lacks) falls back to the default, with a note on the page.
"""
import json
import os
import shlex

STATE_VERSION = 1
# key: (what it sets, example)
KEYS = {
    'shot': ('the shot (the first simulation of that shot)', '203899'),
    'sim': ("the simulation, by its label ('Sim 1', 'Sim 2', '2025 re-run'; case and spaces ignored)", 'sim2'),
    'run': ('the run directory of that simulation', 'prod_th02q4_coll'),
    'frame': ('the frame shown (negative: from the last)', '30'),
    'qty': ("the quantity (its key: n_elc, T_ion, phi, q_par, mach, ...)", 'T_elc'),
    'mode': ('value | change | step (change since the first / the previous frame)', 'change'),
    'log': ('log colour scale for positive values', 'true'),
    'zoom': ("the poloidal view ('whole domain', 'lower divertor', ...)", "'lower divertor'"),
    'x': ('profile x axis: distance | R', 'distance'),
    'theme': ("the look: NSTX-U (the DPP 2025 poster) | PPPL (the lab's slide template)", 'PPPL'),
    'dark': ('dark mode', 'true'),
    'tab': ("the open tab ('Probe', 'Balance & power', ...)", "'Time traces'"),
    'frames': ("the frames the movie plays ('pgkyl load --frame' syntax: ':', '::2', '-50:', '0:60')", "'0:60'"),
    'probe': ('the probed cell: block,i,j', '10,0,15'),
    'blocks': ('the blocks drawn: a list or ranges (blank: every block)', "'1-3,10'"),
    'cmp': ("a run drawn over this one: '<simulation>/<run>', or a run directory", "'Sim 1/prod_coll'"),
    'path': ('a run directory to open: any run, multiblock or single-block (offered as its own simulation)',
             'runs/my_single_block_run'),
}
TRUE = {'1', 'true', 'yes', 'on'}


def as_bool(v):
    return v if isinstance(v, bool) else str(v).strip().lower() in TRUE


def load_state(path):
    """The view a state file holds. Raises OSError (unreadable) or ValueError (not a state file of this version)."""
    with open(os.path.expanduser(path)) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f'{path} is not a viewer state file: {e}') from None
    if not (isinstance(data, dict) and data.get('nstxu_viewer_state') == STATE_VERSION and
            isinstance(data.get('view'), dict)):
        raise ValueError(f'{path} is not a version-{STATE_VERSION} viewer state file')
    return data['view']


def save_state(path, view):
    """Write the view to path (its directory made if needed); returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'nstxu_viewer_state': STATE_VERSION, 'view': view}, f, indent=2)
        f.write('\n')
    return path


def merge(*sources):
    """One dict of view keys from the sources in increasing precedence (missing or None values skipped), and notes
    about keys this viewer does not know."""
    prefs, notes = {}, []
    for src in sources:
        for k, v in (src or {}).items():
            if v is None or v == '':
                continue
            if k not in KEYS:
                notes.append(f'unknown view key {k!r} ignored')
                continue
            prefs[k] = v
    return prefs, notes


# the viewer's own choice when a key is not given (left out of the command line)
DEFAULTS = {'mode': 'change', 'log': False, 'zoom': 'whole domain', 'x': 'distance', 'theme': 'NSTX-U', 'dark': True,
            'tab': 'Midplane profiles', 'frames': '', 'probe': '', 'blocks': '', 'cmp': '', 'path': ''}


def parse_blocks(text, nb):
    """The block numbers a 'blocks' value names ('1-3,10', '7'; blank: None = every block), within 0..nb-1.
    Raises ValueError for anything else."""
    text = str(text or '').replace(' ', '')
    if not text:
        return None
    out = set()
    for part in text.split(','):
        a, _, b = part.partition('-')
        if not a.isdigit() or (b and not b.isdigit()):
            raise ValueError(f'blocks {text!r}: not a list of block numbers or ranges (1-3,10)')
        lo, hi = int(a), int(b or a)
        if lo > hi or hi >= nb:
            raise ValueError(f'blocks {text!r}: this run has blocks 0-{nb - 1}')
        out.update(range(lo, hi + 1))
    return out


def blocks_text(blocks):
    """The 'blocks' value of a set of block numbers ('' for None: every block)."""
    if not blocks:
        return ''
    b, out, i = sorted(blocks), [], 0
    while i < len(b):
        j = i
        while j + 1 < len(b) and b[j + 1] == b[j] + 1:
            j += 1
        out.append(f'{b[i]}' if i == j else f'{b[i]}-{b[j]}')
        i = j + 1
    return ','.join(out)


def command_line(view):
    """The ./view.sh command that opens this view (keys at their defaults left out)."""
    parts = ['./view.sh']
    for k in KEYS:
        v = view.get(k)
        if k == 'shot' or v is None or v == '' or (k in DEFAULTS and v == DEFAULTS[k]):
            continue
        parts += [f'--{k}', shlex.quote(str(v).lower() if isinstance(v, bool) else str(v))]
    return ' '.join(parts)


def norm_label(s):
    return ''.join(str(s).lower().split())
