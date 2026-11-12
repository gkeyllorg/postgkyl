"""The options a run was started with, made readable for the viewer's Options tab (nstxu_viewer.py).

- launch options: options_used.txt, else the -o key=value of the run's command file (gkview.launch_options);
- what each option is, from the driver's own source: every opt_get_<type>_comma_kv(<opts>, "<key>", <target>) call
  names an option and the field it sets; the field's comment in its struct definition says what it is (comment lines
  between members, at the members' indent, head a section); the defaults come from the struct's designated
  initializer (`return (struct x) { .field = value, ... }`) where the driver has one. The source is found next to the
  binary the run's command names (<binary>.c, or the binary's name without a trailing _<date> suffix), with the
  headers it includes from there. It is the source as it is now: when it is newer than the binary, its defaults
  and descriptions may not be the ones the run was built with (said on the page);
- the jobs of the run (restarts.log: one line per job, its kind, restart frame, binary and the options it set);
- what the driver reported while it set the run up: its first job log, every line once (each rank prints them), from
  the launch to 'Gyrokinetic simulation initialized'.
"""
import os
import re
import time

import gkview as gv

OPT_CALL = re.compile(r'opt_get_(\w+?)_comma_kv\(\s*\w+\s*,\s*"(\w+)"\s*,\s*([^,)]+)')
TYPES = {'double': 'number', 'int': 'integer', 'bool': 'true / false', 'string': 'text'}


def command_text(run):
    for name in ('command.txt', 'cmd.txt', 'run.txt'):
        p = os.path.join(run.path, name)
        if os.path.exists(p):
            return open(p, errors='replace').read()
    return ''


def driver_binary(run):
    """The executable the run's command started (the first absolute path that is a file), or None."""
    for tok in command_text(run).split():
        if tok.startswith('/') and os.path.isfile(tok) and not tok.endswith(('.geqdsk', '.csv', '.txt')):
            return tok
    return None


def driver_sources(binary):
    """The driver's source files next to its binary: <binary>.c, or without a trailing _<date>[letter] suffix, and the
    headers it includes with quotes from the same directory. [] when none is found."""
    if not binary:
        return []
    d, name = os.path.split(binary)
    for stem in (name, re.sub(r'_\d{8}[a-z]?$', '', name), re.sub(r'_[^_]+$', '', name)):
        c = os.path.join(d, stem + '.c')
        if os.path.isfile(c):
            out = [c]
            for inc in re.findall(r'#include\s+"([^"]+)"', open(c, errors='replace').read()):
                h = os.path.join(d, inc)
                if os.path.isfile(h):
                    out.append(h)
            return out
    return []


def _struct_fields(src):
    """field name -> (section, description) from every struct definition in a source: a member's trailing comment
    and the comment lines under it (indented past the members), a comment line at the members' indent heading the
    members after it."""
    out = {}
    _struct_fields.prev = prev = {}   # field -> the field declared just before it in its struct
    for m in re.finditer(r'^struct\s+(\w+)\s*\{\s*\n(.*?)^\};', src, re.S | re.M):
        sname, body = m.group(1), m.group(2)
        section, last, indent = sname.replace('_', ' '), None, None
        for line in body.split('\n'):
            s = line.strip()
            if not s:
                continue
            ind = len(line) - len(line.lstrip())
            if s.startswith('//'):
                text = s.lstrip('/').strip()
                if last is not None and (indent is None or ind > indent):
                    out[last] = (out[last][0], (out[last][1] + ' ' + text).strip())
                else:
                    section, last = text, None
                continue
            code, _, comment = s.partition('//')
            if ';' not in code:
                continue
            indent = ind if indent is None else indent
            names = []
            for decl in code.split(';'):   # 'double t_end; int num_frames; // ...': one comment for both
                parts = [p.strip() for p in decl.split(',') if p.strip()]
                if not parts or len(parts[0].split()) < 2:
                    continue
                names += [parts[0].split()[-1]] + parts[1:]
            names = [re.sub(r'[\[=].*$', '', n).lstrip('*').strip() for n in names]
            for n in names:
                if n:
                    out.setdefault(n, (section, comment.strip()))
                    if last is not None and n != last:
                        prev.setdefault(n, last)
                    last = n
    return out


def _defaults(src):
    """field -> default value text from designated initializers 'return (struct x) { .field = value, ... }'."""
    out = {}
    for m in re.finditer(r'return\s*\(struct\s+\w+\)\s*\{', src):
        i, depth = m.end(), 1
        while i < len(src) and depth:
            depth += {'{': 1, '}': -1}.get(src[i], 0)
            i += 1
        block = src[m.end():i - 1]
        for dm in re.finditer(r'^\s*\.(\w+)\s*=\s*(.+?),?\s*(?://.*)?$', block, re.M):
            out.setdefault(dm.group(1), dm.group(2).rstrip(','))
    return out


def driver_options(sources):
    """[dict(key, kind, field, section, what, default)] for every option the driver reads, in source order."""
    text = '\n'.join(open(p, errors='replace').read() for p in sources)
    fields, defaults = _struct_fields(text), _defaults(text)
    rows, seen = [], set()
    for kind, key, target in OPT_CALL.findall(text):
        if key in seen:
            continue
        seen.add(key)
        t = target.strip().lstrip('&')
        field = re.split(r'->|\.', t)[-1].strip() if ('->' in t or '.' in t) else None
        field = re.sub(r'\[.*$', '', field) if field else None
        names_it = lambda f: re.search(rf'\b{re.escape(key)}\b', fields[f][1])  # noqa: E731
        if field not in fields:   # a local read into a field later: the field whose comment names the option
            field = next((f for f in fields if names_it(f)), field)
        section, what = fields.get(field, ('other', ''))
        before = getattr(_struct_fields, 'prev', {}).get(field)
        if field in fields and not what and before in fields and names_it(before):
            what = f'(see {before}) {fields[before][1]}'   # no comment of its own: the field above it names it
        rows.append(dict(key=key, kind=TYPES.get(kind, kind), field=field, section=section, what=what,
                         default=defaults.get(field, '') if field else ''))
    return rows


def options_view(run):
    """dict(binary, sources, stale, rows, unknown): rows = driver_options with 'value' (what the run set at launch,
    '' where it left the default); unknown = launch options the driver source does not read."""
    launch = gv.launch_options(run)
    binary = driver_binary(run)
    sources = driver_sources(binary)
    rows = driver_options(sources) if sources else []
    for r in rows:
        r['value'] = launch.get(r['key'], '')
    known = {r['key'] for r in rows}
    unknown = {k: v for k, v in launch.items() if k not in known}
    stale = bool(sources and binary and max(os.path.getmtime(p) for p in sources) > os.path.getmtime(binary))
    return dict(binary=binary, sources=sources, stale=stale, rows=rows, unknown=unknown, launch=launch,
                ranks=re.search(r'srun\b[^\n]*?-n\s+(\d+)', command_text(run)))


def short_section(sec):
    """'gyroradius rules (opt-in; see ...)' -> 'gyroradius rules'."""
    return re.split(r'\s+\(', sec, maxsplit=1)[0].strip() or sec


def shown_value(v):
    """A value for the table: a path as its last two parts (the full paths: the files table)."""
    v = str(v)
    return '…/' + '/'.join(v.rstrip('/').split('/')[-2:]) if v.startswith('/') and v.count('/') > 2 else v


def effective(view):
    """option -> the value the run used: set at launch, else the driver source's default ('' where unknown)."""
    out = {r['key']: r['value'] or r['default'] for r in view['rows']}
    out.update(view['unknown'])
    return out


def jobs(run):
    """The jobs of the run, from restarts.log: [dict(started, job, kind, restart_frame, bin, set)]."""
    p = os.path.join(run.path, 'restarts.log')
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p, errors='replace'):
        m = re.search(r'\bjob\s+(\d+)', line)
        if not m:
            continue
        kv = dict(re.findall(r"(\w+)=('[^']*'|\S+)", line[m.end():]))
        sets = kv.get('set', '').strip("'")
        out.append(dict(started=line[:m.start()].strip(), job=m.group(1), kind=kv.get('kind', ''),
                        restart_frame=kv.get('restart_frame', ''), binary=kv.get('bin', ''),
                        options_set=', '.join(sets.split())))
    return out


NOISE = re.compile(r'^(The following have been reloaded|\s+\d+\) |\s*Rank \d+ handles|srun\b|\s*$)')
TAGGED = re.compile(r'^\s*[A-Z][A-Z0-9]*(_[A-Z0-9]+)+\b')   # the library's tagged diagnostics (TOK_GEO_..., GKYL_...)


def startup_report(run, limit=250):
    """What the driver printed while it set the run up: its first job log (the oldest), every line once, from the start
    to 'Gyrokinetic simulation initialized' (module messages, the per-rank block lines and the library's tagged
    diagnostic lines left out); its name, the lines, and how many tagged lines were left out."""
    logs = sorted(gv.run_logs(run), key=os.path.getmtime)
    if not logs:
        return None, [], 0
    seen, out, tagged = set(), [], 0
    with open(logs[0], errors='replace') as f:
        for _, line in zip(range(20000), f):
            line = line.rstrip('\n')
            if line.startswith(('Gyrokinetic simulation initialized', 'Taking time-step')):
                break
            if NOISE.match(line) or line in seen:
                continue
            seen.add(line)
            if TAGGED.match(line):
                tagged += 1
                continue
            out.append(line[:400])
            if len(out) >= limit:
                break
    return os.path.basename(logs[0]), out, tagged


def when(path):
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))
