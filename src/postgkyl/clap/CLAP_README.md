# `postgkyl.clap`

A scriptable Python interface to postgkyl, for use in scripts and Jupyter notebook instead of the `pgkyl` command line.

On the command line, pgkyl is a chain of click commands sharing one dataset stack (e.g. `pgkyl file.gkyl gk-rz pl`). This package exposes that same chain as a stateful session object, so each command becomes a typed method call:

```python
from postgkyl.clap import PgkylSession

pg = PgkylSession()
pg.load("file.gkyl")
pg.gk_rz(phi_tor=0.0)
pg.plot(fixaspect=True)
```

Every session also tracks the commands it runs, so `pg.print_cmd()` prints the equivalent `pgkyl ...` command line that reproduces the session (only non-default options are shown). `pg.get_cmd()` returns that string instead of printing it.

## Layout

- **`scripting.py`** — the stable, hand-written runtime core (`_Session`). It
  holds the dataset stack and dispatches commands, plus a few hand-maintained methods like `load`. Edit this file to add or change hand-written behavior.
- **`clap.py`** — **generated, do not edit by hand.** It defines `PgkylSession`
  (which subclasses `_Session`) with one typed method per pgkyl command, so editors/Pylance get full signature and docstring help.
- **`_clap_gen.py`** — the generator that produces `clap.py`.

## Updating `clap.py`

The pgkyl click commands are the single source of truth: each command and option already carries a name, type, default, and help string. `_clap_gen.py` introspects that metadata and emits one typed method per command.

Whenever you add, remove, or change a pgkyl command or its options, regenerate:

```bash
python -m postgkyl.clap._clap_gen          # rewrite clap.py
python -m postgkyl.clap._clap_gen --check  # exit non-zero if clap.py is stale (for CI/tests)
```
