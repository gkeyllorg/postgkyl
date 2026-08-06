"""High-level, scriptable interface to the pgkyl command chain.

The pgkyl command line (e.g. ``pgkyl file.gkyl gk-rz pl``) is a chain of click
commands operating on a shared dataset stack. This module exposes that same
chain to Python/Jupyter as a stateful session object so each command can be
called as a typed method:

    from postgkyl.clap import PgkylSession
    pg = PgkylSession()
    pg.load("file.gkyl")
    pg.gk_rz(phi_tor=0.0)
    pg.plot(fixaspect=True)

The typed methods themselves live in the generated ``postgkyl.clap`` module; this
file holds the stable runtime core they build on. See ``postgkyl._api_gen`` for
the generator that keeps ``api.py`` in sync with the click commands.
"""

import glob
import os
import re
import shlex
import time

import click
import matplotlib.pyplot as plt
import numpy as np

import postgkyl.commands as cmd
import postgkyl.output
from postgkyl.commands import DataSpace
from postgkyl.pgkyl import cli
from postgkyl.utils import load_style


class _Session:
  """Stateful pgkyl stack; the Python equivalent of one CLI invocation.

  Datasets loaded and processed by the command methods accumulate on an internal
  stack (a ``DataSpace``), exactly as they would when chaining commands on the
  command line. Drop to ``session.data`` to reach the underlying ``GData``
  objects and their raw NumPy arrays.

  This class is inherited by ``PgkylSession`` in ``postgkyl.clap``. Since 
  ``PgkylSession`` is generated automatically, this class is a space where one 
  can add stable, hand-written features to the session API.
  """

  def __init__(self, verbose: bool = False, batch_mode: bool = False,
      style: str | None = None):
    """Initialize an empty session.

    Args:
      verbose: Turn on pgkyl verbose output.
      batch_mode: Run in batch mode (no plots are shown).
      style: Path to a Matplotlib style file (defaults to the pgkyl style).
    """
    self.ctx = click.Context(cli)
    self.ctx.obj = {
        "start_time": time.time(),
        "verbose": verbose,
        "batch_mode": batch_mode,
        "saveframes_prefix": os.path.expanduser("~") + "/pg",
        "in_data_strings": [],
        "in_data_strings_loaded": 0,
        "data": DataSpace(),
        "fig": "",
        "ax": "",
        "compgrid": False,
        "global_var_names": (),
        "global_cuts": (None,) * 7,
        "global_c2p": None,
        "global_c2p_vel": None,
        "rcParams": {},
    }
    style_file = style or os.path.join(
        os.path.dirname(postgkyl.output.__file__), "postgkyl.mplstyle")
    load_style(self.ctx, style_file)
    self.cmd_stack = []
    self.figs = []

  def _run(self, command: click.Command, _files=None, **kwargs):
    """Dispatch a click command against this session's stack."""
    self.cmd_stack.append(self._format_command(command, kwargs, files=_files))
    return self._invoke_capturing_figure(command, kwargs)

  def _invoke_capturing_figure(self, command: click.Command, kwargs: dict):
    """Invoke a click command, recording any figure it produces on ``self.figs``."""
    before = set(plt.get_fignums())
    grabbed = []
    real_show = plt.show

    def _show_hook(*args, **kwargs_):
      fig = plt.gcf()
      if fig.get_axes() and fig not in grabbed:
        grabbed.append(fig)

      return real_show(*args, **kwargs_)

    plt.show = _show_hook
    try:
      result = self.ctx.invoke(command, **kwargs)
    finally:
      plt.show = real_show

    if not grabbed:
      for num in plt.get_fignums():
        if num not in before:
          fig = plt.figure(num)
          if fig.get_axes():
            grabbed.append(fig)

    if grabbed:
      self.figs.append(grabbed[-1])
      self.ctx.obj["fig"] = grabbed[-1]

    return result

  def get_fig(self, index: int = -1):
    """Return a figure produced by a previous plotting call.

    Every plotting command appends the figure it creates to ``self.figs``;
    ``get_fig`` returns one of them so it can be inspected or modified after the
    fact, e.g.::

        pg.plot(...)
        fig = pg.get_fig()             # most recent figure
        fig.axes[0].set_title("new")
        fig                            # re-display in a notebook cell

    Args:
      index: Index into the captured figures. Defaults to ``-1`` (most recent);
        any valid list index works (e.g. ``0`` for the first).

    Returns:
      The requested ``matplotlib.figure.Figure``.

    Raises:
      IndexError: If no figure has been captured yet, or ``index`` is out of
        range.
    """
    if not self.figs:
      raise IndexError("No figures have been captured yet; call a plotting "
          "command (e.g. pg.plot(...)) first.")
    # end
    return self.figs[index]

  def get_data(self, idx: int = 0, tag: str | None = None):
    """Return the grid and values of one dataset on the stack.

    Handy for overlaying a dataset onto an existing figure (see :meth:`get_fig`)::

        pg.plot(...)                       # 2D pcolormesh, say (t, v_par)
        grid, values = pg.get_data(0)      # another 1D dataset on the stack
        pg.get_fig().axes[0].plot(grid[0], values[..., 0], "k--")

    The returned grid is cell-centered so its 1D coordinates line up with
    ``values`` (``get_grid`` itself returns nodal edges of length ``cells+1``);
    multi-dimensional (mapped) coordinate arrays are returned unchanged.

    Args:
      idx: Index into the active datasets, in the same order :meth:`plot` sees
        them. Defaults to ``0`` (the first). Negative indices count from the end.
      tag: Restrict to datasets carrying this tag (the ``use``/``tag`` label);
        defaults to all active datasets.

    Returns:
      A ``(grid, values)`` tuple, where ``grid`` is a list with one coordinate
      array per dimension and ``values`` is the ``numpy`` array of components
      (its last axis indexes the components).

    Raises:
      IndexError: If the stack (optionally filtered by ``tag``) is empty, or
        ``idx`` is out of range.
    """
    if tag is not None:
      existing = set(self.data.tag_iterator(only_active=False))
      missing = [t for t in tag.split(",") if t not in existing]
      if missing:
        raise IndexError(f"No datasets on the stack for tag(s) "
            f"{', '.join(missing)}; available tags: {sorted(existing) or 'none'}.")

    datasets = list(self.data.iterator(tag))
    if not datasets:
      raise IndexError("No datasets on the stack"
          + (f" for tag '{tag}'" if tag else "") + "; load some data first.")

    dat = datasets[idx]
    values = dat.get_values()
    cells = values.shape[:-1]
    grid = []
    for d, g in enumerate(dat.get_grid()):
      g = np.asarray(g)
      # Nodal edges -> cell centers so a 1D coordinate aligns with values.
      if g.ndim == 1 and d < len(cells) and g.shape[0] == cells[d] + 1:
        g = 0.5 * (g[:-1] + g[1:])

      grid.append(g)

    return grid, values

  @staticmethod
  def _long_opt(opts) -> str:
    """Pick the most readable CLI flag for a parameter (prefer the long form)."""
    long = [o for o in opts if o.startswith("--")]
    return max(long or opts, key=len)

  def _format_command(self, command: click.Command, kwargs: dict, files=None) -> str:
    """Render one command as the CLI fragment that reproduces it.

    Only options whose value differs from the command default are emitted, so
    the result matches what a user would actually type rather than spelling out
    every defaulted option the generated methods forward.

    Token order matters: pgkyl chains commands, so any token following a
    positional argument is parsed as the *next* command. The fragment is
    therefore emitted as ``name [options] [argument values]`` — options always
    precede positional arguments. For example ``ev`` must read
    ``ev --tag t '<chain>'`` (``ev '<chain>' --tag t`` would treat ``--tag`` as
    a new command), and positional values are emitted bare (no metavar name).
    """
    name_tokens = [shlex.quote(f) for f in files] if files is not None else [command.name]
    opt_tokens = []
    arg_tokens = []

    for param in command.params:
      if param.name == "help" or param.name not in kwargs:
        continue
      value = kwargs[param.name]

      if isinstance(param, click.Argument):
        if value is None:
          continue
        if getattr(param, "multiple", False) or getattr(param, "nargs", 1) == -1:
          arg_tokens.extend(shlex.quote(str(item)) for item in value)
        else:
          arg_tokens.append(shlex.quote(str(value)))
        continue

      default = param.default
      # click >=8.2 marks "no default given" with a Sentinel; treat as None.
      if repr(default).startswith("Sentinel"):
        default = None

      if getattr(param, "is_flag", False):
        secondary = getattr(param, "secondary_opts", [])
        if default is True and secondary:
          if value is False:
            opt_tokens.append(self._long_opt(secondary))
        elif value:
          opt_tokens.append(self._long_opt(param.opts))
        continue

      if value is None or value == default:
        continue

      opt = self._long_opt(param.opts)
      if getattr(param, "multiple", False) or getattr(param, "nargs", 1) == -1:
        for item in value:
          opt_tokens.append(f"{opt} {shlex.quote(str(item))}")
      else:
        opt_tokens.append(f"{opt} {shlex.quote(str(value))}")

    return " ".join(name_tokens + opt_tokens + arg_tokens)

  def set_globals(self, *, c2p: str | None = None, c2p_vel: str | None = None,
      varname: str | tuple[str, ...] | None = None, compgrid: bool | None = None,
      z0: str | None = None, z1: str | None = None, z2: str | None = None,
      z3: str | None = None, z4: str | None = None, z5: str | None = None,
      component: str | None = None) -> None:
    """Set pgkyl group-level options that apply to every file loaded afterwards.

    These mirror the options placed *before* the data on the pgkyl command line,
    e.g. ``pgkyl --c2p-vel map.gkyl file1.gkyl file2.gkyl``. Unlike the same-named
    keyword arguments passed directly to :meth:`load` (which affect only that one
    file), a global applies to all data loaded after it is set, so a subsequent
    chain step such as ``ev`` sees a consistent grid type across its inputs. Call
    this *before* the relevant :meth:`load` calls.

    Args:
      c2p: File with c2p mapped coordinates (global ``--c2p``).
      c2p_vel: File with c2p mapped velocity coordinates (global ``--c2p-vel``).
      varname: Adios variable name(s) (global ``--varname``).
      compgrid: Disregard the mapped grid information (global ``--compgrid``).
      z0, z1, z2, z3, z4, z5: Partial load along each coordinate (int or slice).
      component: Partial load: components (int or slice).
    """
    obj = self.ctx.obj
    if c2p is not None:
      obj["global_c2p"] = c2p
    if c2p_vel is not None:
      obj["global_c2p_vel"] = c2p_vel
    if compgrid is not None:
      obj["compgrid"] = compgrid
    if varname is not None:
      obj["global_var_names"] = (varname,) if isinstance(varname, str) else tuple(varname)
    cuts = list(obj["global_cuts"])
    for idx, value in enumerate((z0, z1, z2, z3, z4, z5, component)):
      if value is not None:
        cuts[idx] = value
    obj["global_cuts"] = tuple(cuts)

  def _global_tokens(self) -> list[str]:
    """CLI tokens for the active group-level options (see :meth:`set_globals`)."""
    obj = self.ctx.obj
    tokens = []
    if obj.get("global_c2p"):
      tokens.append(f"--c2p {shlex.quote(str(obj['global_c2p']))}")
    if obj.get("global_c2p_vel"):
      tokens.append(f"--c2p-vel {shlex.quote(str(obj['global_c2p_vel']))}")
    if obj.get("compgrid"):
      tokens.append("--compgrid")
    for name in obj.get("global_var_names", ()):
      tokens.append(f"--varname {shlex.quote(str(name))}")
    cut_opts = ("--z0", "--z1", "--z2", "--z3", "--z4", "--z5", "--component")
    for opt, value in zip(cut_opts, obj.get("global_cuts", (None,) * 7)):
      if value is not None:
        tokens.append(f"{opt} {shlex.quote(str(value))}")
    return tokens

  def get_cmd(self) -> str:
    """Return the pgkyl CLI command equivalent to this session so far."""
    parts = ["pgkyl"]
    if self.ctx.obj.get("verbose"):
      parts.append("--verbose")
    if self.ctx.obj.get("batch_mode"):
      parts.append("--batch-mode")
    # Group-level options must precede the data they apply to.
    parts.extend(self._global_tokens())
    parts.extend(self.cmd_stack)
    return " ".join(parts)

  def print_cmd(self) -> str:
    """Print the pgkyl CLI command equivalent to this session.

    Reconstructs the chained command line that would reproduce every command run
    on this session so far, e.g. after::

        pg.load("file.gkyl")
        pg.gk_rz(phi_tor=0.0)
        pg.plot(fixaspect=True)

    ``pg.print_cmd()`` prints ``pgkyl file.gkyl gk-rz --fix-aspect``.
    """
    print(self.get_cmd())

  def load(self, *files: str, **kwargs):
    """Load one or more Gkeyll files onto the stack.

    This is the Python counterpart of naming files on the pgkyl command line.
    Any additional keyword arguments are forwarded to the ``load`` command
    (e.g. ``tag``, ``label``, ``mapc2p_name``).

    Args:
      files: One or more paths to Gkeyll output files. e.g.
          ``pg.load("file1.gkyl", "file2.gkyl")``
    """
    self.ctx.obj["in_data_strings"].extend(files)
    result = None
    for f in files:
      result = self._run(cmd.load, _files=(f,), **kwargs)
    return result
  
  def get_framelist(self, name: str, simprefix: str, path: str = ".") -> list[int]:
    """List the available frame numbers for a given output in a directory.

    Scans ``path`` for files named ``{simprefix}-{name}_{frame}.gkyl`` and
    returns the sorted frame numbers. For example, files
    ``rt_gk_alfven_1x2v-apar_0.gkyl``, ``..._1.gkyl``, ``..._2.gkyl`` yield
    ``[0, 1, 2]``.

    Args:
      name: The output name (e.g. ``"apar"``).
      simprefix: The simulation prefix (e.g. ``"rt_gk_alfven_1x2v"``).
      path: The directory to search (default is the current directory).

    Returns:
      The sorted list of frame numbers found for the specified output.
    """
    pattern = os.path.join(path, f"{simprefix}-{name}_*.gkyl")
    regex = re.compile(rf"{re.escape(simprefix)}-{re.escape(name)}_(\d+)\.gkyl$")
    frames = []
    for fn in glob.glob(pattern):
      match = regex.search(os.path.basename(fn))
      if match:
        frames.append(int(match.group(1)))
    return sorted(frames)



  @property
  def data(self) -> DataSpace:
    """The ``DataSpace`` stack holding all loaded/processed datasets."""
    return self.ctx.obj["data"]
