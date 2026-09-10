"""Matplotlib style application -- the ``apply_style`` verb-adjacent helper.

The old ``utils/load_style.py`` hand-parsed an ``.mplstyle`` file line by
line (with a special case for ``cycler(...)`` values) into a Typer context's
``rcParams`` dict. Matplotlib's own style-file parser already supports that
exact ``cycler(...)`` syntax (see ``postgkyl.mplstyle``'s ``axes.prop_cycle``
line), so re-implementing a parser here would be a second, hand-maintained
copy of a fact Matplotlib already owns (DOCTRINE V). This module is a thin,
context-free wrapper: ``apply_style`` resolves the packaged default/name and
forwards to ``matplotlib.pyplot.style.use``.
"""

from __future__ import annotations

import os.path

_STYLE_DIR = os.path.dirname(os.path.realpath(__file__))

# Names this package ships a style sheet for, resolved before falling through
# to Matplotlib's own named styles / arbitrary file paths.
_PACKAGED_STYLES = {
    "postgkyl": os.path.join(_STYLE_DIR, "postgkyl.mplstyle"),
}

DEFAULT_STYLE = "postgkyl"


def apply_style(path_or_name: str | None = None) -> None:
  """Apply a Matplotlib style, mutating ``matplotlib.rcParams`` in place.

  Args:
    path_or_name: A packaged style name (currently only ``"postgkyl"``), a
      name Matplotlib recognizes (e.g. ``"dark_background"``), or a path to
      an ``.mplstyle`` file. ``None`` applies the packaged Postgkyl default.

  This is the module's one documented effect: it mutates global Matplotlib
  rc state (there is no other way to apply a style; see
  ``matplotlib.pyplot.style.use``).
  """
  import matplotlib.pyplot as plt

  name = path_or_name or DEFAULT_STYLE
  target = _PACKAGED_STYLES.get(name, name)
  plt.style.use(target)


__all__ = ["apply_style", "DEFAULT_STYLE"]
