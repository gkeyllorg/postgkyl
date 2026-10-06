"""Matplotlib style scoping -- the ``style_context`` helper.

The old ``utils/load_style.py`` hand-parsed an ``.mplstyle`` file line by
line (with a special case for ``cycler(...)`` values) into a Typer context's
``rcParams`` dict. Matplotlib's own style-file parser already supports that
exact ``cycler(...)`` syntax (see ``postgkyl.mplstyle``'s ``axes.prop_cycle``
line), so re-implementing a parser here would be a second, hand-maintained
copy of a fact Matplotlib already owns (DOCTRINE V). This module is a thin
wrapper: ``style_context`` resolves the packaged default/name and forwards to
``matplotlib.pyplot.style.context``. Postgkyl never mutates global Matplotlib
state; every style it applies is scoped to the figure being drawn.
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


def style_context(path_or_name: str | None = None):
  """A context manager applying a Matplotlib style for its duration only.

  Args:
    path_or_name: A packaged style name (currently only ``"postgkyl"``), a
      name Matplotlib recognizes (e.g. ``"dark_background"``), or a path to
      an ``.mplstyle`` file. ``None`` applies the packaged Postgkyl default.

  ``matplotlib.rcParams`` is restored on exit, so a styled plot never leaks
  into figures drawn afterwards.
  """
  import matplotlib.pyplot as plt

  name = path_or_name or DEFAULT_STYLE
  return plt.style.context(_PACKAGED_STYLES.get(name, name))


__all__ = ["style_context", "DEFAULT_STYLE"]
