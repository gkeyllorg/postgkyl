"""A marimo notebook GUI over the public Postgkyl API.

``pgkyl-gui --path <dir>`` (or ``python -m postgkyl.gui``; the launcher is
:mod:`postgkyl.cli.gui_launch`) opens :mod:`postgkyl.gui.notebook` on a data
directory. The notebook only wires widgets to :mod:`postgkyl.gui.pipeline`,
which turns the choices into ``PostgkylSession`` calls, runs them, and shows
the equivalent session Python and ``pgkyl`` command line.

marimo is an optional dependency: ``pip install 'postgkyl[gui]'``.
"""
