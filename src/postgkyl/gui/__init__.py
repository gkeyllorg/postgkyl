"""A marimo notebook GUI over the public Postgkyl API.

``pgkyl-gui --path <dir>`` (or ``python -m postgkyl.gui``) opens
:mod:`postgkyl.gui.notebook` on a data directory. The notebook only wires
widgets to :mod:`postgkyl.gui.pipeline`, which turns the choices into a chain
of public verbs, runs it, and renders the equivalent Python script.

marimo is an optional dependency: ``pip install 'postgkyl[gui]'``.
"""
