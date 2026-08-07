"""Thin per-verb CLI command shells (one module per verb).

Each module exposes a ``command`` (a ``click.Command``). Adding a new verb is
a drop-in: create ``commands/<verb>.py`` with a ``command`` and add it to
``COMMANDS`` below.

``COMMAND_SECTIONS`` is the one home for the help-listing grouping consumed
by ``PgkylGroup.format_commands`` (``cli/app.py``) -- presentation only; every
command below stays a flat, chainable top-level ``click.Command`` regardless
of which section its name appears in (see ``14-cli.md``, "Help output
organization").
"""

from __future__ import annotations

from . import (
    load, interpolate, local_poly, save, select, plot, info,
    fft, magsq, relchange, mask, collect, sort, grid, val2coord, extractinput,
    fit, growth, differentiate, evaluate, map, integrate, average,
    evalatcoordproj, animate,
    euler, tenmoment, mhd, velocity, agyro, current, energetics,
    parrotate, perprotate, bparrotate, bperprotate, transform_frame,
    laguerre_compose,
    plotly, plotly_animate, pyvista, style,
    gk_distf, gk_energy_balance, gk_fluxsurf, gk_load_quantity, gk_particle_balance,
    gk_rz, gkyl_pkpm,
    listoutputs, status,
)
from . import print as _print

COMMANDS = [
    load.command,
    interpolate.command,
    local_poly.command,
    select.command,
    plot.command,
    info.command,
    save.command,
    fft.command,
    magsq.command,
    relchange.command,
    mask.command,
    collect.command,
    sort.command,
    grid.command,
    val2coord.command,
    extractinput.command,
    fit.command,
    growth.command,
    differentiate.command,
    evaluate.command,
    map.command,
    integrate.command,
    average.command,
    evalatcoordproj.command,
    animate.command,
    euler.command,
    tenmoment.command,
    mhd.command,
    velocity.command,
    agyro.command,
    current.command,
    energetics.command,
    parrotate.command,
    perprotate.command,
    bparrotate.command,
    bperprotate.command,
    transform_frame.command,
    laguerre_compose.command,
    plotly.command,
    plotly_animate.command,
    pyvista.command,
    style.command,
    gk_distf.command,
    gk_energy_balance.command,
    gk_fluxsurf.command,
    gk_load_quantity.command,
    gk_particle_balance.command,
    gk_rz.command,
    gkyl_pkpm.command,
    listoutputs.command,
    status.command,
    _print.command,
]

# Presentation-only grouping for ``pgkyl --help`` (see cli/app.py). Every name
# here must be a registered command's name; ``load``/``info`` are the only
# names split across "Loaders"/"Utility" that also appear implicitly in the
# chain (``load`` is hidden -- see commands/load.py -- so it is omitted here).
COMMAND_SECTIONS: dict[str, list[str]] = {
    "Verbs": [
        "average", "collect", "dg_local_poly", "differentiate", "evalatcoordproj",
        "evaluate", "extractinput", "fft", "fit", "grid", "growth", "integrate",
        "interpolate", "load", "magsq", "map", "mask", "relchange", "select",
        "sort", "val2coord",
    ],
    "Diagnostics": [
        "agyro", "bparrotate", "bperprotate", "current", "energetics",
        "euler", "gk_distf", "gk_energy_balance", "gk_fluxsurf", "gk_load_quantity",
        "gk_particle_balance", "gk_rz", "gkyl_pkpm", "laguerre_compose",
        "mhd", "parrotate", "perprotate", "tenmoment", "transform_frame",
        "velocity",
    ],
    "Render": ["animate", "plot", "plotly", "plotly_animate", "pyvista", "style"],
    "Utility": ["info", "listoutputs", "print", "save", "status"],
}

__all__ = ["COMMANDS", "COMMAND_SECTIONS"]
