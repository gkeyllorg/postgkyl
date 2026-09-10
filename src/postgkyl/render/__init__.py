"""Visualization backends (a backend layer used by the fluent surface).

Re-exporting backend functions here shadows the corresponding submodule
references the import machinery would otherwise set on this package.
``render.animate`` and ``render.plotly`` therefore resolve to the canonical
functions, matching ``render.plot`` and ``render.pyvista``. Other public
Plotly names are re-exported alongside ``plotly`` for the same reason.
"""

from . import labels, style
from .animate import animate
from .matplotlib import plot
from .plotly import open_preview, plotly, plotly_animate, save_rotating_plotly_figure
from .pyvista import pyvista

__all__ = [
    "plot", "animate", "labels", "style", "plotly", "plotly_animate",
    "save_rotating_plotly_figure", "open_preview", "pyvista"
]
