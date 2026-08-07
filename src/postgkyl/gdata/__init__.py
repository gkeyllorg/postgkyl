"""The fluent API surface: the public ``GData``, ``load``, ``GDataGroup``,
and the module-level multi-dataset verbs (``collect``/``evaluate``/``relchange``/
``animate``/``plotly_animate``/``sort``)."""

from .gdata import GData
from .load import load
from .gdatagroup import GDataGroup
from .verbs import animate, collect, evaluate, plot, plotly_animate, relchange, sort

__all__ = ["GData", "load", "GDataGroup", "collect", "evaluate", "relchange",
    "plot", "animate", "plotly_animate", "sort"]
