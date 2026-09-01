"""The data-transformation library -- one function per operation.

Every verb takes a dataset first and returns a dataset (via ``_result``), so the
fluent ``GData`` methods, the operators, and any CLI all delegate here and can
never drift apart. Verbs are typed on ``GDataState`` but return the caller's
concrete (sub)class because ``_result`` rebuilds ``type(self)``.

``interpolate`` is the one-way modal -> NumPy bridge; ``arithmetic`` dispatches
on the container backend (Gkeyll kernels for modal data, NumPy for field data);
``integrate`` is a terminal verb that runs inside Gkeyll on modal data;
``average`` reduces modal data over a dimension subset via
``gkyl_array_average``, producing a new lower-dimensional modal dataset;
``map`` delegates to the grid-mapping engine in ``dg.map``. Flat modules are
domain-independent core verbs; domain subpackages such as ``gyrokinetics``
hold transformations that require domain geometry without interpreting field
components as new physical conclusions. Equation-specific physics (the former
``moments``/``agyro``/``current``/``energetics``/``rotate``/
``transform_frame``/``laguerre`` verbs, folded with the array math they
delegated to) lives one layer up, in ``diagnostics``.
"""

from . import arithmetic, gyrokinetics
from .interpolate import interpolate
from .local_poly import local_poly
from .select import select
from .info import info
from .integrate import integrate, integrate_axis
from .average import average
from .eval_at_coord_proj import eval_at_coord_proj
from .plot import plot
from .animate import animate
from .plotly import plotly
from .plotly_animate import plotly_animate
from .represent import apply, represent

from .fft import fft
from .magsq import magsq
from .relchange import relchange
from .mask import mask
from .collect import collect
from .sort import sort
from .grid import grid
from .val2coord import val2coord
from .extract_input import extract_input
from .fit import fit
from .differentiate import differentiate
from .evaluate import available_operators as available_evaluate_operators, evaluate
from .map import map

__all__ = ["interpolate", "local_poly", "select", "info", "integrate", "integrate_axis", "average",
    "eval_at_coord_proj",
    "plot", "animate", "plotly", "plotly_animate",
    "arithmetic", "represent", "apply",
    "fft", "magsq", "relchange", "mask", "collect", "sort", "grid", "val2coord",
    "extract_input", "fit", "differentiate", "evaluate", "available_evaluate_operators",
    "map", "gyrokinetics"]
