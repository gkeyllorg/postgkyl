"""postgkyl -- a small, layered post-processing library for Gkeyll data.

Public surface (the facade). The golden script::

    import postgkyl as pg
    pg.load('elc_M0_0.gkyl').interpolate().select(z0=0.0).plot()

The facade is **pure re-export** -- every public name is defined in the layer that
owns it and simply gathered here:

    load, GData, GDataGroup            <- api/       (fluent surface)
    collect, evaluate, relchange,      <- api/       (module-level multi-dataset
    animate, plotly_animate, sort                    verbs -- no single ``self``)
    plot                             <- render/    (multi-dataset rendering)
    info                             <- operations/ (the info verb, one-or-many)
    integrate                        <- operations/ (grid integral, via Gkeyll)
    interpolate, select              <- operations/ (functional verb spellings)
    gk_rz                           <- operations/gyrokinetics/ (domain operation)
    represent, apply                 <- operations/ (value_form verbs)
    available_evaluate_operators     <- operations/ (``evaluate``'s RPN token vocabulary)
    save                             <- io/        (file output)
    load_gk_quantity,                <- diagnostics/gyrokinetics/
    load_gk_distf, available_gk_quantities        (equation-internal loaders)
    version_report                    <- _version.py  (``pgkyl --version``'s
                                                      commit/build-info report)

Every computational fluent ``GData`` method delegates to one of these
``operations`` functions, so ``pg.select(a, z0=0.0)`` and
``a.select(z0=0.0)`` are the same call -- the functional and fluent spellings
can never drift apart. ``GData.load(...)`` is the one lifecycle method: it
loads a literal file into an existing object and returns that same object for
chaining. The rest of the
domain-independent ``operations`` verb inventory (``fft``, ``magsq``, ``mask``,
``val2coord``, ``extract_input``, ``fit``, ``differentiate``, ``integrate_axis``,
``map``, plus ``grid`` -- see ``api/gdata.py`` for why ``grid`` has no fluent
spelling) is reachable as a ``GData`` fluent method and via
``postgkyl.operations.<verb>``; this facade does not additionally promote each one to
a bare top-level name (one home per verb-vocabulary fact, not three).

Architecture (strict, cycle-free DAG; see REFACTOR_GKEYLL_FFI.md)::

    floor      gpython/    compiled _gpython extension -> libg0core.so (the only foreign code)
    leaves     numerics/   (pure NumPy; imports nothing internal)
    engine     dg/         interpolation bridge + modal ops -> gpython
    leaves     io/         readers (C-native first)    -> gpython
    container  gdatastate/ GDataState {gkyl|numpy} backend
    seam       operations/ one verb each
    backend    render/     matplotlib
    fluent     api/        GData(GDataState) + operators   <- above operations
    facade     __init__    re-exports only
"""

from postgkyl.gdata import GData, load, GDataGroup, animate, collect, evaluate, plotly_animate, relchange, sort
from postgkyl.operations import (
    apply, available_evaluate_operators, average, differentiate,
    eval_at_coord_proj, extract_input, fft, fit, grid, growth, info, integrate,
    integrate_axis, interpolate, local_poly, magsq, map, mask, represent,
    select, val2coord,
)
from postgkyl.operations.gyrokinetics import gk_fluxsurf, gk_rz
from postgkyl.render import plot, pyvista
from postgkyl.io import save
from postgkyl.diagnostics.gyrokinetics import (
    load_gk_distf, load_gk_quantity, available_quantities as available_gk_quantities)
from postgkyl._version import version_report

__version__ = "2.0.0"

__all__ = ["GData", "load", "GDataGroup", "plot", "info", "integrate",
    "interpolate", "local_poly", "select", "integrate_axis", "average",
    "eval_at_coord_proj", "fft", "magsq", "mask", "grid", "val2coord",
    "extract_input", "fit", "growth", "differentiate", "map",
    "represent", "apply", "gk_rz", "gk_fluxsurf", "save",
    "collect", "evaluate", "relchange", "animate", "plotly_animate", "sort",
    "available_evaluate_operators", "pyvista",
    "load_gk_quantity", "load_gk_distf", "available_gk_quantities",
    "__version__", "version_report"]
