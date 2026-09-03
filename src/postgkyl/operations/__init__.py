"""The data-transformation library -- one function per operation.

Every verb takes a dataset first and returns a dataset (via ``_result``), so the
fluent ``GData`` methods, the operators, and any CLI all delegate here and can
never drift apart. Verbs are typed on ``GDataState`` but return the caller's
concrete (sub)class because ``_result`` rebuilds ``type(self)``.

``interpolate`` is the one-way modal -> NumPy bridge; ``arithmetic`` dispatches
on the container backend (Gkeyll kernels for modal data, NumPy for field data);
``integrate`` performs full or partial integration inside Gkeyll on modal
data (full is terminal; partial stays native and lower-dimensional);
``average`` reduces modal data over a dimension subset via
``gkyl_array_average``, producing a new lower-dimensional modal dataset;
``map`` delegates to the grid-mapping engine in ``dg.map``. Flat modules are
domain-independent core verbs; domain subpackages such as ``gyrokinetics``
hold transformations that require domain geometry without interpreting field
components as new physical conclusions. Equation-specific physics (the former
``moments``/``agyro``/``current``/``energetics``/``rotate``/
``transform_frame``/``laguerre`` verbs, folded with the array math they
delegated to) lives one layer up, in ``diagnostics``.

The terminal renderers (``plot``, ``animate``, ``plotly``, ``plotly_animate``,
and ``pyvista``) are exceptions:
this namespace re-exports their exact canonical callables from
:mod:`postgkyl.render` without wrapping them.
"""

from . import arithmetic, gyrokinetics
from .interpolate import interpolate
from .local_poly import local_poly
from .select import select
from .info import info
from .integrate import integrate
from .average import average
from .eval_at_coord_proj import eval_at_coord_proj
from postgkyl.render import animate, plot, plotly, plotly_animate, pyvista
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
from .growth import growth
from .differentiate import differentiate
from .evaluate import available_operators as available_evaluate_operators, evaluate
from .map import map

# Command metadata is attached at the layer that owns each operation.  This
# block is deliberately declarative: discovery still walks the public API and
# there is no registration side effect or CLI import here.
from typing import Annotated, Literal

from postgkyl.cli_spec import (
    CliArgument, CliType, CommandSpec, DatasetRef, Execution, ResultPolicy,
    Section, command, hidden,
)
from postgkyl.gdatastate.gdatastate import GDataState


def _resolve_receiver_annotations(*functions) -> None:
  """Make the modules' public forward references runtime-resolvable."""
  for function in functions:
    function.__globals__.setdefault("GDataState", GDataState)
    function.__globals__.setdefault("_GDataState", GDataState)
  # end
# end


_MAP = CommandSpec(Section.VERBS, Execution.MAP_REPLACE)
_APPEND = CommandSpec(Section.VERBS, Execution.MAP_APPEND, consumes_inputs=True)
_COMBINE = CommandSpec(Section.VERBS, Execution.COMBINE, consumes_inputs=True)
_TERM_EACH = CommandSpec(Section.UTILITY, Execution.TERMINAL_EACH,
    result=ResultPolicy.VALUE)
_TERM_ALL = CommandSpec(Section.UTILITY, Execution.TERMINAL_ALL,
    result=ResultPolicy.VALUE)

for _function in (
    interpolate, local_poly, select, integrate, average,
    eval_at_coord_proj, fft, magsq, relchange, mask, collect, sort, grid,
    val2coord, extract_input, fit, differentiate, evaluate, map, represent,
    growth,
):
  _resolve_receiver_annotations(_function)
# end

select.__annotations__.update(comp=str | None, z0=str | None, z1=str | None,
    z2=str | None, z3=str | None, z4=str | None, z5=str | None)
integrate.__annotations__["op"] = Literal["none", "abs", "sq"]
integrate.__annotations__["axis"] = Annotated[
    int | tuple | str | None, CliType(str | None), CliArgument()]
evaluate.__annotations__["chain"] = Annotated[str, CliArgument()]
average.__annotations__["dims"] = list[int]
average.__annotations__["weight"] = Annotated[GDataState | None, DatasetRef()]
eval_at_coord_proj.__annotations__.update(
    eval_dirs=list[int], eval_coords=list[float])
relchange.__annotations__.update(
    data0=Annotated[GDataState, DatasetRef()],
    data=Annotated[GDataState, DatasetRef()], comp=str | None)
mask.__annotations__["mask_data"] = Annotated[GDataState | None, DatasetRef()]
fit.__annotations__["guess"] = str | None
map.__annotations__["data"] = GDataState
map.__annotations__["mapping"] = str
represent.__annotations__["to"] = Literal["modal", "nodal", "quad"]

for _function in (interpolate, local_poly, select, average,
    eval_at_coord_proj, fft, magsq, grid, differentiate, map):
  command(_MAP)(_function)
# end
command(_APPEND)(val2coord)
command(_COMBINE)(relchange)
command(_COMBINE)(collect)
command(CommandSpec(Section.VERBS, Execution.COMBINE, consumes_inputs=True))(sort)
command(_COMBINE)(evaluate)
command(_MAP)(mask)
command(CommandSpec(Section.VERBS, Execution.MAP_APPEND))(fit)
command(CommandSpec(Section.VERBS, Execution.MAP_APPEND))(growth)
command(CommandSpec(Section.UTILITY, Execution.TERMINAL_ALL,
    result=ResultPolicy.SILENT))(info)
command(CommandSpec(Section.VERBS, Execution.MAP_OR_TERMINAL_EACH,
    result=ResultPolicy.VALUE))(integrate)
command(_TERM_EACH)(extract_input)
command(_MAP)(represent)

hidden("requires a Python callable and cannot be lowered losslessly")(apply)
hidden("registry provider used by evaluate help and validation")(
    available_evaluate_operators)

__all__ = ["interpolate", "local_poly", "select", "info", "integrate", "average",
    "eval_at_coord_proj",
    "plot", "animate", "plotly", "plotly_animate", "pyvista",
    "arithmetic", "represent", "apply",
    "fft", "magsq", "relchange", "mask", "collect", "sort", "grid", "val2coord",
    "extract_input", "fit", "differentiate", "evaluate", "available_evaluate_operators",
    "map", "growth", "gyrokinetics"]
