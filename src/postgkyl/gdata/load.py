"""``pg.load`` -- load one file or a glob of files into the fluent API."""

from __future__ import annotations

from glob import glob, has_magic
from typing import Annotated, Literal

from postgkyl import operations
from postgkyl.cli_spec import (
    CommandSpec, Execution, KeyValue, Section, command,
)
from postgkyl.gdata.gdata import GData
from postgkyl.gdata.gdatagroup import GDataGroup


@command(CommandSpec(Section.UTILITY, Execution.LOAD))
def load(file_name: str, *, tag: str = "default", label: str = "",
    ctx: Annotated[dict[str, str] | None, KeyValue()] = None,
    value_form: Literal["modal", "nodal", "quad"] | None = None,
    basis_type: str | None = None, poly_order: int | None = None,
    z0: str | None = None, z1: str | None = None, z2: str | None = None,
    z3: str | None = None, z4: str | None = None, z5: str | None = None,
    component: str | None = None,
    read_options: Annotated[dict[str, str] | None, KeyValue()] = None,
    ) -> GData | GDataGroup:
  """Read Gkeyll output into a fluent ``GData`` or ``GDataGroup``.

  ``pg.load('elc_M0_0.gkyl').interpolate().select(z0=0.0).plot()``

  Shell-style glob patterns (``*``, ``?``, and ``[]``) load every matching
  file into a :class:`GDataGroup`, naturally ordered by filename. The group
  broadcasts per-dataset verbs and supplies the multi-dataset ``collect``
  verb, so a frame series can be loaded and stacked in one chain::

      pg.load('elc_M0_*.gkyl').interpolate().collect().plot()

  A pattern always returns a group, even if it matches only one file. A
  literal filename retains the original single-``GData`` return type.

  ``basis_type``, ``poly_order``, and ``value_form`` are properties of the
  data itself, fixed here at load time (from the file's header metadata, or
  the override below) -- no downstream verb (``interpolate``, ``average``,
  ...) ever re-specifies them; they always read ``ctx["basis_type"]``/
  ``ctx["poly_order"]``/``ctx["value_form"]`` off the loaded dataset.

  ``value_form`` overrides the ``"modal"``/``"nodal"``/``"quad"`` tag the
  file's header metadata would otherwise imply -- for files whose writer
  stamps DG basis metadata even though the stored values are already point
  values (e.g. a per-cell diagnostic like a CFL rate), not modal coefficients.

  ``basis_type`` overrides the ``"basis_type"`` (e.g. ``"serendipity"``,
  ``"tensor"``, ``"gkhybrid"``) the file's header metadata would otherwise
  imply -- for files with no basis metadata at all, or metadata that
  mislabels the basis actually used. Setting it also defaults ``value_form``
  to ``"modal"`` (unless ``value_form`` is given too), so downstream verbs
  that read ``ctx["basis_type"]`` resolve the right basis.

  ``poly_order`` overrides the ``"poly_order"`` the file's header metadata
  would otherwise imply. It is independent of ``basis_type``/``value_form`` --
  passing it alone corrects only the polynomial order and asserts nothing
  about whether the dataset is modal.

  Args:
    file_name: Literal filename or shell-style glob pattern to load.
    tag: Tag assigned to every loaded dataset.
    label: Optional display label assigned to every loaded dataset.
    ctx: Initial metadata as repeated key/value entries.
    value_form: Stored representation of the loaded values.
    basis_type: DG basis name overriding file metadata.
    poly_order: Polynomial order overriding file metadata.
    z0: Partial-load selector for coordinate direction 0.
    z1: Partial-load selector for coordinate direction 1.
    z2: Partial-load selector for coordinate direction 2.
    z3: Partial-load selector for coordinate direction 3.
    z4: Partial-load selector for coordinate direction 4.
    z5: Partial-load selector for coordinate direction 5.
    component: Partial-load component selector.
    read_options: Reader-specific options as repeated key/value entries.
  """
  file_name = str(file_name)
  read_kwargs = dict(read_options or {})
  axes = (z0, z1, z2, z3, z4, z5)
  if any(value is not None for value in axes):
    read_kwargs["axes"] = axes
  # end
  if component is not None:
    read_kwargs["comp"] = component
  # end
  if has_magic(file_name):
    matches = glob(file_name)
    if not matches:
      raise FileNotFoundError(f"No files match pattern: '{file_name}'")
    # end
    datasets = [GData(match, tag=tag, label=label, ctx=ctx,
        value_form=value_form, basis_type=basis_type,
        poly_order=poly_order, **read_kwargs) for match in matches]
    return GDataGroup(operations.sort(datasets))
  # end

  return GData(file_name, tag=tag, label=label, ctx=ctx,
      value_form=value_form, basis_type=basis_type,
      poly_order=poly_order, **read_kwargs)
# end
