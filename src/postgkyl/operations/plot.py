"""The ``plot`` verb -- terminal; hands one or more datasets to the render backend.

Point-value forms (nodal/quad) plot **directly**: their values are
materialized at the true physical point locations (a non-uniform mesh whose
cell centers coincide with the points -- ``_materialize.materialize_for_render``),
then rendered by the unchanged backend. Modal data refuses: coefficients are
not plottable; the user chooses ``.interpolate()``, ``.to_nodal()``, or
``.to_quad()`` explicitly.

``plot(a)`` and ``plot(a, b, ...)`` both draw onto **one** figure -- the
multi-dataset spelling is what ``render.plot`` has always supported and what
makes a block family (one field on a decomposed domain, see
``gdatastate.collection.group_blocks``) render as a single picture rather than
one figure per block.
"""

from __future__ import annotations

from postgkyl import render
from postgkyl.gdatastate import flatten_datasets

from ._materialize import materialize_for_render


def plot(*datasets, **kwargs):
  """Render one or more datasets onto a single figure. Returns the figure.

  Args:
    *datasets: Datasets, groups, or nested iterables of them; every dataset
      is drawn onto the same figure (overlaid for 1-D, onto the same panels
      for 2-D -- exactly ``render.plot``'s contract).
    **kwargs: Forwarded verbatim to :func:`postgkyl.render.plot`.

  Raises:
    ValueError: if no datasets were given.
  """
  states = flatten_datasets(datasets)
  if not states:
    raise ValueError("nothing to plot")
  # end
  return render.plot(*[materialize_for_render(d) for d in states], **kwargs)
# end
