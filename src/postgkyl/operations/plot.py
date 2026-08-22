"""The ``plot`` verb -- terminal; hands the dataset to the render backend.

Point-value forms (nodal/quad) plot **directly**: their values are
materialized at the true physical point locations (a non-uniform mesh whose
cell centers coincide with the points -- ``_materialize.materialize_for_render``),
then rendered by the unchanged backend. Modal data refuses: coefficients are
not plottable; the user chooses ``.interpolate()``, ``.to_nodal()``, or
``.to_quad()`` explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import render

from ._materialize import materialize_for_render

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


def plot(data: "GDataState", **kwargs):
  """Render a single dataset and return the Matplotlib figure.

  Pass ``save=True`` for an auto-named PNG or ``saveas=...`` for a PNG/PDF
  output path; the render backend performs the write.
  """
  return render.plot(materialize_for_render(data), **kwargs)
# end
