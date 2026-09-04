"""The ``val2coord`` verb -- build new datasets from columns of a DynVector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from postgkyl.gdatastate.gdatastategroup import GDataStateGroup

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def _get_range(str_in: str, length: int) -> np.ndarray:
  """Parse a comma list, a ``lo:hi[:step]`` slice, or a bare int into indices.

  Pure array/string logic, no ``GData`` coupling; kept local to this verb
  (its grammar -- an optional step -- differs from ``numerics.idx_parser``'s
  slice grammar, and it has no other caller).
  """
  if len(str_in.split(",")) > 1:
    return np.array(str_in.split(","), dtype=int)
  elif str_in.find(":") >= 0:
    parts = str_in.split(":")
    s_idx = 0 if parts[0] == "" else int(parts[0])
    if s_idx < 0:
      s_idx = length + s_idx
    e_idx = length if parts[1] == "" else int(parts[1])
    if e_idx < 0:
      e_idx = length + e_idx
    inc = int(parts[2]) if len(parts) > 2 and parts[2] != "" else 1
    return np.arange(s_idx, e_idx, inc)
  else:
    return np.array([int(str_in)])


def val2coord(data: "GDataState",
              *,
              x: str,
              y: str,
              periodic: bool = False,
              tag: str | None = None,
              label: str | None = None) -> GDataStateGroup:
  """Build new (x, y) datasets from columns of a DynVector.

  Reinterprets columns of ``data`` (typically a DynVector / diagnostic
  table) as plot-ready datasets: the ``x`` column(s) become the grid and the
  ``y`` column(s) become the values. One output dataset is produced per
  selected y-component. When more than one x-component is selected, their
  count must match the number of y-components (paired one-to-one); a single
  x-component is shared across all y-components.

  Args:
    data: the source dataset whose last-axis columns are selected; must be
      NumPy-backed.
    x: component selector for the independent variable: an integer index, a
      comma-separated list (e.g. ``'0,2'``), or a ``'lo:hi[:step]'`` slice.
    y: component selector for the dependent variable(s); same forms as
      ``x``. One output dataset is produced per selected y-component.
    periodic: when True, append the first sample to the end of each output
      (wrapping) so periodic data closes on itself.
    tag: optional tag for the returned datasets.
    label: optional label for the returned datasets.

  Returns:
    A ``GDataStateGroup`` containing one dataset per selected y-component.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed), or if more than
      one x-component is selected and their number does not equal the
      number of y-components.
  """
  if data.backend == "gkyl":
    raise ValueError(
        "val2coord operates on interpolated (NumPy) values; call .interpolate() "
        "first -- raw DG coefficients are not tabular columns.")
  values = data.values
  x_comps = _get_range(x, values.shape[-1])
  y_comps = _get_range(y, values.shape[-1])

  if len(x_comps) > 1 and len(x_comps) != len(y_comps):
    raise ValueError(
        f"val2coord: number of x-components ({len(x_comps):d}) is greater "
        f"than 1 and not equal to the number of y-components "
        f"({len(y_comps):d}).")

  out = []
  for i, yc in enumerate(y_comps):
    xc = x_comps[i] if len(x_comps) > 1 else x_comps[0]
    xv = values[..., xc]
    yv = values[..., yc]
    if periodic:
      xv = np.append(xv, np.atleast_1d(xv[0]), axis=0)
      yv = np.append(yv, np.atleast_1d(yv[0]), axis=0)
    res = data._result([xv], yv[..., np.newaxis], tag=tag, label=label)
    res.color = "C0"
    out.append(res)
  return GDataStateGroup(out)
