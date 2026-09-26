"""The ``mask`` verb -- mask out values by a mask dataset or by thresholds."""

from __future__ import annotations

from typing import Annotated
from postgkyl.cli_spec import DatasetRef

from postgkyl.gdatastate import materialize_point_values
from postgkyl.gdatastate.guards import require_field_domain

import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState


def mask(data: "GDataState",
         mask_data: Annotated[GDataState | None,
                              DatasetRef()] = None,
         *,
         lower: float | None = None,
         upper: float | None = None,
         inplace: bool = False,
         tag: str | None = None,
         label: str | None = None):
  """Mask out values using a mask dataset or numeric thresholds.

  Returns a dataset whose values are a ``numpy.ma`` masked array. Exactly
  one of the masking modes is applied, with ``mask_data`` taking precedence:

  - ``mask_data``: mask cells where the mask dataset's field is negative,
    repeated across ``data``'s components. Load the mask field yourself
    (e.g. ``pg.load(mask_path)``) -- this verb takes data, never a file path
    (``operations`` never touches ``io``).
  - ``lower`` and ``upper``: mask values outside the closed range
    ``[lower, upper]``.
  - ``lower`` only: mask values below ``lower``.
  - ``upper`` only: mask values above ``upper``.

  Args:
    data: the dataset to mask; must contain point values.
    mask_data: an already-loaded dataset whose field selects the mask
      (negative -> masked); it must have exactly one component -- the mask
      is broadcast across every component of ``data`` via
      ``np.repeat(mask_data.values, data.num_comps, axis=-1)``, which only
      produces a shape matching ``data.values`` when ``mask_data`` is
      single-component. A multi-component ``mask_data`` raises from the
      subsequent ``np.ma.masked_where`` broadcast, not from an explicit
      check here.
    lower: lower threshold. Combined with ``upper`` masks outside the range;
      alone masks values below it.
    upper: upper threshold. Combined with ``lower`` masks outside the range;
      alone masks values above it.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset whose values are a masked array.

  Raises:
    ValueError: if ``data`` is unevaluated modal coefficients, or if none of
      ``mask_data``, ``lower``, or ``upper`` is provided.
    IndexError: if ``mask_data`` has more than one component -- the
      repeated mask no longer matches ``data.values``'s shape and
      ``np.ma.masked_where`` rejects the mismatched condition array.
  """
  require_field_domain(data, "mask", "raw coefficients are not field values")
  data = materialize_point_values(data, inplace=inplace)
  values = data.values
  if mask_data is not None:
    mask_data = materialize_point_values(mask_data)
    mask_field = mask_data.values
    mask_rep = np.repeat(mask_field, data.num_comps, axis=-1)
    masked = np.ma.masked_where(mask_rep < 0.0, values)
  elif lower is not None and upper is not None:
    masked = np.ma.masked_outside(values, lower, upper)
  elif lower is not None:
    masked = np.ma.masked_less(values, lower)
  elif upper is not None:
    masked = np.ma.masked_greater(values, upper)
  else:
    raise ValueError(
        "mask: no masking information specified (provide mask_data, lower, "
        "or upper).")
  return data._result(data.grid, masked, inplace=inplace, tag=tag, label=label)
