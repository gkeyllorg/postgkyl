"""The ``average`` verb -- weighted (or plain) average of a native DG field
over a subset of dimensions, via Gkeyll's ``gkyl_array_average``.

A full average is terminal and returns one physical mean per field. Partial
averaging preserves native modal data over the surviving dimensions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated
from postgkyl.cli_spec import CliType, DatasetRef

import numpy as np

from postgkyl import dg
from ._compatibility import average_operands, parse_axes

from postgkyl.gdatastate.gdatastate import GDataState


def average(data: "GDataState",
            dims: Annotated[int | Iterable[int] | str,
                            CliType(str)],
            *,
            weight: Annotated[GDataState | None,
                              DatasetRef()] = None,
            inplace: bool = False,
            tag: str | None = None,
            label: str | None = None):
  """``int f w dx^dims / int w dx^dims`` over the directions in ``dims``.

  Args:
    data: gkyl-backed (native modal) dataset in the modal value_form.
    dims: 0-based direction(s) to average over: an integer, an iterable of
      integers, a comma-separated string (``"0,1"``), or a colon slice
      string (``"0:2"``); ``--dims 0,1`` at the CLI.
    weight: optional gkyl-backed dataset in the modal value_form, same
      ``num_dims``/``basis_type``/``poly_order`` as ``data`` and exactly one
      field (``gkyl_array_average`` takes no field-index argument) -- the
      plain average (dividing by volume) is computed when omitted.
    inplace: Mutate and return ``data`` for partial averaging only.
    tag: Optional tag for a partial-average dataset.
    label: Optional label for a partial-average dataset.

  Returns:
    A float (one field) or NumPy array (multiple fields) containing the
    physical mean when every direction is averaged out. Otherwise a native
    modal dataset over the surviving dimensions. Dataset-only options
    ``inplace``, ``tag``, and ``label`` apply only to partial averaging.

  Raises:
    ValueError: ``data`` (or ``weight``) is NumPy-backed or non-modal, is
      missing basis metadata, or ``weight``'s grid/basis doesn't match
      ``data``'s, ``dims`` is empty, repeated, or out of range, or
      dataset-only options are used for a full average.
  """
  dims = parse_axes(dims, data.num_dims, "average")
  basis_type, poly_order, grid, weight_native = average_operands(
      data, weight, "average")
  keep_dirs, cells_avg, out_native = dg.modal.average(grid,
                                                      basis_type,
                                                      data.num_dims,
                                                      poly_order,
                                                      data.native,
                                                      dims,
                                                      weight=weight_native)

  if not keep_dirs:
    if inplace or tag is not None or label is not None:
      raise ValueError(
          "inplace, tag, and label apply only to partial averaging, which "
          "returns a dataset")
    # The DG layer returns a constant one-cell 1D modal field, including
    # for weighted averages. Reconstruct its physical mean: phi0=1/sqrt(2).
    coefficients = out_native.view().reshape(-1, poly_order + 1)
    means = coefficients[:, 0] / np.sqrt(2.0)
    return float(means[0]) if means.size == 1 else means

  new_grid = [np.asarray(data.grid[d]) for d in keep_dirs]

  return data._result(new_grid,
                      out_native,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      cells=np.asarray(cells_avg))
