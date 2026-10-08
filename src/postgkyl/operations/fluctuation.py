"""The ``fluctuation`` verb -- a native DG field minus its weighted (or plain)
average over a subset of dimensions, on the field's own grid and basis.

The average comes from Gkeyll's ``gkyl_array_average`` (as in ``average``);
it is lifted back onto the full basis, constant along the averaged
directions, and subtracted coefficient by coefficient.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated
from postgkyl.cli_spec import CliType, DatasetRef

from postgkyl import dg
from ._compatibility import average_operands, parse_axes

from postgkyl.gdatastate.gdatastate import GDataState


def fluctuation(data: "GDataState",
                dims: Annotated[int | Iterable[int] | str,
                                CliType(str)],
                *,
                weight: Annotated[GDataState | None,
                                  DatasetRef()] = None,
                inplace: bool = False,
                tag: str | None = None,
                label: str | None = None):
  """``f - <f>_dims``: the field minus its average over the directions in
  ``dims``, where ``<f>_dims = int f w dx^dims / int w dx^dims``.

  Unlike ``average``, the result keeps every dimension of ``data``: the
  lower-dimensional average is lifted back onto ``data``'s basis, constant
  along ``dims``, and subtracted. Averaging the result over the same
  ``dims`` with the same ``weight`` gives zero.

  Args:
    data: gkyl-backed (native modal) dataset in the modal value_form.
    dims: 0-based direction(s) to average over, in ``average``'s grammar:
      an integer, an iterable of integers, ``"0,1"``, or ``"0:2"``;
      ``--dims 0,1`` at the CLI.
    weight: optional gkyl-backed dataset in the modal value_form, same
      ``num_dims``/``basis_type``/``poly_order`` as ``data`` and exactly one
      field (``gkyl_array_average`` takes no field-index argument) -- the
      plain average (dividing by volume) is subtracted when omitted.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A native modal dataset on ``data``'s grid, cells, and basis holding the
    fluctuation of every field.

  Raises:
    ValueError: ``data`` (or ``weight``) is NumPy-backed or non-modal, is
      missing basis metadata, ``weight``'s grid/basis doesn't match
      ``data``'s, or ``dims`` is empty, repeated, or out of range.
  """
  ndim = data.num_dims
  dims = parse_axes(dims, ndim, "fluctuation")
  basis_type, poly_order, grid, weight_native = average_operands(
      data, weight, "fluctuation")
  _keep_dirs, _cells_avg, mean = dg.modal.average(grid,
                                                  basis_type,
                                                  ndim,
                                                  poly_order,
                                                  data.native,
                                                  dims,
                                                  weight=weight_native)
  lifted = dg.modal.lift(basis_type, ndim, poly_order, mean, dims,
                         grid["cells"])
  out = dg.modal.lincomb(1.0, data.native, -1.0, lifted)
  return data._result(data.grid, out, inplace=inplace, tag=tag, label=label)
