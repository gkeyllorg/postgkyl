"""Print stored values or grid coordinates for one or more datasets."""

from __future__ import annotations

import builtins

import numpy as np

from postgkyl.gdatastate import flatten_datasets
from postgkyl.gdatastate.gdatastate import GDataState


def print(*datasets: GDataState,
          use: str | None = None,
          grid: bool = False) -> None:
  """Print the values (or grid) of the selected datasets.

  Values are printed as stored, with singleton dimensions squeezed and
  16-digit precision. For modal data these are DG coefficients; interpolate
  first to print field values. Printing leaves the datasets unchanged.

  Args:
    datasets: Datasets, groups, or nested iterables of datasets to print.
    use: Select only datasets carrying this tag.
    grid: Print each grid axis instead of the values.
  """
  for data in flatten_datasets(datasets):
    if use is not None and data.tag != use:
      continue
    arrays = data.grid if grid else (np.asarray(data.values).squeeze(), )
    for array in arrays:
      builtins.print(np.array2string(array, precision=16))
