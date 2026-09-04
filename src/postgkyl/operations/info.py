"""The ``info`` verb -- print/return summaries for one or more datasets."""

from __future__ import annotations

from postgkyl.gdatastate import flatten_datasets
from postgkyl.gdatastate.gdatastate import GDataState


def info(*datasets: GDataState, header: bool = True) -> list:
  """Print a summary for each dataset; return the list of summary strings.

  Accepts ``info(a, b)`` or ``info([a, b])``. Each dataset's own ``info`` method
  (a pure state reader on the container) does the formatting.

  Args:
    datasets: Datasets whose summaries are returned.
    header: Include the descriptive heading in every summary.
  """
  states = flatten_datasets(datasets)
  return [d.info(index=i, header=header) for i, d in enumerate(states)]
