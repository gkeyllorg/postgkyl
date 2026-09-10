"""The ``info`` verb -- print/return summaries for one or more datasets."""

from __future__ import annotations

from postgkyl.gdatastate import flatten_datasets
from postgkyl.gdatastate.gdatastate import GDataState


def info(*datasets: GDataState, no_header: bool = False) -> list:
  """Print a summary for each dataset; return the list of summary strings.

  Accepts ``info(a, b)`` or ``info([a, b])``. Each dataset's own ``info`` method
  (a pure state reader on the container) does the formatting.

  Args:
    datasets: Datasets whose summaries are returned.
    no_header: Omit the descriptive heading from every summary.
  """
  states = flatten_datasets(datasets)
  return [d.info(index=i, no_header=no_header) for i, d in enumerate(states)]
