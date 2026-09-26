"""The ``info`` verb -- print/return summaries for one or more datasets."""

from __future__ import annotations

from postgkyl.gdatastate import flatten_datasets
from postgkyl.gdatastate.gdatastate import GDataState


def info(*datasets: GDataState,
         no_header: bool = False,
         all: bool = False) -> list:
  """Print a summary for each dataset; return the list of summary strings.

  Accepts ``info(a, b)`` or ``info([a, b])``. Each dataset's own ``info`` method
  (a pure state reader on the container) does the formatting.
  Summaries show current state and assumptions made at load time. With ``all``,
  include original Gkeyll file metadata, explicit load options, and identity
  inferred from the filename.

  Args:
    datasets: Datasets whose summaries are returned.
    no_header: Omit the descriptive heading from every summary.
    all: Include all source metadata and explicit load options.
  """
  states = flatten_datasets(datasets)
  return [
      d.info(index=i, no_header=no_header, all=all)
      for i, d in enumerate(states)
  ]
