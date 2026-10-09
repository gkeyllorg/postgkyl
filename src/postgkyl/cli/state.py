"""Shared CLI state -- the chained pipeline's scratch space (``ctx.obj``)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataSpace:
  """Datasets flowing through a chained command line.

  ``datasets`` is the working set every generated verb transforms;
  ``set_aside`` holds datasets ``activate`` left out, which options naming a
  dataset by tag still find.
  """

  datasets: list = field(default_factory=list)
  set_aside: list = field(default_factory=list)

  def __iter__(self):
    return iter(self.datasets)
