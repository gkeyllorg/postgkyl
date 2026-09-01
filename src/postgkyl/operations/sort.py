"""The ``sort`` verb -- natural/numeric-order datasets by source filename."""

from __future__ import annotations

from postgkyl import numerics
from postgkyl.gdatastate import flatten_datasets
from postgkyl.gdatastate.gdatastate import GDataState


def sort(*datasets: GDataState, reverse: bool = False) -> list[GDataState]:
  """Reorder datasets by the natural/numeric sort of their source filename.

  Fixes the shell-glob/lexicographic-sort trap where ``field_10.gkyl`` sorts
  before ``field_2.gkyl``: digit runs embedded in the filename are compared
  as integers rather than character-by-character, so frame files come out in
  increasing frame order regardless of digit-count padding (see
  ``numerics.natural_sort_key``).

  Accepts ``sort(a, b)`` or ``sort([a, b])`` (flattened via
  ``gdatastate.flatten_datasets``). No dataset is copied or mutated -- only
  the returned list's order differs from the input.

  Args:
    *datasets: the datasets to reorder, or lists/groups thereof.
    reverse: sort in decreasing order instead of increasing.

  Returns:
    The same datasets, reordered by their source filename's natural sort key.
  """
  states = flatten_datasets(datasets)
  return sorted(states, key=lambda d: numerics.natural_sort_key(d.file_name),
      reverse=reverse)
# end
