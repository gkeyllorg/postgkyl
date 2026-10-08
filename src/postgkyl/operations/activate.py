"""Pick datasets by tag; on the command line, choose the working set."""

from __future__ import annotations

from postgkyl.gdatastate import flatten_datasets
from postgkyl.gdatastate.gdatastate import GDataState


def activate(*datasets: GDataState,
             tags: list[str] | None = None) -> list[GDataState]:
  """Keep the datasets carrying one of ``tags``.

  On the command line the kept datasets become the working set and the
  others are set aside: later commands skip them, while an option naming a
  dataset by tag (such as ``average --weight``) still finds them. A bare
  ``activate`` makes every loaded dataset active again::

      pgkyl jacobian.gkyl --tag weight field.gkyl activate --tags default \\
          average --dims 1 --weight weight plot

  Accepts ``activate(a, b)`` or ``activate([a, b])`` (flattened via
  ``gdatastate.flatten_datasets``). No dataset is copied or mutated.

  Args:
    *datasets: The datasets to choose from, or lists/groups thereof.
    tags: Tags to keep (repeat ``--tags`` at the CLI); every dataset when
      omitted or empty.

  Returns:
    The datasets carrying one of ``tags``, in their given order.
  """
  states = flatten_datasets(datasets)
  if not tags:
    return states
  return [data for data in states if data.tag in tags]
