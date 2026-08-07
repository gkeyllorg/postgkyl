"""Helpers for collections of datasets (shared by the multi-dataset verbs).

Lives in ``gdatastate`` because it is generic plumbing over the container type and is
needed by both ``render`` (``pg.plot(a, b)``) and ``operations`` (``pg.info(a, b)``) --
both of which already depend on ``gdatastate``. Keeping it here avoids duplicating the
flatten in two layers or stranding it in the facade.
"""

from __future__ import annotations

from .gdatastate import GDataState


def flatten_datasets(items) -> list:
  """Flatten nested lists/tuples/groups of datasets into a single flat list.

  Lets the multi-dataset entry points accept either ``f(a, b)`` or ``f([a, b])``
  (and nested combinations, including a ``GDataStateGroup`` wherever a dataset is
  expected). Recursion is on any iterable, not just ``list``/``tuple`` -- this is
  what lets a nested ``gdatastate.gdatastategroup.GDataStateGroup`` flatten correctly without this
  module importing that one (it needs no type check, only that groups are
  iterable). Strings pass through whole (never iterated character-by-character);
  non-dataset, non-iterable items also pass through so the downstream consumer
  can raise a clear, contextual error.
  """
  out = []
  for it in items:
    if isinstance(it, GDataState):
      out.append(it)
    # end
    elif isinstance(it, (str, bytes)):
      out.append(it)
    # end
    elif hasattr(it, "__iter__"):
      out.extend(flatten_datasets(it))
    # end
    else:
      out.append(it)
    # end
  # end
  return out
# end


def _family_key(data) -> tuple | None:
  """The key identifying "the same field" across a multiblock decomposition,
  or ``None`` for a dataset that is not part of one.

  Built from the identity ``GDataState`` stamps at load time (``sim``,
  ``quantity``, ``frame`` -- see ``io.naming``) plus the dataset's ``tag``,
  so two differently-tagged results of the same source file (e.g. the raw
  load and a ``gk_rz`` projection of it) never merge. ``block`` is
  deliberately absent: it is what family members differ by.

  Returning ``None`` for single-block data is the property that keeps every
  pre-existing pipeline byte-identical -- with no ``_b<N>`` in the file
  names, every dataset is its own family.
  """
  if not isinstance(data, GDataState) or data.ctx.get("block") is None:
    return None
  # end
  return (data.tag, data.ctx.get("sim"), data.ctx.get("quantity"),
      data.ctx.get("frame"))
# end


def group_blocks(datasets) -> list[list]:
  """Partition datasets into **block families**: one field's blocks together.

  A family is the set of datasets that agree on ``(tag, sim, quantity,
  frame)`` and differ only in ``ctx["block"]`` -- i.e. the pieces of one
  field on a decomposed domain, which terminal verbs (``plot``, ``animate``)
  should treat as a single thing to draw. Datasets with no block index are
  each returned as their own singleton family, so single-block input maps
  1:1 onto the ungrouped list it was before.

  Args:
    datasets: Datasets, groups, or nested iterables of them (flattened via
      :func:`flatten_datasets`).

  Returns:
    A list of lists, in first-appearance order; each family is sorted by
    ascending block index.
  """
  families: dict = {}
  out: list[list] = []
  for data in flatten_datasets(datasets):
    key = _family_key(data)
    if key is None:
      out.append([data])
      continue
    # end
    if key not in families:
      families[key] = []
      out.append(families[key])
    # end
    families[key].append(data)
  # end
  for family in families.values():
    family.sort(key=lambda d: int(d.ctx["block"]))
  # end
  return out
# end
