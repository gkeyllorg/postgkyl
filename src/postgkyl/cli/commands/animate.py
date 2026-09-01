"""Deprecated module path for the generated animation command."""

from postgkyl.cli.commands import command_named

command = command_named("animate")


def _group_by_frame(datasets: list) -> list[list]:
  """Compatibility helper grouping datasets by authoritative frame metadata.

  Datasets without a frame remain together in one trailing group. Readers
  and :class:`GDataState` own filename parsing, so this helper never attempts
  to infer a second frame value from path text.
  """
  groups: dict[int | None, list] = {}
  for dataset in datasets:
    frame = dataset.ctx.get("frame")
    groups.setdefault(int(frame) if frame is not None else None, []).append(dataset)
  # end
  known = sorted(frame for frame in groups if frame is not None)
  return [groups[frame] for frame in known] + (
      [groups[None]] if None in groups else [])
# end
