"""Deprecated module path for the generated animation command."""

from postgkyl.cli.commands import command_named

command = command_named("animate")


def _group_by_frame(datasets: list) -> list[list]:
  """Compatibility helper grouping datasets by their frame metadata."""
  missing = [dataset for dataset in datasets if dataset.ctx.get("frame") is None]
  if missing:
    files = [dataset.file_name for dataset in missing]
    shortest = min(files, key=len)
    diverge_at = len(shortest)
    for file_name in files:
      for index, pair in enumerate(zip(shortest, file_name)):
        if pair[0] != pair[1]:
          diverge_at = min(diverge_at, index)
          break
        # end
      # end
    # end
    for dataset, file_name in zip(missing, files):
      stem = file_name.rsplit(".gkyl", 1)[0]
      dataset.ctx["frame"] = int(stem[diverge_at:].split("_")[0])
    # end
  # end
  groups = {}
  for dataset in datasets:
    groups.setdefault(int(dataset.ctx["frame"]), []).append(dataset)
  # end
  return [groups[key] for key in sorted(groups)]
# end
