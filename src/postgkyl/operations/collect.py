"""The ``collect`` verb -- combine many datasets into one along a new time axis."""

from __future__ import annotations

import numpy as np

from postgkyl.gdatastate import flatten_datasets
from postgkyl.gdatastate.gdatastate import GDataState


def _collect_group(states: list, start: int, *, sumdata: bool,
                   period: float | None, offset: float, tag: str | None,
                   label: str | None) -> GDataState:
  """Collect one group of frames into a single dataset (``collect``'s body,
  factored out so ``chunk`` can call it once per chunk). ``start`` is the
  group's offset into the full input sequence, so the positional-fallback
  time stamp (used when a frame has neither ``ctx['time']`` nor
  ``ctx['frame']``) stays consistent with the ungrouped (no-``chunk``) case."""
  time, values = [], []
  grid = None
  for i, dat in enumerate(states, start=start):
    if dat.backend == "gkyl":
      raise ValueError(
          f"collect operates on interpolated (NumPy) values; call .interpolate() "
          f"first on dataset {i} -- stacking raw DG coefficients would mix "
          f"basis functions.")
    stamp = dat.ctx.get("time", dat.ctx.get("frame", i))
    time.append(stamp)

    val = dat.values
    if sumdata:
      values.append(np.nansum(val, axis=tuple(range(dat.num_dims))))
    else:
      values.append(val)
    if grid is None:
      grid = list(dat.grid)

  time = np.array(time)
  values = np.array(values)

  if period:
    time = (time - offset) % period

  sort_idx = np.argsort(time)
  time = time[sort_idx]
  values = values[sort_idx]

  out_grid = [time] if sumdata else [np.array(time)] + grid
  return states[0]._result(out_grid,
                           values,
                           tag=(tag or "default"),
                           label=(label if label is not None else "collect"))


def collect(*datasets: GDataState,
            sumdata: bool = False,
            period: float | None = None,
            offset: float = 0.0,
            chunk: int | None = None,
            tag: str | None = None,
            label: str | None = None) -> GDataState | list[GDataState]:
  """Collect many single-frame datasets into one with a new leading time axis.

  Accepts ``collect(a, b)`` or ``collect([a, b])`` (flattened via
  ``gdatastate.flatten_datasets``). The per-dataset time stamp is taken from
  ``ctx['time']``, then ``ctx['frame']``, then the dataset's position in the
  sequence as a fallback; frames are sorted by their (possibly folded) time
  stamp. Each result copies the grid/ctx of its group's first frame (via
  ``_result``), so it stays the caller's concrete dataset class.

  Args:
    *datasets: the datasets to collect (each NumPy-backed, sharing a grid
      and component layout), or lists/groups thereof.
    sumdata: when True, sum each frame over all of its spatial axes (keeping
      components) before stacking, so the output grid is just the time
      axis. When False the full spatial data of each frame is retained and
      the time axis becomes a new leading dimension.
    period: when given, fold the time stamps into one period via
      ``(time - offset) % period`` before sorting, producing a phase/epoch
      axis instead of an unfolded time axis.
    offset: phase offset subtracted before the modulo when ``period`` is
      used.
    chunk: when given (and non-zero), split the input into consecutive
      groups of this length and collect each group separately, returning a
      list of datasets (one per chunk; the last chunk may be shorter)
      instead of a single dataset.
    tag: optional tag for the returned dataset(s).
    label: optional label for the returned dataset(s) (defaults to
      ``'collect'``).

  Returns:
    A single dataset with the collected frames stacked along a new leading
    time axis, or, when ``chunk`` is given, a list of such datasets.

  Raises:
    ValueError: if there are no datasets to collect, or one is native modal
      (gkyl-backed).
  """
  states = flatten_datasets(datasets)
  if not states:
    raise ValueError("collect: no datasets to collect.")

  if chunk:
    groups = [(states[i:i + chunk], i) for i in range(0, len(states), chunk)]
    return [
        _collect_group(group,
                       start,
                       sumdata=sumdata,
                       period=period,
                       offset=offset,
                       tag=tag,
                       label=label) for group, start in groups
    ]

  return _collect_group(states,
                        0,
                        sumdata=sumdata,
                        period=period,
                        offset=offset,
                        tag=tag,
                        label=label)
