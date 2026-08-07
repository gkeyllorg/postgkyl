"""Module-level fluent verbs -- the multi-dataset verbs that have no single
``self``.

``collect``, ``evaluate``, ``relchange``, ``plot``, ``animate``, and
``plotly_animate`` each combine *several* datasets into one result (or, for
``plot``/``animate``/``plotly_animate``, into one figure/animation); ``sort``
reorders several datasets rather than combining them. None of these can be one dataset's method the
way ``interpolate``/``select``/``fft``/... are on
:class:`~postgkyl.gdata.gdata.GData`. Each is a one-line delegation to the
matching :mod:`postgkyl.operations` verb, so the functional spelling
(``postgkyl.collect(a, b)``) and this module-level fluent spelling can never
drift apart. :class:`~postgkyl.gdata.gdatagroup.GDataGroup` re-uses these same
functions for its own ``collect``/``evaluate``/``animate``/``plotly_animate``
terminal methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import operations

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


def collect(*datasets: "GDataState", sumdata: bool = False,
    period: float | None = None, offset: float = 0.0, chunk: int | None = None,
    tag: str | None = None, label: str | None = None):
  """Combine many single-frame datasets into one with a new time axis.

  See ``operations.collect``. Accepts ``collect(a, b)`` or ``collect([a, b])``.
  Returns a single dataset, or a list of datasets when ``chunk`` is given.
  """
  return operations.collect(*datasets, sumdata=sumdata, period=period, offset=offset,
      chunk=chunk, tag=tag, label=label)
# end


def sort(*datasets: "GDataState", reverse: bool = False) -> list:
  """Reorder datasets by the natural/numeric sort of their source filename.

  See ``operations.sort``. Accepts ``sort(a, b)`` or ``sort([a, b])``.
  """
  return operations.sort(*datasets, reverse=reverse)
# end


def evaluate(chain: str, *datasets: "GDataState", tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Evaluate an RPN math expression over an explicit list of datasets.

  See ``operations.evaluate``. ``f``/``fN`` tokens in ``chain`` refer to ``datasets[N]``.
  """
  return operations.evaluate(chain, *datasets, tag=tag, label=label)
# end


def relchange(data0: "GDataState", data: "GDataState", *,
    comp: int | str | None = None, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Relative change of ``data`` with respect to the baseline ``data0``.

  See ``operations.relchange``. Returned dataset is built from ``data`` (its
  class propagates, not ``data0``'s).
  """
  return operations.relchange(data0, data, comp=comp, inplace=inplace, tag=tag,
      label=label)
# end


def plot(*datasets, **kwargs):
  """Draw several datasets onto **one** figure (see ``operations.plot``).

  The multi-dataset spelling of :meth:`postgkyl.gdata.gdata.GData.plot`. Used
  by :class:`~postgkyl.gdata.gdatagroup.GDataGroup` so a group -- most
  importantly a multiblock family, one field split across blocks -- renders
  as a single picture instead of broadcasting into one figure per member.
  """
  return operations.plot(*datasets, **kwargs)
# end


def animate(*datasets, **kwargs):
  """Animate a sequence of datasets, one frame per dataset.

  See ``operations.animate``. Each positional argument is a frame; a frame may
  itself be a list of datasets drawn together (mirrors ``operations.animate``'s
  "flat iterable, or iterable of frames" contract).
  """
  return operations.animate(datasets, **kwargs)
# end


def plotly_animate(*datasets, **kwargs):
  """Animate a sequence of datasets with Plotly, one frame per dataset.

  See ``operations.plotly_animate``. Called with no further arguments this
  just builds and returns the figure; pass ``show=True`` to open the
  animation in the browser, or ``save=True``/``saveas=...`` to write it
  instead -- no CLI glue needed either way.
  """
  return operations.plotly_animate(datasets, **kwargs)
# end
