"""Module-level fluent verbs -- the multi-dataset verbs that have no single
``self``.

``collect``, ``evaluate``, ``relchange``, ``plot``, ``animate``, and
``plotly_animate`` each combine *several* datasets into one result (or, for
``plot``/``animate``/``plotly_animate``, into one figure/animation); ``sort``
reorders several datasets rather than combining them. None of these can be one dataset's method the
way ``interpolate``/``select``/``fft``/... are on
:class:`~postgkyl.gdata.gdata.GData`. Each is a direct alias to the matching
:mod:`postgkyl.operations` verb, so the functional spelling
(``postgkyl.collect(a, b)``) and this module-level fluent spelling can never
drift apart. :class:`~postgkyl.gdata.gdatagroup.GDataGroup` re-uses these same
functions for its own ``sort``/``collect``/``evaluate``/``animate``/
``plotly_animate`` methods.
"""

from __future__ import annotations

from postgkyl import operations


# Exact operation aliases: implementation, signature, annotations, docstring,
# and command metadata all remain on the operation that owns them.
collect = operations.collect
sort = operations.sort
evaluate = operations.evaluate
relchange = operations.relchange
animate = operations.animate
plotly_animate = operations.plotly_animate


def plot(*datasets, **kwargs):
  """Draw several datasets onto **one** figure (see ``operations.plot``).

  The multi-dataset spelling of :meth:`postgkyl.gdata.gdata.GData.plot`. Used
  by :class:`~postgkyl.gdata.gdatagroup.GDataGroup` so a group -- most
  importantly a multiblock family, one field split across blocks -- renders
  as a single picture instead of broadcasting into one figure per member.
  """
  kwargs.setdefault("multiblock", True)
  return operations.plot(datasets, **kwargs)
# end
