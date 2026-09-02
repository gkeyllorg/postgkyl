"""Module-level fluent verbs -- the multi-dataset verbs that have no single
``self``.

``collect``, ``evaluate``, ``relchange``, ``plot``, ``animate``, and
``plotly_animate`` each combine *several* datasets into one result (or, for
``plot``/``animate``/``plotly_animate``, into one figure/animation); ``sort``
reorders several datasets rather than combining them. None of these can be one dataset's method the
way ``interpolate``/``select``/``fft``/... are on
:class:`~postgkyl.gdata.gdata.GData`. Each is a direct alias to the canonical
callable exposed through :mod:`postgkyl.operations`, so the functional spelling
(``postgkyl.collect(a, b)``) and this module-level fluent spelling can never
drift apart. :class:`~postgkyl.gdata.gdatagroup.GDataGroup` re-uses these same
functions for its own ``sort``/``collect``/``evaluate``/``plot``/``animate``/
``plotly_animate`` methods.
"""

from __future__ import annotations

from postgkyl import operations


# Exact aliases: implementation, signature, annotations, docstring, and command
# metadata remain on the canonical callable (``render.plot`` owns ``plot``).
collect = operations.collect
sort = operations.sort
evaluate = operations.evaluate
relchange = operations.relchange
animate = operations.animate
plotly_animate = operations.plotly_animate
plot = operations.plot
