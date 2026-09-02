"""The ``plotly_animate`` verb -- terminal; hands a sequence of datasets to the
Plotly render backend's animation engine.

Unlike the canonical Matplotlib ``render.animate`` callable, a frame may not
itself be a list of datasets drawn together: a Plotly animation frame is one
trace-set, so ``data`` is always a flat, one-dataset-per-frame sequence.
"""

from __future__ import annotations

from postgkyl import render

from postgkyl.gdatastate import materialize_point_values


def plotly_animate(data, *, frame_duration: int = 50,
    transition_duration: int = 0, fromcurrent: bool = True,
    redraw: bool = True, save: bool = False,
    saveas: str | None = None, show: bool = False):
  """Animate a flat sequence of datasets, one Plotly frame per dataset.

  Every dataset is bridged through
  :func:`postgkyl.gdatastate.materialize_point_values`
  first (see ``render.plotly.plotly_animate``). ``save``/``saveas``/``show``
  are handled entirely by the render layer, and default to inert -- pass
  ``show=True`` to open the animation in the browser, or
  ``save=True``/``saveas=...`` to write it. Returns the Plotly figure.

  Args:
    data: Selected datasets, one per animation frame.
    frame_duration: Duration of each animation frame in milliseconds.
    transition_duration: Duration of transitions between frames.
    fromcurrent: Start playback from the current slider frame.
    redraw: Redraw traces between frames.
    save: Save the animation to an automatically derived path.
    saveas: Explicit HTML output path.
    show: Open the animation in a browser.
  """
  frames = [materialize_point_values(dat) for dat in data]
  return render.plotly_animate(frames, frame_duration=frame_duration,
      transition_duration=transition_duration, fromcurrent=fromcurrent,
      redraw=redraw, save=save, saveas=saveas, show=show)
# end
