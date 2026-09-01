"""The ``animate`` verb -- terminal; hands a sequence of datasets to the
render backend's animation engine.

Mirrors ``operations/plot.py``: each modal dataset in the sequence is bridged
through its NumPy shadow (point-value forms plot directly; modal
coefficients refuse) via the shared ``_materialize.materialize_for_render``
before the frames reach :func:`postgkyl.render.animate.animate`.
"""

from __future__ import annotations

from postgkyl import render
from postgkyl.gdatastate.gdatastate import GDataState

from ._materialize import materialize_for_render


def animate(data, *, interval: int = 100, fixed_range: bool = True,
    cutoffglobalrange: float | None = None, notitle: bool = False,
    show: bool = True, save: bool = False, saveas: str | None = None,
    fps: int | None = None, dpi: int | None = None,
    saveframes: str | None = None, nproc: int = 1,
    tmpdir: str | None = None):
  """Animate a sequence of frames (see ``render.animate.animate``).

  ``data`` is a flat iterable of datasets (one dataset per frame) or an
  iterable of frames, where each frame is itself a list of datasets drawn
  together. Every dataset is bridged through
  :func:`_materialize.materialize_for_render` first, so the caller may
  freely mix modal and already-interpolated datasets.

  Args:
    data: Selected datasets, one per animation frame.
    interval: Live-animation delay in milliseconds.
    fixed_range: Hold a constant value range across frames.
    cutoffglobalrange: Central percentile band used for the fixed range.
    notitle: Suppress automatic frame titles.
    show: Display the animation interactively.
    save: Save the assembled animation.
    saveas: Animation output path.
    fps: Saved-animation frames per second.
    dpi: Saved-frame resolution.
    saveframes: Prefix for separately written PNG frames.
    nproc: Worker processes used to render frames.
    tmpdir: Parent directory for temporary rendered frames.
  """
  frames = []
  for item in data:
    if isinstance(item, GDataState):
      frames.append(materialize_for_render(item))
    # end
    else:
      frames.append([materialize_for_render(dat) for dat in item])
    # end
  # end
  return render.animate.animate(frames, interval=interval,
      fixed_range=fixed_range, cutoffglobalrange=cutoffglobalrange,
      notitle=notitle, show=show, save=save, saveas=saveas, fps=fps, dpi=dpi,
      saveframes=saveframes, nproc=nproc, tmpdir=tmpdir)
# end
