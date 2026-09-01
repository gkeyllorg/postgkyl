"""The ``animate`` verb -- terminal; hands a sequence of datasets to the
render backend's animation engine.

Mirrors ``operations/plot.py``: each modal dataset in the sequence is bridged
through its NumPy shadow (point-value forms plot directly; modal
coefficients refuse) via the shared ``_materialize.materialize_for_render``
before the frames reach :func:`postgkyl.render.animate.animate`.
"""

from __future__ import annotations

from postgkyl import render
from postgkyl.gdatastate import group_blocks, group_frames
from postgkyl.gdatastate.gdatastate import GDataState

from ._materialize import materialize_for_render


def animate(data, *, multiblock: bool = False, grouptags: bool = False,
    interval: int = 100, fixed_range: bool = True,
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
    multiblock: Force datasets with the same frame index into one frame.
    grouptags: Build a separate animation for each dataset tag.
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
  items = list(data)
  if grouptags and all(isinstance(item, GDataState) for item in items):
    import os

    tags: dict[str, list] = {}
    for item in items:
      tags.setdefault(item.tag, []).append(item)
    # end

    def suffixed(path, tag):
      if path is None:
        return None
      # end
      stem, extension = os.path.splitext(path)
      return f"{stem}_{tag}{extension}"
    # end

    return [animate(tagged, multiblock=multiblock, interval=interval,
        fixed_range=fixed_range, cutoffglobalrange=cutoffglobalrange,
        notitle=notitle, show=show, save=save,
        saveas=suffixed(saveas, tag), fps=fps, dpi=dpi,
        saveframes=suffixed(saveframes, tag), nproc=nproc, tmpdir=tmpdir)
        for tag, tagged in tags.items()]
  # end

  if all(isinstance(item, GDataState) for item in items):
    items = group_frames(items) if multiblock else group_blocks(items)
  # end

  frames = []
  for item in items:
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
