"""Tests for the canonical ``render.animate`` callable through its exact
``operations.animate`` alias (mirrors the aliases of ``render.plot``)."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, operations
from postgkyl.gdatastate.gdatastate import GDataState

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
    reason="no compiled Gkeyll (libg0core.so) found")
pytestmark = needs_gkeyll

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
GEN = os.path.join(DATA, "generated")
F1D = os.path.join(GEN, "1d_ms_p1.gkyl")


@pytest.fixture(autouse=True)
def _close_figs():
  plt.close("all")
  yield
  plt.close("all")
# end


def _three_interpolated_frames():
  return [pg.load(F1D).interpolate().select(comp=c) for c in (0, 0, 0)]
# end


class TestAnimateVerb:
  def test_already_interpolated_frames_pass_through(self):
    from matplotlib.animation import FuncAnimation
    anim = operations.animate(_three_interpolated_frames(), show=False)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 3
  # end

  def test_modal_frames_are_materialized_first(self):
    """A raw (non-interpolated) modal dataset is bridged through its NumPy
    shadow (nodal value_form), just like ``render.plot``."""
    from matplotlib.animation import FuncAnimation
    a = pg.load(F1D).to_nodal()
    b = pg.load(F1D).to_nodal()
    anim = operations.animate([a, b], show=False)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 2
  # end

  def test_raw_modal_frame_without_representation_raises(self):
    a = pg.load(F1D)  # still modal coefficients
    with pytest.raises(ValueError, match="not plottable"):
      operations.animate([a], show=False)
    # end
  # end

  def test_grouped_frames_preserve_structure(self):
    from matplotlib.animation import FuncAnimation
    a = pg.load(F1D).interpolate()
    b = pg.load(F1D).interpolate()
    c = pg.load(F1D).interpolate()
    anim = operations.animate([[a, b], [c]], show=False)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 2
  # end

  def test_saveframes_end_to_end(self, tmp_path):
    prefix = str(tmp_path / "frame")
    paths = operations.animate(_three_interpolated_frames(), saveframes=prefix,
        show=False)
    assert len(paths) == 3
    for p in paths:
      assert os.path.isfile(p)
  # end
# end
    # end
