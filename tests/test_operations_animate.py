"""Tests for the canonical ``render.animate`` callable through its exact
``operations.animate`` alias (mirrors the aliases of ``render.plot``)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import postgkyl as pg
from postgkyl import gpython, operations

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")
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


def _three_interpolated_frames():
  return [pg.load(F1D).interpolate().select(comp=c) for c in (0, 0, 0)]


class TestAnimateVerb:

  def test_already_interpolated_frames_pass_through(self):
    from matplotlib.animation import FuncAnimation
    anim = operations.animate(_three_interpolated_frames(), no_show=True)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 3

  def test_nodal_frames_are_materialized_first(self):
    """Native nodal data is materialized at its true point locations."""
    from matplotlib.animation import FuncAnimation
    a = pg.load(F1D).represent(to="nodal")
    b = pg.load(F1D).represent(to="nodal")
    anim = operations.animate([a, b], no_show=True)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 2

  def test_raw_modal_frames_plot_coefficients(self):
    import numpy as np
    a = pg.load(F1D)
    anim = operations.animate([a], no_show=True)
    anim._func(0, *anim._args)
    for comp, ax in enumerate(anim._fig.axes):
      np.testing.assert_array_equal(ax.lines[0].get_ydata(), a.values[:, comp])
    assert a.backend == "gkyl"
    assert a.ctx.get("value_form", "modal") == "modal"

  def test_grouped_frames_preserve_structure(self):
    from matplotlib.animation import FuncAnimation
    a = pg.load(F1D).interpolate()
    b = pg.load(F1D).interpolate()
    c = pg.load(F1D).interpolate()
    anim = operations.animate([[a, b], [c]], no_show=True)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 2

  def test_saveframes_end_to_end(self, tmp_path):
    prefix = str(tmp_path / "frame")
    paths = operations.animate(_three_interpolated_frames(),
                               saveframes=prefix,
                               no_show=True)
    assert len(paths) == 3
    for p in paths:
      assert os.path.isfile(p)
