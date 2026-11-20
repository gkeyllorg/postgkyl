"""Tests for postgkyl.render.style -- style_context, pin_tick_label_sizes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from postgkyl.render.style import (DEFAULT_STYLE, pin_tick_label_sizes,
                                   style_context)


class TestStyleContext:

  def test_default_applies_packaged_postgkyl_style(self):
    with style_context():
      assert mpl.rcParams["image.cmap"] == "inferno"
      assert mpl.rcParams["image.origin"] == "lower"

  def test_named_postgkyl_style_matches_default(self):
    with style_context(DEFAULT_STYLE):
      assert mpl.rcParams["image.cmap"] == "inferno"

  def test_cycler_line_is_parsed_by_matplotlib(self):
    with style_context():
      cycle = list(mpl.rcParams["axes.prop_cycle"])
    assert len(cycle) == 7

  def test_matplotlib_named_style_is_forwarded(self):
    with style_context("default"):
      # "default" resets to Matplotlib's own baseline cmap.
      assert mpl.rcParams["image.cmap"] == "viridis"

  def test_arbitrary_mplstyle_path_is_applied(self, tmp_path):
    style_file = tmp_path / "custom.mplstyle"
    style_file.write_text("image.cmap: plasma\n")
    with style_context(str(style_file)):
      assert mpl.rcParams["image.cmap"] == "plasma"

  def test_global_rcparams_are_restored_on_exit(self):
    before = dict(mpl.rcParams)
    with style_context():
      pass
    assert dict(mpl.rcParams) == before

  def test_unknown_style_name_raises(self):
    with pytest.raises(OSError):
      with style_context("this-style-does-not-exist"):
        pass


def _styled_figure(projection, pin):
  """A figure built under the packaged style, plus its tick locations."""
  with style_context():
    fig = plt.figure(figsize=(3, 3))  # narrow enough for the font to matter
    ax = fig.add_subplot(projection=projection)
    x = np.linspace(0.03, 0.97, 20)
    if projection == "3d":
      ax.plot_surface(*np.meshgrid(x, x), np.outer(x, x))
    else:
      ax.plot(x, x)
    if pin:
      pin_tick_label_sizes(fig)
    fig.canvas.draw()
    return fig, _tick_locations(fig)


def _tick_locations(fig):
  return [
      axis.get_ticklocs().tolist() for ax in fig.axes
      for axis in (ax.xaxis, ax.yaxis, getattr(ax, "zaxis", None))
      if axis is not None
  ]


@pytest.mark.parametrize("projection", [None, "3d"])
class TestPinTickLabelSizes:
  """A figure drawn after its style exits keeps the styled tick spacing."""

  def test_unpinned_figure_respaces_its_ticks_outside_the_style(
      self, projection):
    # The control: without pinning, the scenario below really does drift.
    fig, styled = _styled_figure(projection, pin=False)
    fig.canvas.draw()
    assert _tick_locations(fig) != styled
    plt.close(fig)

  def test_pinned_figure_keeps_its_ticks_outside_the_style(self, projection):
    fig, styled = _styled_figure(projection, pin=True)
    fig.canvas.draw()
    assert _tick_locations(fig) == styled
    plt.close(fig)
