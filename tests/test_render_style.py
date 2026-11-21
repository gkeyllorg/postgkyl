"""Tests for postgkyl.render.style -- style_context."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import pytest

from postgkyl.render.style import DEFAULT_STYLE, style_context


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
