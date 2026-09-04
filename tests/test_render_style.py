"""Tests for postgkyl.render.style -- apply_style."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from postgkyl.render.style import DEFAULT_STYLE, apply_style


@pytest.fixture(autouse=True)
def _restore_rcparams():
  with mpl.rc_context():
    yield


class TestApplyStyle:

  def test_default_applies_packaged_postgkyl_style(self):
    apply_style()
    assert mpl.rcParams["image.cmap"] == "inferno"
    assert mpl.rcParams["image.origin"] == "lower"

  def test_named_postgkyl_style_matches_default(self):
    apply_style(DEFAULT_STYLE)
    assert mpl.rcParams["image.cmap"] == "inferno"

  def test_cycler_line_is_parsed_by_matplotlib(self):
    apply_style()
    cycle = list(mpl.rcParams["axes.prop_cycle"])
    assert len(cycle) == 7

  def test_matplotlib_named_style_is_forwarded(self):
    apply_style("default")
    # "default" resets to Matplotlib's own baseline cmap.
    assert mpl.rcParams["image.cmap"] == "viridis"

  def test_arbitrary_mplstyle_path_is_applied(self, tmp_path):
    style_file = tmp_path / "custom.mplstyle"
    style_file.write_text("image.cmap: plasma\n")
    apply_style(str(style_file))
    assert mpl.rcParams["image.cmap"] == "plasma"

  def test_unknown_style_name_raises(self):
    with pytest.raises(OSError):
      apply_style("this-style-does-not-exist")
