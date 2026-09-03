"""Postgkyl module for testing the plotting function"""
import os
import matplotlib as mpl
import numpy as np

import postgkyl as pg
from postgkyl.commands.plot import _get_frames


class _FakeData:
  """Minimal stand-in for GData exposing just what '_get_frames' looks at."""

  def __init__(self, frame=None, file_name=None):
    self.ctx = {} if frame is None else {"frame": frame}
    self._file_name = file_name

  def get_file_name(self):
    return self._file_name


class TestGetFrames:
  """Test the frame detection used to color 1D curves by frame number."""

  def test_from_metadata(self):
    datasets = [_FakeData(frame=f, file_name="elc_M0_0.gkyl") for f in (0, 5, 10)]
    assert _get_frames(datasets) == [0, 5, 10]

  def test_from_file_name_when_metadata_missing(self):
    datasets = [_FakeData(file_name=f"elc_M0_{f:d}.gkyl") for f in (0, 5, 10)]
    assert _get_frames(datasets) == [0, 5, 10]

  def test_from_file_name_when_metadata_degenerate(self):
    # Copied files keep the frame of the original in their metadata
    datasets = [_FakeData(frame=5, file_name=f"elc_M0_{f:d}.gkyl") for f in (0, 5)]
    assert _get_frames(datasets) == [0, 5]

  def test_none_when_unusable(self):
    # Same frame, different species; neither source can discriminate
    datasets = [_FakeData(frame=5, file_name=f"{s:s}_M0_5.gkyl") for s in ("elc", "ion")]
    assert _get_frames(datasets) is None

    # Synthesized data has neither a frame nor a file name
    assert _get_frames([_FakeData(), _FakeData()]) is None

class TestPlot:
  """Test Postgkyl plot function.

  Currently, this tests if plots look OK only to some extend (by checking plotted
  values) and mostly tests if plots are created at all. Testing images themselves is
  complicated and differs based on system and/or backend used.
  """
  dir_path = f"{os.path.dirname(__file__)}/test_data"

  def test_plot_pcolormesh(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    img = pg.output.plot(data)
    assert isinstance(img, mpl.collections.QuadMesh)
    mpl.pyplot.close("all")

  def test_plot_contour(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    img = pg.output.plot(data, contour=True)
    assert isinstance(img, mpl.contour.QuadContourSet)
    mpl.pyplot.close("all")

  def test_plot_contour_options(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    img = pg.output.plot(data, contour=True, cnlevels=5, cont_label=True)
    assert isinstance(img, mpl.contour.QuadContourSet)
    mpl.pyplot.close("all")

  def test_plot_line(self):
    data = pg.GData(f"{self.dir_path:s}/twostream-field-energy.gkyl")
    img = pg.output.plot(data)
    assert isinstance(img[0], mpl.lines.Line2D)
    mpl.pyplot.close("all")

    pg.data.select(data, comp=0, overwrite=True)
    img = pg.output.plot(data)
    x_plot, y_plot = img[0].get_xydata().T
    np.testing.assert_array_almost_equal(data.get_grid()[0], x_plot)
    np.testing.assert_array_almost_equal(data.get_values()[...,0], y_plot)
    mpl.pyplot.close("all")

  def test_plot_line_cval(self):
    """Overlaid 1D curves are colored by 'cval' and share a single colorbar."""
    data = pg.GData(f"{self.dir_path:s}/twostream-field-energy.gkyl")
    pg.data.select(data, comp=0, overwrite=True)

    cmap = mpl.pyplot.get_cmap("viridis")
    frames = [0, 5, 10]
    fig = mpl.pyplot.figure()
    for frame in frames:
      img = pg.output.plot(data, figure=fig, cmap="viridis", cval_label="frame",
          cval=float(frame), cval_min=float(frames[0]), cval_max=float(frames[-1]))
      expected = cmap((frame - frames[0]) / (frames[-1] - frames[0]))
      np.testing.assert_array_almost_equal(img[0].get_color(), expected)
    # end

    # One extra axes holding the colorbar, no matter how many curves were drawn
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "frame"
    mpl.pyplot.close("all")

  def test_plot_line_cval_ignored_with_color(self):
    """An explicit 'color' wins over the colormap-based coloring."""
    data = pg.GData(f"{self.dir_path:s}/twostream-field-energy.gkyl")
    pg.data.select(data, comp=0, overwrite=True)

    fig = mpl.pyplot.figure()
    img = pg.output.plot(data, figure=fig, cmap="viridis", color="red",
        cval=1.0, cval_min=0.0, cval_max=2.0)
    assert img[0].get_color() == "red"
    assert len(fig.axes) == 1
    mpl.pyplot.close("all")