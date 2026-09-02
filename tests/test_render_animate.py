"""Tests for postgkyl.render.animate -- FuncAnimation / saved frames / movie
compile.

Builds frames directly as ``GDataState`` (no shim dependency needed for the
render-layer tests. ``ffmpeg``-dependent tests are skipped cleanly when it is
not on ``PATH``.
"""

from __future__ import annotations

import os
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.render import _ffmpeg
anim_mod = import_module("postgkyl.render.animate")

needs_ffmpeg = pytest.mark.skipif(_ffmpeg.resolve_ffmpeg() is None,
    reason="ffmpeg not found on PATH or via imageio-ffmpeg")


def _line_frame(offset: float) -> GDataState:
  d = GDataState()
  d.ctx["frame"] = int(offset)
  d.ctx["time"] = float(offset) * 0.1
  d.push([np.linspace(0.0, 1.0, 9)], (np.arange(8, dtype=float) + offset)[:, None])
  return d
# end


def _three_frames() -> list[GDataState]:
  return [_line_frame(0.0), _line_frame(1.0), _line_frame(2.0)]
# end


@pytest.fixture(autouse=True)
def _close_figs():
  plt.close("all")
  yield
  plt.close("all")
# end


# --------------------------------------------------------------------------
# frame normalization
# --------------------------------------------------------------------------

class TestNormalizeFrames:
  def test_bare_datasets_become_single_dataset_frames(self):
    frames = anim_mod._normalize_frames(_three_frames())
    assert len(frames) == 3
    assert all(len(f) == 1 for f in frames)
  # end

  def test_grouped_frames_kept_as_lists(self):
    grouped = [[_line_frame(0.0), _line_frame(0.5)], [_line_frame(1.0)]]
    frames = anim_mod._normalize_frames(grouped)
    assert len(frames) == 2
    assert len(frames[0]) == 2
    assert len(frames[1]) == 1
  # end

  def test_empty_input_raises(self):
    with pytest.raises(ValueError, match="no datasets"):
      anim_mod._normalize_frames([])
    # end
  # end
# end


# --------------------------------------------------------------------------
# fixed value range
# --------------------------------------------------------------------------

class TestFrameValueRange:
  def test_spans_every_frame(self):
    frames = anim_mod._normalize_frames(_three_frames())
    vmin, vmax = anim_mod._frame_value_range(frames)
    assert vmin == 0.0
    assert vmax == 9.0  # last frame: arange(8) + 2.0 -> max 9.0
  # end

  def test_cutoff_clips_the_range(self):
    frames = anim_mod._normalize_frames(_three_frames())
    vmin_full, vmax_full = anim_mod._frame_value_range(frames)
    vmin_cut, vmax_cut = anim_mod._frame_value_range(frames, cutoff=0.5)
    assert vmin_cut >= vmin_full
    assert vmax_cut <= vmax_full
  # end

  def test_scale_is_applied_before_taking_extrema(self):
    # A fixed range computed on unscaled values would not match what
    # matplotlib.plot actually draws once yscale/zscale is applied.
    frames = anim_mod._normalize_frames(_three_frames())
    vmin, vmax = anim_mod._frame_value_range(frames, yscale=2.0)
    assert vmin == 0.0
    assert vmax == 18.0  # last frame: (arange(8) + 2.0).max() * 2.0
  # end
# end


# --------------------------------------------------------------------------
# live FuncAnimation path
# --------------------------------------------------------------------------

class TestLiveAnimation:
  def test_returns_funcanimation_with_correct_frame_count(self):
    from matplotlib.animation import FuncAnimation
    anim = anim_mod.animate(_three_frames(), show=False)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 3
  # end

  def test_grouped_frames_overlay_per_frame(self):
    from matplotlib.animation import FuncAnimation
    grouped = [[_line_frame(0.0), _line_frame(0.5)],
               [_line_frame(1.0), _line_frame(1.5)]]
    anim = anim_mod.animate(grouped, show=False)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 2
  # end

  def test_multiblock_groups_equal_frame_indices(self):
    from matplotlib.animation import FuncAnimation
    grouped = [_line_frame(0.0), _line_frame(0.5)]
    anim = anim_mod.animate(grouped, multiblock=True, show=False)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 1
  # end

  def test_grouptags_builds_one_animation_per_tag(self):
    from matplotlib.animation import FuncAnimation
    frames = _three_frames()
    frames[0].tag = "left"
    frames[1].tag = "left"
    frames[2].tag = "right"
    animations = anim_mod.animate(frames, grouptags=True, show=False)
    assert len(animations) == 2
    assert all(isinstance(anim, FuncAnimation) for anim in animations)
    assert [anim._save_count for anim in animations] == [2, 1]
  # end

  def test_show_true_does_not_raise_on_agg(self):
    anim = anim_mod.animate(_three_frames(), show=True)
    assert anim is not None
  # end

  @needs_ffmpeg
  def test_live_animation_saves_mp4(self, tmp_path):
    out = tmp_path / "live.mp4"
    anim = anim_mod.animate(_three_frames(), save=True, saveas=str(out),
        fps=5, show=False)
    assert anim is not None
    assert out.exists()
    assert out.stat().st_size > 0
  # end

  def test_notitle_suppresses_frame_time_title(self):
    fig = plt.figure()
    anim_mod._render_frame(0, anim_mod._normalize_frames(_three_frames()), fig,
        {"notitle": True})
    assert fig._suptitle is None
  # end

  def test_title_includes_frame_and_time_by_default(self):
    fig = plt.figure()
    anim_mod._render_frame(1, anim_mod._normalize_frames(_three_frames()), fig, {})
    assert "frame: 1" in fig._suptitle.get_text()
    assert "time:" in fig._suptitle.get_text()
  # end

  def test_explicit_title_is_not_clobbered_by_the_auto_title(self):
    fig = plt.figure()
    anim_mod._render_frame(1, anim_mod._normalize_frames(_three_frames()), fig,
        {"title": "My Animation"})
    assert fig._suptitle.get_text() == "My Animation"
  # end
# end


# --------------------------------------------------------------------------
# saved frames
# --------------------------------------------------------------------------

class TestSaveFrames:
  def test_writes_one_png_per_frame(self, tmp_path):
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, show=False)
    assert len(paths) == 3
    for p in paths:
      assert os.path.isfile(p)
  # end
    # end

  def test_saveframes_path_naming(self, tmp_path):
    prefix = str(tmp_path / "myframe")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, show=False)
    assert paths[0] == f"{prefix}_0.png"
    assert paths[2] == f"{prefix}_2.png"
  # end

  def test_nproc_parallel_writes_the_same_frames(self, tmp_path):
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, nproc=2, show=False)
    assert len(paths) == 3
    for p in paths:
      assert os.path.isfile(p)
    # end
  # end

  def test_nproc_without_saveframes_compiles_through_a_scratch_dir(self, tmp_path):
    out = tmp_path / "parallel.gif"
    result = anim_mod.animate(_three_frames(), nproc=2, tmpdir=str(tmp_path),
        saveas=str(out), show=False)
    assert result == str(out)
    assert out.exists()
    # the scratch directory must not leak its frame PNGs behind.
    assert list(tmp_path.glob("*.png")) == []
  # end
# end


# --------------------------------------------------------------------------
# movie compile
# --------------------------------------------------------------------------

class TestCompileMovie:
  def test_unsupported_extension_raises(self, tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
      anim_mod._compile_movie([], str(tmp_path / "out.bogus"), duration=100.0)
    # end
  # end

  def test_gif_compile_via_pil(self, tmp_path):
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, show=False)
    out = tmp_path / "out.gif"
    anim_mod._compile_movie(paths, str(out), duration=100.0)
    assert out.exists()
  # end

  def test_animate_saves_gif_end_to_end(self, tmp_path):
    out = tmp_path / "movie.gif"
    prefix = str(tmp_path / "frame")
    result = anim_mod.animate(_three_frames(), saveframes=prefix,
        save=True, saveas=str(out), show=False)
    assert out.exists()
    assert len(result) == 3
  # end

  def test_video_extension_raises_clearly_without_ffmpeg(self, monkeypatch,
      tmp_path):
    monkeypatch.setattr(_ffmpeg, "resolve_ffmpeg", lambda: None)
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, show=False)
    with pytest.raises(RuntimeError, match="ffmpeg"):
      anim_mod._compile_movie(paths, str(tmp_path / "out.mp4"), duration=100.0)
    # end
  # end

  @needs_ffmpeg
  def test_mp4_compile_with_ffmpeg(self, tmp_path):
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, show=False)
    out = tmp_path / "out.mp4"
    anim_mod._compile_movie(paths, str(out), fps=10, duration=100.0)
    assert out.exists()
    assert out.stat().st_size > 0
  # end
# end
