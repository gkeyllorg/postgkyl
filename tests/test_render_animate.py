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

needs_ffmpeg = pytest.mark.skipif(
    _ffmpeg.resolve_ffmpeg() is None,
    reason="ffmpeg not found on PATH or via imageio-ffmpeg")
external_tool = pytest.mark.external_tool
slow = pytest.mark.slow


def _line_frame(offset: float) -> GDataState:
  d = GDataState()
  d.ctx["frame"] = int(offset)
  d.ctx["time"] = float(offset) * 0.1
  d.push([np.linspace(0.0, 1.0, 9)], (np.arange(8, dtype=float) + offset)[:,
                                                                          None])
  return d


def _three_frames() -> list[GDataState]:
  return [_line_frame(0.0), _line_frame(1.0), _line_frame(2.0)]


@pytest.fixture(autouse=True)
def _close_figs():
  plt.close("all")
  yield
  plt.close("all")


# --------------------------------------------------------------------------
# frame normalization
# --------------------------------------------------------------------------


class TestNormalizeFrames:

  def test_bare_datasets_become_single_dataset_frames(self):
    frames = anim_mod._normalize_frames(_three_frames())
    assert len(frames) == 3
    assert all(len(f) == 1 for f in frames)

  def test_grouped_frames_kept_as_lists(self):
    grouped = [[_line_frame(0.0), _line_frame(0.5)], [_line_frame(1.0)]]
    frames = anim_mod._normalize_frames(grouped)
    assert len(frames) == 2
    assert len(frames[0]) == 2
    assert len(frames[1]) == 1

  def test_empty_input_raises(self):
    with pytest.raises(ValueError, match="no datasets"):
      anim_mod._normalize_frames([])


# --------------------------------------------------------------------------
# fixed value range
# --------------------------------------------------------------------------


class TestFrameValueRange:

  def test_spans_every_frame(self):
    frames = anim_mod._normalize_frames(_three_frames())
    vmin, vmax = anim_mod._frame_value_range(frames)
    assert vmin == 0.0
    assert vmax == 9.0  # last frame: arange(8) + 2.0 -> max 9.0

  def test_cutoff_clips_the_range(self):
    frames = anim_mod._normalize_frames(_three_frames())
    vmin_full, vmax_full = anim_mod._frame_value_range(frames)
    vmin_cut, vmax_cut = anim_mod._frame_value_range(frames, cutoff=0.5)
    assert vmin_cut >= vmin_full
    assert vmax_cut <= vmax_full

  def test_scale_is_applied_before_taking_extrema(self):
    # A fixed range computed on unscaled values would not match what
    # matplotlib.plot actually draws once yscale/zscale is applied.
    frames = anim_mod._normalize_frames(_three_frames())
    vmin, vmax = anim_mod._frame_value_range(frames, yscale=2.0)
    assert vmin == 0.0
    assert vmax == 18.0  # last frame: (arange(8) + 2.0).max() * 2.0


# --------------------------------------------------------------------------
# live FuncAnimation path
# --------------------------------------------------------------------------


class TestLiveAnimation:

  def test_returns_funcanimation_with_correct_frame_count(self):
    from matplotlib.animation import FuncAnimation
    anim = anim_mod.animate(_three_frames(), no_show=True)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 3

  def test_grouped_frames_overlay_per_frame(self):
    from matplotlib.animation import FuncAnimation
    grouped = [[_line_frame(0.0), _line_frame(0.5)],
               [_line_frame(1.0), _line_frame(1.5)]]
    anim = anim_mod.animate(grouped, no_show=True)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 2

  def test_multiblock_groups_equal_frame_indices(self):
    from matplotlib.animation import FuncAnimation
    grouped = [_line_frame(0.0), _line_frame(0.5)]
    anim = anim_mod.animate(grouped, multiblock=True, no_show=True)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 1

  def test_grouptags_builds_one_animation_per_tag(self):
    from matplotlib.animation import FuncAnimation
    frames = _three_frames()
    frames[0].tag = "left"
    frames[1].tag = "left"
    frames[2].tag = "right"
    animations = anim_mod.animate(frames, grouptags=True, no_show=True)
    assert len(animations) == 2
    assert all(isinstance(anim, FuncAnimation) for anim in animations)
    assert [anim._save_count for anim in animations] == [2, 1]

  def test_show_true_does_not_raise_on_agg(self):
    anim = anim_mod.animate(_three_frames(), no_show=False)
    assert anim is not None

  @needs_ffmpeg
  @external_tool
  @slow
  def test_live_animation_saves_mp4(self, tmp_path):
    out = tmp_path / "live.mp4"
    anim = anim_mod.animate(_three_frames(),
                            save=True,
                            saveas=str(out),
                            fps=5,
                            no_show=True)
    assert anim is not None
    assert out.exists()
    assert out.stat().st_size > 0

  def test_notitle_suppresses_frame_time_title(self):
    fig = plt.figure()
    anim_mod._render_frame(0, anim_mod._normalize_frames(_three_frames()), fig,
                           {"notitle": True})
    assert fig._suptitle is None

  def test_title_includes_frame_and_time_by_default(self):
    fig = plt.figure()
    anim_mod._render_frame(1, anim_mod._normalize_frames(_three_frames()), fig,
                           {})
    assert "frame: 1" in fig._suptitle.get_text()
    assert "time:" in fig._suptitle.get_text()

  def test_explicit_title_is_not_clobbered_by_the_auto_title(self):
    fig = plt.figure()
    anim_mod._render_frame(1, anim_mod._normalize_frames(_three_frames()), fig,
                           {"title": "My Animation"})
    assert fig._suptitle.get_text() == "My Animation"

  @pytest.mark.parametrize(("ctx", "expected"), [
      ({
          "time": 1.25
      }, "time: 1.2500e+00"),
      ({
          "frame": 7
      }, "frame: 7"),
  ])
  def test_generated_title_accepts_either_frame_metadata_field(
      self, ctx, expected):
    frame = _line_frame(0.0)
    frame.ctx.pop("frame")
    frame.ctx.pop("time")
    frame.ctx.update(ctx)
    fig = plt.figure()
    anim_mod._draw_frame([frame], fig, {})
    assert fig._suptitle.get_text() == expected

  def test_variable_range_skips_global_limit_calculation(self, monkeypatch):
    monkeypatch.setattr(
        anim_mod, "_frame_value_range",
        lambda *_args, **_kwargs: pytest.fail("global range was calculated"))
    anim = anim_mod.animate(_three_frames(), variable_range=True, no_show=True)
    assert anim is not None

  def test_live_save_configuration_without_running_a_writer(
      self, monkeypatch, tmp_path):
    saved = []

    def save(self, filename, **kwargs):
      saved.append((filename, kwargs))

    monkeypatch.setattr(anim_mod, "resolve_video_encoder", lambda *_:
                        ("/ffmpeg", "libx264"))
    monkeypatch.setattr("matplotlib.animation.FuncAnimation.save", save)
    out = tmp_path / "movie.mp4"
    anim_mod.animate(_three_frames(),
                     save=True,
                     saveas=str(out),
                     fps=5,
                     dpi=80,
                     no_show=True)
    filename, options = saved[0]
    assert filename == str(out)
    assert options["dpi"] == 80
    assert options["writer"].codec == "libx264"
    assert options["writer"].fps == 5


# --------------------------------------------------------------------------
# saved frames
# --------------------------------------------------------------------------


class TestSaveFrames:

  def test_writes_one_png_per_frame(self, tmp_path):
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, no_show=True)
    assert len(paths) == 3
    for p in paths:
      assert os.path.isfile(p)

  def test_saveframes_path_naming(self, tmp_path):
    prefix = str(tmp_path / "myframe")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, no_show=True)
    assert paths[0] == f"{prefix}_0.png"
    assert paths[2] == f"{prefix}_2.png"

  def test_nproc_parallel_writes_the_same_frames(self, tmp_path):
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(_three_frames(),
                             saveframes=prefix,
                             nproc=2,
                             no_show=True)
    assert len(paths) == 3
    for p in paths:
      assert os.path.isfile(p)

  def test_nproc_without_saveframes_compiles_through_a_scratch_dir(
      self, tmp_path):
    out = tmp_path / "parallel.gif"
    result = anim_mod.animate(_three_frames(),
                              nproc=2,
                              tmpdir=str(tmp_path),
                              saveas=str(out),
                              no_show=True)
    assert result == str(out)
    assert out.exists()
    # the scratch directory must not leak its frame PNGs behind.
    assert list(tmp_path.glob("*.png")) == []

  def test_worker_can_render_one_frame_directly(self, tmp_path):
    prefix = str(tmp_path / "worker")
    path = anim_mod._save_frame_worker(
        (3, [_line_frame(0.0)], {}, prefix, 72, (3.0, 2.0)))
    assert path == f"{prefix}_3.png"
    assert os.path.isfile(path)

  def test_grouped_tags_suffix_saved_frame_prefixes(self, tmp_path):
    frames = _three_frames()
    frames[0].tag = "left"
    frames[1].tag = "left"
    frames[2].tag = "right"
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(frames,
                             grouptags=True,
                             saveframes=prefix,
                             no_show=True)
    assert paths == [[f"{prefix}_left_0.png", f"{prefix}_left_1.png"],
                     [f"{prefix}_right_0.png"]]

  def test_saveas_without_extension_defaults_to_gif(self, tmp_path):
    prefix = str(tmp_path / "frame")
    output = tmp_path / "movie"
    anim_mod.animate(_three_frames(),
                     saveframes=prefix,
                     saveas=str(output),
                     no_show=True)
    assert output.with_suffix(".gif").is_file()


# --------------------------------------------------------------------------
# movie compile
# --------------------------------------------------------------------------


class TestCompileMovie:

  def test_unsupported_extension_raises(self, tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
      anim_mod._compile_movie([], str(tmp_path / "out.bogus"), duration=100.0)

  def test_gif_compile_via_pil(self, tmp_path):
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, no_show=True)
    out = tmp_path / "out.gif"
    anim_mod._compile_movie(paths, str(out), duration=100.0)
    assert out.exists()

  def test_animate_saves_gif_end_to_end(self, tmp_path):
    out = tmp_path / "movie.gif"
    prefix = str(tmp_path / "frame")
    result = anim_mod.animate(_three_frames(),
                              saveframes=prefix,
                              save=True,
                              saveas=str(out),
                              no_show=True)
    assert out.exists()
    assert len(result) == 3

  def test_video_extension_raises_clearly_without_ffmpeg(
      self, monkeypatch, tmp_path):
    monkeypatch.setattr(_ffmpeg, "resolve_ffmpeg", lambda: None)
    monkeypatch.setattr(_ffmpeg, "_bundled_ffmpeg", lambda: None)
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, no_show=True)
    with pytest.raises(RuntimeError, match="ffmpeg"):
      anim_mod._compile_movie(paths, str(tmp_path / "out.mp4"), duration=100.0)

  @pytest.mark.parametrize("fail_encoding", [False, True])
  def test_video_writer_protocol_without_external_process(
      self, monkeypatch, tmp_path, fail_encoding):
    from contextlib import contextmanager
    from PIL import Image
    import matplotlib.animation

    events = []
    images = []

    class FakeImage:
      size = (320, 200)

      def __init__(self):
        self.closed = False
        images.append(self)

      def __enter__(self):
        return self

      def __exit__(self, *_exc):
        self.closed = True

    class FakeAxes:

      def axis(self, value):
        events.append(("axis", value))

      def clear(self):
        events.append(("clear", ))

      def imshow(self, image):
        assert isinstance(image, FakeImage)
        assert not image.closed
        events.append(("imshow", ))

    class FakeFigure:

      def add_axes(self, bounds):
        assert bounds == [0, 0, 1, 1]
        return FakeAxes()

    class FakeWriter:

      def __init__(self, fps, codec, extra_args):
        events.append(("fps", fps))
        assert codec == "libx264"

      @contextmanager
      def saving(self, figure, output_file, dpi):
        assert isinstance(figure, FakeFigure)
        events.append(("saving", output_file, dpi))
        yield

      def grab_frame(self):
        events.append(("grab", ))
        if fail_encoding:
          raise RuntimeError("encoding failed")

    monkeypatch.setattr(anim_mod, "resolve_video_encoder", lambda *_:
                        ("/ffmpeg", "libx264"))
    monkeypatch.setattr(Image, "open", lambda _path: FakeImage())
    monkeypatch.setattr(matplotlib.animation, "FFMpegWriter", FakeWriter)
    monkeypatch.setattr("matplotlib.figure.Figure",
                        lambda **_kwargs: FakeFigure())
    monkeypatch.setattr("matplotlib.backends.backend_agg.FigureCanvasAgg",
                        lambda _figure: None)
    monkeypatch.setattr(plt, "close", lambda figure: events.append(
        ("close", figure)))

    output = str(tmp_path / "movie.mp4")
    if fail_encoding:
      with pytest.raises(RuntimeError, match="encoding failed"):
        anim_mod._compile_movie(["one.png", "two.png"], output, duration=250.0)
    else:
      anim_mod._compile_movie(["one.png", "two.png"], output, duration=250.0)
    assert ("fps", 4.0) in events
    assert ("saving", output, 100) in events
    assert events.count(("grab", )) == (1 if fail_encoding else 2)
    assert len(images) == (2 if fail_encoding else 3)
    assert all(image.closed for image in images)
    assert events[-1][0] == "close"

  @needs_ffmpeg
  @external_tool
  @slow
  def test_mp4_compile_with_ffmpeg(self, tmp_path):
    prefix = str(tmp_path / "frame")
    paths = anim_mod.animate(_three_frames(), saveframes=prefix, no_show=True)
    out = tmp_path / "out.mp4"
    anim_mod._compile_movie(paths, str(out), fps=10, duration=100.0)
    assert out.exists()
    assert out.stat().st_size > 0


def _draw_animation(data, **options):
  animation = anim_mod.animate(data, no_show=True, **options)
  animation._func(0, *animation._args)
  return animation


def _field_frame(offset=0):
  dat = GDataState()
  dat.push(
      [np.linspace(10, 20, 9), np.linspace(30, 40, 9)],
      np.arange(64, dtype=float).reshape(8, 8, 1) + offset)
  return dat


def test_line_controls_and_explicit_limits():
  animation = _draw_animation(_three_frames(),
                              xshift=1,
                              xscale=2,
                              yshift=3,
                              yscale=-2,
                              xlim="2,4",
                              ylim=(-25, 0),
                              color="red",
                              linewidth=3,
                              linestyle="dashed",
                              arg="o",
                              markersize=7,
                              title="Title",
                              xlabel="Position",
                              ylabel="Field",
                              no_showgrid=True,
                              forcelegend=True)
  ax = animation._fig.axes[0]
  line = ax.lines[0]
  np.testing.assert_allclose(line.get_ydata(), -2 * (np.arange(8) + 3))
  assert ax.get_xlim() == (2, 4)
  assert ax.get_ylim() == (-25, 0)
  assert line.get_color() == "red"
  assert line.get_linewidth() == 3
  assert line.get_markersize() == 7
  assert line.get_linestyle() == "--"
  assert animation._fig._suptitle.get_text() == "Title"
  assert ax.get_legend() is not None


def test_shifted_fixed_range_and_transpose():
  animation = _draw_animation(_three_frames(),
                              yshift=3,
                              yscale=-2,
                              transpose=True)
  assert animation._fig.axes[0].get_xlim() == (-24, -6)


def test_2d_range_does_not_clip_spatial_y_axis():
  animation = _draw_animation([_field_frame(), _field_frame(10)],
                              zshift=2,
                              zscale=3,
                              no_colorbar=True)
  ax = animation._fig.axes[0]
  assert ax.get_ylim() == (30, 40)
  assert ax.collections[0].get_clim() == (6, 225)
  assert len(animation._fig.axes) == 1


def test_variable_range_combines_blocks_and_preserves_explicit_bound():
  animation = _draw_animation(
      [[_field_frame(), _field_frame(100)], [_field_frame(200)]],
      variable_range=True,
      zmin=-1,
      no_colorbar=True)
  assert animation._fig.axes[0].collections[0].get_clim() == (-1, 163)
  animation._func(1, *animation._args)
  assert animation._fig.axes[0].collections[0].get_clim() == (-1, 263)


def test_subplots_scatter_and_layout():
  animation = _draw_animation([[_line_frame(0), _line_frame(1)]],
                              subplots=True,
                              num_subplot_col=2,
                              scatter=True,
                              figsize="6,3",
                              no_legend=True)
  assert len(animation._fig.axes) == 2
  np.testing.assert_allclose(animation._fig.get_size_inches(), (6, 3))
  for ax in animation._fig.axes:
    assert len(ax.lines) == 1
    assert ax.lines[0].get_linestyle() == "None"
    assert ax.lines[0].get_marker() == "."
    assert ax.get_legend() is None


def test_collected_and_tag_selection():
  from postgkyl.operations.collect import collect
  collected = collect(*_three_frames(), tag="chosen")
  animation = _draw_animation([collected, _line_frame(9)],
                              use="chosen",
                              collected=True)
  assert animation._save_count == 3
  animation._func(2, *animation._args)
  np.testing.assert_array_equal(animation._fig.axes[0].lines[0].get_ydata(),
                                np.arange(8) + 2)
  assert "time: 2.0000e-01" in animation._fig._suptitle.get_text()


@pytest.mark.parametrize("mode", ["contour", "quiver", "streamline", "group"])
def test_field_modes(mode):
  frame = _field_frame()
  if mode in ("quiver", "streamline"):
    frame.push(list(frame.grid), np.repeat(frame.values, 2, axis=-1))
  options = {mode: 0 if mode == "group" else True}
  if mode == "contour":
    options["clevels"] = "5"
  if mode == "streamline":
    options.update(sdensity=0.5, arrowstyle="->")
  animation = _draw_animation([frame], **options)
  assert animation._fig.axes[0].has_data()


def test_grouped_tags_keep_plot_controls():
  frames = _three_frames()
  frames[0].tag = "first"
  animations = anim_mod.animate(frames,
                                grouptags=True,
                                no_show=True,
                                color="purple",
                                title="fixed",
                                notitle=True)
  for animation in animations:
    animation._func(0, *animation._args)
    assert animation._fig.axes[0].lines[0].get_color() == "purple"
    assert animation._fig._suptitle is None


@pytest.mark.parametrize("extension", ["gif", "webp", "apng"])
def test_image_movie_formats_without_ffmpeg(tmp_path, monkeypatch, extension):
  from PIL import Image
  monkeypatch.setattr(anim_mod, "resolve_video_encoder",
                      lambda *_: pytest.fail("image movie requested ffmpeg"))
  output = tmp_path / f"movie.{extension}"
  anim_mod.animate(_three_frames(),
                   saveas=output,
                   no_show=True,
                   figsize=(3, 2),
                   dpi=40)
  with Image.open(output) as movie:
    assert movie.n_frames == 3


def test_invalid_output_rejected_before_frame_writes(tmp_path):
  with pytest.raises(ValueError, match="unsupported"):
    anim_mod.animate(_three_frames(),
                     saveas=tmp_path / "movie.xyz",
                     saveframes=str(tmp_path / "frame"),
                     no_show=True)
  assert list(tmp_path.iterdir()) == []


def test_log_axes_labels_aspect_and_component_squeeze():
  frame = _line_frame(1)
  frame.push(list(frame.grid), np.repeat(frame.values, 2, axis=-1))
  animation = _draw_animation([frame],
                              squeeze=True,
                              logx=True,
                              logy=True,
                              fixaspect=True,
                              xlabel="x",
                              ylabel="y",
                              no_legend=True)
  ax = animation._fig.axes[0]
  assert len(animation._fig.axes) == 1
  assert len(ax.lines) == 2
  assert ax.get_xscale() == "log"
  assert ax.get_yscale() == "log"
  assert ax.get_xlabel() == "x"
  assert ax.get_ylabel() == "y"
  assert ax.get_aspect() == 1


def test_field_color_controls_and_coordinate_transforms():
  animation = _draw_animation([_field_frame(1)],
                              logz=True,
                              zlim=(1, 100),
                              xshift=2,
                              xscale=3,
                              yshift=4,
                              yscale=2,
                              clabel="density",
                              diverging=True,
                              edgecolors="red")
  ax = animation._fig.axes[0]
  mesh = ax.collections[0]
  assert isinstance(mesh.norm, matplotlib.colors.SymLogNorm)
  assert mesh.get_clim() == (1, 100)
  assert mesh.get_cmap().name == "RdBu_r"
  assert ax.get_xlim() == (36, 66)
  assert ax.get_ylim() == (68, 88)
  assert animation._fig.axes[-1].get_ylabel() == "density"
  np.testing.assert_allclose(mesh.get_edgecolors()[0], [1, 0, 0, 1])


def test_frame_options_match_sequential_and_parallel_output(tmp_path):
  from PIL import Image
  options = dict(color="red",
                 scatter=True,
                 ylim=(-1, 10),
                 figsize=(3, 2),
                 dpi=40,
                 notitle=True,
                 no_show=True)
  sequential = anim_mod.animate(_three_frames(),
                                saveframes=str(tmp_path / "serial"),
                                **options)
  parallel = anim_mod.animate(_three_frames(),
                              nproc=2,
                              saveframes=str(tmp_path / "parallel"),
                              **options)
  for first, second in zip(sequential, parallel):
    with Image.open(first) as serial_image, Image.open(
        second) as parallel_image:
      np.testing.assert_array_equal(np.asarray(serial_image),
                                    np.asarray(parallel_image))


def test_diverging_limits_are_fixed_across_frames():
  animation = _draw_animation(
      [_field_frame(1), _field_frame(10)], diverging=True, no_colorbar=True)
  assert animation._fig.axes[0].collections[0].get_clim() == (-73, 73)
  animation._func(1, *animation._args)
  assert animation._fig.axes[0].collections[0].get_clim() == (-73, 73)


@pytest.mark.parametrize("nproc", [1, 2])
def test_unavailable_codec_fails_before_rendering(monkeypatch, tmp_path, nproc):

  def resolve(*_args):
    raise RuntimeError("no usable 'missing' video encoder")

  monkeypatch.setattr(anim_mod, "resolve_video_encoder", resolve)
  monkeypatch.setattr(anim_mod, "_save_frames",
                      lambda *_args, **_kwargs: pytest.fail("frames rendered"))
  monkeypatch.setattr(plt, "figure",
                      lambda *_args, **_kwargs: pytest.fail("figure created"))
  with pytest.raises(RuntimeError, match="no usable"):
    anim_mod.animate(_three_frames(),
                     saveas=tmp_path / "movie.mp4",
                     codec="missing",
                     nproc=nproc,
                     no_show=True)
  assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("saved_frames", [False, True])
def test_codec_forwarded_once_and_errors_include_encoder(
    monkeypatch, tmp_path, saved_frames):
  import subprocess
  from contextlib import contextmanager
  from matplotlib.animation import FFMpegWriter

  calls = []
  original_path = matplotlib.rcParams["animation.ffmpeg_path"]

  def resolve(context, codec):
    calls.append((context, codec))
    return "/chosen/ffmpeg", "mpeg4"

  def failure():
    raise subprocess.CalledProcessError(1, ["ffmpeg"],
                                        stderr="Encoder initialization failed")

  def save(animation, filename, **kwargs):
    writer = kwargs["writer"]
    assert writer.codec == "mpeg4"
    assert matplotlib.rcParams["animation.ffmpeg_path"] == "/chosen/ffmpeg"
    failure()

  @contextmanager
  def saving(writer, figure, filename, dpi):
    assert writer.codec == "mpeg4"
    failure()
    yield

  monkeypatch.setattr(anim_mod, "resolve_video_encoder", resolve)
  monkeypatch.setattr("matplotlib.animation.FuncAnimation.save", save)
  monkeypatch.setattr(FFMpegWriter, "saving", saving)
  options = {"saveframes": str(tmp_path / "frame")} if saved_frames else {}
  with pytest.raises(RuntimeError,
                     match="Encoder initialization failed") as error:
    anim_mod.animate(_three_frames(),
                     saveas=tmp_path / "movie.mp4",
                     codec="mpeg4",
                     no_show=True,
                     **options)
  assert "codec 'mpeg4'" in str(error.value)
  assert "/chosen/ffmpeg" in str(error.value)
  assert calls == [("animate", "mpeg4")]
  assert matplotlib.rcParams["animation.ffmpeg_path"] == original_path


@needs_ffmpeg
@external_tool
@pytest.mark.parametrize("nproc", [1, 2])
def test_mpeg4_export_accepts_odd_frame_dimensions(tmp_path, nproc):
  output = tmp_path / "odd.mp4"
  anim_mod.animate(_three_frames(),
                   codec="mpeg4",
                   saveas=output,
                   figsize=(3.01, 2.01),
                   dpi=100,
                   nproc=nproc,
                   no_show=True)
  assert output.stat().st_size > 0
