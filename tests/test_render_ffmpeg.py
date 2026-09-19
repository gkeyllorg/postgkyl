"""Tests for postgkyl.render._ffmpeg -- shared ffmpeg discovery."""

from __future__ import annotations

import builtins
import subprocess

import pytest

from postgkyl.render import _ffmpeg


def test_resolve_prefers_path(monkeypatch):
  monkeypatch.setattr(_ffmpeg.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
  assert _ffmpeg.resolve_ffmpeg() == "/usr/bin/ffmpeg"


def test_resolve_falls_back_to_imageio_ffmpeg(monkeypatch):
  import types

  fake_module = types.SimpleNamespace(
      get_ffmpeg_exe=lambda: "/fake/imageio_ffmpeg/ffmpeg")
  monkeypatch.setattr(_ffmpeg.shutil, "which", lambda _name: None)
  monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", fake_module)
  assert _ffmpeg.resolve_ffmpeg() == "/fake/imageio_ffmpeg/ffmpeg"


def test_resolve_returns_none_when_both_sources_are_missing(monkeypatch):
  real_import = builtins.__import__

  def missing_imageio(name, *args, **kwargs):
    if name == "imageio_ffmpeg":
      raise ImportError(name)
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(_ffmpeg.shutil, "which", lambda _name: None)
  monkeypatch.setattr(builtins, "__import__", missing_imageio)
  assert _ffmpeg.resolve_ffmpeg() is None


def test_require_raises_clearly_when_nothing_resolves(monkeypatch):
  monkeypatch.setattr(_ffmpeg, "resolve_ffmpeg", lambda: None)
  with pytest.raises(RuntimeError, match="ffmpeg"):
    _ffmpeg.require_ffmpeg("animate")


def test_require_returns_resolved_path(monkeypatch):
  monkeypatch.setattr(_ffmpeg, "resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
  assert _ffmpeg.require_ffmpeg("animate") == "/usr/bin/ffmpeg"


def _encoder_environment(monkeypatch, system, bundled):
  monkeypatch.setattr(_ffmpeg, "resolve_ffmpeg", lambda: "/system/ffmpeg")
  monkeypatch.setattr(_ffmpeg, "_bundled_ffmpeg", lambda: "/bundled/ffmpeg")
  monkeypatch.setattr(
      _ffmpeg, "_video_encoders", lambda path: system
      if path == "/system/ffmpeg" else bundled)


def test_system_h264_encoder_wins(monkeypatch):
  _encoder_environment(monkeypatch, {"libx264", "mpeg4"}, {"libx264"})
  assert _ffmpeg.resolve_video_encoder("animate") == ("/system/ffmpeg",
                                                      "libx264")


def test_perlmutter_missing_h264_uses_bundled_encoder(monkeypatch):
  _encoder_environment(monkeypatch, {"mpeg4", "h264_nvenc"}, {"libx264"})
  assert _ffmpeg.resolve_video_encoder("animate") == ("/bundled/ffmpeg",
                                                      "libx264")


def test_mpeg4_fallback_without_software_h264(monkeypatch):
  _encoder_environment(monkeypatch, {"mpeg4", "h264_nvenc"}, set())
  assert _ffmpeg.resolve_video_encoder("animate") == ("/system/ffmpeg", "mpeg4")


def test_explicit_codec_is_not_replaced(monkeypatch):
  _encoder_environment(monkeypatch, {"mpeg4"}, {"libx264"})
  assert _ffmpeg.resolve_video_encoder("animate",
                                       "mpeg4") == ("/system/ffmpeg", "mpeg4")
  with pytest.raises(RuntimeError, match="no usable 'vp9'"):
    _ffmpeg.resolve_video_encoder("animate", "vp9")


def test_h264_alias_does_not_fall_back_to_mpeg4(monkeypatch):
  _encoder_environment(monkeypatch, {"mpeg4"}, {"libopenh264"})
  assert _ffmpeg.resolve_video_encoder("animate", "h264") == ("/bundled/ffmpeg",
                                                              "libopenh264")
  _encoder_environment(monkeypatch, {"mpeg4"}, set())
  with pytest.raises(RuntimeError, match="no usable 'h264'"):
    _ffmpeg.resolve_video_encoder("animate", "h264")


def test_encoder_probe_reads_only_video_encoders(monkeypatch):
  import subprocess
  from types import SimpleNamespace

  def run(args, **kwargs):
    assert args == ["/ffmpeg", "-hide_banner", "-encoders"]
    assert kwargs["timeout"] == 15
    return SimpleNamespace(
        stdout=" V....D libx264 H.264\n A..... aac AAC\n V..... mpeg4 MPEG4\n")

  monkeypatch.setattr(subprocess, "run", run)
  assert _ffmpeg._video_encoders("/ffmpeg") == {"libx264", "mpeg4"}


@pytest.mark.parametrize(
    "error",
    [OSError("bad executable"),
     subprocess.TimeoutExpired("ffmpeg", 15)])
def test_broken_system_executable_falls_back(monkeypatch, error):
  _encoder_environment(monkeypatch, set(), {"libx264"})

  def encoders(path):
    if path == "/system/ffmpeg":
      raise error
    return {"libx264"}

  monkeypatch.setattr(_ffmpeg, "_video_encoders", encoders)
  assert _ffmpeg.resolve_video_encoder("animate") == ("/bundled/ffmpeg",
                                                      "libx264")


def test_missing_encoder_error_explains_installation(monkeypatch):
  _encoder_environment(monkeypatch, set(), set())
  with pytest.raises(RuntimeError, match="imageio-ffmpeg") as error:
    _ffmpeg.resolve_video_encoder("animate")
  assert "/system/ffmpeg" in str(error.value)
  assert "package named ffmpeg does not install" in str(error.value)
