"""Tests for postgkyl.render._ffmpeg -- shared ffmpeg discovery."""

from __future__ import annotations

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


def test_require_raises_clearly_when_nothing_resolves(monkeypatch):
  monkeypatch.setattr(_ffmpeg, "resolve_ffmpeg", lambda: None)
  with pytest.raises(RuntimeError, match="ffmpeg"):
    _ffmpeg.require_ffmpeg("animate")


def test_require_returns_resolved_path(monkeypatch):
  monkeypatch.setattr(_ffmpeg, "resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
  assert _ffmpeg.require_ffmpeg("animate") == "/usr/bin/ffmpeg"
