"""Shared ffmpeg discovery, used by both ``animate.py`` and ``plotly.py``.

PATH is checked first, so a system or conda-provided ``ffmpeg`` (the
``environment.yml`` route) always wins; falling back to the bundled
``imageio-ffmpeg`` static binary means a bare ``pip install`` also gets a
working ffmpeg, with no system package manager and no conda involved.
"""

from __future__ import annotations

import shutil


def resolve_ffmpeg() -> str | None:
  path = shutil.which("ffmpeg")
  if path is not None:
    return path
  try:
    import imageio_ffmpeg
  except ImportError:
    return None
  return imageio_ffmpeg.get_ffmpeg_exe()


def require_ffmpeg(context: str) -> str:
  path = resolve_ffmpeg()
  if path is None:
    raise RuntimeError(
        f"{context}: ffmpeg is required but was not found on PATH and "
        "imageio-ffmpeg is not installed.")
  return path
