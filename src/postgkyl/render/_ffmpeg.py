"""Shared ffmpeg discovery, used by both ``animate.py`` and ``plotly.py``.

General discovery prefers PATH, then imageio-ffmpeg. Video export also
checks encoder availability and can use imageio-ffmpeg when the system
executable lacks a suitable software encoder.
"""

from __future__ import annotations

import shutil


def resolve_ffmpeg() -> str | None:
  path = shutil.which("ffmpeg")
  if path is not None:
    return path
  return _bundled_ffmpeg()


def require_ffmpeg(context: str) -> str:
  path = resolve_ffmpeg()
  if path is None:
    raise RuntimeError(
        f"{context}: ffmpeg is required but was not found on PATH and "
        "imageio-ffmpeg is not installed.")
  return path


def _bundled_ffmpeg() -> str | None:
  """Return imageio's executable without requiring a system installation."""
  try:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()
  except (ImportError, RuntimeError, OSError):
    return None


def _video_encoders(path: str) -> set[str]:
  """Read the video encoders compiled into one executable."""
  import subprocess

  result = subprocess.run([path, "-hide_banner", "-encoders"],
                          capture_output=True,
                          text=True,
                          check=True,
                          timeout=15)
  return {
      fields[1]
      for line in result.stdout.splitlines() if len(fields := line.split()) >= 2
      and len(fields[0]) == 6 and fields[0].startswith("V")
  }


def resolve_video_encoder(context: str,
                          codec: str | None = None) -> tuple[str, str]:
  """Choose a binary and a supported software encoder before rendering.

  Prefer H.264 on PATH, then H.264 in imageio's binary. If neither has a
  software H.264 encoder, use MPEG-4 Part 2. Explicit codecs never silently
  change to another format; ``h264`` denotes either software H.264 encoder.
  Hardware encoders are selected only when explicitly requested.
  """
  import subprocess

  primary = resolve_ffmpeg()
  paths = list(
      dict.fromkeys(path for path in (primary, _bundled_ffmpeg())
                    if path is not None))
  preferred = ("libx264", "libopenh264") if codec in (None,
                                                      "h264") else (codec, )
  available = []
  problems = []
  for path in paths:
    try:
      encoders = _video_encoders(path)
    except (OSError, subprocess.SubprocessError) as err:
      problems.append(f"{path}: {err}")
      continue
    available.append((path, encoders))
    for name in preferred:
      if name in encoders:
        return path, name
  if codec is None:
    for path, encoders in available:
      if "mpeg4" in encoders:
        return path, "mpeg4"
  checked = ", ".join(paths) or "no ffmpeg executable found"
  detail = "; ".join(problems)
  requested = repr(codec) if codec else "H.264 or MPEG-4"
  raise RuntimeError(
      f"{context}: no usable {requested} video encoder ({checked}). "
      "Install an ffmpeg build with the required encoder, or install "
      "imageio-ffmpeg with `python -m pip install -U imageio-ffmpeg`. "
      "The PyPI package named ffmpeg does not install the ffmpeg executable. "
      "Use codec='mpeg4' / --codec mpeg4 when available, or save a .gif." +
      (f" Probe errors: {detail}" if detail else ""))
