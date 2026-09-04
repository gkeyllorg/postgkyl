import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel

ROOT_DIR = Path(__file__).parent
BUILD_SCRIPT = ROOT_DIR / "scripts" / "build_gkeyll.sh"


def _build_gkeyll():
  # build_gpython.sh (invoked as build_gkeyll.sh's final step) resolves its
  # NumPy via `${PYTHON:-python3}` -- a bare PATH lookup. Left unset, that
  # can silently resolve to a *different* Python installation than the one
  # actually running this install (macOS commonly has several: system
  # /usr/bin/python3, Homebrew, the actions/setup-python framework build).
  # Such a mismatch passes build_gpython.sh's own `numpy>=2.2` floor check
  # yet still isn't ABI-identical to the NumPy installed at runtime, which
  # has been reproduced to crash the compiled _gpython extension outright
  # (segfault / heap corruption) at some later, unrelated call rather than
  # fail cleanly at import -- see scripts/build_gpython.sh and README.md.
  # Pinning PYTHON to sys.executable here removes the ambiguity entirely:
  # the extension is always built against exactly the NumPy this same
  # interpreter will import at runtime.
  env = os.environ.copy()
  env["PYTHON"] = sys.executable
  subprocess.run(["sh", str(BUILD_SCRIPT)], check=True, cwd=ROOT_DIR, env=env)


class BuildPyWithGkeyll(build_py):

  def run(self):
    _build_gkeyll()
    super().run()


class DevelopWithGkeyll(develop):

  def run(self):
    _build_gkeyll()
    super().run()


class EditableWheelWithGkeyll(editable_wheel):

  def run(self):
    _build_gkeyll()
    super().run()


setup(cmdclass={
    "build_py": BuildPyWithGkeyll,
    "develop": DevelopWithGkeyll,
    "editable_wheel": EditableWheelWithGkeyll,
}, )
