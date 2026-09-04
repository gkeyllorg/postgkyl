import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

ROOT_DIR = Path(__file__).parent
BUILD_SCRIPT = ROOT_DIR / "scripts" / "build_gkeyll.sh"
BUNDLED_LIB = ROOT_DIR / "src" / "postgkyl" / "gpython" / "libg0core.so"
SKIP_BUILD_ENV = "POSTGKYL_SKIP_GKEYLL_BUILD"


def _skip_gkeyll_build():
  """Read the explicit pure-Python CI escape hatch.

  Only ``0`` and ``1`` are accepted so a misspelled value cannot silently
  turn a required native packaging job into a pure-Python build.
  """
  value = os.environ.get(SKIP_BUILD_ENV, "0")
  if value not in {"0", "1"}:
    raise RuntimeError(f"{SKIP_BUILD_ENV} must be 0 or 1, got {value!r}")
  return value == "1"


def _build_gkeyll():
  if _skip_gkeyll_build():
    print(f"# Skipping Gkeyll build ({SKIP_BUILD_ENV}=1)")
    return False
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
  return True


class BuildPyWithGkeyll(build_py):

  def run(self):
    built_native = _build_gkeyll()
    super().run()
    # PEP 660 sets editable_mode and deliberately makes build_py.run() a
    # no-op: the editable wheel points at the source tree, where the build
    # script has already placed both native artifacts.  Trying to copy into
    # build_lib here assumes a directory Setuptools intentionally did not
    # create and makes ``pip install -e`` fail after a successful compile.
    if getattr(self, "editable_mode", False):
      return
    destination = Path(self.build_lib) / "postgkyl" / "gpython"
    if not built_native:
      # A reused build directory may contain output from an earlier native
      # build.  A skip-build lane must remain genuinely pure Python.
      for name in ("_build_info.py", "_gpython.so", "libg0core.so"):
        (destination / name).unlink(missing_ok=True)
      return
    # libg0core is not a Python extension as far as setuptools knows, so copy
    # it explicitly next to _gpython.so.  The extension's relative loader path
    # is deliberately resolved against this exact directory.
    if not BUNDLED_LIB.is_file():
      raise FileNotFoundError(
          f"native build did not produce bundled library: {BUNDLED_LIB}")
    destination.mkdir(parents=True, exist_ok=True)
    self.copy_file(str(BUNDLED_LIB), str(destination / BUNDLED_LIB.name))


class DevelopWithGkeyll(develop):

  def run(self):
    _build_gkeyll()
    super().run()


class BinaryDistribution(Distribution):
  """Mark wheels as interpreter/platform-specific native artifacts.

  ``_gpython.so`` is built by the Gkeyll-aware shell script rather than a
  setuptools ``Extension``, so setuptools cannot infer this itself.  Without
  this declaration it emits an incorrect ``py3-none-any`` wheel.
  """

  def has_ext_modules(self):
    return not _skip_gkeyll_build()


setup(cmdclass={
    "build_py": BuildPyWithGkeyll,
    "develop": DevelopWithGkeyll,
},
      distclass=BinaryDistribution)
