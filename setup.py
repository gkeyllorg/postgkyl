"""Build the Gkeyll core and its ordinary setuptools extension."""

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
GKEYLL = ROOT / "gkeyll"
PACKAGE = "postgkyl.gpython"
BUNDLE_FILES = ("libg0core.so", "_build_info.json")


def _skip_native():
  value = os.environ.get("POSTGKYL_SKIP_GKEYLL_BUILD", "0")
  if value not in {"0", "1"}:
    raise ValueError("POSTGKYL_SKIP_GKEYLL_BUILD must be 0 or 1")
  return value == "1"


def _git(directory, *args):
  result = subprocess.run(["git", "-C", str(directory), *args],
                          capture_output=True,
                          text=True)
  return result.stdout.strip() if result.returncode == 0 else "unknown"


def _build_info():
  import numpy

  branch = _git(GKEYLL, "symbolic-ref", "--short", "HEAD")
  if branch == "unknown":
    branch = "detached"
  if _git(GKEYLL, "status", "--porcelain", "--untracked-files=no"):
    branch += " (locally modified)"
  return {
      "gkeyll_commit": _git(GKEYLL, "rev-parse", "HEAD"),
      "gkeyll_commit_date": _git(GKEYLL, "log", "-1", "--format=%cI"),
      "gkeyll_branch": branch,
      "postgkyl_build_commit": _git(ROOT, "rev-parse", "HEAD"),
      "build_date": datetime.now(timezone.utc).isoformat(),
      "build_cc": os.environ.get("CC", "cc"),
      "build_arch_flags": os.environ.get("ARCH_FLAGS", ""),
      "build_python": platform.python_version(),
      "build_numpy": numpy.__version__,
  }


class BuildGkeyllExtension(build_ext):
  """Keep the native library and provenance beside the extension in all modes."""

  def build_extensions(self):
    import numpy

    subprocess.run(["sh", str(ROOT / "scripts/build_gkeyll.sh")],
                   cwd=ROOT,
                   check=True)
    bundle = Path(self.build_lib) / PACKAGE.replace(".", "/")
    bundle.mkdir(parents=True, exist_ok=True)
    library = bundle / "libg0core.so"
    shutil.copy2(GKEYLL / "build/core/libg0core.so", library)
    if sys.platform == "darwin":
      subprocess.run(
          ["install_name_tool", "-id", "@rpath/libg0core.so",
           str(library)],
          check=True)
    for extension in self.extensions:
      extension.include_dirs = [str(GKEYLL / "core/zero"), numpy.get_include()]
      extension.library_dirs = [str(bundle.resolve())]
    # Reinstalling must relink even if only Gkeyll or the build environment
    # changed. Compiling this single wrapper is cheap; make handles core reuse.
    self.force = True
    super().build_extensions()
    (bundle /
     "_build_info.json").write_text(json.dumps(_build_info(), indent=2) + "\n")

  def _bundle_mapping(self):
    source = Path(
        self.get_finalized_command("build_py").get_package_dir(PACKAGE))
    destination = Path(self.build_lib) / PACKAGE.replace(".", "/")
    return {
        str(destination / name): str(source / name)
        for name in BUNDLE_FILES
    }

  def get_output_mapping(self):
    return {**super().get_output_mapping(), **self._bundle_mapping()}

  def get_outputs(self):
    return sorted(set(super().get_outputs()) | self._bundle_mapping().keys())

  def copy_extensions_to_source(self):
    super().copy_extensions_to_source()
    for built, source in self._bundle_mapping().items():
      self.copy_file(built, source)


setup(
    ext_modules=[] if _skip_native() else [
        Extension(
            "postgkyl.gpython._gpython",
            sources=["src/postgkyl/gpython/csrc/_gpythonmodule.c"],
            libraries=["g0core"],
            runtime_library_dirs=[
                "@loader_path" if sys.platform == "darwin" else "$ORIGIN"
            ],
        )
    ],
    cmdclass={"build_ext": BuildGkeyllExtension},
)
