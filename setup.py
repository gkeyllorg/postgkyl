"""Build the selected Gkeyll stack and its setuptools extension."""

import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
PACKAGE = "postgkyl.gpython"
load_config = runpy.run_path(str(ROOT /
                                 "scripts/gkeyll_config.py"))["load_config"]
build_gkeyll = runpy.run_path(str(ROOT /
                                  "scripts/build_gkeyll.py"))["build_gkeyll"]
PLAN = load_config()
BUNDLE_FILES = tuple(f"libg0{module}.so"
                     for module in PLAN.modules) + ("_build_info.json", )


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


def _gkeyll_paths(plan):
  """Build the internal stack or locate an explicitly selected external build."""
  if plan.mode == "managed":
    build_gkeyll(plan)
    return (plan.directory, plan.directory / "core/zero",
            plan.build_dir / "core/libg0core.so")
  directory = plan.directory
  build = plan.build_dir
  layouts = [("core/zero", build / "core/libg0core.so")]
  if plan.allow_prefix:
    layouts.insert(0, ("include", directory / "lib/libg0core.so"))
  for headers, library in layouts:
    include = directory / headers
    core = directory / library
    if (include / "gkyl_gpython_core.h").is_file() and core.is_file():
      return directory, include, core
  raise ValueError(
      f"POSTGKYL_GKEYLL_DIR={directory} must contain either "
      f"core/zero/gkyl_gpython_core.h and {build}/core/libg0core.so (built checkout), "
      "or include/gkyl_gpython_core.h and lib/libg0core.so (installation prefix). "
      "Build a Gkeyll version with the gpython shim before installing Postgkyl."
  )


def _build_info(gkeyll):
  import numpy

  branch = _git(gkeyll, "symbolic-ref", "--short", "HEAD")
  if branch == "unknown":
    branch = "detached"
  status = _git(gkeyll, "status", "--porcelain", "--untracked-files=no")
  if status not in {"", "unknown"}:
    branch += " (locally modified)"
  return {
      "gkeyll_commit": _git(gkeyll, "rev-parse", "HEAD"),
      "gkeyll_commit_date": _git(gkeyll, "log", "-1", "--format=%cI"),
      "gkeyll_branch": branch,
      "gkeyll_directory": str(gkeyll),
      "gkeyll_modules": PLAN.modules,
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

    gkeyll, headers, core = _gkeyll_paths(PLAN)
    bundle = Path(self.build_lib) / PACKAGE.replace(".", "/")
    bundle.mkdir(parents=True, exist_ok=True)
    # Gkeyll's higher libraries include all prerequisite objects. Link only
    # the highest one to avoid loading multiple copies of the same symbols.
    includes = [headers]
    for module in PLAN.modules:
      name = f"libg0{module}.so"
      source = (core.parent /
                name if headers.name == "include" else core.parent.parent /
                module / name)
      if not source.is_file():
        raise ValueError(f"Selected Gkeyll build is missing {source}")
      library = bundle / name
      shutil.copy2(source, library)
      if module != "core":
        include = (headers if headers.name == "include" else gkeyll / module /
                   "zero")
        if not (include / f"gkyl_gpython_{module}.h").is_file():
          raise ValueError(
              f"Selected Gkeyll build is missing {module} shim headers")
        includes.append(include)
      if sys.platform == "darwin":
        subprocess.run(
            ["install_name_tool", "-id", f"@rpath/{name}",
             str(library)],
            check=True)
    # Remove artifacts from a previous, larger selection before packaging.
    for stale in bundle.glob("libg0*.so"):
      if stale.name not in BUNDLE_FILES:
        stale.unlink()
    for extension in self.extensions:
      extension.include_dirs = [str(path)
                                for path in includes] + [numpy.get_include()]
      extension.library_dirs = [str(bundle.resolve())]
    # Reinstalling must relink even if only Gkeyll or the build environment
    # changed. Compiling this single wrapper is cheap; make handles core reuse.
    self.force = True
    super().build_extensions()
    (bundle / "_build_info.json"
     ).write_text(json.dumps(_build_info(gkeyll), indent=2) + "\n")

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
    mapping = self._bundle_mapping()
    for built, source in mapping.items():
      self.copy_file(built, source)
    source_dir = Path(next(iter(mapping.values()))).parent
    for stale in source_dir.glob("libg0*.so"):
      if stale.name not in BUNDLE_FILES:
        stale.unlink()


setup(
    ext_modules=[] if _skip_native() else [
        Extension(
            "postgkyl.gpython._gpython",
            sources=["src/postgkyl/gpython/csrc/_gpythonmodule.c"],
            libraries=[f"g0{PLAN.module}"],
            define_macros=[(f"POSTGKYL_HAVE_{module.upper()}", "1")
                           for module in PLAN.modules if module != "core"],
            runtime_library_dirs=[
                "@loader_path" if sys.platform == "darwin" else "$ORIGIN"
            ],
        )
    ],
    cmdclass={"build_ext": BuildGkeyllExtension},
)
