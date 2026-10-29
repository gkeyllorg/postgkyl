"""Resolve native build settings for setuptools and the command-line scripts."""

from dataclasses import dataclass
import os
from pathlib import Path
import sys

try:
  import tomllib
except ModuleNotFoundError:  # Python 3.10
  import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BuildPlan:
  """Resolved source ownership, module stack, and native build settings."""

  mode: str
  module: str
  modules: tuple[str, ...]
  directory: Path
  build_dir: Path
  allow_prefix: bool
  lapack_lite: bool


def load_config() -> BuildPlan:
  path = Path(
      os.environ.get("POSTGKYL_BUILD_CONFIG",
                     ROOT / "postgkyl-build.toml")).expanduser().resolve()
  with path.open("rb") as stream:
    document = tomllib.load(stream)
  if set(document) - {"gkeyll"}:
    raise ValueError(f"Unknown build configuration section in {path}")
  settings = document.get("gkeyll", {})
  allowed = {"mode", "module", "directory", "build-dir"}
  if not isinstance(settings, dict) or set(settings) - allowed:
    raise ValueError(f"Unknown Gkeyll configuration setting in {path}")
  if any(not isinstance(value, str) or not value.strip()
         for value in settings.values()):
    raise ValueError("Gkeyll configuration values must be nonempty strings")
  mode = settings.get("mode", "managed")
  module = os.environ.get("POSTGKYL_GKEYLL_MODULE",
                          settings.get("module", "core"))
  directory = os.environ.get("POSTGKYL_GKEYLL_DIR", settings.get("directory"))
  if "POSTGKYL_GKEYLL_DIR" in os.environ:
    mode = "existing"
  if mode not in {"managed", "existing"}:
    raise ValueError("Gkeyll mode must be managed or existing")
  modules = (ROOT / "scripts/gkeyll-modules").read_text().split()
  if module not in modules:
    raise ValueError("POSTGKYL_GKEYLL_MODULE / module must be one of: " +
                     ", ".join(modules))
  build_dir = settings.get("build-dir")
  if mode == "managed" and (directory is not None or build_dir is not None):
    raise ValueError("directory and build-dir require mode = 'existing'")
  if mode == "existing":
    if directory is None or not directory.strip():
      raise ValueError(
          "POSTGKYL_GKEYLL_DIR / directory must be a nonempty path")
    directory = (path.parent / Path(directory).expanduser()).resolve()
    if build_dir is not None:
      build_dir = (directory / Path(build_dir).expanduser()).resolve()
  else:
    directory = ROOT / "gkeyll"
    build_dir = directory / ("build"
                             if module == "core" else "build/postgkyl-full")
  return BuildPlan(
      mode=mode,
      module=module,
      modules=tuple(modules[:modules.index(module) + 1]),
      directory=directory,
      build_dir=build_dir or directory / "build",
      allow_prefix=mode == "existing" and build_dir is None,
      lapack_lite=module == "core",
  )


if __name__ == "__main__":
  try:
    config = load_config()
    if sys.argv[1:] == ["--mode"]:
      print(config.mode)
    elif sys.argv[1:] == ["--module"]:
      if config.mode != "managed":
        raise ValueError(
            "Existing Gkeyll builds must be rebuilt by their owner")
      print(config.module)
    else:
      raise ValueError("usage: gkeyll_config.py --mode|--module")
  except (ValueError, OSError) as exc:
    sys.exit(f"error: {exc}")
