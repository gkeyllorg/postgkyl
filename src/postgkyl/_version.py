"""Debugging statistics behind ``pgkyl --version`` -- not part of the public
computing API (only ``cli/app.py``'s ``--version`` flag reads this).

Reports the postgkyl commit this checkout is at, the vendored Gkeyll commit
it was built against (via ``gpython.build_info()``, generated at build time
by setup.py since gkeyll/ is a build-time-only clone), and
interpreter/platform/dependency versions -- everything a bug report needs
without asking the user to gather it by hand.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import pathlib
import platform
import subprocess
import sys

from postgkyl import gpython
from postgkyl.cli_spec import hidden

_DEPENDENCIES = ("numpy", "scipy", "click", "matplotlib", "msgpack", "plotly",
                 "pyvista")


def _git(repo_dir: pathlib.Path, *args: str) -> str | None:
  if not (repo_dir / ".git").is_dir():
    return None
  try:
    result = subprocess.run(["git", "-C", str(repo_dir), *args],
                            capture_output=True,
                            text=True,
                            timeout=5,
                            check=True)
  except (OSError, subprocess.CalledProcessError):
    return None
  return result.stdout.strip() or None


def _postgkyl_commit() -> str:
  # this file is src/postgkyl/_version.py -- the repo root is two levels up
  repo_dir = pathlib.Path(__file__).resolve().parents[2]
  commit = _git(repo_dir, "rev-parse", "--short=12", "HEAD")
  if commit is None:
    build = gpython.build_info()
    baked = build["postgkyl_build_commit"] if build else None
    if baked and baked != "unknown":
      return f"{baked[:12]} (baked at build time, not a git checkout)"
    return "unknown (not a git checkout)"
  dirty = _git(repo_dir, "status", "--porcelain", "--untracked-files=no")
  return f"{commit}{'-dirty' if dirty else ''}"


def _gkeyll_info() -> str:
  build = gpython.build_info()
  if build is None:
    return "not built (no compiled Gkeyll bridge -- reinstall Postgkyl)"
  return (f"{build['gkeyll_commit'][:12]} ({build['gkeyll_branch']}, "
          f"committed {build['gkeyll_commit_date']})")


def _dependency_versions() -> str:
  versions = []
  for name in _DEPENDENCIES:
    try:
      versions.append(f"{name} {importlib.metadata.version(name)}")
    except importlib.metadata.PackageNotFoundError:
      continue
  return ", ".join(versions)


def _pytest_info() -> str:
  spec = importlib.util.find_spec("pytest")
  if spec is None:
    return ("NOT INSTALLED in this Python environment; from the checkout run "
            "python -m pip install -e '.[test]'")
  try:
    version = importlib.metadata.version("pytest")
  except importlib.metadata.PackageNotFoundError:
    version = "unknown version"
  return f"{version} ({spec.origin})"


def version_report(version: str) -> str:
  """Build the full ``pgkyl --version`` report.

  Args:
    version: Installed postgkyl version string.

  Returns:
    A multiline environment and build report suitable for bug reports.
  """
  build = gpython.build_info()
  bridge = "available" if gpython.available() else "unavailable"
  built_with = "unknown (no build metadata)"
  if build is not None:
    arch = build["build_arch_flags"] or "compiler default"
    bridge += f" (built {build['build_date']}, CC={build['build_cc']}, ARCH_FLAGS={arch})"
    built_with = (f"Python {build.get('build_python', 'unknown')}, "
                  f"NumPy {build.get('build_numpy', 'unknown')}")
  return "\n".join([
      f"pgkyl, version {version}",
      f"postgkyl commit: {_postgkyl_commit()}",
      f"Gkeyll:          {_gkeyll_info()}",
      f"gpython bridge:  {bridge}",
      f"Built with:      {built_with}",
      f"Python:          {platform.python_implementation()} {platform.python_version()}",
      f"Interpreter:     {sys.executable}",
      f"Environment:     {sys.prefix}",
      f"Package:         {pathlib.Path(__file__).resolve().parent}",
      f"Extension:       {gpython.lib_path() or 'unavailable'}",
      f"pytest:          {_pytest_info()}",
      f"Platform:        {platform.platform()}",
      f"Dependencies:    {_dependency_versions()}",
  ])


hidden("version reporting is handled by the manual --version front end")(
    version_report)
