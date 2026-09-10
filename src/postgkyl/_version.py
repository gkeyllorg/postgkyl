"""Debugging statistics behind ``pgkyl --version`` -- not part of the public
computing API (only ``cli/app.py``'s ``--version`` flag reads this).

Reports the postgkyl commit this checkout is at, the vendored Gkeyll commit
it was built against (via ``gpython.build_info()``, generated at build time
by scripts/build_gpython.sh since gkeyll/ is a build-time-only clone), and
interpreter/platform/dependency versions -- everything a bug report needs
without asking the user to gather it by hand.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import platform
import subprocess

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
    return "not built (no compiled Gkeyll bridge -- see scripts/build_gkeyll.sh)"
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


def version_report(version: str) -> str:
  """Build the full ``pgkyl --version`` report.

  Args:
    version: Installed postgkyl version string.

  Returns:
    A multiline environment and build report suitable for bug reports.
  """
  build = gpython.build_info()
  bridge = "available" if gpython.available() else "unavailable"
  if build is not None:
    arch = build["build_arch_flags"] or "compiler default"
    bridge += f" (built {build['build_date']}, CC={build['build_cc']}, ARCH_FLAGS={arch})"
  return "\n".join([
      f"pgkyl, version {version}",
      f"postgkyl commit: {_postgkyl_commit()}",
      f"Gkeyll:          {_gkeyll_info()}",
      f"gpython bridge:  {bridge}",
      f"Python:          {platform.python_implementation()} {platform.python_version()}",
      f"Platform:        {platform.platform()}",
      f"Dependencies:    {_dependency_versions()}",
  ])


hidden("version reporting is handled by the manual --version front end")(
    version_report)
