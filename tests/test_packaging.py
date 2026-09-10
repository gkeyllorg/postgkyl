"""Regression tests for the custom native packaging commands."""

import runpy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import setuptools
from setuptools.dist import Distribution

ROOT_DIR = Path(__file__).parents[1]


def _build_py_command(monkeypatch, tmp_path, *, editable):
  monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
  namespace = runpy.run_path(ROOT_DIR / "setup.py")
  command_type = namespace["BuildPyWithGkeyll"]
  native_library = tmp_path / "libg0core.so"
  native_library.write_bytes(b"native library")
  monkeypatch.setitem(command_type.run.__globals__, "_build_gkeyll",
                      lambda: True)
  monkeypatch.setitem(command_type.run.__globals__, "BUNDLED_LIB",
                      native_library)

  command = command_type(Distribution())
  command.ensure_finalized()
  command.build_lib = str(tmp_path / "missing-build")
  command.editable_mode = editable
  return command


def test_editable_build_uses_native_artifacts_in_source(monkeypatch, tmp_path):
  command = _build_py_command(monkeypatch, tmp_path, editable=True)

  command.run()

  assert not Path(command.build_lib).exists()


def test_wheel_build_creates_native_library_destination(monkeypatch, tmp_path):
  command = _build_py_command(monkeypatch, tmp_path, editable=False)

  command.run()

  bundled = Path(command.build_lib) / "postgkyl/gpython/libg0core.so"
  assert bundled.read_bytes() == b"native library"


def test_sdist_rebuild_preserves_version_without_git(tmp_path):
  """Exercise the real backend on a small package, including the sdist hop."""
  root = tmp_path / "checkout"
  package = root / "src/postgkyl"
  package.mkdir(parents=True)
  for name in ("setup.py", "pyproject.toml", "MANIFEST.in", "README.md",
               "LICENSE"):
    shutil.copyfile(ROOT_DIR / name, root / name)
  shutil.copyfile(ROOT_DIR / "src/postgkyl/_release.py",
                  package / "_release.py")
  (package /
   "__init__.py").write_text("from postgkyl._release import __version__\n")

  def git(*args):
    return subprocess.run(["git", *args],
                          cwd=root,
                          check=True,
                          capture_output=True)

  git("init", "-b", "main")
  git("config", "user.name", "Packaging test")
  git("config", "user.email", "packaging@example.com")
  git("add", ".")
  git("commit", "-m", "release")
  git("tag", "v2.3.4.0")
  git("commit", "--allow-empty", "-m", "change")
  artifacts = tmp_path / "dist"
  result = subprocess.run([
      sys.executable, "-m", "build", "--no-isolation", "--outdir",
      str(artifacts)
  ],
                          cwd=root,
                          env=os.environ | {"POSTGKYL_SKIP_GKEYLL_BUILD": "1"},
                          capture_output=True,
                          text=True)
  assert result.returncode == 0, result.stdout + result.stderr
  with tarfile.open(artifacts / "postgkyl-2.3.4.1.tar.gz") as archive:
    assert archive.extractfile(
        "postgkyl-2.3.4.1/src/postgkyl/_release_version.txt").read(
        ) == b"2.3.4.1\n"
  with zipfile.ZipFile(artifacts /
                       "postgkyl-2.3.4.1-py3-none-any.whl") as wheel:
    assert wheel.read("postgkyl/_release_version.txt") == b"2.3.4.1\n"
    assert b"\nVersion: 2.3.4.1\n" in wheel.read(
        "postgkyl-2.3.4.1.dist-info/METADATA")
    installed = tmp_path / "installed"
    wheel.extractall(installed)
  result = subprocess.run([
      sys.executable, "-c",
      "import postgkyl; from importlib.metadata import version; "
      "assert postgkyl.__version__ == version('postgkyl') == '2.3.4.1'"
  ],
                          cwd=installed,
                          capture_output=True,
                          text=True)
  assert result.returncode == 0, result.stdout + result.stderr
