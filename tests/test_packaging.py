"""Regression tests for the custom native packaging commands."""

import runpy
from pathlib import Path

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
