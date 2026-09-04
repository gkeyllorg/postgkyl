"""Deterministic coverage for version-report fallback paths."""

from __future__ import annotations

import importlib.metadata
from importlib import import_module
import subprocess

import pytest

version = import_module("postgkyl._version")

pytestmark = pytest.mark.compatibility


def test_git_returns_none_outside_a_checkout(tmp_path):
  assert version._git(tmp_path, "status") is None


@pytest.mark.parametrize(
    "error", [OSError("missing git"),
              subprocess.CalledProcessError(1, "git")])
def test_git_converts_process_failures_to_missing(monkeypatch, tmp_path, error):
  (tmp_path / ".git").mkdir()

  def fail(*_args, **_kwargs):
    raise error

  monkeypatch.setattr(version.subprocess, "run", fail)
  assert version._git(tmp_path, "status") is None


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (None, "unknown (not a git checkout)"),
        ({
            "postgkyl_build_commit": "unknown"
        }, "unknown (not a git checkout)"),
        ({
            "postgkyl_build_commit": "1234567890abcdef"
        }, "1234567890ab (baked at build time, not a git checkout)"),
    ],
)
def test_postgkyl_commit_uses_baked_fallback(monkeypatch, build, expected):
  monkeypatch.setattr(version, "_git", lambda *_args: None)
  monkeypatch.setattr(version.gpython, "build_info", lambda: build)
  assert version._postgkyl_commit() == expected


def test_postgkyl_commit_marks_a_dirty_checkout(monkeypatch):
  answers = iter(["abcdef123456", " M changed.py"])
  monkeypatch.setattr(version, "_git", lambda *_args: next(answers))
  assert version._postgkyl_commit() == "abcdef123456-dirty"


def test_gkeyll_info_reports_an_unbuilt_bridge(monkeypatch):
  monkeypatch.setattr(version.gpython, "build_info", lambda: None)
  assert "not built" in version._gkeyll_info()


def test_dependency_versions_omits_missing_distributions(monkeypatch):

  def distribution_version(name):
    if name == "scipy":
      raise importlib.metadata.PackageNotFoundError(name)
    return "1.2.3"

  monkeypatch.setattr(version.importlib.metadata, "version",
                      distribution_version)
  report = version._dependency_versions()
  assert "numpy 1.2.3" in report
  assert "scipy" not in report


def test_version_report_without_build_metadata(monkeypatch):
  monkeypatch.setattr(version.gpython, "build_info", lambda: None)
  monkeypatch.setattr(version.gpython, "available", lambda: False)
  monkeypatch.setattr(version, "_postgkyl_commit", lambda: "unknown")
  monkeypatch.setattr(version, "_gkeyll_info", lambda: "not built")
  monkeypatch.setattr(version, "_dependency_versions", lambda: "numpy 1")
  report = version.version_report("2.0")
  assert "pgkyl, version 2.0" in report
  assert "gpython bridge:  unavailable" in report
  assert "ARCH_FLAGS" not in report
