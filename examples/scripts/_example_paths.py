"""Shared filesystem locations for the executable examples."""

from __future__ import annotations

import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]

TEST_DATA = _REPO_ROOT / "tests" / "test_data"


def prepare_output_dir() -> Path:
  """Return the configured output directory, creating it if necessary."""
  output_dir = Path(
      os.environ.get("PGKYL_EXAMPLE_OUTPUT", _SCRIPT_DIR / "output"))
  output_dir.mkdir(parents=True, exist_ok=True)
  return output_dir
