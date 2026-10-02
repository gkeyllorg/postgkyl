"""The ``pgkyl-gui`` console entry point: serve the notebook with marimo."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys

NOTEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "notebook.py")


def command(path: str) -> list[str]:
  """The command serving the notebook on the data directory ``path``.

  Everything after ``--`` reaches the notebook through ``mo.cli_args()``.
  """
  return [
      sys.executable, "-m", "marimo", "run", NOTEBOOK, "--", "--path",
      os.path.abspath(os.path.expanduser(path))
  ]


def main(argv: list[str] | None = None) -> int:
  """Parse ``--path`` and serve the notebook; return the exit status."""
  parser = argparse.ArgumentParser(
      prog="pgkyl-gui",
      description="Open the Postgkyl notebook GUI on a data directory.")
  parser.add_argument("--path",
                      "-p",
                      default=os.getcwd(),
                      help="Gkeyll data directory (default: the current one).")
  args = parser.parse_args(argv)
  if importlib.util.find_spec("marimo") is None:
    print(
        "pgkyl-gui needs marimo; install it with "
        "\"pip install 'postgkyl[gui]'\".",
        file=sys.stderr)
    return 1
  return subprocess.call(command(args.path))
