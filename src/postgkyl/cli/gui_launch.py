"""The ``pgkyl-gui`` console entry point: serve the notebook with marimo.

It lives beside ``pgkyl`` because the CLI owns the console programs: the
notebook (``postgkyl.gui``) is built on the CLI session, and is reached here
by path, never imported.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.normpath(os.path.join(_HERE, os.pardir, "gui",
                                         "notebook.py"))

PROGRAM = "pgkyl-gui"
DESCRIPTION = "Open the Postgkyl GUI."

# The sample data of a source checkout (src/postgkyl/cli -> tests/test_data).
SAMPLE_DATA = os.path.normpath(
    os.path.join(_HERE, os.pardir, os.pardir, os.pardir, "tests", "test_data"))


def default_path() -> str:
  """The directory opened without ``--path``: the checkout's sample data
  (``tests/test_data``), or the current directory in an installation that
  ships without the tests."""
  return SAMPLE_DATA if os.path.isdir(SAMPLE_DATA) else os.getcwd()


def command(path: str | None, state: str | None = None) -> list[str]:
  """The command serving the notebook on the data directory ``path`` and the
  saved GUI ``state`` file, each when given.

  Everything after ``--`` reaches the notebook through ``mo.cli_args()``.
  marimo's global ``-y`` makes Ctrl+C stop the server at once instead of
  asking for confirmation (``marimo run`` has no other prompt it answers).
  """
  notebook_args = []
  for option, value in (("--path", path), ("--state", state)):
    if value is not None:
      notebook_args += [option, os.path.abspath(os.path.expanduser(value))]
  return [sys.executable, "-m", "marimo", "-y", "run", NOTEBOOK, "--"
          ] + notebook_args


def serve(cmd: list[str]) -> int:
  """Run marimo and return its exit status.

  Ctrl+C reaches every process in the terminal's foreground group. marimo
  shuts down on it; this launcher only waits, so it ignores the interrupt
  instead of dying with a ``KeyboardInterrupt`` traceback. It does so only
  once marimo has started, which keeps the default handling for marimo.
  """
  process = subprocess.Popen(cmd)
  previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
  try:
    return process.wait()
  finally:
    signal.signal(signal.SIGINT, previous)


def main(argv: list[str] | None = None) -> int:
  """Parse ``--path`` and ``--state``, serve the notebook; return the exit
  status."""
  parser = argparse.ArgumentParser(prog=PROGRAM, description=DESCRIPTION)
  parser.add_argument(
      "--path",
      "-p",
      default=None,
      help="Gkeyll data directory (default: the state's directory, else the "
      "source checkout's tests/test_data, or the current directory).")
  parser.add_argument(
      "--state",
      "-s",
      default=None,
      help="GUI state file saved with 'Save state', reopened as it was.")
  args = parser.parse_args(argv)
  if args.state and not os.path.isfile(os.path.expanduser(args.state)):
    print(f"pgkyl-gui: no state file {args.state!r}", file=sys.stderr)
    return 2
  if importlib.util.find_spec("marimo") is None:
    print(
        "pgkyl-gui needs marimo; install it with "
        "\"pip install 'postgkyl[gui]'\".",
        file=sys.stderr)
    return 1
  # A state file brings its own directory; --path still overrides it.
  path = args.path or (None if args.state else default_path())
  return serve(command(path, args.state))
