"""The examples in ``examples/`` are the user-facing tutorial (README +
narrated scripts + a CLI walkthrough). This file is what keeps that tutorial
honest: every script's own ``assert`` statements run for real (via
``runpy``, so a broken example fails here with the same traceback a user
would see), and every ``pgkyl ...`` line quoted in ``examples/cli_tutorial.md``
is replayed through the real CLI. Nothing here re-describes the examples --
it just executes the one copy of them that already exists.
"""

from __future__ import annotations

import glob
import os
import re
import runpy
import shlex

import matplotlib
import pytest
from click.testing import CliRunner

matplotlib.use("Agg")

from postgkyl import gpython
from postgkyl.cli.app import cli

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
SCRIPTS = sorted(
    script for script in glob.glob(os.path.join(EXAMPLES, "scripts", "*.py"))
    if not os.path.basename(script).startswith("_"))
TUTORIAL = os.path.join(EXAMPLES, "cli_tutorial.md")


def _extract_cli_commands(markdown_path):
  """Pull every ``pgkyl ...`` invocation out of the ``` ```bash``` ``` fences
  in a tutorial markdown file, joining ``\\``-continued lines."""
  with open(markdown_path) as fh:
    text = fh.read()

  commands = []
  for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
    pending = []
    for line in block.strip("\n").splitlines():
      line = line.rstrip()
      if not line:
        continue
      if line.endswith("\\"):
        pending.append(line[:-1])
        continue
      pending.append(line)
      full = " ".join(pending).strip()
      pending = []
      if full.startswith("pgkyl "):
        commands.append(full[len("pgkyl "):])
  return commands


CLI_COMMANDS = _extract_cli_commands(TUTORIAL)


class TestTutorialScripts:
  """Every script under ``examples/scripts/`` runs to completion and its
  internal assertions pass -- exercised in-process via ``runpy`` so an
  ``AssertionError`` inside the example surfaces as this test's failure."""

  @needs_gkeyll
  @pytest.mark.parametrize("script",
                           SCRIPTS,
                           ids=[os.path.basename(s) for s in SCRIPTS])
  def test_script_runs_clean(self, script, tmp_path, monkeypatch):
    monkeypatch.setenv("PGKYL_EXAMPLE_OUTPUT", str(tmp_path))
    monkeypatch.syspath_prepend(os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")


class TestCliTutorial:
  """Every ``pgkyl ...`` line quoted in ``examples/cli_tutorial.md`` actually
  runs, from the repository root (the fixture paths in the tutorial are
  written relative to it, exactly as a reader would type them)."""

  def test_tutorial_has_commands(self):
    # A parsing regression (e.g. a fence typo) would otherwise silently
    # leave the parametrized test below with zero cases -- a green suite
    # that covers nothing.
    assert len(CLI_COMMANDS) >= 8

  @needs_gkeyll
  @pytest.mark.parametrize("command", CLI_COMMANDS)
  def test_command_succeeds(self, command, tmp_path, monkeypatch):
    # Symlink 'tests/' into an isolated cwd: the tutorial's relative input
    # paths (``tests/test_data/...``) resolve, but any file the command
    # writes (out.png, distf.npy, ...) lands in tmp_path, not the repo.
    os.symlink(os.path.join(ROOT, "tests"), os.path.join(tmp_path, "tests"))
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, shlex.split(command))
    assert result.exit_code == 0, (
        f"`pgkyl {command}` failed:\n{result.output}\n{result.exception}")
