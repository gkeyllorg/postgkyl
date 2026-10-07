"""``PostgkylSession`` runs the CLI from Python and records its command line."""

from __future__ import annotations

import inspect
from pathlib import Path
import re
import shlex
import textwrap

import click
from click.testing import CliRunner
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest

import postgkyl as pg
from postgkyl.cli import PostgkylSession
from postgkyl.cli.app import MODELS, cli

ROOT = Path(__file__).parents[1]
DATA = ROOT / "tests" / "test_data" / "generated"
DISTF = DATA / "distf_p2_0.gkyl"
TUTORIAL = ROOT / "docs" / "source" / "cli-tutorial.rst"


def _pgkyl(command_line: str) -> str:
  """What running ``command_line`` (``pgkyl ...``) prints."""
  program, *args = shlex.split(command_line)
  assert program == "pgkyl"
  result = CliRunner().invoke(cli, args)
  assert result.exit_code == 0, result.output
  return result.output


def test_session_matches_the_fluent_api_and_the_printed_command(capsys):
  s = PostgkylSession()
  s.load(DISTF)
  s.interpolate()
  s.select(z0=0.5, comp=0)
  assert s.command() == f"pgkyl {DISTF} interpolate select --comp 0 --z0 0.5"

  expected = pg.load(DISTF).interpolate().select(z0=0.5, comp=0)
  (data, ) = s.datasets
  np.testing.assert_array_equal(data.values, expected.values)

  s.print()
  assert capsys.readouterr().out == _pgkyl(s.command())


def test_a_dash_leading_positional_survives_the_command_line(capsys):
  s = PostgkylSession()
  s.load(DISTF)
  s.evaluate("-1 f0 *")
  assert s.command() == f"pgkyl {DISTF} evaluate -- '-1 f0 *'"
  s.print()
  assert capsys.readouterr().out == _pgkyl(s.command())


def test_a_failed_call_is_neither_recorded_nor_applied():
  s = PostgkylSession()
  s.load(DISTF)
  before = s.datasets
  with pytest.raises(click.UsageError, match="no dataset tagged 'missing'"):
    s.average([0], weight="missing")
  with pytest.raises(TypeError, match="no command-line spelling"):
    s.select(z0=slice(1, 2))
  assert s.command() == f"pgkyl {DISTF}"
  assert s.datasets == before


def test_a_missing_file_is_not_recorded():
  s = PostgkylSession()
  with pytest.raises(click.UsageError):
    s.load("does-not-exist.gkyl")
  assert s.command() == "pgkyl"


def test_a_call_that_would_break_the_command_line_is_refused():
  s = PostgkylSession()
  s.load(DISTF)
  s.integrate()
  # On the command line, `integrate info` reads `info` as integrate's axis.
  with pytest.raises(ValueError, match="'integrate' would read 'info'"):
    s.info()
  assert s.command() == f"pgkyl {DISTF} integrate"


def test_methods_are_derived_from_the_cli_models():
  s = PostgkylSession()
  own = {"command", "print_cmd", "datasets"}
  names = {model.name for model in MODELS}
  assert not own & names
  assert names <= set(dir(s))
  for model in MODELS:
    method = getattr(s, model.name)
    exposed = [p.name for p in model.parameters if not p.injected]
    assert list(inspect.signature(method).parameters) == exposed
    assert method.__doc__.startswith(model.long_help)
  with pytest.raises(AttributeError, match="no command or attribute"):
    s.not_a_command


def test_repr_shows_the_command_line():
  s = PostgkylSession()
  s.load(DISTF)
  assert repr(s) == f"PostgkylSession('pgkyl {DISTF}')"


def _code_block(text: str, language: str) -> str:
  """The body of the first ``.. code-block:: language`` in ``text``."""
  match = re.search(rf"\.\. code-block:: {language}\n\n((?:   .*\n|\n)+)", text)
  return textwrap.dedent(match.group(1)).strip() + "\n"


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_the_tutorial_example_prints_a_command_drawing_the_same_figure(
    tmp_path, monkeypatch, capsys):
  section = TUTORIAL.read_text().split("Building a command line from Python")[1]
  script = _code_block(section, "python")
  command = _code_block(section, "bash")
  (tmp_path / "tests").mkdir()
  (tmp_path / "tests" / "test_data").symlink_to(ROOT / "tests" / "test_data")
  monkeypatch.chdir(tmp_path)
  try:
    exec(script, {})
    assert capsys.readouterr().out == command
    session_figure = mpimg.imread("density.png")
    _pgkyl(command)
    np.testing.assert_array_equal(mpimg.imread("density.png"), session_figure)
  finally:
    plt.close("all")
