"""``PostgkylSession`` runs the CLI from Python and records its command line."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import runpy
import shlex

import click
from click.testing import CliRunner
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest

import postgkyl as pg
from postgkyl.cli import PostgkylSession
from postgkyl.cli.app import MODELS, cli
from postgkyl.cli.session import _stub_docstring, render_stub

ROOT = Path(__file__).parents[1]
DATA = ROOT / "tests" / "test_data" / "generated"
DISTF = DATA / "distf_p2_0.gkyl"
EXAMPLE = ROOT / "examples" / "scripts" / "13_postgkyl_session.py"
STUB = ROOT / "src" / "postgkyl" / "cli" / "session.pyi"


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


def test_a_dash_leading_file_name_is_loaded_by_option(tmp_path, monkeypatch,
                                                      capsys):
  monkeypatch.chdir(tmp_path)
  Path("-distf.gkyl").symlink_to(DISTF)
  s = PostgkylSession()
  s.load("-distf.gkyl")
  # A bare "-distf.gkyl" would read as an option.
  assert s.command() == "pgkyl load --file_name -distf.gkyl"
  s.print()
  assert capsys.readouterr().out == _pgkyl(s.command())


def test_a_failed_call_is_neither_recorded_nor_applied():
  s = PostgkylSession()
  s.load(DISTF)
  before = s.datasets
  with pytest.raises(click.UsageError, match="no dataset tagged 'missing'"):
    s.average(0, weight="missing")
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
  own = {"command", "print_cli", "datasets"}
  names = {model.name for model in MODELS}
  assert not own & names
  assert names <= set(dir(s))
  for model in MODELS:
    method = getattr(s, model.name)
    exposed = [p.name for p in model.parameters if not p.injected]
    assert list(inspect.signature(method).parameters) == exposed
    assert method.__doc__ == (inspect.getdoc(model.canonical)
                              or model.long_help)
  with pytest.raises(AttributeError, match="no command or attribute"):
    s.not_a_command


def test_the_ide_stub_is_generated_from_the_current_commands():
  assert STUB.read_text() == render_stub(), (
      "session.pyi is out of date: run "
      "`python scripts/generate_session_stub.py`")


@pytest.mark.parametrize("text", [
    "Summary.\n\n  Indented detail.", 'Quotes """inside""".',
    "Ends with a backslash \\", r"Keeps \d and $\alpha$."
])
def test_a_stub_docstring_reads_back_as_its_text(text):
  statement = _stub_docstring(text, "        ")
  value = ast.literal_eval(statement.strip())
  # rstrip: the closing quotes' indentation, which PEP 257 trimming drops.
  assert inspect.cleandoc(value).rstrip() == inspect.cleandoc(text)


def test_the_stub_declares_every_command_with_its_docstring():
  stub = render_stub()
  compile(stub, str(STUB), "exec")
  for model in MODELS:
    assert f"    def {model.name}(self" in stub
  assert "Interpolate DG (modal/nodal) data" in stub
  assert "def select(self, *, comp: str | float | None = None" in stub


def test_commands_chain_or_stand_alone_alike():
  chained = PostgkylSession()
  assert chained.load(DISTF).interpolate().select(z0=0.5, comp=0) is chained
  stepwise = PostgkylSession()
  stepwise.load(DISTF)
  stepwise.interpolate()
  stepwise.select(z0=0.5, comp=0)
  assert chained.command() == stepwise.command()
  np.testing.assert_array_equal(chained.datasets[0].values,
                                stepwise.datasets[0].values)


def test_result_is_what_the_last_command_returned():
  s = PostgkylSession()
  assert s.result is None
  s.load(DISTF)
  assert s.result is s.datasets[0]
  figure = s.interpolate().plot(no_show=True).result
  try:
    assert figure.axes
  finally:
    plt.close(figure)


def test_repr_shows_the_command_line():
  s = PostgkylSession()
  s.load(DISTF)
  assert repr(s) == f"PostgkylSession('pgkyl {DISTF}')"


def test_the_example_prints_a_command_drawing_the_same_figure(
    tmp_path, monkeypatch, capsys):
  monkeypatch.setenv("PGKYL_EXAMPLE_OUTPUT", str(tmp_path))
  monkeypatch.syspath_prepend(str(EXAMPLE.parent))
  try:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    (command, ) = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("pgkyl ")
    ]
    figure = tmp_path / "13_postgkyl_session.png"
    session_figure = mpimg.imread(figure)
    figure.unlink()
    _pgkyl(command)
    np.testing.assert_array_equal(mpimg.imread(figure), session_figure)
  finally:
    plt.close("all")
