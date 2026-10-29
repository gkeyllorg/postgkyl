"""Quantity discovery and generic eager informational-option contracts."""
from typing import Annotated

import click
from click.testing import CliRunner
import pytest

import postgkyl.cli.compiler as compiler
from postgkyl.cli.app import cli, COMMANDS
from postgkyl.cli.state import DataSpace
from postgkyl.cli_spec import (ChoiceProvider, CommandSpec, Execution,
                               InfoOption, ResultPolicy, Section, command)
from postgkyl.diagnostics.gk.registry import gk_quant_registry


@pytest.mark.parametrize("name", ["gk_load_q", "gk_load_quantity"])
@pytest.mark.parametrize("flag", ["-a", "--avail_quant"])
def test_catalog_requires_no_inputs_and_never_loads(name, flag, monkeypatch):
  """Every registered quantity is described, without calling a data loader."""

  def no_dispatch(*args, **kwargs):
    pytest.fail("An informational flag must not execute the pipeline")

  monkeypatch.setattr(compiler, "execute_model", no_dispatch)
  result = CliRunner().invoke(cli, [name, flag])
  assert result.exit_code == 0, result.output
  rows = [
      line.split(maxsplit=1) for line in result.output.splitlines()
      if line and not line[0].isspace()
  ]
  assert [row[0] for row in rows] == gk_quant_registry.list()
  assert all(len(row) == 2 for row in rows)
  for quantity in gk_quant_registry.list():
    description = gk_quant_registry.get(quantity).description
    assert description.strip(), quantity
    assert "\n" not in description
  # Descriptions make the basis explicit; geometry retains its established name.
  assert "Contravariant" in result.output
  assert "Covariant" in result.output
  assert "geo_int_b_i" in result.output


@pytest.mark.parametrize("width", [50, 80, 120])
def test_catalog_alignment_wrapping_and_complete_descriptions(width):
  """Wrapped descriptions remain in their column and retain every word."""
  result = CliRunner().invoke(cli, ["gk_load_q", "-a"], terminal_width=width)
  assert result.exit_code == 0, result.output
  names = gk_quant_registry.list()
  column = max(map(len, names)) + 2
  rows = {}
  current = None
  for line in result.output.splitlines():
    assert len(line) <= width
    if line[0].isspace():
      assert line[:column] == " " * column
    else:
      current = line.split()[0]
      assert line[len(current):column] == " " * (column - len(current))
      rows[current] = []
    rows[current].append(line[column:])
  assert list(rows) == names
  for name, parts in rows.items():
    # Click may wrap at hyphens as well as whitespace.
    assert "".join("".join(parts).split()) == "".join(
        gk_quant_registry.get(name).description.split())


@pytest.mark.parametrize("name", ["gk_load_q", "gk_load_quantity"])
def test_quantity_help_is_concise(name):
  result = CliRunner().invoke(cli, [name, "-h"])
  assert result.exit_code == 0, result.output
  normalized = " ".join(result.output.split())
  assert "-q, --quantity TEXT" in normalized
  assert "Quantity to load, see -a, --avail_quant for available quantities." in normalized
  assert "-a, --avail_quant" in normalized
  assert "ExB_vel_contra" not in result.output
  assert "geo_int_b_i" not in result.output


def test_choice_validation_and_completion_are_preserved():
  generated = next(command for command in COMMANDS
                   if command.name == "gk_load_quantity")
  option = next(parameter for parameter in generated.params
                if parameter.name == "quantity")
  assert isinstance(option.type, click.Choice)
  assert list(option.type.choices) == gk_quant_registry.list()
  with click.Context(generated) as ctx:
    assert option.type.convert("gradB_vel_contra", option,
                               ctx) == "gradB_vel_contra"
    assert [item.value for item in option.shell_complete(ctx, "gradB")
            ] == ["gradB_vel_contra"]
  result = CliRunner().invoke(cli, ["gk_load_q", "-q", "not_a_quantity"])
  assert result.exit_code == 2
  assert "Invalid value for '--quantity'" in result.output


def test_catalog_exits_before_chained_commands(monkeypatch):

  def no_dispatch(*args, **kwargs):
    pytest.fail("Listing must exit before any command executes")

  monkeypatch.setattr(compiler, "execute_model", no_dispatch)
  result = CliRunner().invoke(
      cli, ["load", "-f", "absent.gkyl", "gk_load_q", "-a", "info"])
  assert result.exit_code == 0, result.output
  assert "geo_int_b_i" in result.output


def test_generic_info_option_uses_owner_rows_and_preserves_python_call():
  """Informational flags are metadata, not artificial loader parameters."""
  calls = []

  @command(
      CommandSpec(Section.UTILITY,
                  Execution.LOAD,
                  result=ResultPolicy.SILENT,
                  info_options=(InfoOption(
                      "available", lambda: [("first", "A short description.")],
                      "List choices and exit."), )))
  def sample(
      required: Annotated[str,
                          ChoiceProvider(lambda: ["first"], metavar="TEXT")]):
    """A sample command.

    Args:
      required: Required choice.
    """
    calls.append(required)

  generated = compiler.build_click_command(compiler.compile_callable(sample))
  result = CliRunner().invoke(generated, ["-a"])
  assert result.exit_code == 0, result.output
  assert result.output == "first  A short description.\n"
  assert calls == []
  # Resilient parsing, used for completion, must not print or exit.
  with click.Context(generated, resilient_parsing=True) as ctx:
    info = next(option for option in generated.params
                if option.name == "available")
    assert info.callback(ctx, info, True) is None
  result = CliRunner().invoke(generated, ["--required", "first"],
                              obj=DataSpace())
  assert result.exit_code == 0, result.output
  sample("first")
  assert calls == ["first", "first"]


def test_info_option_cannot_shadow_api_parameter():

  @command(
      CommandSpec(Section.UTILITY,
                  Execution.LOAD,
                  info_options=(InfoOption("available", lambda: [],
                                           "Show choices."), )))
  def sample(available: str):
    """A sample command.

    Args:
      available: Required input.
    """

  with pytest.raises(compiler.CommandCompilationError,
                     match="conflicts with an API parameter"):
    compiler.compile_callable(sample)


def test_info_option_does_not_steal_an_existing_short_option():

  @command(
      CommandSpec(Section.UTILITY,
                  Execution.LOAD,
                  info_options=(InfoOption("available", lambda: [],
                                           "Show choices."), )))
  def sample(argument: str):
    """A sample command.

    Args:
      argument: Required input.
    """

  generated = compiler.build_click_command(compiler.compile_callable(sample))
  options = {option.name: option.opts for option in generated.params}
  assert options["argument"] == ["--argument", "-a"]
  assert options["available"] == ["--available"]


@pytest.mark.parametrize("name", ["help", "--available", "", "two words"])
def test_info_option_rejects_invalid_names(name):
  with pytest.raises(ValueError, match="identifier"):
    InfoOption(name, lambda: [], "List choices.")


def test_info_option_metadata_validation():
  with pytest.raises(TypeError, match="callable"):
    InfoOption("available", [], "List choices.")
  with pytest.raises(ValueError, match="non-empty help"):
    InfoOption("available", lambda: [], " ")
  info = InfoOption("available", lambda: [], "List choices.")
  with pytest.raises(ValueError, match="duplicate"):
    CommandSpec(Section.UTILITY, Execution.LOAD, info_options=(info, info))
  with pytest.raises(TypeError, match="tuple of InfoOption"):
    CommandSpec(Section.UTILITY, Execution.LOAD, info_options=("available", ))
  with pytest.raises(ValueError, match="non-empty text"):
    ChoiceProvider(lambda: ["first"], metavar="")
