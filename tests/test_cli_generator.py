"""Schema, compilation, discovery, and generated invocation contracts."""

from __future__ import annotations

import ast
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

import click
from click.testing import CliRunner
import pytest

import postgkyl.cli.compiler as compiler
from postgkyl.cli_spec import (
    CliArgument, CliType, CommandSpec, DatasetRef, Execution, KeyValue,
    PipelineInput, ResultPolicy, Section, command,
)
from postgkyl.cli.app import COMMANDS, MODELS
from postgkyl.cli.compiler import (
    CodecKind, CommandCompilationError, build_click_command, compile_callable,
)
from postgkyl.cli.discovery import discover_public_surface
from postgkyl.cli.docstrings import DocstringError
from postgkyl.cli.state import DataSpace
from postgkyl.operations import average


class Format(Enum):
  TEXT = "text"
  BINARY = "binary"
# end


_CALLS = []


@command(CommandSpec(Section.UTILITY, Execution.LOAD,
    result=ResultPolicy.SILENT))
def codec_demo(required: int, *, optional: str | None = None,
    enabled: bool = False, mode: Literal["a", "b"] = "a",
    format: Format = Format.TEXT, paths: list[Path] = [],
    pair: tuple[int, float] = (1, 2.0),
    values: Annotated[dict[str, int] | None, KeyValue()] = None):
  """Exercise every lossless command codec.

  Args:
    required: Required integer value.
    optional: Optional string value.
    enabled: Explicit boolean value.
    mode: Literal mode.
    format: Output format.
    paths: Repeatable filesystem paths.
    pair: Fixed integer/float pair.
    values: Repeatable key/value mapping.
  """
  call = (required, optional, enabled, mode, format, paths, pair, values)
  _CALLS.append(call)
  return call
# end


def test_codec_models_and_round_trip(tmp_path):
  model = compile_callable(codec_demo)
  by_name = {parameter.name: parameter for parameter in model.parameters}
  assert by_name["required"].required
  assert by_name["enabled"].codec.kind is CodecKind.BOOLEAN
  assert by_name["mode"].codec.choices == ("a", "b")
  assert by_name["paths"].codec.multiple
  assert by_name["pair"].codec.nargs == 2
  assert by_name["values"].codec.kind is CodecKind.MAPPING

  path = tmp_path / "x"
  result = CliRunner().invoke(build_click_command(model), [
      "--required", "4", "--optional", "x", "--enabled", "True",
      "--mode", "b", "--format", "binary", "--paths", str(path),
      "--pair", "2", "3.5", "--values", "n=7",
  ], obj=DataSpace())
  assert result.exit_code == 0, result.output
  assert _CALLS[-1] == (
      4, "x", True, "b", Format.BINARY, [path], (2, 3.5), {"n": 7})
# end


def test_exact_option_projection_and_help_provenance():
  model = compile_callable(codec_demo)
  command_obj = build_click_command(model)
  assert {option.opts[0] for option in command_obj.params} == {
      "--required", "--optional", "--enabled", "--mode", "--format",
      "--paths", "--pair", "--values",
  }
  options = {option.name: option.opts for option in command_obj.params}
  assert options == {
      "required": ["--required", "-r"],
      "optional": ["--optional", "-o"],
      "enabled": ["--enabled", "-e"],
      "mode": ["--mode", "-m"],
      "format": ["--format", "-f"],
      "paths": ["--paths"],
      "pair": ["--pair"],
      "values": ["--values", "-v"],
  }
  assert next(p for p in command_obj.params if p.name == "required").help \
      == "Required integer value."
  assert command_obj.short_help == "Exercise every lossless command codec."
# end


def test_short_options_require_a_unique_initial_and_reserve_help():
  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def collisions(*, alpha: int = 0, another: int = 0,
      beta_value: int = 0, hidden: int = 0):
    """Exercise shorthand conflict handling.

    Args:
      alpha: First conflicting option.
      another: Second conflicting option.
      beta_value: Unambiguous option.
      hidden: Option whose initial belongs to help.
    """
  # end

  command_obj = build_click_command(compile_callable(collisions))
  options = {option.name: option.opts for option in command_obj.params}
  assert options == {
      "alpha": ["--alpha"],
      "another": ["--another"],
      "beta_value": ["--beta-value", "-b"],
      "hidden": ["--hidden"],
  }
# end


def test_cli_type_projects_a_broader_python_option():
  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def projected(*,
      output: Annotated[str | list[str] | None, CliType(str | None)] = None):
    """Project direct-Python options explicitly.

    Args:
      output: One command-line output path.
    """
  # end

  model = compile_callable(projected)
  assert [parameter.name for parameter in model.parameters] == ["output"]
  assert model.parameters[0].codec.python_type is str
# end


def test_strict_compilation_rejects_missing_docs_and_unknown_types():
  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def undocumented(value: int):
    """Missing parameter documentation."""
  # end

  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def any_value(value: Any):
    """An unsupported value.

    Args:
      value: Arbitrary value.
    """
  # end

  with pytest.raises(DocstringError, match="value"):
    compile_callable(undocumented)
  # end
  with pytest.raises(CommandCompilationError, match="Any"):
    compile_callable(any_value)
  # end
# end


def test_optional_annotation_does_not_make_a_required_option_optional():
  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def required_optional(value: str | None):
    """Accept an explicitly nullable required value.

    Args:
      value: Required value which may be null in direct Python calls.
    """
  # end

  model = compile_callable(required_optional)
  assert model.parameters[0].required
# end


def test_cli_argument_marker_projects_only_the_declared_parameter_positionally():
  calls = []

  @command(CommandSpec(Section.UTILITY, Execution.LOAD,
      result=ResultPolicy.SILENT))
  def positional(value: Annotated[str, CliArgument()], *, suffix: str = ""):
    """Accept one positional CLI value.

    Args:
      value: Required positional value.
      suffix: Optional suffix.
    """
    calls.append(value + suffix)
  # end

  command_obj = build_click_command(compile_callable(positional))
  assert isinstance(command_obj.params[0], click.Argument)
  assert isinstance(command_obj.params[1], click.Option)
  help_text = CliRunner().invoke(command_obj, ["--help"]).output
  assert "Arguments:" in help_text
  assert "VALUE  Required positional value." in help_text
  result = CliRunner().invoke(command_obj, ["hello", "--suffix", "!"],
      obj=DataSpace())
  assert result.exit_code == 0, result.output
  assert calls == ["hello!"]
  assert CliRunner().invoke(command_obj, ["--value", "hello"],
      obj=DataSpace()).exit_code != 0
# end


def test_cli_argument_marker_requires_a_positional_python_parameter():
  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def invalid(*, value: Annotated[str, CliArgument()]):
    """Reject a positional CLI marker on a keyword-only API parameter.

    Args:
      value: Invalid positional projection.
    """
  # end

  with pytest.raises(CommandCompilationError, match="positional Python"):
    compile_callable(invalid)
  # end
# end


def test_concrete_annotated_alias_is_not_reprocessed(monkeypatch):
  """Keep runtime CLI metadata authoritative over resolved string hints."""
  original = compiler.get_type_hints
  evaluated = None

  def track_evaluated(source, **kwargs):
    nonlocal evaluated
    evaluated = source.__annotations__
    return original(source, **kwargs)
  # end

  monkeypatch.setattr(compiler, "get_type_hints", track_evaluated)

  model = compile_callable(average)
  weight = next(parameter for parameter in model.parameters
      if parameter.name == "weight")
  assert "weight" not in evaluated
  assert weight.dataset_ref
  assert weight.codec.optional
# end


def test_public_inventory_is_total_unique_and_deterministic():
  first = discover_public_surface()
  second = discover_public_surface()
  assert first == second
  assert len({model.name for model in MODELS}) == len(MODELS)
  assert {command_obj.name for command_obj in COMMANDS} >= {
      "interpolate", "five-moment-pressure", "plot", "load",
  }
  assert {command_obj.name for command_obj in COMMANDS} == {
      model.name for model in MODELS}
# end


def test_command_spec_rejects_invalid_loader_state():
  with pytest.raises(ValueError, match="LOAD"):
    CommandSpec(Section.UTILITY, Execution.LOAD, consumes_inputs=True)
  # end
# end


def test_pipeline_input_adapter_receives_the_working_set():
  calls = []

  @command(CommandSpec(Section.UTILITY, Execution.TERMINAL_ALL,
      result=ResultPolicy.VALUE))
  def terminal(data: Annotated[list[object], PipelineInput()], *, value: int = 1):
    """Inspect the pipeline input.

    Args:
      data: Injected command-line working set.
      value: Value to return.
    """
    calls.append(data)
    return value
  # end

  member = object()
  space = DataSpace(datasets=[member])
  result = CliRunner().invoke(build_click_command(compile_callable(terminal)),
      ["--value", "7"], obj=space)
  assert result.exit_code == 0, result.output
  assert calls == [[member]]
  assert result.output == "7\n"
# end


def test_no_scientific_click_decorators_remain():
  allowed = {"app.py"}
  root = Path(__file__).parents[1] / "src" / "postgkyl" / "cli"
  offenders = []
  for path in root.rglob("*.py"):
    tree = ast.parse(path.read_text(), path)
    for node in ast.walk(tree):
      if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        continue
      # end
      if isinstance(node.func.value, ast.Name) and node.func.value.id == "click" \
          and node.func.attr in {"command", "option", "argument"} \
          and path.name not in allowed:
        offenders.append(str(path.relative_to(root)))
      # end
    # end
  # end
  assert offenders == []
# end


def test_every_generated_name_is_the_dashed_api_name():
  discovered = discover_public_surface()
  assert {item.name for item in discovered} == {model.name for model in MODELS}
  for item in discovered:
    assert "_" not in item.name
  # end
  for model, command_obj in zip(MODELS, COMMANDS):
    assert command_obj.name == model.name
    options = {parameter.name: parameter for parameter in command_obj.params}
    expected = {parameter.name for parameter in model.parameters
        if not parameter.injected}
    assert set(options) == expected
    initials = [parameter.name[0] for parameter in model.parameters
        if not parameter.injected and not parameter.argument]
    for parameter in model.parameters:
      if not parameter.injected:
        if parameter.argument:
          assert isinstance(options[parameter.name], click.Argument)
          expected_opts = [parameter.name]
        # end
        else:
          assert isinstance(options[parameter.name], click.Option)
          expected_opts = ["--" + parameter.name.replace("_", "-")]
          if initials.count(parameter.name[0]) == 1 and parameter.name[0] != "h":
            expected_opts.append("-" + parameter.name[0])
          # end
        # end
        assert options[parameter.name].opts == expected_opts
      # end
    # end
  # end
# end


def test_cli_has_no_manual_command_package_or_compatibility_layer():
  root = Path(__file__).parents[1] / "src" / "postgkyl" / "cli"
  assert not (root / "commands").exists()
  assert not (root / "compat.py").exists()
  assert not (root / "legacy.py").exists()
# end
