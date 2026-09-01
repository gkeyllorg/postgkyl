"""Schema, compilation, discovery, and generated invocation contracts."""

from __future__ import annotations

import ast
from enum import Enum
import inspect
from pathlib import Path
from typing import Annotated, Any, Literal

import click
from click.testing import CliRunner
import pytest

from postgkyl.command_spec import (
    CommandSpec, DatasetRef, Execution, KeyValue, PipelineInput, ResultPolicy,
    Section, canonical_callable, command, fluent,
)
from postgkyl.cli.commands import COMMANDS, MODELS
from postgkyl.cli.compiler import (
    CodecKind, CommandCompilationError, build_click_command, compile_callable,
)
from postgkyl.cli.discovery import discover_public_surface
from postgkyl.cli.docstrings import DocstringError
from postgkyl.cli.state import DataSpace


class Format(Enum):
  TEXT = "text"
  BINARY = "binary"
# end


_CALLS = []


@command(CommandSpec(Section.UTILITY, Execution.LOAD, selectable=False,
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
  assert next(p for p in command_obj.params if p.name == "required").help \
      == "Required integer value."
  assert command_obj.short_help == "Exercise every lossless command codec."
# end


def test_strict_compilation_rejects_missing_docs_and_unknown_types():
  @command(CommandSpec(Section.UTILITY, Execution.LOAD, selectable=False))
  def undocumented(value: int):
    """Missing parameter documentation."""
  # end

  @command(CommandSpec(Section.UTILITY, Execution.LOAD, selectable=False))
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
  @command(CommandSpec(Section.UTILITY, Execution.LOAD, selectable=False))
  def required_optional(value: str | None):
    """Accept an explicitly nullable required value.

    Args:
      value: Required value which may be null in direct Python calls.
    """
  # end

  model = compile_callable(required_optional)
  assert model.parameters[0].required
# end


def test_fluent_binding_preserves_the_canonical_contract():
  @command(CommandSpec(Section.VERBS, Execution.MAP_REPLACE))
  def operation(data: object, *, count: int = 1):
    """Repeat a data operation.

    Args:
      data: Pipeline dataset.
      count: Repetition count.
    """
    return data, count
  # end

  class Surface:
    repeat = fluent(operation)
  # end

  assert canonical_callable(Surface.repeat) is operation
  assert tuple(inspect.signature(Surface.repeat).parameters) == ("self", "count")
  assert Surface().repeat(count=3)[1] == 3
# end


def test_public_inventory_is_total_unique_and_deterministic():
  first = discover_public_surface()
  second = discover_public_surface()
  assert first == second
  assert len({model.name for model in MODELS}) == len(MODELS)
  assert {command_obj.name for command_obj in COMMANDS} >= {
      "interpolate", "five-moment-pressure", "plot", "load",
  }
# end


def test_command_spec_rejects_invalid_loader_state():
  with pytest.raises(ValueError, match="LOAD"):
    CommandSpec(Section.UTILITY, Execution.LOAD)
  # end
# end


def test_session_adapter_receives_only_the_dataspace():
  calls = []

  @command(CommandSpec(Section.UTILITY, Execution.SESSION, selectable=False,
      result=ResultPolicy.VALUE))
  def session(space: Annotated[object, PipelineInput()], *, value: int = 1):
    """Inspect session state.

    Args:
      space: Injected command-line session.
      value: Value to return.
    """
    calls.append(space)
    return value
  # end

  space = DataSpace()
  result = CliRunner().invoke(build_click_command(compile_callable(session)),
      ["--value", "7"], obj=space)
  assert result.exit_code == 0, result.output
  assert calls == [space]
  assert result.output == "7\n"
# end


def test_no_scientific_click_decorators_remain():
  allowed = {"app.py", "print.py", "status.py"}
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
