"""Edge contracts for the generated CLI schema and runtime."""

from __future__ import annotations

from enum import Enum
from types import ModuleType, SimpleNamespace
from typing import Annotated, Literal

import click
from click.testing import CliRunner
import pytest

import postgkyl.cli.compiler as compiler
from postgkyl.cli.compiler import (
    CodecKind,
    CommandCompilationError,
    TypeCodec,
    build_click_command,
    compile_callable,
    compile_public_surface,
    group_by_section,
)
from postgkyl.cli.docstrings import DocstringError, parse_docstring
from postgkyl.cli.state import DataSpace
from postgkyl.cli_spec import (
    ChoiceProvider,
    CliArgument,
    CliHidden,
    CliType,
    CommandSpec,
    DatasetRef,
    Execution,
    KeyValue,
    PipelineInput,
    ResultPolicy,
    Section,
    command,
    command_spec,
    hidden,
    hidden_spec,
)
from postgkyl.cli import app as cli_app
from postgkyl.cli.discovery import (
    SurfaceClassificationError,
    _classify,
    _diagnostic_modules,
    discover_public_surface,
)

pytestmark = pytest.mark.compatibility


class _Dataset:
  """Small structural dataset used by the generic pipeline adapter."""

  def __init__(self, tag: str):
    self.tag = tag
    self.ctx = {}
    self.grid = []
    self.values = []


class _Group:

  def __init__(self, *datasets):
    self.datasets = datasets


class _BadEnum(Enum):
  COMPOSITE = (1, 2)


class _Mode(Enum):
  TEXT = "text"
  BINARY = "binary"


def _compiled_option(annotation):

  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def option(value):
    """Compile one option.

    Args:
      value: Value exposed by the command.
    """

  option.__annotations__ = {"value": annotation}
  return compile_callable(option)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({
            "section": "Utility",
            "execution": Execution.LOAD
        }, "enums"),
        ({
            "section": Section.UTILITY,
            "execution": "LOAD"
        }, "enums"),
        ({
            "section": Section.UTILITY,
            "execution": Execution.LOAD,
            "result": "DATA",
        }, "enums"),
        ({
            "section": Section.UTILITY,
            "execution": Execution.COMBINE,
            "consumes_inputs": 1,
        }, "bool value"),
        ({
            "section": Section.UTILITY,
            "execution": Execution.COMBINE,
            "order": True,
        }, "integer"),
        ({
            "section": Section.UTILITY,
            "execution": Execution.MAP_REPLACE,
            "consumes_inputs": True,
        }, "only MAP_APPEND and COMBINE"),
        ({
            "section": Section.UTILITY,
            "execution": Execution.TERMINAL_ALL,
        }, "non-DATA"),
    ],
)
def test_command_spec_rejects_invalid_states(kwargs, message):
  with pytest.raises((TypeError, ValueError), match=message):
    CommandSpec(**kwargs)


def test_cli_marker_records_reject_invalid_values():
  with pytest.raises(ValueError, match="non-empty"):
    DatasetRef("")
  with pytest.raises(TypeError, match="callable"):
    ChoiceProvider(1)
  with pytest.raises(ValueError, match="non-empty"):
    hidden("   ")


def test_command_and_hidden_markers_are_mutually_exclusive():
  spec = CommandSpec(Section.UTILITY, Execution.LOAD)

  def first():
    pass

  assert command(spec)(first) is first
  assert command(spec)(first) is first
  assert command_spec(first) == spec
  assert hidden_spec(first) is None
  with pytest.raises(ValueError, match="already has a CommandSpec"):
    hidden("not public")(first)
  with pytest.raises(ValueError, match="different CommandSpec"):
    command(CommandSpec(Section.VERBS, Execution.LOAD))(first)

  @hidden("not public")
  def second():
    pass

  assert hidden_spec(second).reason == "not public"
  assert command_spec(second) is None
  with pytest.raises(ValueError, match="already marked CliHidden"):
    command(spec)(second)
  with pytest.raises(TypeError, match="requires a CommandSpec"):
    command("load")


def test_discovery_rejects_unclassified_and_doubly_classified_callables():

  def unclassified():
    pass

  with pytest.raises(SurfaceClassificationError, match="unclassified"):
    _classify(unclassified, "example.unclassified")

  def doubly_classified():
    pass

  doubly_classified.__postgkyl_command_spec__ = CommandSpec(
      Section.UTILITY, Execution.LOAD)
  doubly_classified.__postgkyl_cli_hidden__ = CliHidden("synthetic conflict")
  with pytest.raises(SurfaceClassificationError,
                     match="both exposed and hidden"):
    _classify(doubly_classified, "example.conflict")


def test_discovery_handles_cycles_facade_diagnostics_and_nonfunction_variables(
):
  root = ModuleType("postgkyl.diagnostics")
  child = ModuleType("postgkyl.diagnostics.synthetic")
  root.__all__ = ["child", "alias"]
  root.child = child
  root.alias = child
  child.__all__ = []
  child.VARIABLES = {"not_callable": object()}

  modules = list(_diagnostic_modules(root))
  assert modules == [root, child]

  @hidden("classified at its diagnostic home")
  def diagnostic_alias():
    pass

  diagnostic_alias.__module__ = child.__name__

  class EmptySurface:
    pass

  facade = SimpleNamespace(GData=EmptySurface,
                           GDataGroup=EmptySurface,
                           __all__=["diagnostic_alias"],
                           diagnostic_alias=diagnostic_alias,
                           diagnostics=root)
  assert discover_public_surface(facade) == ()


def test_group_alias_with_missing_target_falls_through():
  group = cli_app.PgkylGroup()
  context = click.Context(group)
  assert group.get_command(context, "pl") is None


def test_group_help_ignores_missing_commands_and_empty_sections(monkeypatch):
  group = cli_app.PgkylGroup()
  context = click.Context(group)
  formatter = context.make_formatter()
  monkeypatch.setattr(cli_app, "COMMAND_SECTIONS", {
      "Missing": ["not_registered"],
      "Empty": [],
  })
  group.format_commands(context, formatter)
  assert formatter.getvalue() == ""


def test_group_resolve_command_with_no_arguments_delegates_to_click():
  group = cli_app.PgkylGroup()
  context = click.Context(group)
  with pytest.raises(IndexError):
    group.resolve_command(context, [])


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        (None, "missing docstring"),
        ("Args:", "missing first-paragraph"),
        ("Summary.\n\nArgs:\n  value: one\n\nArgs:\n  value: two",
         "duplicate Args"),
        ("Summary.\n\nArgs:\n  value:", "has no description"),
        ("Summary.\n\nArgs:\n  value: one\n  value: two", "documented twice"),
        ("Summary.\n\nArgs:\nnot an entry", "malformed Args entry"),
        ("Summary.\n\nArgs:\n  absent: no such parameter",
         "absent from signature"),
        ("Summary.", "is undocumented"),
    ],
)
def test_docstring_parser_reports_each_invalid_contract(doc, message):

  def target(value):
    pass

  target.__doc__ = doc
  with pytest.raises(DocstringError, match=message):
    parse_docstring(target, required={"value"}, signature_names={"value"})


def test_docstring_parser_collects_multiline_text_and_stops_at_section():

  def target(value):
    """Summary line continued
    on the next line.

    Longer narrative.

    Args:
      value: First part.
        Second part.

    Returns:
      Nothing.
    """

  parsed = parse_docstring(target,
                           required={"value"},
                           signature_names={"value"})
  assert parsed.summary == "Summary line continued on the next line."
  assert parsed.long_help.endswith("Longer narrative.")
  assert parsed.parameters == {"value": "First part. Second part."}


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        (Annotated[int, object()], "unsupported Annotated marker"),
        (Annotated[int,
                   ChoiceProvider(lambda: (1, )),
                   ChoiceProvider(lambda: (2, ))], "duplicate Annotated"),
        (Annotated[dict[str, int], KeyValue(),
                   KeyValue()], "duplicate Annotated"),
        (Annotated[int, CliType(int), CliType(str)], "duplicate Annotated"),
        (Annotated[dict[str, int],
                   ChoiceProvider(lambda: (1, )),
                   KeyValue()], "cannot be combined"),
        (Annotated[int,
                   ChoiceProvider(lambda: "abc")], "choice provider failed"),
        (Annotated[int, ChoiceProvider(lambda: ())], "returned no choices"),
        (Annotated[list[int],
                   ChoiceProvider(lambda: (1, ))], "requires a scalar"),
        (Annotated[int, ChoiceProvider(lambda: ("one", ))], "do not match"),
        (Annotated[int, ChoiceProvider(lambda: (True, ))], "do not match"),
        (Annotated[int, ChoiceProvider(lambda: (1, 1))], "duplicate choices"),
        (Annotated[int, KeyValue()], "requires a mapping"),
        (_BadEnum, "Enum values must be CLI scalars"),
        (Literal[()], "Literal choices"),
        (Literal[object()], "Literal choices"),
        (list[()], "list must have exactly one"),
        (tuple[()], "tuple must declare"),
        (dict[str, int], "mapping needs Annotated"),
        (Annotated[dict[()], KeyValue()], "mapping must declare"),
        (Annotated[dict[str, list[int]], KeyValue()], "must be CLI scalars"),
        (int | str, "unsupported union"),
        (complex, "unsupported annotation"),
    ],
)
def test_compiler_rejects_lossy_option_annotations(annotation, message):
  with pytest.raises(CommandCompilationError, match=message):
    _compiled_option(annotation)


def test_choice_and_variadic_tuple_codecs_round_trip():
  calls = []

  @command(
      CommandSpec(Section.UTILITY, Execution.LOAD, result=ResultPolicy.SILENT))
  def options(*,
              level: Annotated[int, ChoiceProvider(lambda: (1, 2))] = 1,
              values: tuple[int, ...] = ()):
    """Use provider and repeated-value codecs.

    Args:
      level: Registry-provided level.
      values: Repeated integer values.
    """
    calls.append((level, values))

  result = CliRunner().invoke(
      build_click_command(compile_callable(options)),
      ["--level", "2", "--values", "3", "--values", "4"],
      obj=DataSpace())
  assert result.exit_code == 0, result.output
  assert calls == [(2, [3, 4])]


def test_mapping_codec_rejects_malformed_and_duplicate_entries():
  model = _compiled_option(Annotated[dict[str, int] | None, KeyValue()])
  command_obj = build_click_command(model)
  for arguments, message in (
      (["--value", "missing-separator"], "expected key=value"),
      (["--value", "=1"], "expected key=value"),
      (["--value", "a=1", "--value", "a=2"], "duplicate mapping key"),
  ):
    result = CliRunner().invoke(command_obj, arguments, obj=DataSpace())
    assert result.exit_code != 0
    assert message in result.output


def test_compile_callable_reports_signature_and_name_errors():

  def unmarked():
    pass

  with pytest.raises(CommandCompilationError, match="has no CommandSpec"):
    compile_callable(unmarked)

  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def keywords(**values: int):
    """Reject keyword capture.

    Args:
      values: Arbitrary values.
    """

  with pytest.raises(CommandCompilationError, match=r"\*\*kwargs"):
    compile_callable(keywords)

  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def unresolved(value: "MissingType"):  # noqa: F821
    """Reject an unresolved annotation.

    Args:
      value: Unresolvable value.
    """

  with pytest.raises(CommandCompilationError, match="could not be resolved"):
    compile_callable(unresolved)

  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def named():
    """Reject an invalid projected name."""

  with pytest.raises(CommandCompilationError, match="invalid command name"):
    compile_callable(named, name="-named")


@pytest.mark.parametrize(
    ("annotation", "execution", "parameter", "message"),
    [
        (Annotated[object, PipelineInput(),
                   PipelineInput()], Execution.COMBINE, "value",
         "duplicate PipelineInput"),
        (object, Execution.LOAD, "*value", "not a declared pipeline"),
        (Annotated[object, DatasetRef(), DatasetRef()], Execution.COMBINE,
         "value", "duplicate DatasetRef"),
        (Annotated[object, DatasetRef(), PipelineInput()], Execution.COMBINE,
         "value", "both DatasetRef and PipelineInput"),
        (Annotated[str, CliArgument(), CliArgument()], Execution.COMBINE,
         "value", "duplicate CliArgument"),
        (Annotated[str, CliArgument(), PipelineInput()], Execution.COMBINE,
         "value", "both CliArgument and PipelineInput"),
        (Annotated[str, CliArgument(), DatasetRef()], Execution.COMBINE,
         "value", "both CliArgument and DatasetRef"),
        (Annotated[list[str], CliArgument()], Execution.LOAD, "value",
         "supports scalar values only"),
    ],
)
def test_compiler_rejects_invalid_pipeline_markers(annotation, execution,
                                                   parameter, message):
  namespace = {}
  exec(
      f"def target({parameter}):\n"
      "  'Compile pipeline metadata.\\n\\n  Args:\\n    value: Pipeline value.'\n",
      namespace,
  )
  target = namespace["target"]
  target.__annotations__ = {"value": annotation}
  command(CommandSpec(Section.UTILITY, execution))(target)
  with pytest.raises(CommandCompilationError, match=message):
    compile_callable(target)


def test_load_rejects_an_explicit_pipeline_receiver():

  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def loader(value: Annotated[object, PipelineInput()]):
    """Reject a receiver on a loader.

    Args:
      value: Invalid pipeline receiver.
    """

  with pytest.raises(CommandCompilationError, match="LOAD commands cannot"):
    compile_callable(loader)


def test_compile_public_surface_deduplicates_aliases_and_rejects_collisions():

  @command(CommandSpec(Section.VERBS, Execution.LOAD, order=2))
  def first():
    """First command."""

  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def second():
    """Second command."""

  surface = [
      SimpleNamespace(name="alias", callable=first),
      SimpleNamespace(name="alias", callable=first),
      SimpleNamespace(name="second", callable=second),
  ]
  models = compile_public_surface(surface)
  assert [model.name for model in models] == ["alias", "second"]
  assert group_by_section(models) == {
      "Verbs": ["alias"],
      "Utility": ["second"],
  }

  surface[-1] = SimpleNamespace(name="alias", callable=second)
  with pytest.raises(CommandCompilationError, match="command name collision"):
    compile_public_surface(surface)


def _invoke_pipeline(fn, datasets, arguments=()):
  space = DataSpace(list(datasets))
  result = CliRunner().invoke(build_click_command(compile_callable(fn)),
                              list(arguments),
                              obj=space)
  assert result.exit_code == 0, result.output
  return result, space


@pytest.mark.parametrize("execution",
                         [Execution.MAP_REPLACE, Execution.MAP_APPEND])
def test_map_execution_policies_transform_each_dataset(execution):

  @command(
      CommandSpec(Section.VERBS,
                  execution,
                  consumes_inputs=execution is Execution.MAP_APPEND))
  def transform(data: object):
    """Transform one dataset.

    Args:
      data: Current dataset.
    """
    return _Dataset(data.tag + "-new")

  _, space = _invoke_pipeline(transform, [_Dataset("a"), _Dataset("b")])
  assert [dataset.tag for dataset in space.datasets] == ["a-new", "b-new"]


def test_map_append_can_preserve_inputs_and_flatten_groups():

  @command(CommandSpec(Section.VERBS, Execution.MAP_APPEND))
  def append(data: object):
    """Append a group for each dataset.

    Args:
      data: Current dataset.
    """
    return _Group(_Dataset(data.tag + "-one"), _Dataset(data.tag + "-two"))

  _, space = _invoke_pipeline(append, [_Dataset("a")])
  assert [dataset.tag for dataset in space.datasets] == ["a", "a-one", "a-two"]


def test_map_or_terminal_replaces_data_and_prints_scalar_values():

  @command(
      CommandSpec(Section.VERBS,
                  Execution.MAP_OR_TERMINAL_EACH,
                  result=ResultPolicy.VALUE))
  def maybe(data: object):
    """Return either data or a value.

    Args:
      data: Current dataset.
    """
    return _Dataset("replaced") if data.tag == "data" else 7

  result, space = _invoke_pipeline(maybe, [_Dataset("data"), _Dataset("value")])
  assert result.output == "7\n"
  assert [dataset.tag for dataset in space.datasets] == ["replaced", "value"]


def test_terminal_each_presents_nested_values_without_mutating_data():

  @command(
      CommandSpec(Section.UTILITY,
                  Execution.TERMINAL_EACH,
                  result=ResultPolicy.VALUE))
  def inspect_one(data: object):
    """Inspect one dataset.

    Args:
      data: Current dataset.
    """
    return [None, data, (data.tag, )]

  original = _Dataset("kept")
  result, space = _invoke_pipeline(inspect_one, [original])
  assert result.output == "kept\n"
  assert space.datasets == [original]


def test_terminal_all_silent_result_neither_prints_nor_mutates_data():

  @command(
      CommandSpec(Section.UTILITY,
                  Execution.TERMINAL_ALL,
                  result=ResultPolicy.SILENT))
  def inspect_all(data: Annotated[list[object], PipelineInput()]):
    """Inspect the entire working set.

    Args:
      data: Current working set.
    """
    assert [dataset.tag for dataset in data] == ["kept"]
    return "not printed"

  original = _Dataset("kept")
  result, space = _invoke_pipeline(inspect_all, [original])
  assert result.output == ""
  assert space.datasets == [original]


def test_load_value_adds_dataset_results_and_presents_only_values():

  @command(
      CommandSpec(Section.UTILITY, Execution.LOAD, result=ResultPolicy.VALUE))
  def load_value():
    """Load and present a mixed result."""
    return (_Dataset("loaded"), "message")

  result, space = _invoke_pipeline(load_value, [])
  assert result.output == "message\n"
  assert space.datasets == []


def test_combine_preserves_returned_dataset_order():

  @command(CommandSpec(Section.VERBS, Execution.COMBINE))
  def reverse(*datasets: object):
    """Reverse datasets.

    Args:
      datasets: Current working set.
    """
    return tuple(reversed(datasets))

  first, second = _Dataset("first"), _Dataset("second")
  _, space = _invoke_pipeline(reverse, [first, second])
  assert space.datasets == [second, first]


def test_dataset_reference_resolves_uniquely_and_limits_consumption():

  @command(CommandSpec(Section.VERBS, Execution.COMBINE, consumes_inputs=True))
  def take(reference: Annotated[object, DatasetRef()] = None):
    """Consume one referenced dataset.

    Args:
      reference: Tagged dataset to consume.
    """
    return _Dataset("result")

  kept, consumed = _Dataset("kept"), _Dataset("chosen")
  _, space = _invoke_pipeline(take, [kept, consumed], ["--reference", "chosen"])
  assert [dataset.tag for dataset in space.datasets] == ["kept", "result"]

  for datasets, message in (([kept], "no dataset tagged"),
                            ([_Dataset("chosen"),
                              _Dataset("chosen")], "matches 2 datasets")):
    result = CliRunner().invoke(build_click_command(compile_callable(take)),
                                ["--reference", "chosen"],
                                obj=DataSpace(datasets))
    assert result.exit_code != 0
    assert message in result.output


def test_pipeline_errors_keep_click_errors_and_wrap_api_errors():

  @command(
      CommandSpec(Section.UTILITY,
                  Execution.TERMINAL_ALL,
                  result=ResultPolicy.SILENT))
  def fail(*datasets: object):
    """Raise an API error.

    Args:
      datasets: Current working set.
    """
    raise ValueError("bad input")

  result = CliRunner().invoke(build_click_command(compile_callable(fail)), [],
                              obj=DataSpace([_Dataset("one")]))
  assert result.exit_code != 0
  assert "bad input" in result.output

  @command(
      CommandSpec(Section.UTILITY,
                  Execution.TERMINAL_ALL,
                  result=ResultPolicy.SILENT))
  def fail_with_click(*datasets: object):
    """Raise a Click error.

    Args:
      datasets: Current working set.
    """
    raise click.UsageError("click input")

  with pytest.raises(click.UsageError, match="click input"):
    compiler.execute_model(SimpleNamespace(obj=DataSpace([_Dataset("one")])),
                           compile_callable(fail_with_click), {})


def test_generated_help_skips_undocumented_internal_arguments():
  argument = compiler._DocumentedArgument(["value"], help=None)
  assert argument.get_help_record(None) is None
  command_obj = compiler._GeneratedCommand("internal", params=[argument])
  result = CliRunner().invoke(command_obj, ["--help"])
  assert result.exit_code == 0
  assert "Arguments:" not in result.output


def test_unreachable_codec_kinds_fail_loudly_and_none_stays_none():
  codec = TypeCodec(CodecKind.STRING, str)
  assert compiler._convert_scalar(None, codec) is None
  invalid = TypeCodec("invalid", str)
  with pytest.raises(AssertionError, match="invalid"):
    compiler._click_scalar(invalid)


def test_pipeline_receiver_requires_a_concrete_annotation():

  @command(CommandSpec(Section.VERBS, Execution.MAP_REPLACE))
  def transform(data):
    """Transform data.

    Args:
      data: Current dataset.
    """

  with pytest.raises(CommandCompilationError, match="concrete annotation"):
    compile_callable(transform)


def test_self_receiver_requires_exactly_one_direct_input():

  @command(CommandSpec(Section.VERBS, Execution.MAP_REPLACE))
  def transform(self):
    """Transform one receiver.

    Args:
      self: Current dataset.
    """

  model = compile_callable(transform)
  with pytest.raises(click.UsageError, match="exactly one"):
    compiler._call(model, [_Dataset("one"), _Dataset("two")], {}, None)


def test_keyword_only_pipeline_receiver_gets_the_working_set():
  received = []

  @command(
      CommandSpec(Section.UTILITY,
                  Execution.TERMINAL_ALL,
                  result=ResultPolicy.SILENT))
  def inspect(*, data: Annotated[list[object], PipelineInput()]):
    """Inspect all data.

    Args:
      data: Current working set.
    """
    received.append(data)

  _, space = _invoke_pipeline(inspect, [_Dataset("one")])
  assert received == [space.datasets]


def test_optional_dataset_reference_can_be_omitted():

  @command(CommandSpec(Section.VERBS, Execution.COMBINE))
  def inspect(reference: Annotated[object, DatasetRef()] = None):
    """Inspect an optional reference.

    Args:
      reference: Optional tagged dataset.
    """
    assert reference is None
    return ()

  _invoke_pipeline(inspect, [_Dataset("one")])


@pytest.mark.parametrize(
    "execution", [Execution.MAP_OR_TERMINAL_EACH, Execution.TERMINAL_EACH])
def test_each_policies_can_silence_scalar_results(execution):

  @command(CommandSpec(Section.UTILITY, execution, result=ResultPolicy.SILENT))
  def inspect(data: object):
    """Inspect one dataset silently.

    Args:
      data: Current dataset.
    """
    return data.tag

  result, _ = _invoke_pipeline(inspect, [_Dataset("one")])
  assert result.output == ""


def test_combine_can_consume_unreferenced_inputs():

  @command(CommandSpec(Section.VERBS, Execution.COMBINE, consumes_inputs=True))
  def combine(*datasets: object):
    """Replace all inputs.

    Args:
      datasets: Current working set.
    """
    return _Dataset("combined")

  _, space = _invoke_pipeline(combine, [_Dataset("one"), _Dataset("two")])
  assert [dataset.tag for dataset in space.datasets] == ["combined"]


def test_combine_and_terminal_all_apply_their_result_policies():

  @command(
      CommandSpec(Section.VERBS, Execution.COMBINE, result=ResultPolicy.VALUE))
  def combine(*datasets: object):
    """Present a combined value.

    Args:
      datasets: Current working set.
    """
    return len(datasets)

  result, _ = _invoke_pipeline(combine, [_Dataset("one")])
  assert result.output == "1\n"

  @command(
      CommandSpec(Section.UTILITY,
                  Execution.TERMINAL_ALL,
                  result=ResultPolicy.SILENT))
  def silent(*datasets: object):
    """Ignore a terminal value.

    Args:
      datasets: Current working set.
    """
    return len(datasets)

  result, _ = _invoke_pipeline(silent, [_Dataset("one")])
  assert result.output == ""


def test_optional_enum_argument_lowers_its_default_value():

  @command(CommandSpec(Section.UTILITY, Execution.LOAD))
  def choose(mode: Annotated[_Mode, CliArgument()] = _Mode.TEXT):
    """Choose a mode.

    Args:
      mode: Optional mode.
    """

  command_obj = build_click_command(compile_callable(choose))
  assert command_obj.params[0].default == "text"
