"""Strict callable-to-command compilation and generic pipeline execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import inspect
from pathlib import Path
import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

import click

from postgkyl.cli_spec import (
    ChoiceProvider,
    CliArgument,
    CliType,
    CommandSpec,
    DatasetRef,
    Execution,
    KeyValue,
    PipelineInput,
    ResultPolicy,
    Section,
    command_spec,
)

from .docstrings import parse_docstring


class CommandCompilationError(ValueError):
  """A public callable cannot be represented by the closed CLI contract."""


class CodecKind(Enum):
  STRING = "string"
  INTEGER = "integer"
  FLOAT = "float"
  BOOLEAN = "boolean"
  PATH = "path"
  CHOICE = "choice"
  ENUM = "enum"
  SEQUENCE = "sequence"
  TUPLE = "tuple"
  MAPPING = "mapping"


@dataclass(frozen=True)
class TypeCodec:
  kind: CodecKind
  python_type: object
  optional: bool = False
  choices: tuple[object, ...] = ()
  items: tuple["TypeCodec", ...] = ()
  multiple: bool = False
  nargs: int = 1


@dataclass(frozen=True)
class ParameterModel:
  name: str
  kind: inspect._ParameterKind
  annotation: object
  codec: TypeCodec | None
  help: str | None
  required: bool
  default: object
  injected: bool = False
  dataset_ref: bool = False
  argument: bool = False


@dataclass(frozen=True)
class CommandModel:
  name: str
  callable: object
  canonical: object
  qualname: str
  spec: CommandSpec
  help: str
  long_help: str
  parameters: tuple[ParameterModel, ...]

  @property
  def section(self) -> Section:
    return self.spec.section


class _DocumentedArgument(click.Argument):
  """Click argument which retains its API parameter's help text."""

  def __init__(self, param_decls, *, help: str | None = None, **attrs):
    super().__init__(param_decls, **attrs)
    self.help = help

  def get_help_record(self, ctx):
    if not self.help:
      return None
    return self.make_metavar(ctx), self.help


class _GeneratedCommand(click.Command):
  """Render positional parameter docs separately from option docs."""

  def format_options(self, ctx, formatter) -> None:
    arguments = []
    options = []
    for parameter in self.get_params(ctx):
      record = parameter.get_help_record(ctx)
      if record is None:
        continue
      target = arguments if isinstance(parameter, click.Argument) else options
      target.append(record)
    if arguments:
      with formatter.section("Arguments"):
        formatter.write_dl(arguments)
    if options:
      with formatter.section("Options"):
        formatter.write_dl(options)


_NONE_TYPE = type(None)
_SCALARS = {
    str: CodecKind.STRING,
    int: CodecKind.INTEGER,
    float: CodecKind.FLOAT,
    bool: CodecKind.BOOLEAN,
    Path: CodecKind.PATH,
}
_RESERVED_SHORT_OPTIONS = frozenset({"h"})


def _short_option_names(
    parameters: tuple[ParameterModel, ...]) -> dict[str, str]:
  """Assign each available initial to its first exposed parameter."""
  assigned: dict[str, str] = {}
  claimed = set(_RESERVED_SHORT_OPTIONS)
  for parameter in parameters:
    if parameter.injected or parameter.argument:
      continue
    initial = parameter.name[0]
    if initial in claimed:
      continue
    assigned[parameter.name] = f"-{initial}"
    claimed.add(initial)
  return assigned


def _error(fn, parameter: str, message: str) -> CommandCompilationError:
  qualname = f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', fn)}"
  return CommandCompilationError(
      f"{qualname}: parameter {parameter!r}: {message}")


def _unwrap_annotated(annotation):
  markers: list[object] = []
  while get_origin(annotation) is Annotated:
    args = get_args(annotation)
    annotation = args[0]
    markers.extend(args[1:])
  return annotation, tuple(markers)


def _optional(annotation):
  origin = get_origin(annotation)
  if origin not in (Union, types.UnionType):
    return annotation, False
  args = get_args(annotation)
  non_none = tuple(arg for arg in args if arg is not _NONE_TYPE)
  if len(non_none) == 1 and len(non_none) != len(args):
    return non_none[0], True
  return annotation, False


def _codec(fn, name: str, annotation, markers: tuple[object, ...]) -> TypeCodec:
  providers = [
      marker for marker in markers if isinstance(marker, ChoiceProvider)
  ]
  key_values = [marker for marker in markers if isinstance(marker, KeyValue)]
  cli_types = [marker for marker in markers if isinstance(marker, CliType)]
  unknown = [
      marker for marker in markers
      if not isinstance(marker, (ChoiceProvider, CliArgument, CliType, KeyValue,
                                 DatasetRef, PipelineInput))
  ]
  if unknown:
    raise _error(fn, name, f"unsupported Annotated marker {unknown[0]!r}")
  if len(providers) > 1 or len(key_values) > 1 or len(cli_types) > 1:
    raise _error(fn, name, "duplicate Annotated codec marker")
  if providers and key_values:
    raise _error(fn, name, "ChoiceProvider and KeyValue cannot be combined")
  if cli_types:
    annotation = cli_types[0].annotation
  annotation, optional = _optional(annotation)
  if annotation in (Any, inspect.Parameter.empty):
    raise _error(fn, name, "missing or Any annotation")

  if providers:
    try:
      provided = providers[0].provider()
      if isinstance(provided, (str, bytes)):
        raise TypeError("provider returned text instead of a choice collection")
      values = tuple(provided)
    except Exception as exc:
      raise _error(fn, name, f"choice provider failed: {exc}") from exc
    if not values:
      raise _error(fn, name, "choice provider returned no choices")
    if annotation not in _SCALARS:
      raise _error(fn, name, "choice provider requires a scalar annotation")
    if not all(
        isinstance(value, annotation)
        and not (annotation is int and isinstance(value, bool))
        for value in values):
      raise _error(
          fn, name,
          "choice provider values do not match the annotated scalar type")
    if len(set(values)) != len(values):
      raise _error(fn, name, "choice provider returned duplicate choices")
    return TypeCodec(CodecKind.CHOICE, annotation, optional, choices=values)
  if key_values and get_origin(annotation) not in (dict, Mapping):
    raise _error(fn, name, "KeyValue requires a mapping annotation")
  if annotation in _SCALARS:
    return TypeCodec(_SCALARS[annotation], annotation, optional)
  if inspect.isclass(annotation) and issubclass(annotation, Enum):
    values = tuple(member.value for member in annotation)
    if not all(isinstance(value, (str, int, float)) for value in values):
      raise _error(fn, name, "Enum values must be CLI scalars")
    return TypeCodec(CodecKind.ENUM, annotation, optional, choices=values)
  origin = get_origin(annotation)
  args = get_args(annotation)
  if origin is Literal:
    if not args or not all(
        isinstance(value, (str, int, float)) for value in args):
      raise _error(fn, name, "Literal choices must be strings or numbers")
    return TypeCodec(CodecKind.CHOICE,
                     annotation,
                     optional,
                     choices=tuple(args))
  if origin is list:
    if len(args) != 1:
      raise _error(fn, name, "list must have exactly one item type")
    item = _codec(fn, name, args[0], ())
    return TypeCodec(CodecKind.SEQUENCE,
                     annotation,
                     optional,
                     items=(item, ),
                     multiple=True)
  if origin is tuple:
    if not args:
      raise _error(fn, name, "tuple must declare its item type(s)")
    if len(args) == 2 and args[1] is Ellipsis:
      item = _codec(fn, name, args[0], ())
      return TypeCodec(CodecKind.SEQUENCE,
                       annotation,
                       optional,
                       items=(item, ),
                       multiple=True)
    items = tuple(_codec(fn, name, arg, ()) for arg in args)
    return TypeCodec(CodecKind.TUPLE,
                     annotation,
                     optional,
                     items=items,
                     nargs=len(items))
  if origin in (dict, Mapping):
    if not key_values:
      raise _error(fn, name, "mapping needs Annotated[..., KeyValue()]")
    if len(args) != 2:
      raise _error(fn, name, "mapping must declare key and value types")
    key = _codec(fn, name, args[0], ())
    value = _codec(fn, name, args[1], ())
    composite = {CodecKind.SEQUENCE, CodecKind.TUPLE, CodecKind.MAPPING}
    if key.kind in composite or value.kind in composite:
      raise _error(fn, name, "mapping keys and values must be CLI scalars")
    return TypeCodec(CodecKind.MAPPING,
                     annotation,
                     optional,
                     items=(key, value),
                     multiple=True)
  if get_origin(annotation) in (Union, types.UnionType):
    raise _error(
        fn, name,
        f"unsupported union {annotation!r}; only T | None is lossless")
  raise _error(fn, name, f"unsupported annotation {annotation!r}")


def _type_hints(fn) -> dict[str, object]:
  annotations = dict(getattr(fn, "__annotations__", {}))
  deferred = {
      name: annotation
      for name, annotation in annotations.items()
      if isinstance(annotation, str)
  }
  if not deferred:
    return annotations
  try:
    source = types.SimpleNamespace(__annotations__=deferred)
    resolved = get_type_hints(source,
                              globalns=getattr(fn, "__globals__", None),
                              include_extras=True)
  except Exception as exc:
    qualname = f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', fn)}"
    raise CommandCompilationError(
        f"{qualname}: annotations could not be resolved: {exc}") from exc
  # Concrete annotations installed by an API owner are already resolved.
  # Evaluating them again is observably broken for
  # ``Annotated[T | None, ...]`` on Python 3.10, so only deferred strings pass
  # through ``get_type_hints`` and concrete objects remain authoritative.
  for name, annotation in annotations.items():
    if isinstance(annotation, str):
      annotations[name] = resolved[name]
  return annotations


def compile_callable(fn, *, name: str | None = None) -> CommandModel:
  """Inspect one marked callable into an immutable, Click-free model."""
  spec = command_spec(fn)
  if spec is None:
    raise CommandCompilationError(f"{fn!r} has no CommandSpec")
  canonical = fn
  signature = inspect.signature(canonical)
  hints = _type_hints(canonical)
  raw: list[tuple[inspect.Parameter, object, tuple[object, ...], bool, bool,
                  bool]] = []
  for index, parameter in enumerate(signature.parameters.values()):
    if parameter.kind is inspect.Parameter.VAR_KEYWORD:
      raise _error(canonical, parameter.name, "**kwargs is not representable")
    annotation = hints.get(parameter.name, parameter.annotation)
    base, markers = _unwrap_annotated(annotation)
    marker_injected = any(
        isinstance(marker, PipelineInput) for marker in markers)
    if sum(isinstance(marker, PipelineInput) for marker in markers) > 1:
      raise _error(canonical, parameter.name, "duplicate PipelineInput marker")
    is_receiver = parameter.name == "self" or (
        index == 0 and spec.execution
        in (Execution.MAP_REPLACE, Execution.MAP_APPEND,
            Execution.MAP_OR_TERMINAL_EACH, Execution.TERMINAL_EACH))
    is_variadic_input = parameter.kind is inspect.Parameter.VAR_POSITIONAL and (
        spec.execution in (Execution.COMBINE, Execution.TERMINAL_ALL))
    injected = marker_injected or is_receiver or is_variadic_input
    if injected and parameter.name != "self" and base in (
        Any, inspect.Parameter.empty):
      raise _error(canonical, parameter.name,
                   "pipeline receivers need a concrete annotation")
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL and not injected:
      raise _error(canonical, parameter.name,
                   "*args is not a declared pipeline receiver")
    is_dataset_ref = any(isinstance(marker, DatasetRef) for marker in markers)
    if sum(isinstance(marker, DatasetRef) for marker in markers) > 1:
      raise _error(canonical, parameter.name, "duplicate DatasetRef marker")
    if is_dataset_ref and injected:
      raise _error(canonical, parameter.name,
                   "cannot be both DatasetRef and PipelineInput")
    is_argument = any(isinstance(marker, CliArgument) for marker in markers)
    if sum(isinstance(marker, CliArgument) for marker in markers) > 1:
      raise _error(canonical, parameter.name, "duplicate CliArgument marker")
    if is_argument and injected:
      raise _error(canonical, parameter.name,
                   "cannot be both CliArgument and PipelineInput")
    if is_argument and is_dataset_ref:
      raise _error(canonical, parameter.name,
                   "cannot be both CliArgument and DatasetRef")
    if is_argument and parameter.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD):
      raise _error(canonical, parameter.name,
                   "CliArgument requires a positional Python parameter")
    raw.append(
        (parameter, base, markers, injected, is_dataset_ref, is_argument))

  injected_parameters = [item for item in raw if item[3]]
  if spec.execution is Execution.LOAD and injected_parameters:
    raise CommandCompilationError(
        f"{canonical.__module__}.{canonical.__qualname__}: "
        "LOAD commands cannot declare a pipeline receiver")
  docs = parse_docstring(canonical,
                         required=set(signature.parameters),
                         signature_names=set(signature.parameters))
  models: list[ParameterModel] = []
  for parameter, base, markers, injected, is_dataset_ref, is_argument in raw:
    required = parameter.default is inspect.Parameter.empty
    default = None if required else parameter.default
    codec = None
    if not injected:
      codec = (TypeCodec(CodecKind.STRING, str,
                         optional=not required) if is_dataset_ref else _codec(
                             canonical, parameter.name, base, markers))
      if is_argument and (codec.multiple or codec.nargs != 1):
        raise _error(canonical, parameter.name,
                     "CliArgument currently supports scalar values only")
    models.append(
        ParameterModel(name=parameter.name,
                       kind=parameter.kind,
                       annotation=base,
                       codec=codec,
                       help=docs.parameters.get(parameter.name),
                       required=required,
                       default=default,
                       injected=injected,
                       dataset_ref=is_dataset_ref,
                       argument=is_argument))

  command_name = name or getattr(canonical, "__name__", "")
  if not command_name or command_name.startswith("-"):
    raise CommandCompilationError(
        f"{canonical!r}: invalid command name {command_name!r}")
  qualname = f"{canonical.__module__}.{canonical.__qualname__}"
  return CommandModel(command_name, fn, canonical, qualname, spec, docs.summary,
                      docs.long_help, tuple(models))


def compile_public_surface(callables) -> tuple[CommandModel, ...]:
  """Compile and validate a complete discovered surface atomically."""
  models = tuple(
      compile_callable(item.callable, name=item.name) for item in callables)
  seen: dict[str, CommandModel] = {}
  for model in models:
    previous = seen.get(model.name)
    if previous is not None and previous.canonical is not model.canonical:
      raise CommandCompilationError(
          f"command name collision {model.name!r}: {previous.qualname} and {model.qualname}"
      )
    seen[model.name] = model
  return tuple(
      sorted(
          seen.values(),
          key=lambda model:
          (list(Section).index(model.section), model.spec.order, model.name)))


def _click_scalar(codec: TypeCodec):
  if codec.kind is CodecKind.STRING:
    return click.STRING
  if codec.kind is CodecKind.INTEGER:
    return click.INT
  if codec.kind is CodecKind.FLOAT:
    return click.FLOAT
  if codec.kind is CodecKind.BOOLEAN:
    return click.BOOL
  if codec.kind is CodecKind.PATH:
    return click.Path(path_type=Path)
  if codec.kind in (CodecKind.CHOICE, CodecKind.ENUM):
    return click.Choice(codec.choices, case_sensitive=True)
  if codec.kind in (CodecKind.SEQUENCE, CodecKind.MAPPING):
    return _click_scalar(
        codec.items[0]) if codec.kind is CodecKind.SEQUENCE else click.STRING
  if codec.kind is CodecKind.TUPLE:
    return click.Tuple([_click_scalar(item) for item in codec.items])
  raise AssertionError(codec.kind)


def _convert_scalar(value, codec: TypeCodec):
  if value is None:
    return None
  if codec.kind is CodecKind.ENUM:
    return codec.python_type(value)
  return value


def _convert(value, codec: TypeCodec):
  if value is None:
    return None
  if codec.kind is CodecKind.SEQUENCE:
    return [_convert_scalar(item, codec.items[0]) for item in value]
  if codec.kind is CodecKind.TUPLE:
    return tuple(
        _convert_scalar(item, sub) for item, sub in zip(value, codec.items))
  if codec.kind is CodecKind.MAPPING:
    result = {}
    key_codec, value_codec = codec.items
    for entry in value:
      key, separator, item = entry.partition("=")
      if not separator or not key:
        raise click.BadParameter("expected key=value")
      click_key = _click_scalar(key_codec).convert(key, None, None)
      click_value = _click_scalar(value_codec).convert(item, None, None)
      if click_key in result:
        raise click.BadParameter(f"duplicate mapping key {click_key!r}")
      result[_convert_scalar(click_key, key_codec)] = _convert_scalar(
          click_value, value_codec)
    return result or None if codec.optional else result
  return _convert_scalar(value, codec)


def _resolve_tag(ctx, parameter: str, tag: str):
  matches = [dataset for dataset in ctx.obj.datasets if dataset.tag == tag]
  if not matches:
    raise click.UsageError(f"--{parameter}: no dataset tagged {tag!r}")
  if len(matches) != 1:
    raise click.UsageError(
        f"--{parameter}: tag {tag!r} matches {len(matches)} datasets")
  return matches[0]


def _is_dataset(value) -> bool:
  return all(hasattr(value, name) for name in ("ctx", "grid", "values"))


def _datasets_from_result(result) -> list:
  members = getattr(result, "datasets", None)
  if members is not None:
    return list(members)
  if _is_dataset(result):
    return [result]
  if isinstance(result,
                (list, tuple)) and all(_is_dataset(item) for item in result):
    return list(result)
  return []


def _present(value) -> None:
  if value is None:
    return
  if isinstance(value, (list, tuple)):
    for item in value:
      _present(item)
    return
  if not _is_dataset(value):
    click.echo(value)


def _selected(ctx) -> list:
  return list(ctx.obj.datasets)


def _call(model: CommandModel, selected: list, values: dict, ctx):
  args: list[object] = []
  kwargs: dict[str, object] = {}
  referenced: list[object] = []
  for parameter in model.parameters:
    if parameter.injected:
      if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        args.extend(selected)
      elif parameter.name == "self":
        if len(selected) != 1:
          raise click.UsageError(
              f"{model.name}: expected exactly one pipeline input")
        args.append(selected[0])
      elif model.spec.execution in (Execution.MAP_REPLACE, Execution.MAP_APPEND,
                                    Execution.MAP_OR_TERMINAL_EACH,
                                    Execution.TERMINAL_EACH):
        args.append(selected[0])
      elif parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD):
        args.append(selected)
      else:
        kwargs[parameter.name] = selected
      continue
    value = values[parameter.name]
    if parameter.dataset_ref:
      if value is not None:
        value = _resolve_tag(ctx, parameter.name, value)
        referenced.append(value)
    elif parameter.codec is not None:
      value = _convert(value, parameter.codec)
    if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
      args.append(value)
    else:
      kwargs[parameter.name] = value
  return model.canonical(*args, **kwargs), referenced


def execute_model(ctx, model: CommandModel, values: dict):
  """Invoke one model through its generic working-set execution adapter."""
  selected = _selected(ctx)
  execution = model.spec.execution
  needs_input = execution is not Execution.LOAD
  if needs_input and not selected and not any(p.dataset_ref
                                              for p in model.parameters):
    raise click.UsageError(f"{model.name}: no datasets selected")

  try:
    if execution in (Execution.MAP_REPLACE, Execution.MAP_APPEND,
                     Execution.MAP_OR_TERMINAL_EACH, Execution.TERMINAL_EACH):
      outputs: list[object] = []
      for dataset in selected:
        result, _ = _call(model, [dataset], values, ctx)
        outputs.append(result)
      if execution is Execution.MAP_REPLACE:
        replacements = iter(outputs)
        chosen = {id(dataset) for dataset in selected}
        ctx.obj.datasets = [
            next(replacements) if id(dataset) in chosen else dataset
            for dataset in ctx.obj.datasets
        ]
      elif execution is Execution.MAP_APPEND:
        if model.spec.consumes_inputs:
          consumed = {id(dataset) for dataset in selected}
          ctx.obj.datasets = [
              dataset for dataset in ctx.obj.datasets
              if id(dataset) not in consumed
          ]
        for result in outputs:
          ctx.obj.datasets.extend(_datasets_from_result(result))
      elif execution is Execution.MAP_OR_TERMINAL_EACH:
        by_input = {
            id(dataset): result
            for dataset, result in zip(selected, outputs)
        }
        ctx.obj.datasets = [
            by_input[id(dataset)] if id(dataset) in by_input
            and _is_dataset(by_input[id(dataset)]) else dataset
            for dataset in ctx.obj.datasets
        ]
        if model.spec.result is ResultPolicy.VALUE:
          for result in outputs:
            _present(result)
      else:
        if model.spec.result is ResultPolicy.VALUE:
          for result in outputs:
            _present(result)
      return outputs

    result, referenced = _call(model, selected, values, ctx)
    if execution is Execution.LOAD:
      ctx.obj.datasets.extend(_datasets_from_result(result))
      if model.spec.result is ResultPolicy.VALUE:
        _present(result)
    elif execution is Execution.COMBINE:
      result_datasets = _datasets_from_result(result)
      if len(result_datasets) == len(selected) and sorted(map(id, result_datasets)) \
          == sorted(map(id, selected)):
        ordered = iter(result_datasets)
        selected_ids = {id(dataset) for dataset in selected}
        ctx.obj.datasets = [
            next(ordered) if id(dataset) in selected_ids else dataset
            for dataset in ctx.obj.datasets
        ]
        return result
      if model.spec.consumes_inputs:
        consumed = {id(dataset) for dataset in (referenced or selected)}
        ctx.obj.datasets = [
            dataset for dataset in ctx.obj.datasets
            if id(dataset) not in consumed
        ]
      ctx.obj.datasets.extend(result_datasets)
      if model.spec.result is ResultPolicy.VALUE:
        _present(result)
    elif execution is Execution.TERMINAL_ALL:
      if model.spec.result is ResultPolicy.VALUE:
        _present(result)
    return result
  except click.ClickException:
    raise
  except (ValueError, TypeError, OSError) as exc:
    raise click.UsageError(str(exc)) from exc


def build_click_command(model: CommandModel,
                        *,
                        command_class=_GeneratedCommand) -> click.Command:
  """Lower an immutable model to a Click command."""
  params: list[click.Parameter] = []
  short_options = _short_option_names(model.parameters)
  for parameter in model.parameters:
    if parameter.injected:
      continue
    codec = parameter.codec
    assert codec is not None
    if parameter.argument:
      attrs = dict(type=_click_scalar(codec), required=parameter.required)
      if not parameter.required:
        default = parameter.default
        if codec.kind is CodecKind.ENUM and isinstance(default, Enum):
          default = default.value
        attrs["default"] = default
      params.append(
          _DocumentedArgument([parameter.name], help=parameter.help, **attrs))
      continue
    attrs = dict(
        type=_click_scalar(codec),
        required=parameter.required,
        help=parameter.help,
        show_default=not parameter.required,
        multiple=codec.multiple,
    )
    if not parameter.required:
      default = parameter.default
      if codec.kind is CodecKind.ENUM and isinstance(default, Enum):
        default = default.value
      if codec.multiple and default is None:
        default = ()
      attrs["default"] = default
    declarations = [f"--{parameter.name}"]
    if parameter.name in short_options:
      declarations.append(short_options[parameter.name])
    declarations.append(parameter.name)
    params.append(click.Option(declarations, **attrs))

  @click.pass_context
  def callback(click_context, **kwargs):
    return execute_model(click_context, model, kwargs)

  return command_class(model.name,
                       params=params,
                       callback=callback,
                       help=model.long_help,
                       short_help=model.help)


def group_by_section(models: tuple[CommandModel, ...]) -> dict[str, list[str]]:
  """Derive the flat help presentation from compiled models."""
  return {
      section.value:
      [model.name for model in models if model.section is section]
      for section in Section
      if any(model.section is section for model in models)
  }


__all__ = [
    "CodecKind",
    "CommandCompilationError",
    "CommandModel",
    "ParameterModel",
    "TypeCodec",
    "build_click_command",
    "compile_callable",
    "compile_public_surface",
    "execute_model",
    "group_by_section",
]
