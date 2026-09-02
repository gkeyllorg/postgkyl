"""Frozen, dependency-free records describing the generated CLI surface.

This package deliberately knows nothing about Click or postgkyl datasets.
API-owning modules attach these records to their public callables; the CLI is
the only layer that interprets them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable


class Section(Enum):
  """Top-level help sections, in display order."""

  VERBS = "Verbs"
  DIAGNOSTICS = "Diagnostics"
  RENDER = "Render"
  UTILITY = "Utility"
# end


class Execution(Enum):
  """Closed set of working-set adapters used by the CLI runtime."""

  MAP_REPLACE = auto()
  MAP_APPEND = auto()
  COMBINE = auto()
  TERMINAL_EACH = auto()
  TERMINAL_ALL = auto()
  LOAD = auto()
# end


class ResultPolicy(Enum):
  """How a returned API value is presented at the command-line edge."""

  DATA = auto()
  VALUE = auto()
  SILENT = auto()
# end


@dataclass(frozen=True)
class CommandSpec:
  """Pipeline metadata attached to one canonical API callable."""

  section: Section
  execution: Execution
  consumes_inputs: bool = False
  result: ResultPolicy = ResultPolicy.DATA
  order: int = 0

  def __post_init__(self) -> None:
    if not isinstance(self.section, Section) or not isinstance(
        self.execution, Execution) or not isinstance(self.result, ResultPolicy):
      raise TypeError("CommandSpec section, execution, and result must be enums")
    # end
    if not isinstance(self.consumes_inputs, bool):
      raise TypeError("CommandSpec consumes_inputs must be a bool value")
    # end
    if not isinstance(self.order, int) or isinstance(self.order, bool):
      raise TypeError("CommandSpec order must be an integer")
    # end
    if self.execution is Execution.LOAD:
      if self.consumes_inputs:
        raise ValueError("LOAD commands cannot consume working-set inputs")
    # end
    if self.consumes_inputs and self.execution not in (
        Execution.MAP_APPEND, Execution.COMBINE):
      raise ValueError(
          "only MAP_APPEND and COMBINE commands may consume their inputs")
    # end
    if self.execution in (Execution.TERMINAL_EACH, Execution.TERMINAL_ALL):
      if self.consumes_inputs:
        raise ValueError("terminal commands cannot consume working-set inputs")
      # end
      if self.result is ResultPolicy.DATA:
        raise ValueError("terminal commands need a non-DATA result policy")
      # end
  # end
# end


@dataclass(frozen=True)
class DatasetRef:
  """Resolve this API dataset parameter from a unique working-set tag."""

  default_tag: str | None = None

  def __post_init__(self) -> None:
    if self.default_tag is not None and not self.default_tag:
      raise ValueError("DatasetRef default_tag must be non-empty when supplied")
    # end
# end


@dataclass(frozen=True)
class PipelineInput:
  """Inject this parameter from the selected CLI working set."""
# end


@dataclass(frozen=True)
class CliType:
  """Lossless CLI input type for a broader direct-Python annotation."""

  annotation: object
# end


@dataclass(frozen=True)
class ChoiceProvider:
  """Obtain an option's choices from the API registry returned by provider."""

  provider: Callable[[], object]

  def __post_init__(self) -> None:
    if not callable(self.provider):
      raise TypeError("ChoiceProvider requires a callable provider")
    # end
# end


@dataclass(frozen=True)
class KeyValue:
  """Parse a mapping as repeated ``key=value`` option values."""
# end


@dataclass(frozen=True)
class CliHidden:
  """Explicitly exclude a public callable from command generation."""

  reason: str

  def __post_init__(self) -> None:
    if not self.reason.strip():
      raise ValueError("CliHidden requires a non-empty reason")
    # end
  # end
# end


_COMMAND_ATTR = "__postgkyl_command_spec__"
_HIDDEN_ATTR = "__postgkyl_cli_hidden__"


def command(spec: CommandSpec):
  """Attach ``spec`` to a callable without wrapping or registering it."""
  if not isinstance(spec, CommandSpec):
    raise TypeError("command() requires a CommandSpec")
  # end

  def decorate(fn):
    if hasattr(fn, _HIDDEN_ATTR):
      raise ValueError(f"{fn!r} is already marked CliHidden")
    # end
    previous = getattr(fn, _COMMAND_ATTR, None)
    if previous is not None and previous != spec:
      raise ValueError(f"{fn!r} already has a different CommandSpec")
    # end
    setattr(fn, _COMMAND_ATTR, spec)
    return fn
  # end
  return decorate
# end


def hidden(reason: str):
  """Explicitly mark a public callable as unavailable from the CLI."""
  marker = CliHidden(reason)

  def decorate(fn):
    if hasattr(fn, _COMMAND_ATTR):
      raise ValueError(f"{fn!r} already has a CommandSpec")
    # end
    setattr(fn, _HIDDEN_ATTR, marker)
    return fn
  # end
  return decorate
# end


def command_spec(fn) -> CommandSpec | None:
  """Return the immutable command metadata attached to ``fn``."""
  return getattr(fn, _COMMAND_ATTR, None)
# end


def hidden_spec(fn) -> CliHidden | None:
  """Return the explicit CLI exclusion attached to ``fn``."""
  return getattr(fn, _HIDDEN_ATTR, None)
# end


__all__ = [
    "ChoiceProvider", "CliHidden", "CliType", "CommandSpec", "DatasetRef",
    "Execution", "KeyValue", "PipelineInput", "ResultPolicy", "Section",
    "command", "command_spec", "hidden", "hidden_spec",
]
