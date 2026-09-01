"""Runtime-compiled command inventory plus the bounded front-end commands."""

from __future__ import annotations

import click

from postgkyl.cli.compat import rewrite_argv
from postgkyl.cli.compiler import (
    build_click_command, compile_public_surface, execute_model, group_by_section,
)
from postgkyl.cli.discovery import discover_public_surface
from postgkyl.cli.legacy import LEGACY_MODELS, build_legacy_command

from . import print as _print
from . import status


class GeneratedCommand(click.Command):
  """A generated command with the temporary declarative argv translator."""

  def __init__(self, model):
    self.model = model
    built = build_click_command(model)
    super().__init__(built.name, params=built.params, callback=built.callback,
        help=built.help, short_help=built.short_help, hidden=built.hidden)
  # end

  def parse_args(self, ctx, args):
    return super().parse_args(ctx, rewrite_argv(self.model, ctx, list(args)))
  # end
# end


# Compilation validates the complete discovered surface before any command is
# registered on the Click group.
MODELS = compile_public_surface(discover_public_surface())
GENERATED_COMMANDS = tuple(GeneratedCommand(model) for model in MODELS)
_MODEL_BY_NAME = {model.name: model for model in MODELS}
LEGACY_COMMANDS_BY_NAME = {
    name: build_legacy_command(name) for name in LEGACY_MODELS
}


def _legacy_diagnostic(name, namespace, variables, extra_options):
  """Build one temporary variable-dispatch adapter over direct models."""
  params = [click.Option(["--variable-name", "-v"], required=True,
      type=click.Choice(sorted(variables)), help="Legacy quantity alias.")]
  params.extend(extra_options)
  params.extend([
      click.Option(["--use"], default=None),
      click.Option(["--tag"], default=None),
      click.Option(["--label"], default=None),
  ])

  @click.pass_context
  def callback(click_context, variable_name, **values):
    from postgkyl.cli.compat import warn_legacy

    warn_legacy(click_context, name)
    function = variables[variable_name]
    target = _MODEL_BY_NAME[
        f"{namespace}-{function.__name__.replace('_', '-')}"]
    accepted = {parameter.name for parameter in target.parameters
        if not parameter.injected}
    call_values = {parameter.name: values.get(parameter.name, parameter.default)
        for parameter in target.parameters if not parameter.injected}
    call_values["use"] = values.get("use")
    return execute_model(click_context, target, call_values)
  # end
  return click.Command(name, params=params, callback=callback,
      help=f"Deprecated {name} variable dispatcher; use direct {namespace}-* commands.")
# end


import postgkyl as _pg

LEGACY_DISPATCHERS = (
    _legacy_diagnostic("euler", "five-moment",
        _pg.diagnostics.five_moment.VARIABLES, [
          click.Option(["--gas-gamma", "-g"], type=float, default=5.0 / 3.0),
          click.Option(["--num-moms"], type=int, default=None),
        ]),
    _legacy_diagnostic("tenmoment", "ten-moment",
        _pg.diagnostics.ten_moment.VARIABLES, [
          click.Option(["--gas-gamma", "-g"], type=float, default=5.0 / 3.0),
        ]),
    _legacy_diagnostic("mhd", "mhd", _pg.diagnostics.mhd.VARIABLES, [
      click.Option(["--gas-gamma", "-g"], type=float, default=5.0 / 3.0),
      click.Option(["--mu-0"], type=float, default=1.0),
    ]),
)
FRONT_END_COMMANDS = (status.command, _print.command)
COMMANDS = GENERATED_COMMANDS + LEGACY_DISPATCHERS + FRONT_END_COMMANDS
COMMAND_SECTIONS = group_by_section(MODELS)
COMMAND_SECTIONS.setdefault("Diagnostics", []).extend(
    command.name for command in LEGACY_DISPATCHERS)
COMMAND_SECTIONS.setdefault("Utility", []).extend(
    command.name for command in FRONT_END_COMMANDS)

_BY_NAME = {command.name: command for command in COMMANDS}


def command_named(name: str) -> click.Command:
  """Return one compiled command by canonical name."""
  return _BY_NAME[name]
# end


__all__ = [
    "COMMANDS", "COMMAND_SECTIONS", "FRONT_END_COMMANDS", "GENERATED_COMMANDS",
    "LEGACY_COMMANDS_BY_NAME", "LEGACY_DISPATCHERS", "MODELS", "command_named",
]
