"""``pgkyl`` command-line entry point -- a chained pipeline on pure Click.

The chained syntax mirrors the fluent script API 1:1::

    pg.load('f.gkyl').interpolate().select(z0=0).plot()      # script
    pgkyl   f.gkyl    interpolate select --z0 0 plot   # CLI

Chaining and callback-before-dispatch are native to ``click.Group(chain=True)``,
so the only custom code is a small :class:`PgkylGroup.get_command` override for
command-name abbreviation and treating a bare filename as an implicit ``load``.
Every subcommand is compiled from a public API callable at import time. This
module owns only chaining, command aliases, and bare-file dispatch; it
contains no per-command option or execution definitions.
"""

from __future__ import annotations

from glob import glob
from types import MappingProxyType

import click

from postgkyl import __version__, version_report
from postgkyl.cli.compiler import (
    build_click_command,
    compile_public_surface,
    group_by_section,
)
from postgkyl.cli.discovery import discover_public_surface
from postgkyl.cli.state import DataSpace

# Compilation validates the complete discovered surface before registration.
# These aliases add spellings only; they never replace a generated command or
# alter its options.
MODELS = compile_public_surface(discover_public_surface())
COMMANDS = tuple(build_click_command(model) for model in MODELS)
COMMAND_SECTIONS = group_by_section(MODELS)
COMMAND_ALIASES = MappingProxyType({"pl": "plot", "ev": "evaluate"})


class PgkylGroup(click.Group):
  """Click's chained group with spelling-only command aliases."""

  def get_command(self, ctx, name):
    cmd = super().get_command(ctx, name)
    if cmd is not None:
      return cmd
    if name in COMMAND_ALIASES:
      target = COMMAND_ALIASES[name]
      command = super().get_command(ctx, target)
      if command is not None:
        return command
    matches = [c for c in self.list_commands(ctx) if c.startswith(name)]
    if len(matches) == 1:
      return super().get_command(ctx, matches[0])
    if matches:
      ctx.fail(f"Ambiguous command '{name}': {', '.join(sorted(matches))}")
    return None

  def resolve_command(self, ctx, args):
    """Expand a bare file pattern to the canonical ``load --file_name`` form."""
    if args:
      token = args[0]
      exact = click.Group.get_command(self, ctx, token)
      alias = COMMAND_ALIASES.get(token)
      if exact is None and alias is None and glob(token):
        args[:1] = ["load", "--file_name", token]
    return super().resolve_command(ctx, args)

  def format_commands(self, ctx, formatter) -> None:
    """Group ``pgkyl --help``'s command listing under section headers.

    Presentation only (see ``commands/__init__.py``'s ``COMMAND_SECTIONS``
    and "14-cli.md"'s "Help output organization"): every command stays a
    flat, chainable top-level ``click.Command`` resolved exactly as before;
    only how they are *printed* changes, mirroring how ``git``/``docker``
    group their subcommand help.
    """
    for section, names in COMMAND_SECTIONS.items():
      rows = []
      for name in names:
        cmd = self.get_command(ctx, name)
        if cmd is None:
          continue
        rows.append((name, cmd.get_short_help_str(limit=formatter.width - 6)))
      if rows:
        with formatter.section(section):
          formatter.write_dl(rows)


def _print_version(ctx, param, value) -> None:
  if not value or ctx.resilient_parsing:
    return
  click.echo(version_report(__version__))
  ctx.exit()


@click.group(cls=PgkylGroup,
             chain=True,
             context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--version",
              is_flag=True,
              expose_value=False,
              is_eager=True,
              callback=_print_version,
              help="Show version, commit, Gkeyll build info, and exit.")
@click.pass_context
def cli(ctx) -> None:
  """Postprocessing and plotting tool for Gkeyll data.

  Datasets are loaded, processed and plotted by chaining commands, e.g.::

      pgkyl file.gkyl interpolate select --z0 0 plot
  """
  ctx.obj = DataSpace()


for _command in COMMANDS:
  cli.add_command(_command)

__all__ = [
    "COMMANDS",
    "COMMAND_ALIASES",
    "COMMAND_SECTIONS",
    "MODELS",
    "PgkylGroup",
    "cli",
]

if __name__ == "__main__":
  cli()
