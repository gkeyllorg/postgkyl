"""``pgkyl`` command-line entry point -- a chained pipeline on pure Click.

The chained syntax mirrors the fluent script API 1:1::

    pg.load('f.gkyl').interpolate().select(z0=0).plot()      # script
    pgkyl   f.gkyl    interpolate select --z0 0 plot   # CLI

Chaining and callback-before-dispatch are native to ``click.Group(chain=True)``,
so the only custom code is a small :class:`PgkylGroup.get_command` override for
command-name abbreviation and treating a bare filename as an implicit ``load``.
Scientific commands are compiled from public API callables at import time;
this module owns only chaining, abbreviation, bare-file dispatch, and session
policy.
"""

from __future__ import annotations

from glob import glob

import click

from postgkyl import __version__, version_report
from postgkyl.cli.state import DataSpace
from postgkyl.cli.commands import (
    COMMANDS, COMMAND_SECTIONS, LEGACY_COMMANDS_BY_NAME,
)
from postgkyl.cli.compat import COMMAND_ALIASES, warn_legacy


class PgkylGroup(click.Group):
  """Click's chained group + two conveniences: abbreviation & bare-filename load."""

  def get_command(self, ctx, name):
    cmd = super().get_command(ctx, name)
    if cmd is not None:
      return cmd
    # end
    if name in LEGACY_COMMANDS_BY_NAME:
      warn_legacy(ctx, name)
      return LEGACY_COMMANDS_BY_NAME[name]
    # end
    if name in COMMAND_ALIASES:
      target = COMMAND_ALIASES[name]
      command = super().get_command(ctx, target)
      if command is not None:
        warn_legacy(ctx, name)
        return command
      # end
    # end
    matches = [c for c in self.list_commands(ctx) if c.startswith(name)]
    if len(matches) == 1:
      return super().get_command(ctx, matches[0])
    # end
    if matches:
      ctx.fail(f"Ambiguous command '{name}': {', '.join(sorted(matches))}")
    # end
    if glob(name):
      ctx.obj.in_data_strings.append(name)
      return super().get_command(ctx, "load")
    # end
    ctx.fail(f"'{name}' is not a command name nor a data file")
  # end

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
        # end
        rows.append((name, cmd.get_short_help_str(limit=formatter.width - 6)))
      # end
      if rows:
        with formatter.section(section):
          formatter.write_dl(rows)
  # end
# end


def _print_version(ctx, param, value) -> None:
  if not value or ctx.resilient_parsing:
    return
  # end
  click.echo(version_report(__version__))
  ctx.exit()
# end


@click.group(cls=PgkylGroup, chain=True,
    context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--version", is_flag=True, expose_value=False, is_eager=True,
    callback=_print_version,
    help="Show version, commit, Gkeyll build info, and exit.")
@click.option("--batch-mode", "-b", is_flag=True, help="Do not show plots; save them instead.")
@click.option("--saveframes-prefix", default="pgkyl", help="Output prefix used in batch mode.")
@click.option("--value-form", "-v", default=None,
    type=click.Choice(["modal", "nodal", "quad"]),
    help="Override every loaded file's modal/nodal/quad tag -- for files "
    "whose header carries DG basis metadata even though the stored values "
    "are already point values (e.g. a per-cell diagnostic like a CFL rate).")
@click.pass_context
def cli(ctx, batch_mode, saveframes_prefix, value_form) -> None:
  """Postprocessing and plotting tool for Gkeyll data.

  Datasets are loaded, processed and plotted by chaining commands, e.g.::

      pgkyl file.gkyl interpolate select --z0 0 plot
  """
  ctx.obj = DataSpace(batch=batch_mode, prefix=saveframes_prefix,
      value_form=value_form)
# end


for _command in COMMANDS:
  cli.add_command(_command)
# end


if __name__ == "__main__":
  cli()
# end
