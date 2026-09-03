#!/usr/bin/env python3
"""Command line entry point for postgkyl.

Uses click (https://click.palletsprojects.com/en) commands to wrap pgkyl functions.
"""

from glob import glob
import click
import os.path
import sys
import time

from postgkyl import __version__
from postgkyl.commands import DataSpace
from postgkyl.utils import load_style, verb_print
import postgkyl.commands as cmd


def _print_version(ctx, param, value):
  if not value or ctx.resilient_parsing:
    return
  # end
  click.echo(f"Postgkyl {__version__} ({sys.platform})")
  click.echo(f"Python version: {sys.version}".format())
  click.echo("Copyright 2016-2024 Gkeyll Team")
  click.echo("Postgkyl can be used freely for research at universities,")
  click.echo("national laboratories, and other non-profit institutions.")
  click.echo("There is NO warranty.\n")
  click.echo("Spam, egg, sausage, and spam.")
  ctx.exit()


class PgkylCommandGroup(click.Group):
  """Custom pgkyl click command group class.

  It allows to:
    - use shortened versions of command names
    - use a file name as a command
  """

  def get_command(self, ctx, cmd_name):
    # cmd_name is a full name of a pgkyl command
    rv = click.Group.get_command(self, ctx, cmd_name)
    if rv is not None:
      return rv
    # end

    # cmd_name is an abreviation of a pgkyl command
    matches = [x for x in self.list_commands(ctx) if x.startswith(cmd_name)]
    if matches and len(matches) == 1:
      return click.Group.get_command(self, ctx, matches[0])
    elif matches:
      ctx.fail(f"Too many matches for '{cmd_name}': {', '.join(sorted(matches))}")
    # end

    # cmd_name is a data set
    if glob(cmd_name):
      ctx.obj["in_data_strings"].append(cmd_name)
      return click.Group.get_command(self, ctx, "load")
    # end

    ctx.fail(f"'{cmd_name}' does not match either command name nor a data file")


# The command line mode entry command
@click.command(name="pgkyl", cls=PgkylCommandGroup, chain=True,
    context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--verbose", "-v", is_flag=True, help="Turn on verbosity.")
@click.option("--batch-mode", is_flag=True, help="Run in batch mode (no plots will be shown).")
@click.option("--saveframes-prefix", default=os.path.expanduser("~")+"/pg",
              help="Output prefix to use for plot output in batch mode.")
@click.option("--version", is_flag=True, callback=_print_version, expose_value=False,
    is_eager=True, help="Print the version information.")
@click.option("--z0", help="Partial file load: 0th coord (either int or slice)")
@click.option("--z1", help="Partial file load: 1st coord (either int or slice)")
@click.option("--z2", help="Partial file load: 2nd coord (either int or slice)")
@click.option("--z3", help="Partial file load: 3rd coord (either int or slice)")
@click.option("--z4", help="Partial file load: 4th coord (either int or slice)")
@click.option("--z5", help="Partial file load: 5th coord (either int or slice)")
@click.option("--component", "-c", help="Partial file load: comps (either int or slice)")
@click.option("--compgrid", is_flag=True, help="Disregard the mapped grid information")
@click.option("--varname", "-d", multiple=True,
    help="Specify the Adios variable name (default is 'CartGridField')")
@click.option("--c2p", help="Specify the file name containing c2p mapped coordinates")
@click.option("--c2p-vel", "c2p_vel",
    help="Specify the file name containing c2p mapped velocity coordinates")
@click.option("--style", help="Sets Maplotlib rcParams style file.")
@click.pass_context
def cli(ctx, **kwargs):
  """Postprocessing and plotting tool for Gkeyll data.

  Datasets can be loaded, processed and plotted using a command chaining mechanism. For
  full documentation see the Gkeyll documentation webpages
  (https://gkeyll.readthedocs.io). Help for individual commands can be obtained using
  the --help option for that command.
  """
  ctx.obj = {}  # The main contex object
  ctx.obj["start_time"] = time.time()  # Timings are written in the verbose mode
  if kwargs["verbose"]:
    ctx.obj["verbose"] = True
    verb_print(ctx, "This is Postgkyl running in verbose mode.")
  else:
    ctx.obj["verbose"] = False
  # end

  ctx.obj["batch_mode"] = False
  if kwargs["batch_mode"]:
    ctx.obj["batch_mode"] = True
  #end

  ctx.obj["saveframes_prefix"] = kwargs["saveframes_prefix"]

  ctx.obj["in_data_strings"] = []
  ctx.obj["in_data_strings_loaded"] = 0

  ctx.obj["data"] = DataSpace()

  ctx.obj["fig"] = ""
  ctx.obj["ax"] = ""

  ctx.obj["compgrid"] = kwargs["compgrid"]
  ctx.obj["global_var_names"] = kwargs["varname"]
  ctx.obj["global_cuts"] = (kwargs["z0"], kwargs["z1"], kwargs["z2"],
      kwargs["z3"], kwargs["z4"], kwargs["z5"], kwargs["component"])
  ctx.obj["global_c2p"] = kwargs["c2p"]
  ctx.obj["global_c2p_vel"] = kwargs["c2p_vel"]

  ctx.obj["rcParams"] = {}
  fn = kwargs["style"] if kwargs["style"] else f"{os.path.dirname(os.path.realpath(__file__))}/output/postgkyl.mplstyle"
  load_style(ctx, fn)


# Hook the individual commands into pgkyl
cli.add_command(cmd.config)
cli.add_command(cmd.activate)
cli.add_command(cmd.agyro)
cli.add_command(cmd.mom_agyro)
cli.add_command(cmd.animate)
cli.add_command(cmd.collect)
cli.add_command(cmd.current)
cli.add_command(cmd.deactivate)
cli.add_command(cmd.dg_evproj)
cli.add_command(cmd.dg_avg)
cli.add_command(cmd.differentiate)
cli.add_command(cmd.energetics)
cli.add_command(cmd.euler)
cli.add_command(cmd.mhd)
cli.add_command(cmd.ev)
cli.add_command(cmd.extractinput)
cli.add_command(cmd.fft)
cli.add_command(cmd.gk_nodes)
cli.add_command(cmd.dg_local_poly)
cli.add_command(cmd.gk_distf)
cli.add_command(cmd.gk_load_quantity)
cli.add_command(cmd.grid)
cli.add_command(cmd.gui)
cli.add_command(cmd.growth)
cli.add_command(cmd.info)
cli.add_command(cmd.integrate)
cli.add_command(cmd.interpolate)
cli.add_command(cmd.laguerrecompose)
cli.add_command(cmd.listoutputs)
cli.add_command(cmd.load)
cli.add_command(cmd.magsq)
cli.add_command(cmd.mask)
cli.add_command(cmd.gk_energy_balance)
cli.add_command(cmd.gk_particle_balance)
cli.add_command(cmd.gk_rz)
cli.add_command(cmd.gk_fluxsurf)
cli.add_command(cmd.plot)
cli.add_command(cmd.pr)
cli.add_command(cmd.relchange)
cli.add_command(cmd.select)
cli.add_command(cmd.style)
cli.add_command(cmd.tenmoment)
cli.add_command(cmd.trajectory)
cli.add_command(cmd.val2coord)
cli.add_command(cmd.velocity)
cli.add_command(cmd.write)
cli.add_command(cmd.transformframe)
cli.add_command(cmd.pkpm)

if __name__ == "__main__":
  ctx = []
  cli(ctx)
# end
