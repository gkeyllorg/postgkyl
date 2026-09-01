"""``print`` -- print the values (or grid) of the active datasets."""

from __future__ import annotations

import click
import numpy as np

from .._apply import active_datasets

np.set_printoptions(precision=16)


@click.command("print")
@click.option("--use", default=None, help="Select active datasets carrying this tag.")
@click.option("--grid", "-g", "show_grid", is_flag=True, default=False,
    help="Print the grid instead of the values.")
@click.pass_context
def command(ctx, use, show_grid) -> None:
  """Print the values (or, with --grid, the grid) of the active datasets."""
  pool = active_datasets(ctx)
  if use is not None:
    pool = [d for d in pool if d.tag == use]
  # end
  for d in pool:
    if show_grid:
      for axis in d.grid:
        click.echo(axis)
    # end
      # end
    else:
      click.echo(np.asarray(d.values).squeeze())
# end
    # end
  # end
