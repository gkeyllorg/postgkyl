import click
import numpy as np

from postgkyl.data import GData
from postgkyl.data import GInterpModal, GInterpNodal
from postgkyl.utils import verb_print


@click.command()
@click.option("--basis_type","-b",
    type=click.Choice(["ms", "ns", "mo", "mt", "gkhyb", "gkhyb_vel", "pkpmhyb"]),
    help="Specify DG basis.")
@click.option("--poly_order", "-p", type=click.INT, help="Specify polynomial order.")
@click.option("--interp", "-i", type=click.INT,
     help="Interpolation onto a general mesh of specified amount.")
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option("--tag", "-t", help="Optional tag for the resulting array")
@click.option("--label", "-l", help="Custom label for the result")
@click.option("--read", "-r", type=click.BOOL, help="Read from general interpolation file.")
@click.pass_context
def interpolate(ctx, **kwargs):
  """Interpolate DG data onto a uniform mesh."""
  verb_print(ctx, "Starting interpolate")
  data = ctx.obj["data"]

  basis_type = None
  is_modal = None
  if kwargs.get("basis_type"):
    if kwargs["basis_type"] == "ms":
      basis_type = "serendipity"
      is_modal = True
    elif kwargs["basis_type"] == "ns":
      basis_type = "serendipity"
      is_modal = False
    elif kwargs["basis_type"] == "mo":
      basis_type = "maximal-order"
      is_modal = True
    elif kwargs["basis_type"] == "mt":
      basis_type = "tensor"
      is_modal = True
    elif kwargs["basis_type"] == "gkhyb":
      basis_type = "gkhybrid"
      is_modal = True
    elif kwargs["basis_type"] == "gkhyb_vel":
      basis_type = "gkhybrid_vel"
      is_modal = True
    elif kwargs["basis_type"] == "pkpmhyb":
      basis_type = "hybrid"
      is_modal = True
    # end
  # end

  for dat in data.iterator(kwargs["use"]):
    if kwargs["basis_type"] is None and dat.ctx["basis_type"] is None:
      ctx.fail(
          click.style(f"ERROR in interpolate: no 'basis_type' was specified and dataset {dat.get_label():s} does not have required ctxdata",
              fg="red")
      )
    # end


    if is_modal or dat.ctx["is_modal"]:
      dg = GInterpModal(dat, kwargs["poly_order"], kwargs["basis_type"],
          kwargs["interp"], kwargs["read"])
    else:
      dg = GInterpNodal(dat, kwargs["poly_order"], basis_type, kwargs["interp"], kwargs["read"])
    # end

    num_nodes = dg.num_nodes
    num_comps = int(dat.get_num_comps() / num_nodes)

    grid, values = dg.interpolate(tuple(range(num_comps)))
    # Handle 0D field for having collect turns into a time trace.
    if dat.ctx.get("num_cdim") == 0 and not dat.ctx.get("num_vdim"):
      grid = [np.array([grid[0][0], grid[0][-1]])]
      values = values.mean(axis=0, keepdims=True)
    # end

    if kwargs["tag"]:
      out = GData(tag=kwargs["tag"], label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"], ctx=dat.ctx)
      out.push(grid, values)
      data.add(out)
    else:
      dat.push(grid, values)
    # end
  # end
  verb_print(ctx, "Finishing interpolate")
