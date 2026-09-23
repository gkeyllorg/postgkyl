import os

import click

from postgkyl.data import GData
from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops
from postgkyl.utils import verb_print
from postgkyl.commands.dg_avg import _jacobgeo_path


@click.command(name="dg-fluct")
@click.option("--z0", is_flag=True, help="Subtract the average over direction 0.")
@click.option("--z1", is_flag=True, help="Subtract the average over direction 1.")
@click.option("--z2", is_flag=True, help="Subtract the average over direction 2.")
@click.option("--z3", is_flag=True, help="Subtract the average over direction 3.")
@click.option("--z4", is_flag=True, help="Subtract the average over direction 4.")
@click.option("--z5", is_flag=True, help="Subtract the average over direction 5.")
@click.option("--weight", "-w", default=None,
    help="Weight file for the average. Defaults to <prefix>-geo_int_jacobgeo.gkyl "
         "found next to the dataset; pass a path to override, or 'none' to disable.")
@click.option("--use", "-u", help="Tag to apply to. [default: all active]")
@click.option("--tag", "-t", help="Tag for the output dataset.")
@click.option("--label", "-l", help="Label for the output dataset.")
@click.pass_context
def dg_fluct(ctx, **kwargs):
  """
  Fluctuation of a DG field about its average over specified directions.

  Computes dA = A - <A>, where <A> is the average over the directions given
  by the flags --z0, --z1, ... --z5, lifted back onto the full grid. The output
  keeps the dimensionality of the input.
  """
  verb_print(ctx, "Starting dg-fluct")
  data = ctx.obj["data"]

  z_opts   = [kwargs[f"z{d}"] for d in range(6)]
  avg_dirs = [i for i, z in enumerate(z_opts) if z]

  if not avg_dirs:
    ctx.fail("dg-fluct requires at least one direction flag (--z0 ... --z5).")

  ops = GkeyllDGops()

  weight_opt   = kwargs["weight"]
  weight_off   = weight_opt is not None and str(weight_opt).strip().lower() == "none"
  weight_cache = {}  # file path -> loaded GData (avoid re-reading per dataset)

  def _load_weight(dat):
    """Resolve and load the weight GData for a dataset, or None."""
    if weight_off:
      return None
    if weight_opt:  # explicit override: must exist
      path = weight_opt
      if not os.path.isfile(path):
        ctx.fail(f"weight file '{path}' not found.")
    else:  # auto-detect the geometric Jacobian
      path = _jacobgeo_path(dat.get_file_name())
      if path is None or not os.path.isfile(path):
        verb_print(ctx, f"No jacobgeo weight found ({path}); using unweighted average.")
        return None
    if path not in weight_cache:
      verb_print(ctx, f"Loading average weight from {path}")
      weight_cache[path] = GData(file_name=path, comp_grid=ctx.obj["compgrid"])
    return weight_cache[path]

  for dat in data.iterator(kwargs["use"]):
    out = ops.fluctuation(avg_dirs, dat, weight=_load_weight(dat))

    # Carry the provenance of the source over to the fluctuation.
    out._file_name = dat.get_file_name()
    out.set_label(dat.get_label())
    out.set_tag(kwargs["tag"] if kwargs["tag"] else dat.get_tag())
    if kwargs["label"]:
      out.set_label(kwargs["label"])

    dat.deactivate()
    data.add(out)

  verb_print(ctx, "Finishing dg-fluct")
