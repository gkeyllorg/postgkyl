import os

import click

from postgkyl.data import GData
from postgkyl.data import select as pgkyl_select
from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops
from postgkyl.utils import verb_print


def _jacobgeo_path(file_name):
  """Return the <prefix>-geo_int_jacobgeo.gkyl path next to file_name, or None."""
  if not file_name:
    return None
  dirname  = os.path.dirname(file_name)
  basename = os.path.basename(file_name)
  prefix   = basename.split("-")[0]
  return os.path.join(dirname, f"{prefix}-geo_int_jacobgeo.gkyl")


@click.command(name="dg-avg")
@click.option("--z0", is_flag=True, help="Average over direction 0.")
@click.option("--z1", is_flag=True, help="Average over direction 1.")
@click.option("--z2", is_flag=True, help="Average over direction 2.")
@click.option("--z3", is_flag=True, help="Average over direction 3.")
@click.option("--z4", is_flag=True, help="Average over direction 4.")
@click.option("--z5", is_flag=True, help="Average over direction 5.")
@click.option("--comp", "-c", default=None,
    help="Component index to select from the result (int or slice).")
@click.option("--weight", "-w", default=None,
    help="Weight file for the average. Defaults to <prefix>-geo_int_jacobgeo.gkyl "
         "found next to the dataset; pass a path to override, or 'none' to disable.")
@click.option("--use", "-u", help="Tag to apply to. [default: all active]")
@click.option("--tag", "-t", help="Tag for the output dataset.")
@click.option("--label", "-l", help="Label for the output dataset.")
@click.pass_context
def dg_avg(ctx, **kwargs):
  """
  Average a DG field over specified directions.

  Directions to average over are specified using the flags --z0, --z1, ... --z5.
  The output has a reduced dimensionality corresponding to the averaged directions.

  The geometric Jacobian <prefix>-geo_int_jacobgeo.gkyl (found next to the dataset)
  is used as the weight if present, giving the weighted average int(f J dx) / int(J dx).
  Override the weight file with --weight, or disable weighting with '--weight none'.
  """
  verb_print(ctx, "Starting dg-avg")
  data = ctx.obj["data"]

  z_opts   = [kwargs["z0"], kwargs["z1"], kwargs["z2"],
              kwargs["z3"], kwargs["z4"], kwargs["z5"]]
  avg_dirs = [i for i, z in enumerate(z_opts) if z]

  if not avg_dirs:
    ctx.fail("dg-avg requires at least one direction flag (--z0 ... --z5).")

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
    out = dat
    weight = _load_weight(dat)
    
    # Perform the averaging directly on the specified directions
    out = ops.average(avg_dirs, out, weight=weight,
                      comp_grid=ctx.obj["compgrid"])

    # Carry the provenance of the source over to the average.
    out._file_name = dat.get_file_name()
    out.set_label(dat.get_label())

    if kwargs["tag"]:
      out.set_tag(kwargs["tag"])
    if kwargs["label"]:
      out.set_label(kwargs["label"])

    if kwargs["comp"] is not None:
      pgkyl_select(out, overwrite=True, comp=kwargs["comp"])

    dat.deactivate()
    data.add(out)

  verb_print(ctx, "Finishing dg-avg")