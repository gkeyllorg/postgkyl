import os

import click
import numpy as np

from postgkyl.data import GData
from postgkyl.tools.gk_transport import compute_transport, TRANSPORT_OUTPUTS
from postgkyl.utils import verb_print
from postgkyl.commands.gk_load_quantity import parse_extra


@click.command(name="gk-transport")
@click.option("--name", "-n", required=True, type=click.STRING,
  help="Simulation name prefix (e.g. gk_tcv_3x2v_p1).")
@click.option("--species", "-s", required=True, type=click.STRING,
  help="Species name, or a comma-separated list (e.g. elc,ion).")
@click.option("--frame", "-f", default=None, type=click.STRING,
  help="Frames to average over: a frame, a comma-separated list, or a range "
       "'start:stop[:step]'. Default: every available frame.")
@click.option("--path", "-p", default="./", type=click.STRING,
  help="Directory containing the simulation files.")
@click.option("--outputs", "-o", default="gamma,q,D,chi", type=click.STRING,
  help=f"Comma-separated profiles to push to the stack, among: {', '.join(TRANSPORT_OUTPUTS)}.")
@click.option("--fluct", default="none", type=click.Choice(["none", "y", "yz"]),
  help="'none' for the total fluxes, 'y' or 'yz' for the turbulent part only, i.e. the "
       "correlation of the fluctuations about the y or (y,z) average.")
@click.option("--conv", default=1.5, type=click.FLOAT, show_default=True,
  help="Coefficient c of the convective energy flux c*<T>*Gamma removed from Q to form q "
       "(0, 3/2 or 5/2).")
@click.option("--grad-tol", default=1e-3, type=click.FLOAT, show_default=True,
  help="D (chi) is masked where |d<n>/dx| (|d<T>/dx|) is below grad-tol times its maximum.")
@click.option("--per-frame", is_flag=True, default=False,
  help="Push the flux-surface averaged profiles of each frame instead of their time "
       "average (e.g. to 'collect' them into a space-time diagram).")
@click.option("--interp", "-i", default=None, type=click.INT,
  help="Number of radial nodes per cell (default poly_order+1).")
@click.option("--extra", "-e", default=None, type=click.STRING,
  help="Extra key=value pairs for the fetch functions, e.g. mass=...,charge=... A key may "
       "be given one value per species, in the order of --species.")
@click.option("--tag", "-t", default="transport", type=click.STRING,
  help="Tag prefix for the output datasets: <tag>_<output>[_<species>].")
@click.option("--label", "-l", default=None, type=click.STRING,
  help="Label override for the output datasets.")
@click.pass_context
def gk_transport(ctx, **kwargs):
  """
  Gyrokinetics: radial turbulent transport of a 3x simulation.

  \b
  Flux-surface (y,z, Jacobian-weighted) and time averaged radial profiles of
    gamma: particle flux <Gamma^x>, ExB plus magnetic flutter if apar exists,
    Q:     energy flux <Q^x> = <(m/2) M2 v^x>,
    q:     heat flux q = Q - conv*<T>*Gamma,
    D:     particle diffusivity -Gamma/(<g^xx> d<n>/dx) (m^2/s),
    chi:   heat diffusivity -q/(<n> <g^xx> d<T>/dx) (m^2/s),
    n, T, gxx: the averaged density, temperature and <|grad x|^2>.
  The fluxes are contravariant radial components (.grad x).

  \b
  Command line example:
    pgkyl gk-transport -n gk_tcv_3x2v_p1 -s elc,ion -f 100:200 -o D,chi plot
  """
  data = ctx.obj["data"]

  outputs = [o.strip() for o in kwargs["outputs"].split(",") if o.strip()]
  unknown = [o for o in outputs if o not in TRANSPORT_OUTPUTS]
  if unknown:
    ctx.fail(f"Unknown output(s) {', '.join(unknown)}. Choose among: "
             f"{', '.join(TRANSPORT_OUTPUTS)}.")

  species_list = [s.strip() for s in kwargs["species"].split(",") if s.strip()]
  user_extra = parse_extra(kwargs["extra"])

  for species_idx, species in enumerate(species_list):
    verb_print(ctx, f"gk-transport: computing the transport of {species}")
    results = compute_transport(
      kwargs["path"], kwargs["name"], species, frame=kwargs["frame"], fluct=kwargs["fluct"],
      conv=kwargs["conv"], grad_tol=kwargs["grad_tol"],
      extra=dict(user_extra, species_idx=species_idx),
      per_frame=kwargs["per_frame"], num_interp=kwargs["interp"])

    for res in results:
      verb_print(ctx, f"  frames {res['frames'][0]}..{res['frames'][-1]} ({len(res['frames'])})")
      for output in outputs:
        out_ctx = {"frame": res["frames"][-1], "grid_type": "uniform"}
        if res["time"] is not None:
          out_ctx["time"] = res["time"]
        out = GData(ctx=out_ctx)
        out.push([res["grid"][0]], res[output][..., np.newaxis])

        label = kwargs["label"] if kwargs["label"] is not None else (
          TRANSPORT_OUTPUTS[output] % species[0])
        if kwargs["label"] is not None and len(species_list) > 1:
          label += f" {species}"
        if len(results) > 1:
          label += f" f{res['frames'][-1]}"
        out.set_label(label)

        tag = f"{kwargs['tag']}_{output}"
        if len(species_list) > 1:
          tag += f"_{species}"
        out.set_tag(tag)

        # Stamp a file name so later commands can locate sibling files.
        out._file_name = os.path.join(kwargs["path"],
                                      f"{kwargs['name']}-{species}_{output}_{res['frames'][-1]}.gkyl")
        data.add(out)

  verb_print(ctx, "Finished gk-transport")
