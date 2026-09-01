"""``gk_rz`` -- interpolate DG data and map it onto the R-Z plane."""

from __future__ import annotations

import inspect

import click

import postgkyl as pg

from .._apply import active_datasets, apply
from .._options import label_option, tag_option, use_option


def _default(name: str):
  """Derive CLI defaults from the authoritative operation signature."""
  return inspect.signature(pg.gk_rz).parameters[name].default
# end


@click.command("gk_rz")
@click.option("--mapc2p", "-m", default=_default("mapc2p"), type=click.STRING,
    help="Use a modal mapc2p file as the geometry source instead of the default "
         "nodes file; pass '' to look up '<prefix>-geo_int_mapc2p.gkyl' from the "
         "first processed dataset's prefix.")
@click.option("--nodes", "-n", "nodes_file", default=_default("nodes_file"), type=click.STRING,
    help="Path to a nodal geometry file, overriding the default "
         "'<prefix>-geo_int_nodes.gkyl' lookup.")
@click.option("--z-axis", "-z", type=float, default=_default("z_axis"),
    help="Vertical position of the magnetic axis (m), added to the geometry Z. "
         "mapc2p files store Z relative to the axis; pass Z_axis from the "
         "simulation input file to plot in machine coordinates.")
@click.option("--phi-tor", "-p", type=float, default=_default("phi_tor"),
    help="Toroidal angle (radians) of the poloidal plane to project 3-D data onto.")
@click.option("--nz-interp", type=int, default=_default("nz_interp"),
    help="Parallel (z) up-sampling factor used to smooth the projected 3-D surfaces.")
@use_option
@tag_option(default="rz")
@label_option()
@click.pass_context
def command(ctx, mapc2p, nodes_file, z_axis, phi_tor, nz_interp, use, tag,
    label) -> None:
  """Gyrokinetics: interpolate DG dataset(s) and map them to the R-Z plane.

  Assumes DG data (not yet interpolated) has been loaded onto the working
  set by a preceding command.

  The geometry is found from the prefix of the first matching dataset's
  source file: the pointwise '<prefix>-geo_int_nodes.gkyl' is preferred
  (exact node coordinates, robust at coarse z resolution), falling back to
  the modal '<prefix>-geo_int_mapc2p.gkyl'. Use '-n path' to point at a
  specific nodes file, '-m path' at a specific mapc2p file, or "-m ''" to
  force the default mapc2p lookup.

  For 3-D (field-aligned) data the field is reconstructed on the poloidal
  plane at toroidal angle --phi-tor (default 0) by interpolating along the
  binormal direction, up-sampled in z (--nz-interp) for smooth surfaces.
  """
  targets = [d for d in active_datasets(ctx) if use is None or d.tag == use]
  if not targets:
    return
  # end

  gk_ops = pg.operations.gyrokinetics
  geo = gk_ops.resolve_geometry(
      targets[0].file_name, mapc2p=mapc2p, nodes_file=nodes_file)
  projection = gk_ops.resolve_rz_projection(
      targets[0], geo, z_axis=z_axis, nz_interp=nz_interp)

  apply(ctx, lambda d: gk_ops.map_to_rz(
      d, projection, phi_tor=phi_tor, tag=tag, label=label), use=use)
# end
