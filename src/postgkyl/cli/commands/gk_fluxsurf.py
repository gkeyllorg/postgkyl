"""``gk_fluxsurf`` -- extract a 2-D theta-phi flux surface from 3-D data."""

from __future__ import annotations

import inspect

import click

import postgkyl as pg

from .._apply import active_datasets, apply
from .._options import label_option, tag_option, use_option


def _default(function, name: str):
  """Derive CLI defaults from the authoritative operation signature."""
  return inspect.signature(function).parameters[name].default
# end


@click.command("gk_fluxsurf")
@click.option("--mapc2p", "-m",
    default=_default(pg.operations.gyrokinetics.resolve_geometry, "mapc2p"),
    type=click.STRING,
    help="Use a modal mapc2p file as the geometry source instead of the default "
         "nodes file.")
@click.option("--nodes", "-n", "nodes_file",
    default=_default(pg.operations.gyrokinetics.resolve_geometry, "nodes_file"),
    type=click.STRING,
    help="Path to a nodal geometry file, overriding the default lookup.")
@click.option("--x-idx", "-x", type=int,
    default=_default(pg.operations.gyrokinetics.resolve_flux_surface_grid, "x_idx"),
    help="Cell index in the radial (x) direction representing the flux surface.")
@click.option("--nphi", type=int,
    default=_default(pg.operations.gyrokinetics.resolve_flux_surface_grid, "nphi"),
    help="Number of toroidal angle (phi) slices.")
@click.option("--nz-interp", type=int,
    default=_default(pg.operations.gyrokinetics.resolve_flux_surface_grid, "nz_interp"),
    help="Parallel (z) up-sampling factor used to smooth the projected 3-D surfaces.")
@use_option
@tag_option(default="fluxsurf")
@label_option()
@click.pass_context
def command(ctx, mapc2p, nodes_file, x_idx, nphi, nz_interp, use, tag, label) -> None:
  """Gyrokinetics: extract a 2-D theta-phi flux surface.

  Extracts data along a specific radial flux surface (constant x) for 3-D
  field-aligned data, by performing a binormal projection over a scan of
  toroidal angles (phi), creating a 2-D grid of phi vs z (where z maps
  along the poloidal/theta direction).

  The geometry is found the same way as ``gk_rz``: from the prefix of the
  first matching dataset's source file, preferring the pointwise
  '<prefix>-geo_int_nodes.gkyl', falling back to the modal
  '<prefix>-geo_int_mapc2p.gkyl'.
  """
  targets = [d for d in active_datasets(ctx) if use is None or d.tag == use]
  if not targets:
    return
  # end

  gk_ops = pg.operations.gyrokinetics
  geo = gk_ops.resolve_geometry(
      targets[0].file_name, mapc2p=mapc2p, nodes_file=nodes_file)
  fs_grid = gk_ops.resolve_flux_surface_grid(
      targets[0], geo, x_idx=x_idx, nphi=nphi, nz_interp=nz_interp)

  apply(ctx, lambda d: gk_ops.extract_flux_surface(
      d, fs_grid, tag=tag, label=label), use=use)
# end
