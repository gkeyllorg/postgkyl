"""``load`` -- drain queued file globs into the working set (bare-filename dispatch).

Basis, polynomial order, and value_form are properties of the data itself
(HIERARCHY docs / CLAUDE.md): this is the one place the CLI lets a caller
set them, exactly once, at the moment a file is read. No downstream verb
(``interpolate``, ``average``, ...) ever re-specifies them again -- they
always read ``ctx["basis_type"]``/``ctx["poly_order"]``/``ctx["value_form"]``
off the already-loaded dataset.
"""

from __future__ import annotations

from glob import glob

import click

import postgkyl as pg

from .._options import label_option, tag_option

# Short DG basis code -> (long basis name, default value_form). The short
# codes name whether the on-disk basis is modal or nodal (e.g. "ns" =
# non-modal/nodal serendipity); an explicit --value-form still overrides
# this implied default -- see the precedence in ``command`` below.
_BASIS_CODES = {
    "ms": ("serendipity", "modal"),
    "ns": ("serendipity", "nodal"),
    "mo": ("maximal-order", "modal"),
    "mt": ("tensor", "modal"),
    "gkhyb": ("gkhybrid", "modal"),
    "pkpmhyb": ("hybrid", "modal"),
}


@click.command("load", hidden=True)
@click.option("--z0", default=None, help="Partial file load: 0th coord (either int or slice).")
@click.option("--z1", default=None, help="Partial file load: 1st coord (either int or slice).")
@click.option("--z2", default=None, help="Partial file load: 2nd coord (either int or slice).")
@click.option("--z3", default=None, help="Partial file load: 3rd coord (either int or slice).")
@click.option("--z4", default=None, help="Partial file load: 4th coord (either int or slice).")
@click.option("--z5", default=None, help="Partial file load: 5th coord (either int or slice).")
@click.option("--component", "-c", default=None,
    help="Partial file load: component(s) (either int or slice).")
@tag_option("default")
@label_option()
@click.option("--basis", "-b", default=None,
    type=click.Choice(sorted(_BASIS_CODES)),
    help="DG basis code (ms, ns, mo, mt, gkhyb, pkpmhyb). Overrides the "
    "file header's basis_type -- required for files with no basis metadata "
    "at all (e.g. a velocity-space mapc2p_vel file). Default: from file.")
@click.option("--poly-order", "-p", "poly_order", type=int, default=None,
    help="Polynomial order. Overrides the file header's poly_order. "
    "Default: from file.")
@click.option("--value-form", "-v", default=None,
    type=click.Choice(["modal", "nodal", "quad"]),
    help="Override this load's modal/nodal/quad tag, taking precedence over "
    "any default implied by '--basis' and over the session-wide "
    "'--value-form' -- for files whose header carries DG basis metadata "
    "even though the stored values are already point values (e.g. a "
    "per-cell diagnostic like a CFL rate).")
@click.pass_context
def command(ctx, z0, z1, z2, z3, z4, z5, component, tag, label, basis,
    poly_order, value_form) -> None:
  """Load queued data files (invoked implicitly by bare filenames)."""
  ds = ctx.obj
  patterns, ds.in_data_strings = list(ds.in_data_strings), []
  axes = (z0, z1, z2, z3, z4, z5)
  basis_type, basis_value_form = _BASIS_CODES.get(basis, (None, None))
  vform = (value_form if value_form is not None
      else basis_value_form if basis_value_form is not None
      else ds.value_form)
  for pattern in patterns:
    loaded = [pg.load(fn, tag=tag, label=label, value_form=vform,
        basis_type=basis_type, poly_order=poly_order, axes=axes,
        comp=component) for fn in glob(pattern)]
    # Natural (numeric-aware) order, via the same one-home sort the ``sort``
    # verb uses: a plain lexicographic sort would order a glob's matches
    # '..._b0, _b1, _b10, _b2' and frames '_1, _10, _2', so blocks and frames
    # would enter the working set out of order.
    ds.datasets.extend(pg.sort(*loaded))
  # end
# end
