"""Gyrokinetic diagnostics: named quantities and distribution functions.

``postgkyl.diagnostics.gk`` is another equation-specific module in
the diagnostics layer (like ``five_moment`` in
``examples/scripts/03_diagnostics_five_moment.py``), but for gyrokinetic
simulations it owns its own *loading* too -- a simulation's files follow a
naming convention (``<name>-<species>_<quantity>_<frame>.gkyl``), so instead
of calling ``pg.load`` on individual filenames, you ask for a quantity by
name and the loader resolves which files it needs:

* ``pg.gk.available_quantities()`` lists the registered quantity names.
* ``pg.gk.load_quantity(quantity, species, name, frame, path=...)`` resolves
  the source file(s) for that quantity, computes it, and returns one
  ``GData`` per requested species -- already interpolated, ready to
  ``.select()``/``.plot()`` like anything else.
* ``pg.gk.load_distf(name=..., species=..., frame=..., ...)`` reconstructs a
  full distribution function from the saved ``Jf``-times-Jacobian(s) files
  (what the CLI's ``gk_load_distf`` command wraps).

This uses the ``rt_gk_tcv_iwl*`` fixtures staged in ``tests/test_data`` --
two related simulations: ``rt_gk_tcv_iwl_adapt_source_1x2v_p1`` wrote ion
Hamiltonian moments (``M0``/``M1`` are derivable from those), and
``rt_gk_tcv_iwl_1x2v_p1`` wrote the electron distribution function plus its
geometry (``geo_int_jacobtot_inv``).

Run directly:
    MPLBACKEND=Agg PYTHONPATH=src python examples/scripts/04_gyrokinetics.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import postgkyl as pg

from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()

HMOM_NAME = "rt_gk_tcv_iwl_adapt_source_1x2v_p1"  # wrote ion Hamiltonian moments
GK_NAME = "rt_gk_tcv_iwl_1x2v_p1"                 # wrote elc distf + geometry

print("registered quantities:", pg.gk.available_quantities())

# 1. A moment quantity ("M0", the density) resolved straight from the
#    HamiltonianMoments file the registry knows how to read; already
#    interpolated, so it's a regular one-component field.
m0, = pg.gk.load_quantity(
    "M0", "ion", HMOM_NAME, "250", path=str(TEST_DATA))
print("M0:", repr(m0), " label:", m0.get_label())

fig = m0.plot(title=m0.get_label(), show=False)
fig.savefig(OUTPUT_DIR / "04_gyrokinetics_M0.png")

# 2. "M1" needs the species mass to convert a momentum-like moment into a
#    velocity -- extra per-quantity parameters go through **extra.
m1, = pg.gk.load_quantity(
    "M1", "ion", HMOM_NAME, "250", path=str(TEST_DATA), mass=2.0)
print("M1:", repr(m1), " label:", m1.get_label())

# 3. A species-independent geometric factor, from the other simulation --
#    ``species=None`` since geometry isn't per-species.
jacobtot_inv, = pg.gk.load_quantity(
    "geo_int_jacobtot_inv", None, GK_NAME, path=str(TEST_DATA))
print("(J B)^-1:", repr(jacobtot_inv), " label:", jacobtot_inv.get_label())

fig = jacobtot_inv.plot(title=jacobtot_inv.get_label(), show=False)
fig.savefig(OUTPUT_DIR / "04_gyrokinetics_jacobtot_inv.png")

# 4. The full electron distribution function: 3D (x, vpar, mu), built from
#    the saved Jf-times-Jacobian(s) file plus the geometry factor above.
distf = pg.gk.load_distf(
    name=str(TEST_DATA / GK_NAME), species="elc",
    frame=250,
    jacobtot_inv_file=TEST_DATA / f"{GK_NAME}-geo_int_jacobtot_inv.gkyl")
print("distf:", repr(distf))

# It's a regular GData from here on -- e.g. select a fixed-mu slice down to
# the (x, vpar) plane and plot it, same as any other 2D field.
slice_2d = distf.select(z2=0.0)
fig = slice_2d.plot(title="elc distf, mu=0 slice", show=False)
fig.savefig(OUTPUT_DIR / "04_gyrokinetics_distf_slice.png")

print("04_gyrokinetics: OK")
