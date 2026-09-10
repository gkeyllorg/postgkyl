"""Gyrokinetic R-Z mapping through fluent, functional, and reusable APIs.

Run directly:
    MPLBACKEND=Agg PYTHONPATH=src python examples/scripts/05_gk_rz.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np

import postgkyl as pg

from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()

data = pg.load(TEST_DATA / "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")

# The common path: geometry is inferred from FIELD's simulation prefix.
mapped = pg.gk.rz(data, z_axis=0.0, phi_tor=0.0, nz_interp=2)
fig = mapped.plot(title="Electron density in the poloidal plane",
                  xlabel="R [m]",
                  ylabel="Z [m]",
                  clabel=r"$n_e$ [m$^{-3}$]",
                  fixaspect=True,
                  no_show=True)
fig.savefig(OUTPUT_DIR / "05_gk_rz.png")

# Reuse geometry and projection when several fields/frames or toroidal angles
# share one computational grid.
geometry = pg.gk.resolve_geometry(data.file_name)
projection = pg.resolve_rz_projection(data, geometry, z_axis=0.0, nz_interp=2)
at_zero = data.map_to_rz(projection=projection, phi_tor=0.0)
at_quarter_turn = pg.map_to_rz(data, projection=projection, phi_tor=np.pi / 2)
np.testing.assert_allclose(at_zero.values, mapped.values)
assert not np.allclose(
    at_zero.values, at_quarter_turn.values, rtol=1e-12, atol=0.0)

print("05_gk_rz: OK")
