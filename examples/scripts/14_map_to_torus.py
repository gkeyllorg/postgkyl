"""Toroidal-wedge rendering through fluent and reusable APIs.

Run directly:
    PYTHONPATH=src python examples/scripts/14_map_to_torus.py
"""

from __future__ import annotations

import numpy as np

import postgkyl as pg

from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()

data = pg.load(TEST_DATA / "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")

# The wedge spans interpolated radial indices 4 to -4 (counted back from the
# outer edge) and toroidal angles 0 to 3*pi/2. Geometry is inferred from the
# dataset's simulation prefix. The p1 solution dips below zero at a few
# samples next to the parallel boundary; cmin keeps the color scale on the
# physical range.
torus = data.map_to_torus_3d(x_idx=(4, -4),
                             phi_tor=(0.0, 1.5 * np.pi),
                             nphi=160)
torus.pyvista(surface=True,
              no_normalize=True,
              is_log=True,
              cmin=5e16,
              no_show=True,
              no_spin=True,
              hide_axes=True,
              theme="document",
              camera_elevation=15.0,
              camera_azimuth=0.0,
              title="Electron density on a toroidal wedge",
              clabel=r"$n_e$ [m$^{-3}$]",
              saveas=str(OUTPUT_DIR / "14_map_to_torus.png"))

# Reuse geometry and projection when several fields or frames share one
# computational grid. The first cap is the poloidal projection at phi = 0.
geometry = pg.resolve_geometry(data.file_name)
projection = pg.resolve_torus_projection(data,
                                         geometry,
                                         x_idx=(4, -4),
                                         phi_tor=(0.0, 1.5 * np.pi),
                                         nphi=160)
reused = data.map_to_torus_3d(projection=projection)
np.testing.assert_allclose(reused.values, torus.values)
poloidal = data.map_to_torus_rz(phi_tor=0.0).values[4:93, :, 0]
np.testing.assert_allclose(reused.values[:89, :-1, 0, 0], poloidal)

print("14_map_to_torus: OK")
