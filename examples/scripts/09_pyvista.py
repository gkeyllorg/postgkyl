"""Render Gaussian isosurfaces and a translucent volume with PyVista."""

import postgkyl as pg
from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()
data = pg.load(TEST_DATA / "generated/gaussian_volume.gkyl").interpolate()
data.pyvista(contour_levels=5,
             no_show=True,
             no_spin=True,
             hide_axes=True,
             title="Gaussian density isosurfaces",
             xlabel="x",
             ylabel="y",
             zlabel="z",
             clabel="Density",
             theme="document",
             camera_elevation=10.0,
             camera_azimuth=20.0,
             saveas=str(OUTPUT_DIR / "09_isosurfaces.png"))
data.pyvista(volume=True,
             opacity="linear",
             no_show=True,
             no_spin=True,
             hide_axes=True,
             title="Gaussian density volume",
             xlabel="x",
             ylabel="y",
             zlabel="z",
             clabel="Density",
             theme="document",
             camera_elevation=10.0,
             camera_azimuth=20.0,
             saveas=str(OUTPUT_DIR / "09_volume.png"))
