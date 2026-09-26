"""Interactive height surfaces from 2D data and isosurfaces from a 3D field."""

import postgkyl as pg
from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()
surface = pg.load(TEST_DATA / "generated/wave_surface_000.gkyl").interpolate()
surface.plotly(background="light",
               title="Travelling-wave height surface",
               xlabel="x",
               ylabel="y",
               zlabel="Density [normalized]",
               clabel="Density",
               saveas=str(OUTPUT_DIR / "08_surface.html"))

volume = pg.load(TEST_DATA / "generated/gaussian_volume.gkyl").interpolate()
volume.plotly(background="light",
              title="Gaussian density isosurfaces",
              xlabel="x",
              ylabel="y",
              zlabel="z",
              clabel="Density",
              surface_count=8,
              opacity=0.15,
              cmin=0.1,
              cmax=0.9,
              saveas=str(OUTPUT_DIR / "08_volume.html"))
