"""Load numbered frames, assemble an x–t diagram, and animate 1D and 2D data."""

import numpy as np
import postgkyl as pg
from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()
paths = sorted((TEST_DATA / "generated").glob("travelling_wave_*.gkyl"))
frames = [pg.load(path).interpolate() for path in paths]
assert len(frames) == 16

# collect adds a leading time coordinate, taken from each file's metadata.
history = pg.collect(frames)
np.testing.assert_allclose(history.grid[0], [d.ctx["time"] for d in frames])
np.testing.assert_allclose(history.values, np.stack([d.values for d in frames]))
history.plot(title="Travelling wave: space–time diagram",
             xlabel="Time",
             ylabel="x",
             clabel="Density [normalized]",
             no_show=True,
             saveas=OUTPUT_DIR / "07_collect.png",
             dpi=100)

# animate takes the individual frames, not the dataset returned by collect.
# saveframes also keeps every PNG for analysis or a presentation.
pg.animate(frames,
           saveframes=str(OUTPUT_DIR / "07_wave_frame"),
           saveas=str(OUTPUT_DIR / "07_wave.gif"),
           fps=8,
           dpi=90,
           figsize=(6, 4),
           no_show=True)

surface_paths = sorted((TEST_DATA / "generated").glob("wave_surface_*.gkyl"))
surface_frames = [pg.load(path).interpolate() for path in surface_paths]
assert len(surface_frames) == 12
pg.animate(surface_frames,
           saveframes=str(OUTPUT_DIR / "07_surface_frame"),
           saveas=str(OUTPUT_DIR / "07_surface.gif"),
           fps=6,
           dpi=90,
           figsize=(6, 4),
           no_show=True)
