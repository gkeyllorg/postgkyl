"""Quickstart: the golden script path.

    pg.load(...).interpolate().select(...).plot()

This walks through the same chain the CLI uses (see
``examples/cli_tutorial.md``), one step at a time:

1. ``pg.load`` reads a ``.gkyl`` file. Raw DG data lands in the *modal*
   backend -- coefficients, not values you can plot or feed to NumPy.
2. ``.interpolate()`` is the one-way bridge to a uniform mesh of plain
   values (the *numpy* backend).
3. ``.select()`` picks out one component/coordinate.
4. ``.plot()`` renders it (a terminal verb -- returns the Matplotlib
   ``Figure``, which this script then saves to disk).
5. ``.save()`` writes a dataset back out; reloading it recovers the same
   values, byte for byte.

Run directly:
    MPLBACKEND=Agg PYTHONPATH=src python examples/scripts/01_quickstart.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; drop this line to see the plot windows

import postgkyl as pg

from _example_paths import TEST_DATA, prepare_output_dir

DATA = TEST_DATA / "generated" / "2d_c2p_rot45_ms_p1.gkyl"
OUTPUT_DIR = prepare_output_dir()

# 1. Load -- raw DG coefficients, native to Gkeyll. This 2D file carries a
#    2-component vector field (a rotated coordinate map).
d = pg.load(DATA)
print("loaded:", repr(d))

# 2. Interpolate -- the modal -> NumPy bridge.
field = d.interpolate()
print("interpolated:", repr(field))

# 3a. Select by component -- pick out the x-component as its own 2D scalar
#     field, and plot it.
comp0 = field.select(comp=0)
fig = comp0.plot(title="x-component", show=False)
png_path = OUTPUT_DIR / "01_quickstart_comp0.png"
fig.savefig(png_path)
print("saved plot:", png_path)

# 3b. Select by coordinate -- fix y=0.5 to take a lineout across x, keeping
#     both components. The y axis survives as a size-1 slice (``select``
#     narrows a coordinate, it doesn't drop the axis -- ``plot`` squeezes it
#     away when rendering).
lineout = field.select(z1=0.5)
fig = lineout.plot(title="lineout at y=0.5", show=False)
lineout_path = OUTPUT_DIR / "01_quickstart_lineout.png"
fig.savefig(lineout_path)
print("saved plot:", lineout_path)

# 4. Save / reload round trip -- writing an interpolated field to ``.gkyl``
#    and reading it back recovers the same values exactly.
gkyl_path = comp0.save(str(OUTPUT_DIR / "01_quickstart_out.gkyl"))
reloaded = pg.load(gkyl_path)
print("round-tripped through", gkyl_path)

print("01_quickstart: OK")
