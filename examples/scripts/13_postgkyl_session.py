"""Build a ``pgkyl`` command line from Python with ``PostgkylSession``.

A ``PostgkylSession`` is the ``pgkyl`` command line driven from Python. Each
command is a method taking that command's options -- an IDE shows them, with
the command's docstring, as you type, and ``help(s.map_to_rz)`` lists them --
and the session records the equivalent command line. Develop a figure in a
script or notebook, then print the one command that redraws it, e.g. in a
batch job.

This script maps the electron density fluctuation of a 3x2v gyrokinetic
turbulence simulation onto the poloidal (R-Z) plane. The fluctuation is
taken about the average over the binormal direction y,

    delta n_e = n_e - <n_e>_y,

so a diverging color map centered on zero separates over- from
under-densities.

Run directly:
    MPLBACKEND=Agg python examples/scripts/13_postgkyl_session.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; drop this line to see the plot window

from matplotlib.figure import Figure
import numpy as np

import postgkyl as pg
from postgkyl.cli import PostgkylSession

from _example_paths import TEST_DATA, prepare_output_dir

# The electron density M0 at frame 5; its R-Z geometry is inferred from the
# simulation prefix (rt_gk_tcv_nt_iwl_3x2v_p1-geo_int_mapc2p.gkyl).
DATA = TEST_DATA / "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl"
OUTPUT_DIR = prepare_output_dir()

# 1. One method per command, in pipeline order. Each returns the session, so
#    calls chain (as here) or stand alone, one per line. Directions are
#    0-based: (x, y, z) = (0, 1, 2).
s = PostgkylSession()
s.load(DATA).fluctuation(dims=1)

# 2. The working set is what the next command applies to; inspect it with the
#    fluent API at any step. Averaging the fluctuation over y gives zero, to
#    roundoff.
(fluctuation, ) = s.datasets
y_average = fluctuation.average(1)
np.testing.assert_allclose(y_average.values,
                           0.0,
                           atol=1e-12 * np.abs(fluctuation.values).max())

# 3. Map onto the R-Z plane at toroidal angle 0, up-sampling the parallel
#    direction by 12 points per cell. The result matches the fluent API's.
s.map_to_rz(nz_interp=12)
(mapped, ) = s.datasets
expected = pg.load(DATA).fluctuation(dims=1).map_to_rz(nz_interp=12)
np.testing.assert_array_equal(mapped.values, expected.values)

# 4. Render, labeling the subplot rather than the figure: empty xlabel and
#    ylabel drop the figure-wide coordinate labels. What the last command
#    returned, here the figure, is s.result.
s.plot(xlabel="",
       ylabel="",
       subplot_xlabels="R [m]",
       subplot_ylabels="Z [m]",
       clabel=r"$\delta n_e$ [m$^{-3}$]",
       no_legend=True,
       diverging=True,
       fixaspect=True,
       saveas=OUTPUT_DIR / "13_postgkyl_session.png")
assert isinstance(s.result, Figure)

# 5. Print the command line that draws the same figure.
s.print_cli()

print("13_postgkyl_session: OK")
