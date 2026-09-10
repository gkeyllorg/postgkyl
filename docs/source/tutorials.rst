Scientific workflows
======================

All scripts below run against the :download:`example bundle
<downloads/postgkyl-examples.zip>`. Unzip it, activate the environment where
Postgkyl is installed, and run the commands from the extracted directory.
The Python and CLI outputs below are both generated and compared when this website is built.

Compare profiles across calculations
--------------------------------------

Compare density, parallel velocity, and two temperatures while retaining
visibility into low-density tails. These are analytic synthetic profiles
generated for plotting tests, not evidence of simulation convergence.
The logarithmic side masks nonpositive values, including negative velocity.

.. include:: _pairs/mirror_comparison.inc

Recover pressure from conserved moments
-----------------------------------------

A fluid file stores density, momentum components, and total energy.
``five_moment.pressure`` subtracts kinetic energy and applies the specified
ratio of specific heats. The generated fixture is a stationary shock-tube
initial condition in dimensionless units, with gamma = 5/3. It does not show
the evolved shock, contact, or rarefaction.

.. include:: _pairs/03_diagnostics_five_moment.inc

Inspect a gyrokinetic distribution
------------------------------------

This example uses committed TCV output: ion Hamiltonian moments from
``rt_gk_tcv_iwl_adapt_source_1x2v_p1`` and an electron distribution plus geometry
from ``rt_gk_tcv_iwl_1x2v_p1``. These are separate runs; do not combine their
moments into one conservation calculation. Frame 250 identifies saved output,
not a time in seconds.

``load_quantity`` resolves named quantities from available files.
``load_distf`` reconstructs the distribution using its matching Jacobians.
Coordinates remain computational in this example; the optional velocity
mapping is not enabled. Inspect the returned grid before assigning units or
interpreting a velocity-space slice.

.. include:: _pairs/04_gyrokinetics.inc

Plot in physical R–Z coordinates
----------------------------------

Computational coordinates are useful for analysis, but a poloidal view places
the field in its physical geometry. Keep the companion
``rt_gk_tcv_nt_iwl_3x2v_p1-geo_int_mapc2p.gkyl`` beside the electron density
file so the operation can resolve it by the simulation prefix.

.. include:: _pairs/05_gk_rz.inc

Measure exponential growth
----------------------------

The generator supplies an analytic energy history with a known exponential
rate. Fit the logarithm of positive energy to time; its slope is the energy
growth rate. If energy is proportional to squared mode amplitude, the
amplitude growth rate is half this slope. This exact example checks the
calculation; real data require selecting a justified growth interval and
examining residuals before saturation dominates. The script checks the known rate and residuals numerically, then plots the reconstructed fitted energy.

.. include:: _pairs/06_growth.inc

Choose the arithmetic representation
--------------------------------------

The script compares weak DG algebra with pointwise operations after
interpolation. A projected product does not in general preserve all terms
of the full polynomial product. A nonzero divisor alone does not guarantee
an exact multiply/divide round trip.

.. include:: _pairs/02_arithmetic_and_numpy.inc
