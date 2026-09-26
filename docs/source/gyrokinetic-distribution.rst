Inspect a gyrokinetic distribution
==================================

All scripts below run against the :download:`example bundle
<downloads/postgkyl-examples.zip>`. Unzip it, activate the environment where
Postgkyl is installed, and run the commands from the extracted directory.
The Python and CLI outputs below are both generated and compared when this website is built.

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
