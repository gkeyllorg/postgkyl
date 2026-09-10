Recover pressure from conserved moments
=======================================

All scripts below run against the :download:`example bundle
<downloads/postgkyl-examples.zip>`. Unzip it, activate the environment where
Postgkyl is installed, and run the commands from the extracted directory.
The Python and CLI outputs below are both generated and compared when this website is built.

A fluid file stores density, momentum components, and total energy.
``five_moment.pressure`` subtracts kinetic energy and applies the specified
ratio of specific heats. The generated fixture is a stationary shock-tube
initial condition in dimensionless units, with gamma = 5/3. It does not show
the evolved shock, contact, or rarefaction.

.. include:: _pairs/03_diagnostics_five_moment.inc
