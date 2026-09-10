Measure exponential growth
==========================

All scripts below run against the :download:`example bundle
<downloads/postgkyl-examples.zip>`. Unzip it, activate the environment where
Postgkyl is installed, and run the commands from the extracted directory.
The Python and CLI outputs below are both generated and compared when this website is built.

The generator supplies an analytic energy history with a known exponential
rate. Fit the logarithm of positive energy to time; its slope is the energy
growth rate. If energy is proportional to squared mode amplitude, the
amplitude growth rate is half this slope. This exact example checks the
calculation; real data require selecting a justified growth interval and
examining residuals before saturation dominates. The script checks the known rate and residuals numerically, then plots the reconstructed fitted energy.

.. include:: _pairs/06_growth.inc
