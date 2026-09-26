Inspect a field and extract a lineout
=======================================

The first question after a simulation finishes is often: what is stored in
this file, and how does one component vary along a particular cut?

Install Postgkyl using :doc:`installation`. Download and unzip the
:download:`example bundle <downloads/postgkyl-examples.zip>` into a working
directory. It contains the scripts and data with the same paths as the repository.
From that directory, run:

.. code-block:: bash

   python examples/scripts/01_quickstart.py

The script loads an analytic two-component coordinate map, interpolates its
DG coefficients, and plots the first component. This small synthetic field
makes it easy to see which direction and component a selection acts on.

``comp=0`` selects a component; ``z1=0.5`` extracts a spatial cut.
The script also saves and reloads the selected dataset and checks its values.
The paired figures below are compared pixel by pixel during the build.

.. include:: _pairs/01_quickstart.inc
