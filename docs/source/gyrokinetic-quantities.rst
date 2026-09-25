Load derived gyrokinetic quantities
===================================

Download and unzip the :download:`example bundle
<downloads/postgkyl-examples.zip>`, then run the Python script or ``pgkyl``
commands below from the extracted directory. In a source checkout, first run
``python tests/generate_test_data.py`` to create the small synthetic inputs.

``pg.gk.load_quantity`` and its CLI command ``gk_load_quantity`` resolve the
source files and compute a named quantity. This example loads three electron
quantities at frame 0:

.. math::

   v_{t,e} = \sqrt{T_e/m_e}, \qquad
   \phi_{\mathrm{norm}} = e\phi/T_e, \qquad
   \rho_e = \frac{\sqrt{m_e T_e}}{|q_e| B}.

Temperature is in joules, potential in volts, magnetic field in tesla, mass
in kilograms, and charge in coulombs. Thermal speed is in m/s, normalized
potential is dimensionless, and Larmor radius is in metres. Here ``vt`` uses
the convention without a factor of two under the square root.

The generated simulation prefix is ``gk_quantity_1d_p1``. Its sources are:

* ``gk_quantity_1d_p1-elc_MaxwellianMoments_0.gkyl``: density, parallel flow,
  and temperature divided by mass; metadata supplies electron mass and charge.
* ``gk_quantity_1d_p1-field_0.gkyl``: electrostatic potential.
* ``gk_quantity_1d_p1-geo_int_bmag.gkyl``: frame-independent magnetic field.

Supply ``species="elc"`` (CLI: ``--species elc``) for all three requests:
even ``phi_norm`` needs a species to choose the temperature. The ``e`` in
its definition is the positive elementary charge. When source metadata lacks
mass or charge, pass ``mass=...``/``charge=...`` in Python or
``--mass ...``/``--charge ...`` on the CLI, using the simulation's units.

The synthetic fields are constant within each cell of a p1 modal basis.
The script checks the results against their analytic values. Calculations
preserve modal DG representation; ``interpolate`` supplies point values after
the calculation. Python returns a list of datasets, including when only one
species and frame are requested. The CLI adds those datasets to the command
chain for subsequent operations and plotting.

The website build executes both interfaces and compares all three plots.

.. include:: _pairs/12_gk_load_quantity.inc
