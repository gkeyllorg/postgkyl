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


Drift direction selectors
-------------------------

``ExB_vel``, ``gradB_vel`` and ``diamag_vel`` accept:

* ``direction="0"``, ``"1"``, ``"2"``: contravariant coordinate rates,
  retaining previous numerical behavior. Python also accepts integers.
* ``direction="n0"``, ``"n1"``, ``"n2"``: velocity along a unit normal
  toward increasing computational coordinate, in m/s. ``n0`` measures
  transport across flux surfaces in a flux-coordinate simulation.
* ``direction="R"``, ``"Z"``, ``"phi"``: cylindrical components in m/s.
  Use ``R`` and ``Z`` to plot physical drift arrows on an R-Z plane.

Physical projections support 2x and 3x data. In 2x, the grid contains
the radial and parallel coordinates, while geometry retains all three
vectors. Selectors keep the full geometry indexing: ``0``/``n0`` refer
to the radial coordinate, ``1``/``n1`` to the binormal coordinate, and
``2``/``n2`` to the parallel coordinate. Thus ``n1`` is not the normal
of the second stored grid dimension. The omitted binormal derivative
is zero, but the binormal drift can be nonzero and is included when
reconstructing the physical velocity.

The loader reads Cartesian
coordinate tangents from ``geo_int_dxdz`` and coordinate unit normals from
``geo_int_normals``. Cylindrical projections use ``geo_int_mapc2p`` positions:
Cartesian MAPC2P positions supply X/R and Y/R directly; TOKAMAK positions
supply the stored toroidal angle. No new geometry output is required.
Missing required geometry and cylindrical R/phi projections at R=0 produce
errors. Each packed vector contains its X, Y, Z components consecutively.

The reconstructed velocity is ``v = sum_i v^i e_i``. Normal components are
``v dot n_i = v^i/sqrt(g^{ii})`` (no sum); coordinate normals need not be
orthogonal. Unit vectors are evaluated at quadrature points and projected;
products use the retained DG basis. Transport-flux diagnostics remain
contravariant.

Numerical selectors do not universally have units of m/s. For TOKAMAK
geometry ``0`` has units Wb/(rad s), and ``1`` and ``2`` have units rad/s.
Arbitrary MAPC2P coordinate units cannot be inferred and are labeled
``coordinate units/s``. Use a physical selector for an unambiguous m/s result.

For example::

   pgkyl gk_load_q -n sim -s ion -f 0 -q gradB_vel -d n0 map_to_rz plot

The legacy read-option spelling ``-r dir=n0`` also works. Other diagnostics
continue to accept their original numerical direction selectors.

.. literalinclude:: ../../examples/scripts/20_drift_projections.py
   :language: python
