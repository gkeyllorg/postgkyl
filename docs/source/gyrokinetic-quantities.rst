Load derived gyrokinetic quantities
===================================

Download and unzip the :download:`example bundle
<downloads/postgkyl-examples.zip>`, then run the Python script or ``pgkyl``
commands below from the extracted directory. In a source checkout, first run
``python tests/generate_test_data.py`` to create the small synthetic inputs.

``pg.gk.load_quantity`` and its CLI command ``gk_load_quantity`` resolve the
source files and compute a named quantity. List every supported quantity and
its brief description without loading simulation files::

   pgkyl gk_load_q -a
   pgkyl gk_load_q --avail_quant

The catalog is alphabetical, with quantity names on the left and descriptions
on the right. ``gk_load_q -h`` shows concise option help; select a quantity with
``-q, --quantity``. The same options work with ``gk_load_quantity``.

This example loads three electron quantities at frame 0:

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


Vector names and component conventions
---------------------------------------

Names ending in ``_contra`` return contravariant components :math:`V^i`
for numerical direction selectors; names ending in ``_cov`` return covariant
components :math:`V_i`. These components use the coordinate basis, which need
not be orthogonal or normalized. The following are clean renames: update
scripts and commands to the new names; the old names are not aliases.

.. list-table:: Renamed quantities
   :header-rows: 1
   :widths: 35 40 25

   * - Previous name
     - New name
     - Representation
   * - ``ExB_vel``
     - ``ExB_vel_contra``
     - Contravariant
   * - ``gradB_vel``
     - ``gradB_vel_contra``
     - Contravariant
   * - ``diamag_vel``
     - ``diamag_vel_contra``
     - Contravariant
   * - ``flux_particle_ExB``
     - ``flux_particle_ExB_contra``
     - Contravariant
   * - ``flux_energy_ExB``
     - ``flux_energy_ExB_contra``
     - Contravariant
   * - ``flux_particle_dB``
     - ``flux_particle_dB_contra``
     - Contravariant
   * - ``flux_energy_dB``
     - ``flux_energy_dB_contra``
     - Contravariant
   * - ``flux_particle``
     - ``flux_particle_contra``
     - Contravariant
   * - ``flux_energy``
     - ``flux_energy_contra``
     - Contravariant
   * - ``inv_L_n``
     - ``inv_L_n_cov``
     - Covariant
   * - ``inv_L_T``
     - ``inv_L_T_cov``
     - Covariant

The naming audit follows the formulas, rather than the presence of a
``dir`` argument alone:

* Drifts compute :math:`v^i=\mathbf v\cdot\nabla x^i`. Particle and energy
  fluxes multiply contravariant drift or magnetic-flutter components by
  scalar moments, so their components are contravariant too.
* Inverse gradient lengths compute :math:`-\partial_i X/X`, a covariant
  logarithmic gradient. Their units are inverse coordinate units, not
  universally inverse metres.
* ``E_field_cov`` computes :math:`-\partial_i\phi`, and ``E_field_contra``
  raises the index with :math:`g^{ij}`. ``dB_perp_contra`` computes the
  contravariant curl of :math:`A_\parallel\mathbf b`; ``dB_perp_cov`` lowers
  the index with :math:`g_{ij}`. ``B_tot_cov`` and ``B_tot_contra`` add the
  equilibrium field in the corresponding representation. These names
  already specify the basis and remain unchanged.
* ``geo_int_b_i`` retains its name: the subscript already denotes covariant
  components. It still returns all three packed components; ``dir`` does
  not select a component. Geometry tensors, including ``geo_int_g_ij``
  and ``geo_int_gij``, and actual Gkeyll output filenames are unchanged.
* ``particle_D``, ``heat_chi``, ``particle_D_gB`` and ``heat_chi_gB`` are
  directional transport coefficients, not vector components. Their names
  remain unchanged, as do scalar parallel velocities and magnitudes.

Fluxes and transport coefficients retain their existing 3x, cross-field
``dir=0,1`` restriction and fluctuation conventions. Inverse gradient lengths
retain the numerical selectors for coordinates present in the data. These
quantities do not yet accept physical projection selectors.

Physical direction selectors
-----------------------------

``ExB_vel_contra``, ``gradB_vel_contra``, ``diamag_vel_contra``,
``E_field_cov``, ``E_field_contra``, ``dB_perp_cov``, ``dB_perp_contra``,
``B_tot_cov`` and ``B_tot_contra`` accept:

* ``direction="0"``, ``"1"``, ``"2"``: coordinate components in the basis
  named by the suffix, retaining previous numerical behavior. Python also
  accepts integers.
* ``direction="n0"``, ``"n1"``, ``"n2"``: physical component along the unit
  normal toward increasing computational coordinate. ``n0`` measures
  transport across flux surfaces in a flux-coordinate simulation.
* ``direction="R"``, ``"Z"``, ``"phi"``: physical cylindrical components.
  Use ``R`` and ``Z`` to plot vector arrows on an R-Z plane.

Physical projections have units m/s for drifts, V/m for the electrostatic
field, and T for magnetic fields. The suffix describes the representation
used for numerical selectors; a physical selector returns the requested
projection irrespective of that suffix. For example, ``E_field_cov`` and
``E_field_contra`` with ``direction="R"`` describe the same physical component.
With spatially varying DG geometry, separate weak basis transformations can
introduce projection/truncation differences between the two calculations.

Physical projections support 2x and 3x data. In 2x, the grid contains
the radial and parallel coordinates, while geometry retains all three
vectors. Selectors keep the full geometry indexing: ``0``/``n0`` refer
to the radial coordinate, ``1``/``n1`` to the binormal coordinate, and
``2``/``n2`` to the parallel coordinate. Thus ``n1`` is not the normal
of the second stored grid dimension. The omitted binormal derivative
is zero, but the binormal vector component can be nonzero and is included
when reconstructing the physical vector.

The loader reads Cartesian coordinate tangents from ``geo_int_dxdz`` for
contravariant components, reciprocal vectors from ``geo_int_dzdx`` for
covariant components, and coordinate unit normals from ``geo_int_normals``.
Cylindrical R/phi projections use ``geo_int_mapc2p`` positions:
Cartesian MAPC2P positions supply X/R and Y/R directly; TOKAMAK positions
supply the stored toroidal angle. No new geometry output is required, and
Z projections do not require position data. Missing required geometry and
cylindrical R/phi projections at R=0 produce errors. Each packed basis vector
contains its Cartesian X, Y, Z components consecutively.

The reconstructed vector is
:math:`\mathbf V=\sum_i V^i\mathbf e_i=\sum_i V_i\nabla x^i`.
Normal components are
:math:`\mathbf V\cdot\hat{\mathbf n}_i=V^i/\sqrt{g^{ii}}` (no sum);
coordinate normals need not be orthogonal. For modal data, cylindrical unit
vectors are evaluated at quadrature points and projected back; contractions
use DG weak products. No silent interpolation occurs.

Numerical selectors have coordinate-dependent units. For TOKAMAK drift
components, ``0`` has units Wb/(rad s), and ``1`` and ``2`` have units rad/s.
Arbitrary MAPC2P coordinate units cannot be inferred and are labeled
``coordinate units/s``. Numeric electric/magnetic components are labeled
with their covariant/contravariant component-unit convention. Use a physical
selector for unambiguous m/s, V/m, or T units.

For example::

   pgkyl gk_load_q -n sim -s ion -f 0 -q gradB_vel_contra -d n0 map_to_rz plot
   pgkyl gk_load_q -n sim -s ion -f 0 -q E_field_cov -d R map_to_rz plot

The read-option spelling ``-r dir=n0`` also works. ``B_tot_*`` retains its
existing equilibrium-only fallback when no ``apar`` output is available.

Reusable Python projector
-------------------------

``pg.operations.project_vector`` projects any three scalar component datasets,
with explicit ``basis="contravariant"``, ``"covariant"`` or ``"cartesian"``.
Supply ``tangents`` for contravariant reconstruction or ``duals`` for covariant
reconstruction, plus ``normals`` for n0/n1/n2. For R/phi supply ``positions``
and ``position_system="cartesian"`` for (X,Y,Z) or ``"cylindrical"`` for
(R,Z,phi). All geometry uses the packing described above.

The operation accepts native modal, nodal or quadrature data, or interpolated
point data. All inputs must have compatible grids, backends and representations;
convert them explicitly before mixing layouts. It returns a scalar dataset in
the same representation and does not infer physical units or discover files.
It is a Python operation, not a separate CLI command. GK diagnostics own file
loading and labels and call this shared operation after assembling the vector.

The runnable example covers the loader and the general Python operation:

.. literalinclude:: ../../examples/scripts/20_drift_projections.py
   :language: python
