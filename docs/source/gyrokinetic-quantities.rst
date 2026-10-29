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

Derived vector quantity names have no ``_cov`` or ``_contra`` suffix.
Each vector has one entry;
``gk_load_q -a`` reports its default in the description column. The ``rep``
option overrides that default. Covariant components :math:`V_i` and
contravariant components :math:`V^i` use a coordinate basis, which need not
be orthogonal or normalized.

.. list-table:: Default vector representations
   :header-rows: 1
   :widths: 70 30

   * - Quantities
     - Default
   * - ``ExB_vel``, ``gradB_vel``, ``diamag_vel``
     - Contravariant
   * - ``flux_particle_ExB``, ``flux_energy_ExB``, ``flux_particle_dB``,
       ``flux_energy_dB``, ``flux_particle``, ``flux_energy``
     - Contravariant
   * - ``dB_perp``
     - Contravariant
   * - ``E_field``, ``B_tot``, ``inv_L_n``, ``inv_L_T``, ``geo_int_b_i``
     - Covariant

The default representations follow the formulas used to compute each
quantity, rather than the presence of a ``dir`` argument alone:

* Drifts compute :math:`v^i=\mathbf v\cdot\nabla x^i`. Particle and energy
  fluxes multiply contravariant drift or magnetic-flutter components by
  scalar moments, so their components are contravariant too.
* Inverse gradient lengths compute :math:`-\partial_i X/X`, a covariant
  logarithmic gradient. Their units are inverse coordinate units, not
  universally inverse metres.
* ``E_field`` computes :math:`-\partial_i\phi`; ``rep=contra`` raises
  the index with :math:`g^{ij}`. ``dB_perp`` computes the contravariant curl
  of :math:`A_\parallel\mathbf b`; ``rep=covar`` lowers the index with
  :math:`g_{ij}`. ``B_tot`` adds the equilibrium field and defaults to
  covariant components. All three support the same representation options.
* ``geo_int_b_i`` retains its name: the subscript already denotes covariant
  components. Without direction or representation options it returns all three
  packed components. Supplying a direction selects one component; requesting
  a representation requires a direction. Geometry tensors, including ``geo_int_g_ij``
  and ``geo_int_gij``, and actual Gkeyll output filenames are unchanged.
* ``particle_D``, ``heat_chi``, ``particle_D_gB`` and ``heat_chi_gB`` are
  directional transport coefficients, not vector components. Their names
  remain unchanged, as do scalar parallel velocities and magnitudes.

Representation and direction
----------------------------

Select a representation with ``-r, --rep`` (Python: ``rep=...``), independently
of the numeric direction ``-d, --direction`` (or ``-o dir=...``):

.. list-table:: Components for direction 0, 1, 2
   :header-rows: 1

   * - rep
     - Components
   * - ``covar``
     - Covariant computational components
   * - ``contra``
     - Contravariant computational components
   * - ``cyl``
     - Cylindrical R, phi, Z components.
   * - ``cart``
     - Cartesian X, Y, Z components.
   * - ``norm``
     - Projections onto unit normals.

Omitting ``rep`` retains each quantity's native representation. The catalog
``pgkyl gk_load_q -a`` reports it at the end of the description, for example
``ExB drift velocity. (default: contravariant)``. The metric-tensor entries
also identify their stored representation, but tensor conversions are not
supported by this vector option. Scalars and directional transport coefficients
reject ``rep``.

Representation changes are enabled for all registered vectors. Labels and result
metadata identify the requested output representation.

For example::

   pgkyl gk_load_q -n sim -s ion -f 0 -q gradB_vel -r cyl -d 0 map_to_rz plot
   pgkyl gk_load_q -n sim -s ion -f 0 -q E_field -r contra -d 1
   pgkyl gk_load_q -n sim -s ion -f 0 -q flux_particle -o rep=norm -o dir=0

Conversions support 2x and 3x data. In 2x, the grid contains radial and parallel
coordinates while vector components retain all three geometry indices:
0=radial, 1=binormal, 2=parallel for computational components and normals.
Thus normal 1 is not the normal of the second stored grid dimension.
Missing binormal derivatives contribute zero, but transformed binormal
components can be nonzero. Existing native 1x calculations remain available.

ExB and magnetic-flutter particle and energy fluxes support 2x and 3x data.
They provide all three vector components; equilibrium parallel streaming is
excluded, including in component 2. With ``fluct=none`` (the default), each
contribution is the appropriate moment times its drift or flutter component.
With ``fluct=y`` or ``fluct=yz``, both factors have their Jacobian-weighted
mean subtracted before multiplication. In 3x, the average is along y or
(y,z), respectively. In 2x, **both choices average along stored dimension 1
(z)**, keeping radial dimension 0. This is a correlation of deviations from
the parallel mean, not a binormal fluctuation.

For a 2x field :math:`f(x,z)`, the mean is
:math:`\langle f\rangle_z=\int J f\,dz/\int J\,dz`. The fluctuation
contribution to a particle flux is therefore
:math:`(n-\langle n\rangle_z)(v^i-\langle v^i\rangle_z)` for ExB, with
the corresponding moment and flutter component used for the other fluxes.

Directional diffusivities retain their restriction to directions 0 and 1.
Fluctuations are computed in the native basis before the final representation
change. Averaging and a spatially varying basis transformation need not commute.
``B_tot`` retains its equilibrium-only fallback when ``apar`` is absent.

Geometry and units
------------------

All these representations describe the same vector. Covariant and
contravariant components use coordinate basis vectors; Cartesian and
cylindrical components, and unit-normal projections, use unit vectors.
The distinction is the basis and its normalization.

Raising and lowering indices uses the saved metric tensors:

.. math::

   V^i = \sum_j g^{ij} V_j, \qquad V_i = \sum_j g_{ij} V^j.

``geo_int_gij`` supplies the contravariant metric and ``geo_int_g_ij`` the
covariant metric. Each packs six independent components as 00,01,02,11,12,22.
The vector is reconstructed in Cartesian components as
:math:`\mathbf V=\sum_i V^i\mathbf e_i=\sum_i V_i\nabla x^i`, using
``geo_int_dxdz`` for tangents or ``geo_int_dzdx`` for reciprocal vectors.
``geo_int_normals`` supplies coordinate unit normals; they need not be
mutually orthogonal. Every packed basis vector contains Cartesian X,Y,Z
components consecutively.

Cylindrical R/phi projections use ``geo_int_mapc2p`` positions. Cartesian
MAPC2P positions supply X/R and Y/R; TOKAMAK positions supply the stored
angle. Cartesian components and cylindrical Z need no positions. Only
geometry required by the requested conversion is loaded. Missing geometry,
incompatible layouts, and cylindrical R/phi at R=0 produce errors.

Cartesian, cylindrical, and unit-normal projections have units m/s for
drifts, V/m for electric fields,
T for magnetic fields, inverse metres for logarithmic gradients, inverse
square metres per second for particle fluxes, W/m^2 for energy fluxes, and
are dimensionless for the magnetic-field unit vector. Computational components
have coordinate-dependent units. For contravariant TOKAMAK drift components,
0 has units Wb/(rad s), and 1 and 2 have units rad/s.

For modal data, contractions use DG weak products and cylindrical unit vectors
are evaluated at quadrature points before projection back to the retained
basis. No silent interpolation occurs. With variable geometry, separate weak
basis transformations can introduce truncation differences: a conversion and
its inverse are not guaranteed to recover every original coefficient exactly.

Reusable Python operation
-------------------------

``pg.operations.project_vector`` accepts three scalar component datasets,
``basis="contravariant"|"covariant"|"cartesian"`` describing the input,
and ``rep`` and numeric ``direction`` selecting the output. Supply
``metric_cov`` or ``metric_contra`` to lower or raise indices. For Cartesian
reconstruction supply ``tangents`` or ``duals``; normal projections additionally
need ``normals``, and cylindrical R/phi need ``positions`` with explicit
``position_system="cartesian"`` for (X,Y,Z) or ``"cylindrical"`` for (R,Z,phi).

All inputs must have compatible grids, backends and representations. Native
modal, nodal and quadrature data, and interpolated point data, retain their
arithmetic semantics and representation. The operation performs no file
discovery or unit inference; GK diagnostics provide those. It is a Python
operation, not a separate CLI command.

.. literalinclude:: ../../examples/scripts/20_drift_projections.py
   :language: python
