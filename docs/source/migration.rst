Migrating existing workflows
============================

The refactored API resolves the basis and stored representation when loading,
then composes operations on that dataset. The following inventory describes
current support; an unsupported legacy feature is not an implicit conversion.

Compatibility inventory
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Previous workflow
     - Current status
     - Supported replacement or limit
   * - Native Gkeyll ``.gkyl`` files
     - Supported, including partial loading and dynvectors.
     - Supply missing or incorrect basis metadata once to ``pg.load``.
   * - Legacy Gkeyll HDF5 and FLASH HDF5
     - Readers remain available.
     - Header/layout detection chooses the reader; inspect ``info --all``.
   * - ADIOS ``.bp`` reading/writing
     - Removed from this implementation; no ADIOS reader or writer is registered.
     - Use a compatible legacy tool to export evaluated arrays and coordinates,
       if that tool supports the file, or regenerate supported Gkeyll output.
       This release does not convert ADIOS files.
   * - ``maximal-order`` / ``mo`` interpolation
     - Not implemented by the current native basis bridge.
     - Evaluate with a compatible legacy implementation or regenerate output.
       Changing the basis name alone does not convert coefficients.
   * - ``gkhybrid_vel`` / ``gkhyb_vel``
     - Not implemented by the current native basis bridge.
     - Preserve the original basis meaning; do not relabel it ``gkhybrid``.
   * - Historical basis abbreviations
     - Load uses full basis names and a separate ``value_form``.
     - ``ms`` → ``serendipity`` + ``modal``; ``ns`` → ``serendipity`` +
       ``nodal``; ``mt`` → ``tensor`` + ``modal``; ``gkhyb`` → ``gkhybrid``;
       ``pkpmhyb`` → ``hybrid``. Unsupported ``mo`` and ``gkhyb_vel`` stay
       unsupported. Aliases must not be used to change the actual file basis.
   * - Per-command ``interpolate -p/-b`` and load flags
     - Basis/order/representation belong to ``load``.
     - Use ``load --poly_order ... --basis_type ... --value_form ...``.
       Partial-load options are ``--z0`` through ``--z5`` and ``--component``;
       downstream selection uses ``select --comp``. Reader options use
       ``--read_options key=value``.
   * - Separate interpolator objects and tuple results
     - ``interpolate()`` and ``local_poly()`` return new datasets.
     - Access ``result.grid`` and ``result.values``. Repeated interpolation of
       already evaluated values raises instead of reinterpreting components.
   * - ``to_quad()`` / ``to_modal()`` in early refactor examples
     - Replaced by the single representation operation.
     - Use ``represent(to="quad")`` / ``represent(to="modal")``;
       CLI: ``represent --to quad`` / ``represent --to modal``.
   * - CLI aliases and abbreviated commands
     - Explicit aliases are ``pl`` → ``plot`` and ``ev`` → ``evaluate``.
     - Unique command prefixes are accepted; ambiguous prefixes raise.
       Use canonical command names in maintained scripts.
   * - Generic CLI ``--use TAG``
     - Not provided by the generated executor; transforms use the working set.
     - Use separate CLI invocations or filter a Python ``GDataGroup`` by tag.
   * - ``write --filename ... --mode ... --single``
     - Replaced by ``save --out_name ... --extension ...``.
     - Formats: ``gkyl``, ``txt``, ``npy``, ``vtk``. ADIOS, append and combined
       ``--single`` output are unavailable. Use explicit distinct paths when
       saving several datasets; ``var_name`` currently has no effect.

Loading legacy metadata
-----------------------

Move the old command-local basis override to the load boundary:

.. code-block:: bash

   # Previous command vocabulary:
   pgkyl legacy.gkyl interpolate --basis_type ms --poly_order 1 plot

   # Current command vocabulary:
   pgkyl load --file_name legacy.gkyl --basis_type serendipity \
     --poly_order 1 --value_form modal interpolate plot --no_show

The Python equivalent is:

.. code-block:: python

   import postgkyl as pg

   data = pg.load("legacy.gkyl", basis_type="serendipity",
                  poly_order=1, value_form="modal")
   evaluated = data.interpolate()
   grid, values = evaluated.grid, evaluated.values
   quadrature = data.represent(to="quad")
   recovered = quadrature.represent(to="modal")

These options assert the known meaning of existing coefficients. They do not
repair unsupported basis layouts or translate an unrelated file format. Hybrid
phase-space files must resolve their configuration/velocity split; explicit
Python metadata can be supplied as ``ctx={"num_cdim": 1, "num_vdim": 2}`` for
a known 1x2v Vlasov hybrid file. Missing or ambiguous information is not a basis
conversion. Interpolation and representation transforms preserve this split;
native weak/integration kernels that cannot express it raise explicitly. Basis
support for transforms does not imply support for every native operation.

Selecting tagged data
---------------------

Tags label datasets; they do not narrow the generated CLI working set. Filter
explicitly in Python when different species or representations need different
operations:

.. code-block:: python

   group = pg.GDataGroup()
   group.load("sim-elc_*.gkyl", tag="electron")
   group.load("sim-ion_*.gkyl", tag="ion")
   electrons = pg.GDataGroup(d for d in group if d.get_tag() == "electron")
   electron_values = electrons.interpolate()

Return values and saving
------------------------

``pg.load(literal_path)`` returns a ``GData``; a glob always returns a
``GDataGroup``. Most transformations return a dataset, with ``inplace=True``
requesting mutation. Full ``integrate()`` returns a scalar for one physical
field or a NumPy array for several; partial modal integration returns a
lower-dimensional modal dataset. ``plot()`` returns a figure. ``save()`` returns
the written path. Group transformations return groups, and terminal operations
usually return lists of per-member results.

The Gkeyll field header can reconstruct only uniform Cartesian cell edges.
Saving nonuniform, point-coordinate, or mapped geometry as ``.gkyl`` raises
before opening the destination. Old mapped files that omitted their coordinates
cannot be repaired from metadata alone; reload the original data and mapping.
Text export retains the physical coordinates of supported point/edge layouts.
``.npy`` exports values only and is not a complete dataset round trip. The
registered loaders do not provide a general ``.txt`` or ``.npy`` dataset reader.

.. code-block:: python

   path = data.save("preserved.gkyl")        # supported uniform DG geometry
   total = data.integrate()                   # terminal scalar/array
   path = evaluated.save("samples", extension="txt")

Modal multiplication and division remain weak DG operations. Pointwise nonlinear
work uses ``represent(to="quad")`` or ``apply`` as described in :doc:`concepts`.
Plotting a physical field likewise requires explicit evaluation; raw modal
plots show labeled coefficient channels.
