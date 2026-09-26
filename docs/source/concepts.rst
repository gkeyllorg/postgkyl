Working with data
===================

Coordinates, components, and metadata
---------------------------------------

``pg.load(path)`` returns a ``GData`` containing coordinates, values, and
metadata in ``ctx``. Inspect the file before interpreting component numbers:
a five-moment fluid state and a gyrokinetic distribution do not use the same
layout. Diagnostics know the equation-specific interpretation.

Basis type, polynomial order, and ``value_form`` are properties of the data.
They are read from the file or supplied once to ``load`` when missing.
Downstream operations use that metadata. Do not guess a higher-order basis
for a file whose header lacks this information.

``pgkyl file.gkyl info`` shows the current data summary and any assumptions made
by the loader. Use ``info --all`` to also inspect the original file header,
verbatim metadata, explicit load options, and identity inferred from the filename.
The Python equivalents are ``data.info(all=True)`` and ``pg.info(data, all=True)``.

Coefficients and point values
-------------------------------

``value_form`` distinguishes three representations:

* ``modal`` stores DG expansion coefficients. Multiplication and division
  use weak DG operations; coefficients are not samples for NumPy ufuncs.
* ``nodal`` stores values at basis nodes.
* ``quad`` stores values at Gauss–Legendre quadrature points.

Nodal and quadrature values support pointwise arithmetic, including native
data. The native/NumPy backend and the value representation are separate
facts. ``represent(to=...)`` explicitly changes representation to ``'modal'``,
``'nodal'``, or ``'quad'`` in both the functional and fluent Python APIs:
``pg.represent(data, to='quad')`` or ``data.represent(to='quad')``.
The CLI uses the same verb and parameter: ``represent --to quad`` (or
``represent -t quad``). ``interpolate()`` creates a new NumPy-backed field on a
refined mesh for selection, plotting, and general analysis.

Plotting raw modal data draws the stored coefficients as separate channels
on the cell grid, without changing the representation. For example,
``pgkyl file.gkyl pl`` plots coefficients. To plot the evaluated field, use
``pgkyl file.gkyl interpolate pl`` or ``pgkyl file.gkyl local_poly pl``.
``represent(to='nodal')`` and ``represent(to='quad')`` also provide field values
at basis nodes and quadrature points respectively.

``represent(to='quad')`` (CLI: ``represent -t quad``) uses the quadrature count,
ordering, and modal/quadrature transforms supplied by the Gkeyll basis.
For example, a 1x1v hybrid p1 field has six quadrature values per cell
(two configuration points and three velocity points), and converting back
with ``represent(to='modal')`` preserves all six modal coefficients to roundoff.
Some basis/order combinations lack native quadrature kernels and raise
``NotImplementedError``. An explicit ``num_quad=N`` selects a custom uniform
Gauss rule with N points per direction; it must resolve the projection
integrand, including the quadratic velocity dependence of hybrid p1.

Saved native quadrature data records ``quad_rule="gkeyll"`` and the total
node count in ``num_quad``. Custom rules record ``quad_rule="gauss"`` and
the per-direction order; files predating this marker retain their original
custom-rule interpretation.

For nonlinear operations on a modal field, ``apply(fn, num_quad=...)`` spells
out evaluation at quadrature points followed by projection back to modal
coefficients. Mixing representations in arithmetic raises an error.

Integration and cuts
----------------------

Integrating a modal field uses the DG representation directly.
``average`` and partial ``integrate`` can reduce dimensionality while keeping
a modal dataset for further operations. ``eval_at_coord_proj`` evaluates at
specified coordinates and projects into the basis of the surviving directions.

After interpolation, ``select(comp=...)`` selects components and
``select(z0=..., z1=...)`` selects positions. A selected axis may retain a
singleton dimension. Rendering squeezes it as needed.
Use ``local_poly()`` when the plot must preserve jumps between DG cells.

Composition
-------------

Most transformations return a new dataset; ``inplace=True`` explicitly
requests mutation. ``plot`` returns a figure, and full integration returns
integrated values. A chain ends when it reaches a terminal result.

``GDataGroup`` broadcasts operations over its members. CLI pipelines similarly
maintain a working set of loaded datasets; use ``evaluate`` to combine them
with an RPN expression. Public boolean parameters default to ``False``:
``--no_show`` means ``True`` and disables the plot window. Explicit
``--no_show True`` and ``--no_show False`` are also accepted.

See :doc:`reference/api` for exact arguments and :doc:`reference/cli` for
the generated command spellings.
