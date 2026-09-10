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

Coefficients and point values
-------------------------------

``value_form`` distinguishes three representations:

* ``modal`` stores DG expansion coefficients. Multiplication and division
  use weak DG operations; coefficients are not samples for NumPy ufuncs.
* ``nodal`` stores values at basis nodes.
* ``quad`` stores values at Gauss–Legendre quadrature points.

Nodal and quadrature values support pointwise arithmetic, including native
data. The native/NumPy backend and the value representation are separate
facts. ``to_modal()``, ``to_nodal()``, and ``to_quad()`` explicitly change
representation. ``interpolate()`` creates a new NumPy-backed field on a
refined mesh for selection, plotting, and general analysis.

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
