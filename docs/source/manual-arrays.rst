Access grids and values directly
==================================

Postgkyl does not require you to express every calculation as a pipeline.
After loading a dataset, ``data.grid`` exposes its coordinate arrays and
``data.values`` exposes the stored numbers. What those numbers mean depends
on the representation: a raw modal file holds coefficients, while an
interpolated dataset holds field values. ``np.asarray(data)`` also exposes
point values; it deliberately refuses modal coefficients.

Use ``.copy()`` when you want independent writable arrays. Never assume that
editing a view will leave its dataset unchanged. The example below copies
both the grid and values, performs a NumPy calculation, and uses ``clone``
and ``push`` to create a plotted result while retaining the field metadata.

An axis with one more coordinate than its value-array dimension contains
cell edges; use adjacent-edge midpoints to associate one coordinate with each
cell value. If the coordinates already align one-to-one with the values,
use them directly. ``np.meshgrid(..., indexing="ij")`` preserves the data's
axis order. The final array dimension indexes components, not space.

This example converts density to a normalized perturbation. Its CLI equivalent
uses ``evaluate`` for the same arithmetic. The rendered pixels are checked
against the manually manipulated Python arrays.

Download the :download:`example bundle <downloads/postgkyl-examples.zip>` first.

.. include:: _pairs/10_manual_arrays.inc
