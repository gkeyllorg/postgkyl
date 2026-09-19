Plot a multiblock dataset
=========================

A multiblock simulation writes one file per block. Load all blocks of the
same quantity and frame, interpolate each block, then pass them together to
``pg.plot(*blocks)``. Each block keeps its own grid; the resulting figure
uses one shared color scale and one colorbar.

Postgkyl recognizes the Gkeyll naming convention
``<sim>_b<N>-<quantity>_<frame>.gkyl`` automatically. For example,
``mb_sim_b*-elc_M0_0.gkyl`` selects every block of ``elc_M0`` at frame zero.
Keep the frame fixed when plotting a snapshot. Quote the wildcard in the
CLI so Postgkyl receives the complete pattern.

This example uses generated, synthetic p1 modal data on three adjacent
rectangular blocks spanning ``0 <= x <= 3`` and ``0 <= y <= 1``.
Coordinates and field values have arbitrary units. Interpolation converts
the modal coefficients into field samples before rendering.

Download and unzip the :download:`example bundle
<downloads/postgkyl-examples.zip>`, then run from the extracted directory.
In a repository checkout, first run ``python tests/generate_test_data.py``.
The Python and CLI figures below are generated and compared during the
documentation build.

.. include:: _pairs/11_multiblock.inc
