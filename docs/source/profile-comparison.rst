Compare profiles across calculations
====================================

All scripts below run against the :download:`example bundle
<downloads/postgkyl-examples.zip>`. Unzip it, activate the environment where
Postgkyl is installed, and run the commands from the extracted directory.
The Python and CLI outputs below are both generated and compared when this website is built.

Compare density, parallel velocity, and two temperatures while retaining
visibility into low-density tails. These are analytic synthetic profiles
generated for plotting tests, not evidence of simulation convergence.
The logarithmic side masks nonpositive values, including negative velocity.

.. include:: _pairs/mirror_comparison.inc

Legend labels in the CLI and Python API
---------------------------------------

Pass a JSON array to ``--legend_labels``, in dataset input order. Quote the
entire array so the shell preserves the brackets and double quotes. For
example, compare a new result with an older result in another directory::

   pgkyl zzim-ion_BiMaxwellianMoments_65.gkyl \
     old-with-bug/zzim-ion_BiMaxwellianMoments_65.gkyl \
     interp pl -f0 --legend_labels '["new","old"]'

Spaces and commas within labels work naturally:
``--legend_labels '["new result","old, corrected"]'``.
Repeating the option also works: ``--legend_labels new --legend_labels old``.
Arrays and repeated options can be mixed; their entries are combined in order.
``--legend_labels "new,old"`` and ``--legend_labels "new","old"`` both
supply one literal label containing a comma. Unquoted
``--legend_labels ["new","old"]`` is unreliable: shells strip the double
quotes and may interpret the brackets as filename patterns. Use the outer
single quotes shown above. To supply a literal label beginning with ``[``,
put it inside an array, for example ``--legend_labels '["[reference]"]'``.

In Python, pass a list of strings:

.. code-block:: python

   import postgkyl as pg

   new = pg.load("zzim-ion_BiMaxwellianMoments_65.gkyl").interpolate()
   old = pg.load("old-with-bug/zzim-ion_BiMaxwellianMoments_65.gkyl").interpolate()
   pg.plot(new, old, figure=0, legend_labels=["new", "old"])

Each explicit label is reused verbatim on every component subplot, without a
component suffix. ``legend_subplot=0`` (CLI: ``--legend_subplot 0``) limits the
legend to the first subplot; ``no_legend=True`` (CLI: ``--no_legend``) hides it.

Without explicit legend labels, a figure with one dataset uses ``c0``, ``c1``,
etc. A figure with multiple datasets uses ``<filename>_c0``,
``<filename>_c1``, etc., including for single-component data. Here, filename
means the basename including its extension. Files with the same basename,
as in this example, need explicit labels to distinguish them. In-memory
datasets without a filename use their dataset label, or ``dataset 0``,
``dataset 1``, etc. if that is empty. If fewer legend labels than datasets are
provided, the remaining datasets use these defaults. Subplot titles and axis
labels do not change legend entries.
