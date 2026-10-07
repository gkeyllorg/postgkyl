Command-line workflows
========================

Every figure in :doc:`examples` has both a complete Python script
and a copyable ``pgkyl`` command. Their generated outputs appear together,
with the result of the build's equivalence check. See
:doc:`interface-equivalence` for the complete comparison report.

The CLI keeps a working set of datasets: loading adds files, transformations
operate on them, ``collect`` combines frames, and ``plot`` or ``animate``
renders the result. A quoted filename wildcard loads multiple matching files.
The Python spelling is a list of loaded datasets passed to the corresponding
function. The animation tutorial demonstrates both forms.

Command and option names preserve Python underscores. A boolean flag such as
``--no_show`` means ``True``; explicit ``--no_show True`` and ``--no_show False``
are also accepted. ``--saveas`` chooses the output path. The gallery's CLI
commands use ``output/``; the scripts use ``examples/scripts/output/`` by
default, or the directory specified by ``PGKYL_EXAMPLE_OUTPUT``.

Use :doc:`reference/cli` to inspect every command and
:doc:`reference/api` for the corresponding Python calls. The example bundle
also contains the longer, executable CLI walkthrough in
``examples/cli_tutorial.md``.

List-valued options
-------------------

Options corresponding to Python lists accept a quoted JSON array. For example,
Python's ``legend_labels=["old", "new"]`` becomes
``--legend_labels '["old","new"]'``. The outer single quotes protect the
array from the shell; strings inside the array use JSON double quotes.
Spaces and commas inside those strings are preserved.

You may also repeat an option once per item, or combine arrays and individual
items; entries retain their command-line order. Bare comma-separated text is
one item, not an array. A literal string starting with ``[`` must itself be
an array entry, for example ``--legend_labels '["[reference]"]'``.
Array items use the option's declared type, so numeric list options accept
numeric arrays and reject invalid numbers. Fixed-length tuple options such as
``--figsize 12 10`` retain their existing syntax.

Building a command line from Python
-----------------------------------

``PostgkylSession`` runs the CLI from Python, one command per call, and
records the equivalent ``pgkyl`` command line. Run from the repository root
(or the extracted example bundle), this takes the electron density of a 3x2v
gyrokinetic simulation along the first coordinate and plots it:

.. code-block:: python

   from postgkyl.cli import PostgkylSession

   s = PostgkylSession()
   s.load("tests/test_data/rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")
   s.interpolate()
   s.select(z1=0.0, z2=0.0)
   s.plot(title="Electron density", saveas="density.png")
   s.print_cmd()

``s.print_cmd()`` prints the command line that draws the same figure:

.. code-block:: bash

   pgkyl tests/test_data/rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl interpolate select --z1 0.0 --z2 0.0 plot --title 'Electron density' --saveas density.png

Every command is a method with that command's options, so ``help(s.select)``
lists them. Each call parses and runs the very tokens it records, on the same
working set as the command line, so ``s.command()`` is exactly what ran. The
current datasets are ``s.datasets``; after the ``select`` above it holds one
dataset of 96 cells along the first coordinate. A dataset passed to an option
such as ``average --weight`` is named by its ``tag``. A value with no
command-line spelling, such as a NumPy array or a ``slice``, is refused with a
``TypeError``; use the string selector ``"1:3"`` instead. A failed call is not
recorded.
