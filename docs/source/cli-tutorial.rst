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
