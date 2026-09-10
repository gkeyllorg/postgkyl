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
