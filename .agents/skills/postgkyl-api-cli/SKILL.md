---
name: postgkyl-api-cli
description: Add or change public Postgkyl verbs, signatures, fluent aliases, or generated CLI behavior while preserving API/CLI parity.
---

# Keep one executable public API

Core operations use `op(data: GDataState, *, ..., inplace=False, tag=None,
label=None) -> GDataState`; funnel mutation/new-result decisions through `_result`.

- Alias exact fluent verbs in the GData class body to canonical operations; no
  wrappers, runtime setattr, or lazy imports. GDataGroup broadcasts member verbs.
  Multi-dataset verbs share the aliases in `gdata/verbs.py`.
- Rendering lives in `render/`. Facade, operations, fluent methods, and generated
  CLI refer to the same canonical callables.
- Every public boolean defaults to `False`. For enabled default behavior, name
  its inverse (`no_show`, `volume`, `nodal`) and implement it under the false
  branch. CLI bare boolean options mean True; explicit True/False remain accepted.
- Expose basis/order/value_form overrides only at load time, never downstream.

The CLI discovers the public API, constructs CommandSpec records, and lowers them
through one generic compiler. Keep scientific command knowledge in API signatures,
annotations, docstrings, and command metadata. Do not add handwritten subcommands
or a `cli/commands/` package.

Preserve underscores exactly (`local_poly`, `--num_moms`). Assign a short option
to the first parameter with each initial in signature order; reserve `-h` for
help. Aliases/abbreviations change spelling only. Bare filenames expand to
`load --file_name`; they do not define different loading semantics.

Use native Click chaining and callback-before-dispatch. Help groups the flat
inventory into Verbs, Diagnostics, Render, and Utility. The console entry point
is `postgkyl.cli.app:cli`. Version reporting is owned by `_version.py`, exported
through the facade, and receives the version string explicitly to avoid an import
ordering dependency; preserve its build/dependency diagnostics.

Verify signature changes with `tests/test_cli_generator.py`,
`tests/test_cli_commands.py`, and relevant diagnostic CLI tests; check canonical
callable identities in `tests/test_postgkyl.py` when changing aliases.
