---
name: postgkyl-architecture
description: Place modules and enforce import boundaries when adding or refactoring Postgkyl operations, diagnostics, rendering, or public surfaces.
---

# Preserve layer ownership

`tests/test_postgkyl.py::_ALLOWED` and its AST checks are the authority for
allowed imports and acyclicity. Consult that mapping for exact edges; when
adding a top-level module or allowed edge, update the contract and this guidance
in the same change if ownership changes. Do not maintain a second import matrix.

| Layer | Owns |
| --- | --- |
| `__init__.py`, `cli/`, `gui/` | Public surfaces; the facade only re-exports, with no function/class definitions |
| `diagnostics/` | Equation-specific physical conclusions and compositions |
| `gdata/` | Fluent GData/GDataGroup API and load entry point |
| `operations/` | Data transformations; flat equation-blind verbs |
| `render/` | Canonical plotting and animation callables |
| `gdatastate/` | State, clone, `_result`, shared guards and point-value materialization; no verbs |
| `dg/`, `io/` | Kernel orchestration and file I/O respectively |
| `numerics/`, `cli_spec.py` | Pure math and frozen command metadata; no internal imports |
| `gpython/` | Sole foreign-library boundary |

Imports point downward along allowed edges, never back to a higher surface.
Operations accept GDataState; `_result` constructs `type(self)` so fluent results
remain GData without importing gdata. Reuse `gdatastate/guards.py` and
`materialize.py` instead of duplicating capability checks or native bridges.
Readers return `(grid, values)` and fill plain metadata; they never import state.

Diagnostics are free functions under model families `gk`, `vm`, `pkpm`, or `mom`,
not GData methods. Model-specific loading belongs beside its physics. Resolve
output stems/frames through `diagnostics/discovery.py`; keep quantity vocabulary
in the equation module's `VARIABLES` table. Use public functions from lower
layers and return a state via `_result`, or a Figure for program diagnostics.
for the current major version.

Run the import, foreign-floor, facade, and canonical-callable contracts in
`tests/test_postgkyl.py` after structural changes.
