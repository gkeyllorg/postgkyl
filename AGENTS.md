# Coding Doctrine

**0. Locality of reasoning.** Every principle below is a projection of
one axiom: a reader must be able to understand a fragment without the
whole program. Whatever keeps a local conclusion sound — a frozen
record, an honest signature, a stated law — is doctrine. Whatever
forces a global search — ambient state, a leaky layer, a second copy
of a fact — is the enemy.

*Data — what it does, and what it may say*

**I. Data is inert. Functions transform.** No objects that know
things and do things. Data is a frozen record. Behavior is a function
that takes data in and returns data out. If you're reaching for
inheritance, you've taken a wrong turn.

**II. Make illegal states unrepresentable.** The shape of a datum is
its strongest invariant. Constructors refuse invalid states; a checked
fact becomes a type; downstream never re-proves what upstream
established. Parse, don't validate.

*Functions — one idea, honestly declared*

**III. A function is one idea.** It takes exactly what it needs and
returns exactly what it computes. If the signature has two concepts in
it, you have two functions.

**IV. The signature tells the whole truth.** Inward: if something
needs a value, it receives it as a parameter — no spooky action at a
distance, no stringly-typed interfaces, no implicit state. Outward:
same inputs, same outputs; effects and failure appear in the type, not
in the fine print. Pure core, effects at the edges.

*Knowledge — one home per fact*

**V. Every fact has one home.** One authoritative representation of
each decision and each piece of knowledge; everything else inherits or
is derived mechanically — never maintained by hand in parallel.
Configuration is decided once, at the highest level, and threaded
down; no module ever decides its own context. If the design and the
implementation can disagree, you have two sources of truth and zero.

*Layers — what above, how below*

**VI. Separate what from how.** Logic and machinery are different
concerns with a hard boundary. The layer that says *what* to compute
should be readable by someone who has never seen the machinery
underneath. The layer that says *how* lives below, stays below, and
nothing leaks up from it.

**VII. Notation is execution; lowering is transliteration.** Looking
up: the spec layer reads like the math or logic it implements — when
notation *is* the executable object, not a comment beside it, bugs
have nowhere to hide. Looking down: the layer that executes the spec
reproduces it exactly — nothing added, nothing dropped, nothing
reinterpreted; no opinions, no defaults, no helpful conversions. If
the lowering changes anything, the spec is a lie.

*Abstraction — earned, and binding*

**VIII. Earn your abstractions.** No abstraction before the second
use. Three similar lines is better than a premature helper. The right
amount of complexity is the minimum the current task demands — not the
current task plus three hypothetical future ones.

**IX. An abstraction is a contract.** It is defined by what it
guarantees, not what it hides. If you can't state what is always true
of it — properties a client may rely on without reading the
implementation — it isn't an abstraction, it's indirection. Two
implementations that honor the contract must be interchangeable; and
its outputs stay in its vocabulary, so uses compose.

*Verification — formal first*

**X. Trust the most formal thing first.** Types over tests, tests
over docs, docs over comments. Invest in whichever layer catches the
bug earliest with the least ongoing maintenance cost.

## Commands

```bash
# Install for development (editable) + test deps
pip install -e .[test]

# Run the tests
pytest tests/
# Without an install, point Python at the src layout:
PYTHONPATH=src python -m pytest tests/

# Run the CLI (chained pipeline; mirrors the fluent script API)
pgkyl file.gkyl interpolate select --z0 0 plot
pgkyl file.gkyl info

# A diagnostics chain (equation-specific physics; see diagnostics/)
# (diagnostics take NumPy-backed data, so interpolate always runs first)
pgkyl euler_5m_0.gkyl interpolate five-moment-pressure --num-moms 5 plot

# An RPN chain over the working set (see operations/evaluate.py)
pgkyl a.gkyl b.gkyl evaluate --chain "f0 f1 +" interpolate plot

# `pgkyl --help` lists every registered command, grouped by section
# (Verbs / Diagnostics / Render / Utility).
pgkyl --help
pgkyl --version
```

## Architecture — a strict, one-way layered DAG

Every folder has **one job**, and imports point in **one direction only** (leaves at the
bottom). There is **no import cycle** — this is enforced by a
test (see "Import contract"). Arrow = "may import":

> **Keeping the picture honest:** the two diagrams below and the prose after them are a
> mirror of `tests/test_postgkyl.py::_ALLOWED` — that dict (and the AST walk that checks
> every real import against it) is the enforced source of truth; this file is only a
> readable projection of it. The two *can* drift (they already had:
> `operations/animate.py`, `operations/average.py`, `operations/eval_at_coord_proj.py`, `operations/local_poly.py`,
> `gdatastate/guards.py`, `gdata/gdatagroup.py`, and `gdata/verbs.py` existed in the tree before they
> were added here). Whenever you add a new top-level module file or a new allowed import
> edge, update `_ALLOWED` and this section in the same commit — don't let the picture
> outlive the code it describes.

```
src/postgkyl/
│
├─ command_spec/       frozen command records/decorators · no imports [LEAF]
│
├─ __init__.py          facade · `import postgkyl as pg`              [SURFACE]
│
├─ cli/                 generated Click surface + generic runtime         [SURFACE]
│   ├─ app.py
│   ├─ discovery.py
│   └─ compiler.py
│
├─ diagnostics/        equation-specific physics · one module        [COMPOSITION]
│                      per equation model
│
├─ gdata/                ★ THE FLUENT SURFACE  (sits ABOVE operations)  [FLUENT API]
│   ├─ gdata.py          class GData(GDataState) + .interpolate()/.plot()/
│   │                    .plotly()/.pyvista()
│   ├─ gdatagroup.py     fluent GDataGroup: broadcasts verbs over its members
│   ├─ verbs.py          module-level fluent verbs with no single `self`
│   │                    (collect/evaluate/relchange/animate) — one-line
│   │                    delegations to `operations`, shared by GData and GDataGroup
│   └─ load.py           pg.load(...) → returns a GData
│
├─ operations/         data transformations · the single seam          [VERBS]
│   ├─ interpolate.py    interpolate(d: GDataState) -> GDataState      (flat core
│   ├─ select.py                                                       verbs are
│   ├─ animate.py        terminal: sequence of datasets → render's animation engine
│   ├─ average.py        terminal-adjacent: weighted average over a dim subset,
│   │                    stays modal/gkyl-native (composes with further verbs)
│   ├─ eval_at_coord_proj.py  terminal-adjacent: eval at coords, project to the
│   │                    lower-dim basis for survivors, stays modal/gkyl-native
│   ├─ local_poly.py     modal coefficients → discontinuity-preserving plot mesh
│   └─ gyrokinetics/     domain geometry transformations: R-Z + flux surfaces
│
├─ render/             canonical matplotlib · plotly · pyvista callables   [BACKEND]
│
├─ gdatastate/         ★ THE CONTAINER  (state only, NO verbs)        [CONTAINER]
│   ├─ gdatastate.py          class GDataState: grid·values·ctx·_result·dunders
│   ├─ gdatastategroup.py     GDataStateGroup
│   ├─ guards.py         shared field-domain guard (backend=="gkyl" -> raise);
│                        one home for the ".interpolate() first" check reused
│                        across operations/diagnostics instead of retyped per verb
│   └─ materialize.py    shared native point-values → NumPy-shadow bridge
│
├─ numerics/           pure NumPy math · 0 internal imports           [LEAF]
├─ dg/                 interpolation bridge + modal ops → gpython     [ENGINE]
├─ io/                 readers (C-native first) + writer → gpython    [ENGINE]
└─ gpython/            ★ THE FOREIGN FLOOR · compiled shim            [FLOOR]
    ├─ csrc/             _gpythonmodule.c — CPython extension over gkyl_gpython.h
    │                    (the shim itself lives in gkeyll/core/zero/)
    ├─ _gpython.so       built extension (scripts/build_gpython.sh)
    ├─ _lib.py           loads _gpython · GPYTHON_API_VERSION handshake
    ├─ array.py          GkylArray — capsule owner of a gkyl_array
    ├─ basis.py          basis cache + interpolation matrices via the shim
    ├─ rio.py            file loading via gkyl_array_rio
    └─ kernels.py        weak mul/div/inv · lincomb · reduce · integrate
```

```
╔══════════════════════════════════════════════════════════════════════════════╗
║ SURFACES        __init__.py  (pg facade)        cli/  (argv → verbs)         ║
║                                                 gui/  (argv → graphics)      ║
╚════════════════════════════╦══════════════════════════════╦══════════════════╝
                             │ imports                      │ imports
                             ▼                              ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ COMPOSITION                                                                  ║
║   diagnostics/   ★ EQUATION-SPECIFIC PHYSICS · one module per equation model ║
║      five_moment.py  density(d), pressure(d, gas_gamma=…), mach(d) …         ║
║      ten_moment.py   p_par(d, field), agyro(species, field) …                ║
║      mhd.py · plasma.py · multispecies.py · rotations.py · kinetic.py        ║
║      pkpm.py         laguerre_compose(…) + load_pkpm(…)                      ║
║      gyrokinetics/   load_gk_distf · load_gk_quantity + quantity registry    ║
║                      (Tpar, beta, ExB_vel, …) · energy_balance → Figure …    ║
║      discovery.py    shared naming-convention stem/frame discovery           ║
╚════════════════════════════╦═════════════════════════════════════════════════╝
                             │ imports
                             ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ FLUENT API   ★ the fluent surface lives HERE, above operations               ║
║                                                                              ║
║   gdata/load.py    pg.load(path) ───────────────────► returns gdata.GData    ║
║   gdata/gdata.py   class GData(GDataState):                                  ║
║                   interpolate = operations.interpolate (static alias)        ║
║                   plot/plotly/pyvista = operations = render (one each)       ║
║   gdata/gdatagroup.py  GDataGroup(gdatastate.GDataStateGroup): broadcasts any verb ║
║                  over its members via __getattr__ — no verb body duplicated  ║
║   gdata/verbs.py module-level verbs with no single `self` (collect/evaluate/ ║
║                  relchange/animate) — direct aliases to operations,          ║
║                  shared by GData and GDataGroup so spellings can't drift     ║
╚════════════════════════════╦═══════════════════════════════════╦═════════════╝
                             │ imports                           │ extends
                             ▼                                   │ (subclass)
╔══════════════════════════════════════════════════════════════════════════════╗
║ VERBS · data transformations                                                 ║
║   operations/interpolate.py   def interpolate(d: GDataState) -> GDataState   ║
║   operations/select.py                                                       ║
║   operations/gyrokinetics/     domain-specific geometry transformations       ║
╚════════════════════════════╦═══════════════════════════════════╩═════════════╝
                             │ imports                            │
                             ▼                                    │
╔══════════════════════════════════════════════════════════════════════════════╗
║ BACKEND                                                                      ║
║   render/ (mpl · plotly · pyvista) — owns the canonical render callables;    ║
║   facade, GData, operations, and CLI views share their exact identities      ║
╚════════════════════════════╦════════════════════════════════════╦════════════╝
                             │ imports                            │
                             ▼                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ CONTAINER                                                                    ║
║   gdatastate/gdatastate.py   GDataState: grid · values · ctx ·_result·dunders║
║   gdatastate/gdatagroup.py   GDataStateGroup                                 ║
║   gdatastate/guards.py  shared field-domain guard: backend=="gkyl" -> raise  ║
║                   the ".interpolate() first" message — one home for a check  ║
║                   operations/diagnostics verbs used to retype independently  ║
║   gdatastate/materialize.py  native nodal/quad → NumPy point-value state     ║
╚════════════════════════════╦═════════════════════════════════════════════════╝
                             │ imports
                             ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ ENGINE / LEAVES                                                              ║
║   command_spec/ (frozen command metadata · imports nothing)                 ║
║   numerics/ (pure math · imports nothing)                                    ║
║   dg/ (interpolation bridge + modal ops)   io/ (readers · writer)            ║
╚═════════════╦══════════════════════════════════════╦═════════════════════════╝
              │ imports                              │ imports
              ▼                                      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ FOREIGN FLOOR   gpython/  — the compiled gpython bridge (GKEYLL_C_SHIM.md)   ║
║   GkylArray (capsule RAII) · basis cache · rio · kernels                     ║
║                     ▼ import _gpython  (extension over gkyl_gpython.h only)  ║
║   gkeyll/core/zero/{gkyl_gpython.h, gpython.c} — the shim, compiled by       ║
║   Gkeyll's own make core INTO:      ▼ linked -lg0core                        ║
║   libg0core.so  (gkeyll/ clone · built by scripts/build_gkeyll.sh)           ║
╚══════════════════════════════════════════════════════════════════════════════╝

```

### The two-domain lifecycle (REFACTOR_GKEYLL_FFI.md)

Every dataset lives in one of two backends, discriminated by
`GDataState.backend`:

- **`"gkyl"` (modal domain)** — DG coefficients as a native `gkyl_array`
  (`gpython.GkylArray`). Loading lands here. All math runs inside Gkeyll:
  `*`/`/` → weak kernels (`gkyl_dg_mul_op`/`div_op`), `+`/`-` → coefficient
  lin-combs (`gkyl_array_accumulate`), scalars → `scale`/mean-shift, integer
  powers → repeated weak multiply, `.integrate()` → `gkyl_array_integrate`.
  `values` is a read-only view; `np.asarray`/ufuncs/`select` refuse with
  ".interpolate() first".
- **`"numpy"` (field domain)** — post-`interpolate()` values as a plain ndarray;
  the unchanged NumPy stack (`select`, `plot`, ufuncs, arithmetic).

`interpolate()` is the **one-way bridge**: matrix from Gkeyll's basis functions,
applied per cell with NumPy `tensordot`, returning a *new, by-value* array.

Every dataset has one **`value_form`** (`ctx["value_form"]`): `modal`
coefficients, `nodal` values at the basis `node_list` points, or `quad`
values at Gauss–Legendre points. This is a single, backend-agnostic fact —
there is no separate `is_modal` flag duplicating it; a numpy-backed dataset
that was never native (e.g. a plain nodal-basis file read without the
Gkeyll library) carries the same three-valued `value_form` as a gkyl-native
one, and every consumer (`_require_operable`, `interpolate`, `average`, …)
reads that one key. **The capability boundary is modal vs point-values, not
gkyl vs NumPy:**

- **modal** — only Gkeyll's DG operations: weak `* /`, coefficient `+ -`/scalar
  kernels, `.integrate()`, `.interpolate()`. Ufuncs/`np.asarray`/`plot` refuse.
- **nodal / quad** — the values ARE the field at points, so *every* pointwise
  NumPy operation is exact and allowed (ufuncs, `* / **`, `np.asarray`) —
  computed on the views, wrapped back native, **staying in-value_form** —
  and they `plot()` directly at their true point locations (non-tensor node
  sets, e.g. serendipity p2 in 2-D, plot via `.to_quad()`).

Conversions are **never implicit** — only `.to_modal()/.to_nodal()/.to_quad()`
change `value_form` (nodal↔modal exact; quad round-trip exact for degree
≤ 2·num_quad−1); `.apply(fn, num_quad=…)` is the one-shot modal → quad → fn →
project-back spelling (≡ `fn(d.to_quad()).to_modal()`). Datasets combine only
within one value_form. See REFACTOR_GKEYLL_FFI.md §3b.

`basis_type`, `poly_order`, and `value_form` are properties **of the data
itself** — read from a file's header metadata, or set once via `pg.load(...,
basis_type=..., poly_order=..., value_form=...)` / the CLI's bare-filename
`load` (`-b`/`-p`/`-v`) — never re-specified by a downstream verb. `.interpolate()`,
`.local_poly()`, `.average()`, `.eval_at_coord_proj()`, `.integrate()`, … all
read `ctx["basis_type"]`/`ctx["poly_order"]`/`ctx["value_form"]` off the
dataset and raise a clear error if a required one is missing; none of them
take a `basis`/`poly_order` override argument. Loading a dataset that has a
spatial grid (`ctx["cells"]`) without `basis_type`/`poly_order`/`value_form`
resolvable (neither in the file header nor given explicitly) warns and
defaults to `basis_type="serendipity"`, `poly_order=0`, `value_form="nodal"`
— the trivial one-point-per-cell basis, since there is no modal structure to
assume otherwise. When `value_form` is defaulted this way, the grid is
re-expressed as cell centers (p0 nodal's one point per cell) instead of the
reader's raw cell-edge grid, so it lines up one-to-one with `values`. A
dynvector/diagnostic file has no spatial grid and thus no DG basis to speak
of, so it is exempt from this defaulting. This is distinct from the readers'
own narrower default (`gkyl_c_reader.py`/`gkyl_reader.py`: `basis_type`
resolved but no `value_form` tag in the file -> assume `"modal"`, silently,
no warning) — there the data is *known* to be real DG output (the header
has a basis), so the stored numbers are almost certainly modal coefficients
already, not point values; that assumption is safe enough not to warn about.
The `gdatastate.py`-level default above only fires when nothing about a
basis was found at all.

### `gdatastate/` — the container (`gdatastate/state.py`)
`GDataState` holds one dataset: a nodal `grid` (list of 1-D edge arrays), values in
one of the two backends (`gpython.GkylArray` or `np.ndarray`), and metadata in `ctx`.
It is **verb-less** and imports only downward (`io` to construct itself, `gpython` for
the backend type, and `dg` for shared point-value materialization). It owns:
- shape properties (`num_dims`/`num_comps`/`num_cells`/`bounds`), `grid`/`values`,
- `backend` (`"gkyl"`/`"numpy"`) and `native` (the raw `GkylArray` for the kernels),
- `push`, `clone` (backend-aware deep copy via `type(self)`), and **`_result(...)`** —
  the one "mutate-self vs. emit-new" decision point every verb funnels through,
- pure state readers only: `__array__` (refuses on gkyl-backed data),
  `__repr__`/`__str__`, `info`, `is_interpolated`.
`gdatastate/collection.py` has `flatten_datasets` (shared by the multi-dataset entry points).
`gdatastate/guards.py` centralizes the field-domain check (`backend == "gkyl"` → raise with
the standard ".interpolate() first" message) that several `operations`/`diagnostics` verbs
need but that isn't itself a verb, so it lives here rather than in `operations`.
`gdatastate/materialize.py` owns the shared conversion from native nodal/quad values to
a NumPy-backed state used by every terminal consumer.

### `api/` — the fluent surface (`api/gdata.py`, `api/load.py`)
`class GData(GDataState)` adds the **fluent verb methods** (`.interpolate()`, `.select()`,
`.plot()`, `.save()`, `.info` inherited) and the **computing operators**
(`+ - * / **`, reflected, `__neg__`/`__abs__`, `__array_ufunc__`). Because it lives
*above* `operations`, exact verbs are static class-body aliases to their canonical
operation — no wrapper, runtime `setattr`, or lazy import. `pg.load(...)`
returns a `GData`.

`gdata/gdatagroup.py` mirrors the same move one level up: `class GDataGroup(gdatastate.GDataStateGroup)`
adds broadcasting — any attribute not defined on the class is resolved by `__getattr__`,
looked up on every member, so a verb call broadcasts across the whole group without a
single verb body being duplicated. `gdata/verbs.py` holds the handful of verbs that
combine *several* datasets and so have no single `self` to hang off of a class —
`collect`, `evaluate`, `relchange`, `animate` — each a direct alias to the matching
`operations` function; `GData` and `GDataGroup` both call through these same
module-level functions for their own methods, so the functional and fluent spellings
of a multi-dataset verb can never drift apart.

**The trick that removes the cycle:** `operations` verbs are typed on `GDataState` but *return*
the caller's concrete class, because `_result` builds `type(self)()`. So `operations` never needs
to import `gdata`, yet the whole fluent chain stays `GData`. See `HIERARCHY_2.md`.

### `operations/` — the data-transformation library (the single seam)
Flat core verbs have one module each and domain transformations live in named
subpackages exposed from `operations/__init__.py`. Contract:
`op(data: GDataState, *, ..., inplace=False, tag=None, label=None) -> GDataState`.
Flat modules hold equation-blind core verbs. Domain subpackages may know the
geometry or representation of one equation system while still only
transforming/re-expressing data; `operations/gyrokinetics/` owns the R-Z and
flux-surface projections. Code that interprets components to derive a new
physical conclusion belongs in `diagnostics/`.
Implemented: `interpolate` (the bridge verb: gkyl-backed in, numpy-backed out),
`select` (field-domain only), `info`, `integrate`
(terminal; runs inside Gkeyll on modal data), `represent`/`apply` (the explicit
value_form verbs behind `.to_modal()/.to_nodal()/.to_quad()/.apply()`),
`arithmetic` (`binary` + `apply_ufunc`), which **dispatches on `backend`**: modal
operands → `dg.modal` kernel calls; numpy operands → the NumPy path; mixed
domains or mixed value_forms → error, plus the field-domain analysis verbs
(`fft`, `magsq`, `relchange`, `mask`, `collect`, `grid`, `val2coord`,
`extract_input`, `fit` (its `window=True` mode covers growth-rate-style
leading-window fits), `differentiate`, `evaluate`, `map`); and the modal-native
verbs `average` (weighted average over a dimension subset via Gkeyll's
`gkyl_array_average`) and `eval_at_coord_proj` (eval at physical coordinates,
projected onto the lower-dimensional basis for the surviving directions) —
both terminal-adjacent like `represent`: they emit a new, lower-dimensional
dataset that stays modal/gkyl-native, so it composes with further
`.to_nodal()`/`.interpolate()`/`.average()`/`.eval_at_coord_proj()` calls
rather than dropping to NumPy. `local_poly` bridges modal coefficients to a
discontinuity-preserving plotting mesh. `operations.plot`/`plotly`/`pyvista`
are direct aliases of their canonical `render` callables, while animation
remains a terminal operation over a sequence.
All terminal consumers share `gdatastate.materialize_point_values`, so the
point-value capability rule has one home. Verbs wrap the layers below; they
don't reimplement.

### `diagnostics/` — equation-specific physics (COMPOSITION, above `api`)
The layer that knows what the numbers *mean* — and the ONLY package in the
COMPOSITION tier. One module (or subpackage) per equation model:
`five_moment`, `ten_moment`, `mhd`, `plasma` (plasma parameters), `multispecies`
(`energetics`, `accumulate_current`), `rotations` (par/perp to B), `kinetic`
(frame transforms), `pkpm` (Laguerre reconstruction + `load_pkpm`),
`trajectory`, `enstrophy`, `ke_dke` (program-scale figures ported from the
old `apps/trajectory.py`/`tools/calc_*.py`), and `gyrokinetics/` (distf/
quantity loaders, the quantity registry — Tpar, beta, drift velocities — plus
its own program-scale analyses: `energy_balance`/`particle_balance`/`nodes`,
ported from the old `apps/gk_*.py`). R-Z mapping and theta-phi flux-surface
extraction are gyrokinetic operations; their old diagnostic module paths are
compatibility aliases for the current major version. Contract: a diagnostic takes loaded
data — one or several `GData` — plus physical scalars as keyword-only
options, and returns `GDataState` (via `_result`, same inplace/tag/label
contract as a verb) or a Figure; it is built entirely from the public
vocabulary below it (`operations`, `gdatastate`, `numerics`, `gdata`) and nothing below the
surfaces imports it. The `render` edge is pre-authorized for this layer (a
program diagnostic may want `render.plot()`'s generic panel layout), but as
of this writing every program module builds its own bespoke figure directly
with `matplotlib` instead.

**Each equation model owns its loading internally** — there is no `loaders/`
package. Entry points like `gyrokinetics.load_gk_quantity(...)` (naming-
convention load + registry dispatch, "physics-ready data by name") and
`pkpm.load_pkpm(...)` live beside the physics they feed, because a quantity's
ingredient files and its formula are one piece of equation knowledge. The
only shared piece is `diagnostics/discovery.py` — equation-blind
output-stem/frame discovery, the one home for Gkeyll's file-naming
convention; equation loaders and programs resolve files through it, never
with private globbing.

Functions have real names (`five_moment.pressure(d, gas_gamma=…)`), never
string dispatch; each equation module's `VARIABLES` table maps the CLI's
quantity-name vocabulary (`"density"`, `"pressure"`, …) to those functions —
the one home for that vocabulary. These are **free functions, not `GData`
methods**: the layer sits above the fluent surface. (This layer absorbed the
former `models/` package — array physics now lives as private helpers inside
the equation module that uses it.)

### Engine layers — `dg/`, `io/` (may import `gpython` only)
- **`dg/`** — Gkeyll-kernel orchestration. `dg/interpolate.py` is the one-way modal→NumPy
  bridge (matrix from `gpython.basis`, applied per cell with `tensordot`; nodal-basis
  files convert through the exact `nodal_to_modal` matrix first); `dg/modal.py`
  holds the operations that stay modal (weak algebra, `lincomb`, `shift_mean`,
  `power`, `integrate`); `dg/rep.py` holds the explicit value_form changes
  (modal·nodal·quad) and `apply_pointwise` — all on native arrays.
- **`io/`** — file I/O: `read()` dispatches over a reader registry. `GkylCReader`
  (first) reads field files entirely inside Gkeyll (`gkyl_grid_array_new_from_file`)
  and returns a native `GkylArray`; the pure-Python `GkylReader` is the fallback for
  no-library installs, partial loads, and dynvectors. `save()` supports
  `gkyl`/`txt`/`npy`/`vtk`. Readers fill a plain `ctx` dict and return
  `(grid, values)` — they never import `gdatastate`.

### Leaves — `numerics/` (imports nothing), `gpython/` (the foreign floor)
- **`numerics/`** — pure NumPy: `idx_parser` (selection strings) and `elementwise`
  (`grids_compatible`). No `GData`, ever.
- **`gpython/`** — **the only doorway to the foreign world** (a test enforces this),
  and it is a *compiled* one (GKEYLL_C_SHIM.md): the gpython shim
  (`gkeyll/core/zero/{gkyl_gpython.h, gpython.c}`) lives **in the gkeyll tree** and is
  compiled by Gkeyll's own `make core` *into* `libg0core.so` — it holds every
  struct access, the by-value `struct gkyl_basis` convention, and the basis
  function-pointer dispatch, all checked by the C compiler against the headers
  in the same tree (shim and library can never drift apart).
  `csrc/_gpythonmodule.c` wraps `gkyl_gpython.h` (opaque handles + scalars + buffers
  only) into the `_gpython` extension, built by `scripts/build_gpython.sh` against
  the cloned `gkeyll/`'s `libg0core.so` (linked + rpath-bound, not dlopened).
  `_lib.py` imports the extension and performs the `GPYTHON_API_VERSION`
  handshake. `array.py`'s
  `GkylArray` holds the owning capsule (its destructor releases the C array;
  zero-copy constructions pin their NumPy buffer in the capsule, and `view()`
  ties the ndarray's `base` chain to the capsule so views outlive their dataset
  — never hand out C memory without that pin). `basis.py` builds every matrix
  by evaluating Gkeyll's own basis through the shim: `eval_matrix(points)`,
  nodal↔modal, modal↔quad (+ Gauss rules). No struct layout, signature, or
  ctypes declaration exists in Python. `gpython.available()` is the single
  capability switch.

### `render/` — visualization backend (`render/matplotlib.py`)
`plot(*datasets, ...)` is the one canonical plotting function: 1-D lines / 2-D
pcolormesh, one panel per component, multi-dataset overlay/grouping, saving,
display, and every public plot option. The facade, fluent method, operations
namespace, and generated CLI all point to this same callable.

### `__init__.py` — the facade (pure re-export)
Gathers the public names from the layer that owns each: `load`/`GData` ← `gdata`,
`plot` ← `render`, `info` ← `operations`, `save` ← `io`, `load_gk_quantity`/
`load_gk_distf`/`available_gk_quantities` ← `diagnostics.gyrokinetics`. **It
contains no function or class definitions** (a test enforces this).

`_version.py` sits beside `__init__.py` (same `""` layer) and supplies
`version_report` — `pgkyl --version`'s debugging-statistics report, not a
computing verb, but re-exported through the facade like everything else here
(so `cli/app.py` stays a pure facade consumer). It reads `gpython.available()`/
`gpython.build_info()` (hence `""`'s extra allowed edge, `-> gpython`), git
(the postgkyl commit this checkout is at) and `importlib.metadata` (key
dependency versions). `version_report(version)` takes the version string as
a parameter rather than importing `postgkyl.__version__` itself, so
`__init__.py`'s own `from postgkyl._version import version_report` line has
no ordering dependency on the `__version__` assignment below it.

### `cli/` — the CLI (chained pipeline on pure Click)
`cli/app.py` defines `PgkylGroup(click.Group)` with `chain=True`; the chaining loop and
callback-before-dispatch are **native to Click**. At import time it discovers the
public script API, compiles its `CommandSpec` records, and lowers every record through
the one generic compiler. There is no `cli/commands/` package and no hand-authored
subcommand. Python underscores are mechanically rendered as CLI dashes in command and
option names (`local_poly` → `local-poly`, `num_moms` → `--num-moms`). The custom
group code handles spelling-only aliases/unambiguous abbreviations and expands a bare
filename to `load --file-name`; aliases never change a generated command's parameters
or execution. `format_commands` groups `pgkyl --help` under Verbs / Diagnostics /
Render / Utility; presentation does not change the flat, chainable inventory.
`--version` is a custom eager `click.option` (not `click.version_option`, since
the output is more than one string): its callback calls the facade's
`version_report(__version__)` (both imported via `from postgkyl import
__version__, version_report`, same as any other facade name) to print the
postgkyl commit,
the vendored Gkeyll commit/branch/build-date this build was linked against
(`gpython.build_info()`, generated by `scripts/build_gpython.sh` since
`gkeyll/` is a build-time-only clone — absent, it reports "not built"),
gpython-bridge availability, and interpreter/platform/dependency versions.
`--help` is wired through Click's own group help. The console entry point
object is `postgkyl.cli.app:cli`.

`load` (or its bare-filename shorthand) exposes the script parameters
`--basis-type`, `--poly-order`, and `--value-form` — `basis_type`/`poly_order`/
`value_form` are properties of the data fixed once here at load time, so no
other verb command (`interpolate`, `local-poly`, …) repeats them.
