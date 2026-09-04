# `pgkyl` CLI tutorial

Every command below is real: it runs against fixture files under
`tests/test_data/` -- most committed directly, a few under
`tests/test_data/generated/` and synthesized by `tests/generate_test_data.py`
(run once, or on the first `pytest` invocation, which does it automatically)
-- and `tests/test_examples.py` replays each one (via `click.testing.CliRunner`,
from the repository root) as a regression check. If the CLI's surface ever
changes in a way that breaks one of these commands, that test fails -- this
file cannot silently drift out of date the way a hand-maintained tutorial can.

Run any line yourself from the repository root, after `pip install -e .[test]`
and (for the `tests/test_data/generated/` fixtures) `python
tests/generate_test_data.py`.

## 1. Inspect a file

`info` is the "what is this?" command -- dimensions, components, grid,
value range, and the DG basis/order it was written with.

```bash
pgkyl tests/test_data/rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl info
```

## 2. The chain: interpolate -> select -> plot

Raw `.gkyl` files hold DG *coefficients*; `interpolate` bridges them onto a
uniform mesh of plain values, `select` narrows down to one component (or one
coordinate slice), and `plot` renders it. The CLI options are the Python
parameters with dashes: `--show False` runs headless and `--saveas` writes a
PNG.

```bash
pgkyl tests/test_data/rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl \
    interpolate select --comp 0 plot --show False --saveas out.png
```

Any unambiguous command-name prefix is accepted as a spelling-only alias
(`interp` -> `interpolate`, `sel` -> `select`):

```bash
pgkyl tests/test_data/rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl \
    interp sel --comp 0 info
```

## 3. Discontinuity-preserving plots with `local-poly`

`interpolate` produces a continuous refined mesh; `local-poly` instead
evaluates the DG polynomial cell-by-cell and splices a NaN at every
inter-cell interface, so a plot shows genuine discontinuities instead of
smoothing over them -- useful for shocks or anything with jumps at cell
boundaries.

```bash
pgkyl tests/test_data/generated/distf_p2_0.gkyl \
    local-poly select --z1 0.0 --z2 0.0 plot --show False --saveas out.png
```

## 4. DynVector inspection and `fit`

`.gkyl` files without a spatial grid (diagnostics like a field-energy history)
are DynVectors: `info` summarizes the data, and `fit` fits a model to it (here,
a straight line to the series vs. time -- the growth-rate use case).

```bash
pgkyl tests/test_data/generated/energy_dynvec.gkyl info
pgkyl tests/test_data/generated/energy_dynvec.gkyl fit --fit-type linear
```

## 5. Combining datasets: `evaluate`

`evaluate` runs a Reverse Polish Notation expression over every dataset
currently loaded: `fN` refers to the `N`-th one in load order (`f` alone
means `f0`), so `"f0 f1 -"` subtracts the second dataset from the first --
e.g. two frames of the same field, to see how it changed between them.
(`fN[c]` selects component `c` of dataset `N`; don't confuse that with
indexing datasets themselves -- there is no `f[N]` form.) Data must be
`interpolate`d first, same as `select`/`plot`.

```bash
pgkyl tests/test_data/generated/distf_p2_0.gkyl tests/test_data/generated/distf_p2_1.gkyl \
    interpolate evaluate "f0 f1 -" info
```

## 6. Gyrokinetics: pre-named quantities and distribution functions

`gk-load-quantity` loads one of a registry of named gyrokinetic
quantities (listed in its generated `--quantity` choices) straight from a simulation's naming convention --
no manual file paths. `--name` is the simulation's *name prefix* (not a path);
`--path` is the directory to look in.

```bash
pgkyl gk-load-quantity --help
pgkyl gk-load-quantity --quantity geo_int_jacobtot_inv --species "" \
    --name rt_gk_tcv_iwl_1x2v_p1 --path tests/test_data info
```

`gk-load-distf` reconstructs a full distribution function from the saved
`Jf`-times-Jacobian(s) files (here `--name` *does* include the directory, since
the simulation name itself includes the directory):

```bash
pgkyl gk-load-distf --name tests/test_data/rt_gk_tcv_iwl_1x2v_p1 \
    --species elc --frame 250 \
    --jacobtot-inv-file tests/test_data/rt_gk_tcv_iwl_1x2v_p1-geo_int_jacobtot_inv.gkyl \
    info
```

## 7. Map a gyrokinetic field to R-Z

`gk-rz` is a data transformation: it interpolates one raw DG component and
maps it onto the physical poloidal plane. Geometry is inferred from the
field's filename, preferring nodal geometry and falling back to modal
`mapc2p` geometry. The CLI and Python calls below use the same operation and
defaults:

```bash
pgkyl tests/test_data/rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl \
    gk-rz --nz-interp 2 info
```

```python
import postgkyl as pg

mapped = pg.load(
    "tests/test_data/rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl"
).gk_rz(nz_interp=2)
```

## 8. Saving to another format

`save` writes the current dataset(s) out as `gkyl`/`txt`/`npy`/`vtk`.

```bash
pgkyl tests/test_data/generated/distf_p2_0.gkyl save --out-name distf --extension npy
```

## 9. One API-derived command inventory

Every loaded file becomes a dataset in the current chain. The command list is
compiled from the script API, so `--help` is the authoritative inventory and
every Python underscore appears as a CLI dash.

```bash
pgkyl --help
```

## See also

- `pgkyl --help` lists every registered command, grouped by section
  (Verbs / Diagnostics / Render / Utility).
- `pgkyl <command> --help` documents that command's options -- most carry a
  worked example in their docstring, e.g. `pgkyl local-poly --help`.
- `examples/scripts/` is the Python-script equivalent of this tutorial (the
  fluent `GData` API instead of the chained CLI).
