# Postgkyl examples

Two parallel tutorials over the same golden path -- `load -> interpolate ->
select -> plot` -- one per interface:

- **`scripts/`** -- the fluent Python API (`import postgkyl as pg`).
- **`cli_tutorial.md`** -- the equivalent `pgkyl` command-line chains.

Both are executable, and both are tested: `tests/test_examples.py` runs
every script and replays every command quoted in the markdown tutorial, so
an example that stops working (an API rename, a removed option) fails the
test suite instead of quietly rotting. Treat that file as the single source
of truth for "does the tutorial still work," not just this README.

## Scripts

| Script | What it covers |
| --- | --- |
| [`01_quickstart.py`](scripts/01_quickstart.py) | `pg.load` → `.interpolate()` → `.select()` → `.plot()` → `.save()`/reload |
| [`02_arithmetic_and_numpy.py`](scripts/02_arithmetic_and_numpy.py) | Weak DG algebra on raw modal data (`*`, `/`, `+`, `.integrate()`) vs. plain NumPy math after `.interpolate()`, and the guardrail between them |
| [`03_diagnostics_five_moment.py`](scripts/03_diagnostics_five_moment.py) | The `diagnostics` layer: equation-specific physics (`postgkyl.diagnostics.mom.five_moment`) on top of a `GData`, on a generated shock-tube initial state |
| [`04_gyrokinetics.py`](scripts/04_gyrokinetics.py) | The gyrokinetic diagnostics: `pg.gk.load_quantity` (named moments/geometry, resolved by naming convention) and `pg.gk.load_distf` (full distribution function), on the `rt_gk_tcv_iwl*` fixtures |
| [`05_gk_rz.py`](scripts/05_gk_rz.py) | Gkeyll geometry discovery with `pg.gk.rz`, plus explicit projection reuse with `map_to_rz` |
| [`06_growth.py`](scripts/06_growth.py) | Recover a known energy growth rate and inspect log-space residuals |
| [`07_collect_animate.py`](scripts/07_collect_animate.py) | Load many frames, collect a space–time diagram, and animate 1D/2D travelling waves |
| [`08_plotly.py`](scripts/08_plotly.py) | Interactive 2D height surfaces and 3D volume isosurfaces |
| [`09_pyvista.py`](scripts/09_pyvista.py) | Off-screen VTK isosurface and volume screenshots |
| [`10_manual_arrays.py`](scripts/10_manual_arrays.py) | Copy grids/values, manipulate NumPy arrays, and wrap the result |
| [`mirror_comparison.py`](scripts/mirror_comparison.py) | A four-panel algorithm-sensitivity figure from two analytic, symmetric 1-D p1 modal-serendipity datasets, with joined linear/log axes |

Run one directly:

```bash
pip install --no-build-isolation -e '.[test]'
MPLBACKEND=Agg PYTHONPATH=src python examples/scripts/01_quickstart.py
```

Each script prints what it's doing as it goes and asserts the invariants it
demonstrates, so a successful run ending in `... OK` is itself a (manual)
confirmation that the example still holds. Output files (PNGs, a `.gkyl`
round trip) land in `examples/scripts/output/` by default, or wherever
`PGKYL_EXAMPLE_OUTPUT` points if that environment variable is set (this is
how `tests/test_examples.py` redirects them into a temp directory instead of
the repo).

## CLI

See [`cli_tutorial.md`](cli_tutorial.md) -- inspecting a file, the
`interpolate`/`select`/`plot` chain, discontinuity-preserving plots with
`local_poly`, DynVector `info`/`fit`, the gyrokinetic loaders
(`gk_load_quantity`, `gk_load_distf`), the `gk_rz`
diagnostic, `save`, and the generated command inventory.

## Running the tests

```bash
pytest tests/test_examples.py -v
```

Both halves need a compiled Gkeyll (`libg0core.so`) to run -- every fixture
here is a native `.gkyl` file, so the tests are skipped (not failed)
when `postgkyl.gpython.available()` is `False`, matching the rest of the
test suite's `needs_gkeyll` convention.

## Script/CLI figure equivalence

`figure_commands.json` holds a `pgkyl` pipeline for every published figure.
`compare_interfaces.py` runs those pipelines and the Python scripts separately,
then compares decoded pixels and GIF timing, or Plotly traces/layout/configuration.
Both outputs and the passing comparison report appear on the website.

```bash
python tests/generate_test_data.py
python examples/compare_interfaces.py
```

Results land in `build/figure-comparison/`, with CLI versions under `cli/`.
The PyVista examples need OpenGL; headless CI uses Mesa/EGL with
`VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow` and `LIBGL_ALWAYS_SOFTWARE=1`.
The Plotly HTML examples require no browser executable to generate them.
