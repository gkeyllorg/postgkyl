---
name: gui
description: Build, change, or review a GUI, notebook viewer, or dashboard over Gkeyll output. A GUI only calls Postgkyl commands and displays what they return; every calculation belongs in Postgkyl's API/CLI.
---

# A GUI displays; Postgkyl computes

A graphical front end (marimo notebook, run viewer, dashboard, web page)
performs no calculations on simulation data. It turns the user's choices into
Postgkyl commands, runs them, and displays their results. Every number a GUI
shows is the output of a command a user could run with `pgkyl`: the same
canonical callable with the same options.

A calculation inside a GUI is a second implementation of a fact (design
doctrine V). Nobody reviews or tests it against Postgkyl, it can silently
disagree with the command, scripts and the CLI cannot reach it, and the
numbers on the screen cannot be reproduced from a shell.

## What a GUI does

- Discover runs, stems, and frames through `postgkyl.io` and
  `diagnostics.discovery`.
- Turn widget values into a chain of `PostgkylSession` calls, run it, and
  show the equivalent session Python and `pgkyl` command line (the pattern of
  `postgkyl.gui.pipeline`: `Step`, `run`, `python_script`). Program-scale
  results come from diagnostic callables (`gk.balance`, `gk.dt_limits`, ...)
  called with explicit options.
- Presentation: colours, colour and axis limits taken from the values a
  command returned, labels, layout, decimating a long trace for drawing,
  caching a command's output keyed by its inputs. Unit changes use the
  `units` verb.
- Keep its own settings and saved views.

## What a GUI must not do

- Arithmetic on loaded data: temperatures or flows from moments, sums over
  species or blocks, ratios and relative changes, derivatives, integrals and
  averages, interpolation to points, surfaces, or the midplane, fits, budgets,
  time-step or CFL reconstructions, statistics reported as results.
- Read `.gkyl` files with its own reader, index modal coefficients (`c0 / 2`
  as a cell average), or carry its own basis functions, weak products, or
  geometry transforms.
- Re-implement a command "for speed". Make the command faster instead.

## When the GUI needs a number Postgkyl cannot produce

1. Add it to Postgkyl first: a canonical callable in its layer (architecture
   skill; equation-specific work goes in `diagnostics/<model>/`), with CLI
   metadata (api-cli skill) and analytic tests (testing skill).
2. Then call it from the GUI. The Postgkyl change lands first; the GUI change
   may follow in its own pull request.
3. Do not leave a temporary calculation in the GUI until the command exists.

## Review

Search the GUI for NumPy on data arrays (`np.mean`, `np.sum`, `np.sqrt`,
`np.interp`, `np.percentile`, array `*` and `/` on values read from files).
Each hit is presentation of a command's output or a defect. Each displayed
quantity should have a `pgkyl` command the page can show.
