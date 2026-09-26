# Documentation integration

The website builds documentation from **Postgkyl main**, as requested. There is
no pinned Postgkyl dependency or submodule. The code, function docstrings,
compiled CLI, tutorials, fixtures, and figures come from one checkout per
build. Local and pull-request previews use the proposed working checkout.

- `docs/source/` owns guides and tutorial narratives.
- `scripts/build_docs.py` runs the examples and prepares a Sphinx source tree.
- `docs/conf.py` builds a standalone preview of that tree; `docs/_ext/` copies
  interactive HTML beside its referring pages in either website layout.
- `examples/` owns executable Python and CLI tutorials; published code is
  included directly from those files.
- `examples/figure_commands.json` owns the paired CLI pipelines.
  `examples/compare_interfaces.py` executes scripts and CLI independently,
  compares pixels, GIF frames/timings, and Plotly data/layout/configuration,
  and produces the report used by the website.
- `tests/generate_test_data.py` owns synthetic fixtures, including the
  shock-tube state, exponential energy history, travelling waves, and 3D Gaussian.
- `tests/test_docs_build.py` checks strict Sphinx builds, command coverage,
  per-command API pages, navigation, portable downloads, and preservation of unrelated directories.
- `.github/workflows/docs.yml` runs the documentation checks on Python 3.12
  for pull requests and main pushes, and uploads an HTML preview.

In gkyl-doc, `scripts/prepare_postgkyl.py` fetches main, installs that checkout,
and stages the generated section. `make html`, Read the Docs, and the host's
GitHub Actions use that script. The host also tests daily to catch upstream
changes without requiring a host commit. Its README describes the optional
Read the Docs token for publishing after those scheduled checks.

Merge the Postgkyl side first, then the host integration. Hosted build time and
account configuration must be checked on Read the Docs; the local integration
uses an explicit checkout so both sides can be tested before merging.

The hosting configuration selects Python 3.12. Builds reject Sphinx warnings,
missing native support, broken examples, and a mismatched installed checkout.
A recorded source SHA explains a particular generated result; each normal
website build still fetches main afresh.

See `docs/source/contributing.rst` for local commands and the gkyl-doc README
for the complete website build.
