# Postgkyl

![pytest](https://github.com/ammarhakim/postgkyl/actions/workflows/test.yml/badge.svg)

This is the Postgkyl project. It is both Python library and command-line tool
designed to provide unified access to Gkeyll data together with a broad variety
of analytical and visualization tools.

## Documentation

Full documentation of the Gkeyll project is available at
[ReadTheDocs](http://gkeyll.rtfd.io).

## Dependencies and Installation

Postgkyl requires the packages listed in pyproject.toml

Postgkyl requires NumPy >= 2.2.6. In addition, there is one optional
dependency:

* [pytest](https://pypi.org/project/pytest/)

[pytest](https://docs.pytest.org/en/stable/) is required only for developers.

### Setting up virtual environment (recommended)

We strongly recommend creating a virtual Python environment for everybody
working with more than one Python project (this includes even using both
Postgkyl and Sphinx). The two recommended options are
[venv](https://docs.python.org/3/library/venv.html) and
[mamba](https://mamba.readthedocs.io/en/latest/).

With `venv`, one can create the virtual environment with:

```bash
python -m venv /path/to/new/virtual/environments/pgkyl
```

then activate it with:

| bash/zsh | `source <venv>/bin/activate`      |
| fish     | `source <venv>/bin/activate.fish` |
| csh/tcsh | `source <venv>/bin/activate.csh`  |

and deactivate with:

```bash
deactivate
```

With `mamba`, one can create the virtual environment with:

```bash
mamba create -n pgkyl
```

then activate with:

```bash
mamba activate pgkyl
```

and deactivate with:

```bash
mamba deactivate
```

With `mamba`, the provided `environment.yml` creates the Python build
environment. Runtime and test dependencies remain authoritative in
`pyproject.toml` and are installed by the `pip install` step below:

```bash
mamba env create -f environment.yml
```

### Installing Postgkyl

Postgkyl itself is installed with `pip`.[^1] Developers and users who want to
have the most up-to-date version should install Postgkyl from the source code:

```bash
git clone https://github.com/ammarhakim/postgkyl.git
cd postgkyl
pip install --upgrade numpy setuptools wheel
pip install -e '.[test]' --no-build-isolation
```

Alternatively, Postgkyl can be installed directly from [PyPI](https://pypi.org/project/postgkyl/):

```bash
pip install --upgrade numpy setuptools wheel
pip install 'postgkyl[test]' --no-build-isolation
```

#### The Gkeyll bridge (native `.gkyl` reading, `interpolate`, weak algebra)

Postgkyl talks to Gkeyll through a small compiled bridge (`gpython`), not a path you configure. **This is
built automatically** as part of `pip install`/`pip install -e .`. `setup.py` runs
`scripts/build_gkeyll.sh`, which:

1. fetches the exact [Gkeyll](https://github.com/ammarhakim/gkeyll) commit in
   `scripts/gkeyll-revision` into `./gkeyll/` (a sparse, blobless clone of just
   the `core/` app — a few tens of MB, not a submodule),
2. `./configure`s and `make core`s it into `gkeyll/build/core/libg0core.so`
   with no external dependencies (`--use-lapack-lite=yes`, so no
   MPI/CUDA/SuperLU/Lua/system LAPACK are required), then
3. bundles `libg0core.so` beside and compiles postgkyl's `_gpython` CPython
   extension (`src/postgkyl/gpython/csrc/_gpythonmodule.c`) against a relative
   loader path, so a built wheel does not depend on the source checkout.

This step needs **network access** (to clone Gkeyll) and **a C compiler**.
It defaults to `cc`; if your system doesn't have `cc`, set `CC=gcc` (or any compiler you have) before
installing:
```bash
CC=gcc pip install -e '.[test]' --no-build-isolation
```

**Always install with `--no-build-isolation`** (as above). Without it, `pip`
builds the extension in a throwaway environment that resolves `numpy`
independently of the one that ends up installed for running Postgkyl. The
extension targets NumPy's `>=2.2` ABI explicitly (matching the `numpy>=2.2.6`
floor above) so that a same-major mismatch fails loudly at import with a clear
`numpy.dtype size changed` error rather than silently — but this is a
best-effort backstop, not a guarantee: a build/runtime NumPy skew has been
observed to crash the native bridge outright (segfault or memory corruption
inside Gkeyll's own C code, surfacing anywhere from the next file read to an
unrelated `matplotlib` call much later) instead of raising cleanly. Building
against the exact NumPy already installed is the only reliable fix, which is
what `--no-build-isolation` gives you.

If this step fails or is skipped, Postgkyl still imports and works — reading
files falls back to a pure-Python reader, and anything that needs the
compiled bridge (`.interpolate()`, weak `* /` on modal data, native `.gkyl`
reading, `.integrate()`, …) raises a `RuntimeError` naming the missing piece
instead of the pipeline silently doing the wrong thing. Check whether the
bridge is active with:
```bash
python -c "from postgkyl import gpython; print(gpython.available())"
```

To rebuild by hand (e.g. after pulling a Postgkyl or Gkeyll update, or after
fixing a compiler issue), re-run either script from the repo root — both are
safe to re-run:
```bash
scripts/build_gkeyll.sh   # full: re-clone/build libg0core.so, then the extension
scripts/build_gpython.sh  # just the extension, if libg0core.so is already built
```

Pure-Python compatibility testing can explicitly omit the native build with
`POSTGKYL_SKIP_GKEYLL_BUILD=1`. This switch is intended for test lanes that
select the `compatibility` marker; normal installs continue to build the
bridge.

To verify a release artifact independently of the checkout, build it and run
the clean-environment smoke test:

```bash
python -m build --no-isolation
scripts/smoke_wheel.sh dist/*.whl
```

If `gpython.available()` is `False`, the printed error explains which of the
two prerequisites (compiler, or the clone) is missing, or whether the built
extension is stale relative to the shim header — the fix in that last case
is always `scripts/build_gpython.sh`.

## Formatting

Install the repository's Git hook and run both formatters over all tracked
Python and C sources with:

```bash
python -m pip install -e ".[test]"
pre-commit install
pre-commit run --all-files
```

The test extra pins the supported pre-commit runner; pre-commit installs the
pinned YAPF, clang-format, Ruff, and repository-sanity hooks in isolated
environments. YAPF reads `.style.yapf`; clang-format reads `.clang-format`;
Ruff reads `pyproject.toml`. CI checks the exact pull-request commit and fails
with a formatter diff when that commit is not clean.

## Testing

Postgkyl utilizes [pytest](https://docs.pytest.org/) for testing. The tests can
be called manually from the root Postgkyl directory simply by using:

```bash
pytest [-v]
```

The default suite treats unexpected warnings as errors and uses strict marker
and configuration validation. Useful CI-equivalent subsets are:

```bash
POSTGKYL_SKIP_GKEYLL_BUILD=1 pytest -m compatibility
POSTGKYL_REQUIRE_GKEYLL=1 pytest -m native
pytest -m "render and not external_tool"
pytest -m external_tool  # invokes Chrome and/or ffmpeg
pytest -m "not external_tool" --cov=postgkyl --cov-branch --cov-fail-under=93
```

The external-tool lane has explicit timeouts in CI. Native lanes set
`POSTGKYL_REQUIRE_GKEYLL=1`, turning a missing bridge into a session failure
instead of allowing the native test inventory to skip silently.

## API and CLI documentation

Public command documentation lives on the Python function that implements the
operation. The equivalent `GData` spelling is a class-body alias to that same
function, so editor hover help, `help(pg.interpolate)`,
`help(data.interpolate)`, and `pgkyl interpolate --help` cannot maintain
separate descriptions.
The installed distribution includes a `py.typed` marker so language servers
consume these inline signatures and aliases from a virtual environment too.

Command docstrings use `Args:` entries in Google style. Every CLI-visible
parameter needs one entry; command compilation rejects missing, duplicate, or
unknown parameter documentation. `tests/test_documentation.py` additionally
checks the public Python surface, static fluent aliases, source/runtime
docstring identity, and deterministic CLI lowering. Run it directly with:

```bash
pytest tests/test_documentation.py
```

## Authors

The full list of authors can be found [here](AUTHORS.md).

## License

Postgkyl is distributed under the MIT License.

[^1]: This does *not* require any additional modifications of `PYTHONPATH`. If
    Postgkyl was used previously through `PYTHONPATH`, we strongly recommend
    removing the path to the Postgkyl repository from the variable.
