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

Note that with `mamba`, one can also use the provided `environment.yml` file,
which also includes dependency specifications:

```bash
mamba env create -f environment.yml
```

### Installing Postgkyl

The Postgkyl itself is installed with `pip`.[^1] Developers and uses who want to
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
pip install -e postgkyl[test] --no-build-isolation
```

#### The Gkeyll bridge (native `.gkyl` reading, `interpolate`, weak algebra)

Postgkyl talks to Gkeyll through a small compiled bridge (`gpython`), not a path you configure. **This is
built automatically** as part of `pip install`/`pip install -e .`. `setup.py` runs
`scripts/build_gkeyll.sh`, which:

1. clones [Gkeyll](https://github.com/ammarhakim/gkeyll) into `./gkeyll/`
   (a sparse, blobless, depth-1 clone of just the `core/` app — a few tens
   of MB, not a submodule),
2. `./configure`s and `make core`s it into `gkeyll/build/core/libg0core.so`
   with no external dependencies (`--use-lapack-lite=yes`, so no
   MPI/CUDA/SuperLU/Lua/system LAPACK are required), then
3. compiles postgkyl's `_gpython` CPython extension
   (`src/postgkyl/gpython/csrc/_gpythonmodule.c`) against it.

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

If `gpython.available()` is `False`, the printed error explains which of the
two prerequisites (compiler, or the clone) is missing, or whether the built
extension is stale relative to the shim header — the fix in that last case
is always `scripts/build_gpython.sh`.

## Testing

Postgkyl utilizes [pytest](https://docs.pytest.org/) for testing. The tests can
be called manually from the root Postgkyl directory simply by using:

```bash
pytest [-v]
```

## Authors

The full list of authors can be found [here](AUTHORS.md).

## License

Postgkyl is distributed under the MIT License.

[^1]: This does *not* require any additional modifications of `PYTHONPATH`. If
    Postgkyl was used previously through `PYTHONPATH`, we strongly recommend
    removing the path to the Postgkyl repository from the variable.
