# Postgkyl

![pytest](https://github.com/ammarhakim/postgkyl/actions/workflows/test.yml/badge.svg)

This is the Postgkyl project. It is both Python library and command-line tool
designed to provide unified access to Gkeyll data together with a broad variety
of analytical and visualization tools.

## Installation

Follow these steps to install the current source version of Postgkyl. Run
the commands one line at a time in a terminal, using the same terminal
throughout. These instructions use bash or zsh on Linux or macOS. On Windows,
first set up [Ubuntu in WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
and use its terminal.

You need an internet connection, Git (to download the code), and build tools
(to build part of Postgkyl). Install the tools for your system:

- **Ubuntu / Debian, including WSL:** run `sudo apt update`, then
  `sudo apt install git build-essential`.
- **macOS:** run `xcode-select --install` and complete the installer.
- **Other Linux distributions:** install Git, Make, and a C compiler using
  your distribution's package manager.

Download Postgkyl and enter its folder:

```bash
git clone https://github.com/ammarhakim/postgkyl.git
cd postgkyl
```

### 1. Create an environment

An environment keeps Postgkyl's Python packages separate from those used by
other projects. Choose **one** of the following options, then continue to
step 2. If you are new to Python environments, use **mamba**.

#### pyenv

[pyenv](https://github.com/pyenv/pyenv#installation) lets you install a
specific Python version. If it is not installed, follow its installation
instructions, including the
[Python build prerequisites](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)
and shell setup, then reopen your terminal and return to the `postgkyl`
folder.

Install Python 3.12, select it for this folder, and use Python's built-in
`venv` tool to create and activate the environment:

```bash
pyenv install 3.12
pyenv local 3.12
python -m venv .venv
source .venv/bin/activate
```

#### mamba

If you do not have mamba, install
[Miniforge](https://github.com/conda-forge/miniforge#install), which includes
it. Allow the installer to initialize your shell, then reopen your terminal
and return to the `postgkyl` folder. Create and activate the environment:

```bash
mamba env create -f environment.yml
mamba activate pgkyl
```

If you already use **conda** (for example, through Anaconda or Miniconda),
you can use `conda` in place of `mamba` in both commands.

### 2. Install dependencies

Dependencies are the other packages Postgkyl needs. The two configuration
files have different jobs:

- [environment.yml](environment.yml) creates the mamba/conda environment
  with Python, pip (the Python package installer), and setuptools (a build
  tool). The pyenv/venv option above sets up Python and pip without this file.
- [pyproject.toml](pyproject.toml) lists the packages Postgkyl uses, such as
  NumPy and Matplotlib, and the optional developer tools. pip reads this file
  when installing Postgkyl, so these dependency lists are maintained here.

With your environment active, install NumPy and the Python build tools:

```bash
python -m pip install --upgrade "numpy>=2.2.6" setuptools wheel
```

NumPy must be installed before building Postgkyl. The remaining dependencies
will be installed automatically in step 3; you do not need to install them
one by one.

### 3. Install Postgkyl

From the `postgkyl` folder, run:

```bash
python -m pip install --no-build-isolation .
```

The final `.` means "install from this folder." Keep `--no-build-isolation`
so Postgkyl builds using the NumPy you just installed. This step also
downloads and builds the Gkeyll bridge automatically and may take several
minutes.

Check the installation:

```bash
pgkyl --version
pgkyl --help
```

The first command prints version information; the second lists the available
commands. Each time you open a new terminal, activate your environment again:
run `source .venv/bin/activate` from the `postgkyl` folder for pyenv/venv, or
`mamba activate pgkyl` (or `conda activate pgkyl`) for mamba/conda.

## Documentation

Full documentation of the Gkeyll project, including Postgkyl, is available at
[ReadTheDocs](https://gkeyll.readthedocs.io/en/latest/postgkyl/index.html). The repository also contains
[examples](examples/README.md) and [notebooks](notebooks/README.md).

The website generates this section from Postgkyl's `main` branch. Guides live
in [docs/source](docs/source), reference pages come from the Python API and
compiled CLI, and figures come from the tested example scripts. See the
[documentation build instructions](docs/source/contributing.rst) for a local
preview and [integration notes](docs/integration-plan.md) for the website setup.

For help with a particular command, add `--help`, for example:

```bash
pgkyl interpolate --help
```

### Additional installation notes

To install the published version from [PyPI](https://pypi.org/project/postgkyl/),
complete the environment and dependency steps above, then replace the install
command in step 3 with:

```bash
python -m pip install --no-build-isolation postgkyl
```

To leave an environment, run `deactivate` for pyenv/venv, `mamba deactivate`
for mamba, or `conda deactivate` for conda. In shells other than bash/zsh,
venv activation uses `source .venv/bin/activate.fish` for fish or
`source .venv/bin/activate.csh` for csh/tcsh.

Installing with pip does not require changes to `PYTHONPATH`. If you
previously added a Postgkyl checkout to that variable, remove that entry so
Python uses the installed package.

#### Gkeyll bridge

The Gkeyll bridge (`gpython`) connects Postgkyl to Gkeyll's compiled code for
native `.gkyl` reading, interpolation, integration, and DG arithmetic.
Installing from source builds it automatically. A prebuilt wheel includes
the bridge already.

During a source build, `setup.py` runs `scripts/build_gkeyll.sh`, which:

1. Downloads the [Gkeyll](https://github.com/ammarhakim/gkeyll) revision
   recorded in `scripts/gkeyll-revision` into `gkeyll/`.
2. Builds its core library with the bundled LAPACK implementation. No
   separate MPI, CUDA, SuperLU, Lua, or system LAPACK installation is needed.
3. Builds the Python extension and bundles the core library beside it, so
   the installed package can run without the Gkeyll source folder.

The build needs Git, Make, a C compiler, and network access. It uses `cc` by
default. To select another installed compiler, for example GCC, run:

```bash
CC=gcc python -m pip install --no-build-isolation .
```

Always use `--no-build-isolation` when building from source so the bridge
builds against the NumPy in your active environment. A different NumPy at
build time can cause import errors or crashes. After changing NumPy, rebuild
the bridge in that environment.

Check whether the bridge is available:

```bash
python -c "from postgkyl import gpython; print(gpython.available())"
```

This should print `True`. If it prints `False`, get the error details with:

```bash
python -c "from postgkyl import gpython; gpython.require()"
```

A failed source build stops installation. An installation with an unavailable
bridge can still read files through the Python reader, but operations that
require the bridge raise an error.

To rebuild and reinstall from the repository folder with your environment
active, run:

```bash
python -m pip install --no-build-isolation .
```

For an editable developer installation (see below), you can rebuild in place
with `PYTHON=python scripts/build_gkeyll.sh`. If the Gkeyll core library is
already built and only the Python extension needs rebuilding, use
`PYTHON=python scripts/build_gpython.sh` instead.

## Developing for Postgkyl

Complete the installation steps above, then run this from the `postgkyl`
folder with your environment active:

```bash
python -m pip install --no-build-isolation -e '.[test]'
```

The `-e` option makes the installation use your source files directly, so
Python edits take effect without reinstalling. The `[test]` option also
installs the testing, formatting, and packaging tools listed in
`pyproject.toml`.

### pytest

[pytest](https://docs.pytest.org/) runs the automated tests. From the
`postgkyl` folder, run:

```bash
python -m pytest tests/
```

Add `-v` to see a separate result for each test.

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

For pure-Python compatibility testing, skip the native build at installation
time with `POSTGKYL_SKIP_GKEYLL_BUILD=1`. Use this only when testing the
`compatibility` subset; normal installations build the bridge.

### Formatting

After installing the developer tools above, enable the checks that run
before a Git commit and run them over all tracked files:

```bash
pre-commit install
pre-commit run --all-files
```

pre-commit installs the pinned YAPF, clang-format, Ruff, and repository
checks. YAPF reads `.style.yapf`; clang-format reads `.clang-format`; Ruff
reads `pyproject.toml`. The automated pull-request checks use these same
tools and report any formatting changes needed.

### API and CLI documentation

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

### Automatic version numbers

Versions use `major.minor.PR.commit`, starting at `2.0.0.0`. Merged PRs
automatically increment the third component; later commits increment the fourth.
Use the **Version** workflow on `main` to request a minor or major release.
See [versioning](docs/source/versioning.md) for examples and local checkout behavior.

### Checking a release package

Build a wheel (an installable package) and test it in a clean environment:

```bash
python -m build --no-isolation
scripts/smoke_wheel.sh dist/*.whl
```

## Authors

The full list of authors can be found [here](AUTHORS.md).

## License

Postgkyl is distributed under the MIT License.
