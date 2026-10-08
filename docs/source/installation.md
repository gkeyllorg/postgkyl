# Installation

Postgkyl requires Python 3.10 or newer. These commands use bash or zsh on
Linux or macOS.

## Install from source

Install Git, Make, a C compiler, and Python development headers first:

- Ubuntu/Debian: `sudo apt install git build-essential python3-venv python3-dev`.
- macOS: `xcode-select --install`.
- Other Linux distributions: install the equivalent development packages.

```bash
git clone https://github.com/ammarhakim/postgkyl.git
cd postgkyl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
pgkyl --version
pgkyl --help
```

pip installs build dependencies in an isolated environment and runtime dependencies
in your active environment. No separate NumPy installation is required.
Dependency versions live in
[pyproject.toml](https://github.com/ammarhakim/postgkyl/blob/main/pyproject.toml).
For an editable checkout, follow [Developer installation](#developer-installation).

The first native build may take several minutes and downloads the Gkeyll branch named in
[scripts/gkeyll-branch](https://github.com/ammarhakim/postgkyl/blob/main/scripts/gkeyll-branch)
(currently `main`). Later builds reuse the existing checkout without fetching,
switching branches, or discarding local edits. The core uses bundled LAPACK;
MPI, CUDA, SuperLU, Lua, and system LAPACK are not required.
The installed package includes the core library and needs no Gkeyll checkout
at runtime. `pgkyl --version` reports its build provenance.

Activate the environment again in each terminal with `source .venv/bin/activate`.
Run `deactivate` to leave it. Installation does not require `PYTHONPATH` changes;
remove old Postgkyl entries if they shadow the installed package.

## Alternative environments

An existing Python environment works too; use the same pip installation command.

For mamba or conda, from the checkout:

```bash
mamba env create -f environment.yml
mamba activate pgkyl
python -m pip install .
```

Use `conda` in place of `mamba` if preferred. Install
[Miniforge](https://github.com/conda-forge/miniforge#install) if needed.
`environment.yml` supplies Python and pip; project dependencies remain in
`pyproject.toml`.

If you use [pyenv](https://github.com/pyenv/pyenv#installation), select your Python
version before creating the venv, for example `pyenv install 3.12` and
`pyenv local 3.12`. Follow pyenv's prerequisites and shell setup first.

(developer-installation)=
## Developer installation

After creating and activating an environment from the source checkout, install
in editable mode with the test and development tools:

```bash
python -m pip install -e '.[test]'
pre-commit install
```

Python edits take effect immediately. After native changes, follow
[Updating and rebuilding](#updating-and-rebuilding). See the
[development guide](development.md) for tests, formatting, and release workflows.

For documentation development, use Python 3.12 and include the documentation
extra:

```bash
python -m pip install -e '.[docs,test]'
```

Then follow the [documentation build guide](contributing.rst).

For pure-Python compatibility testing, use a fresh checkout:

```bash
POSTGKYL_SKIP_GKEYLL_BUILD=1 python -m pip install -e '.[test]'
```

This switch skips compilation; it does not remove native artifacts from an
existing editable checkout. Normal installations build the bridge. Run the
`compatibility` test subset as described in the development guide.

For local portable Linux wheel builds, make Docker available and install
cibuildwheel before following the development guide's release workflow:

```bash
python -m pip install cibuildwheel
```

## Notebooks

After installing Postgkyl, install Marimo in the same environment to use the
[interactive notebooks](https://github.com/ammarhakim/postgkyl/tree/main/notebooks):

```bash
python -m pip install marimo
```

The `gui` extra installs Marimo for the notebook GUI, which browses a data
directory, loads files or gyrokinetic quantities, and shows each figure with
the equivalent `PostgkylSession` Python and `pgkyl` command line, each with a
copy button:

```bash
python -m pip install '.[gui]'
pgkyl-gui --path /path/to/simulation/output
```

"Save state" writes every choice to a file; reopen the GUI exactly as it was
with:

```bash
pgkyl-gui --state pgkyl_gui_state.json
```

The state remembers its data directory; `--path` overrides it.
`python -m postgkyl.gui` is equivalent to `pgkyl-gui`.

## Rendering dependencies

PyVista requires an OpenGL context even for off-screen screenshots. On headless
Ubuntu/Debian, install Mesa/EGL and select software rendering:

```bash
sudo apt install libegl1 libgl1-mesa-dri
export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
export LIBGL_ALWAYS_SOFTWARE=1
```

Video export requires an ffmpeg executable. A pip package that includes one is
available through:

```bash
python -m pip install -U imageio-ffmpeg
```

The `ffmpeg` Python package does not supply an executable. GIF, WebP, and APNG
require no ffmpeg installation. See [Animation](animation.rst) for encoder
selection and [PyVista](pyvista.rst) for rendering examples.

(updating-and-rebuilding)=
## Updating and rebuilding

With the environment active, update both Postgkyl and Gkeyll, rebuild, and
reinstall as an editable installation:

```bash
bash scripts/update_pgkyl.sh
# Install a copy of the checkout instead:
bash scripts/update_pgkyl.sh --no-editable
```

Updates use fast-forward-only Git operations and refuse tracked Gkeyll edits
or local commits on the branch being updated. Set `PYTHON=/path/to/python`
to select an interpreter explicitly.

To rebuild local native changes, including uncommitted Gkeyll changes, simply
repeat the editable installation:

```bash
python -m pip install -e '.[test]'
```

This uses the same build path as a regular installation. For just the producer
core, run `sh scripts/build_gkeyll.sh`. To explicitly update only the producer,
run `sh scripts/update_gkeyll.sh`, then reinstall Postgkyl.
Use `CC=gcc` to select a compiler or `BUILD_JOBS=2` to limit parallel compilation.
`ARCH_FLAGS` defaults to empty for portable binaries; developers may opt into
machine-specific flags for local use.

## Troubleshooting

To diagnose an unavailable native bridge:

```bash
python -c "from postgkyl import gpython; gpython.require()"
```

A source build failure stops installation. An unavailable bridge still permits
some Python readers; operations requiring native kernels report an error.
Reinstall in the active environment after changing Python or encountering an
incompatible native dependency.

For an offline rebuild, obtain the Gkeyll checkout and install all build/runtime
dependencies in advance. Then use `python -m pip install --no-build-isolation .`.
This optional advanced mode uses the active environment's build dependencies.
