# Installation

Requires Python 3.10+ on Linux or macOS. Commands below use bash or zsh.

## Install from source

Install Git, Make, a C compiler, and Python development headers:

- Ubuntu/Debian: `sudo apt install git build-essential python3-venv python3-dev`
- macOS: `xcode-select --install`

```bash
git clone https://github.com/ammarhakim/postgkyl.git
cd postgkyl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
pgkyl --version
```

pip installs Python dependencies and builds Gkeyll's core library. The first
build downloads Gkeyll and may take several minutes; later builds reuse the
checkout, including local edits. The default core needs no external numerical
libraries, MPI, CUDA, or Lua.

Run `source .venv/bin/activate` in each new terminal. For Conda/Mamba, use
`mamba env create -f environment.yml` and `mamba activate pgkyl` instead of
creating a venv, then run the same pip command.

(developer-installation)=
## Developer installation

Install editable Python sources and development tools:

```bash
python -m pip install -e '.[test]'
pre-commit install
```

Python edits take effect immediately. Native changes require reinstallation.
For documentation work, use `'.[docs,test]'`. See the
[development guide](development.md) for testing and release workflows and
[contributing guide](contributing.rst) for building documentation.

## Configure Gkeyll

Edit `postgkyl-build.toml`, then reinstall. pip and the build/update scripts
share this configuration. The default builds the internal checkout:

```toml
[gkeyll]
mode = "managed"
module = "core"
```

Choose `core`, `moments`, `vlasov`, `gyrokinetic`, or `pkpm`; each includes all
preceding modules. Modules above core require full LAPACK (OpenBLAS on Linux,
Accelerate on macOS) and SuperLU, normally under `~/gkylsoft`. For custom
features or dependency paths, configure Gkeyll separately and use existing mode.

### Use an installed Gkeyll

```toml
[gkeyll]
mode = "existing"
module = "pkpm"
directory = "~/gkylsoft/gkeyll"
```

Point `directory` at the installation's `include/` and `lib/` parent and omit
`build-dir`. Gkeyll's `make install` places these under `~/gkylsoft/gkeyll`
by default. The selected modules must have compatible
`include/gkyl_gpython_<module>.h` headers and `lib/libg0<module>.so` libraries;
an older installation without the gpython shims must be rebuilt and installed.
Postgkyl does not need the executables in `bin/`.

Relative `directory` paths start at the configuration file; `build-dir` is
relative to `directory`. `~` expands to your home directory; `$HOME` is not
expanded in TOML.

Postgkyl copies the selected libraries without modifying the existing build.
After rebuilding Gkeyll, reinstall Postgkyl and restart Python. Third-party
shared dependencies must remain available at runtime.

### Overrides

| Variable | Effect |
| --- | --- |
| `POSTGKYL_BUILD_CONFIG` | Use another TOML file |
| `POSTGKYL_GKEYLL_MODULE` | Override the selected module |
| `POSTGKYL_GKEYLL_DIR` | Override the directory and select existing mode |

For a one-time full-stack build:

```bash
POSTGKYL_GKEYLL_MODULE=pkpm python -m pip install -e '.[test]'
```

Configuration applies to source builds. Prebuilt wheels keep their bundled
native libraries. Source distributions include `postgkyl-build.toml`; use a
separate file for machine-specific settings when packaging releases.

On Python 3.10, standalone scripts need `tomli`, included in the `test` extra.
pip installs it automatically for isolated builds; Python 3.11+ uses `tomllib`.

(updating-and-rebuilding)=
## Updating and rebuilding

Rebuild the configured native bridge and reinstall:

```bash
python -m pip install -e '.[test]'
```

Managed mode builds the current Gkeyll sources. Existing mode requires you to
rebuild Gkeyll first. To also fetch upstream updates:

```bash
bash scripts/update_pgkyl.sh
```

The updater fast-forwards Postgkyl and managed Gkeyll sources, then reinstalls.
It leaves existing Gkeyll builds alone and refuses tracked Gkeyll edits or
local commits. Add `--no-editable` for a regular installation.

For managed Gkeyll alone, use `sh scripts/build_gkeyll.sh` to build or
`sh scripts/update_gkeyll.sh` to fetch updates. Reinstall Postgkyl afterward.
`BUILD_JOBS`, `CC`, and `ARCH_FLAGS` control compilation; `PYTHON` selects the
scripts' interpreter.

## Verify the native build

```python
from postgkyl import gpython

gpython.require()
print(gpython.module_api_versions())
print(gpython.build_info())
```

`require()` reports loading errors. The other calls show available modules
and build provenance. See [Native physics operations](native-physics.md) for
optional bindings and examples.

```{toctree}
:hidden:

native-physics
```

For an offline rebuild, obtain Gkeyll and all build dependencies first, then
run `python -m pip install --no-build-isolation .`.

For pure-Python compatibility testing, use a fresh checkout:

```bash
POSTGKYL_SKIP_GKEYLL_BUILD=1 python -m pip install -e '.[test]'
```

This skips compilation but does not remove existing native artifacts.

## Notebooks and rendering

Install Marimo for the [interactive notebooks](https://github.com/ammarhakim/postgkyl/tree/main/notebooks):

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

For headless PyVista on Ubuntu/Debian:

```bash
sudo apt install libegl1 libgl1-mesa-dri
export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
export LIBGL_ALWAYS_SOFTWARE=1
```

Video export needs an ffmpeg executable, available through `imageio-ffmpeg`:

```bash
python -m pip install -U imageio-ffmpeg
```

See [Animation](animation.rst) and [PyVista](pyvista.rst) for rendering options.
