import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

# This notebook uses a compiles and runs (the initialization of) a Gkeyll
# gyrokinetic input file with twistshift BCs, and plots:
#   - The region covered by shifting the lower-y cells.
#   - The selected quantity to be shifted.
#   - The shifted quantity.
# Requires:
#   - postgkyl.
#   - gkeyll libraries (core and gyrokinetic).
#   - Marimo.


@app.cell
def _imports():
  import glob
  import re
  import subprocess
  from pathlib import Path

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  from matplotlib.lines import Line2D
  import numpy as np
  import postgkyl as pg
  import marimo as mo
  return (
      glob,
      re,
      subprocess,
      Path,
      matplotlib,
      plt,
      patches,
      Line2D,
      np,
      pg,
      mo,
  )


@app.cell
def _helpers(pg, np):

  def getRawGrid(dataFile, **opKey):
    pgData = pg.GData(dataFile)
    dimOut = pgData.get_num_dims()
    xNodal = pgData.get_grid()
    if 'location' in opKey:
      if opKey['location'] == 'center':
        xOut = [[] for _ in range(dimOut)]
        for i in range(dimOut):
          nNodes = np.shape(xNodal[i])[0]
          xOut[i] = np.zeros(nNodes - 1)
          xOut[i] = np.multiply(0.5,
                                xNodal[i][0:nNodes - 1] + xNodal[i][1:nNodes])
      else:
        xOut = xNodal
    else:
      xOut = xNodal
    nxOut = np.zeros(dimOut, dtype='int')
    lxOut = np.zeros(dimOut, dtype='double')
    dxOut = np.zeros(dimOut, dtype='double')
    for i in range(dimOut):
      nxOut[i] = np.size(xOut[i])
      lxOut[i] = xOut[i][-1] - xOut[i][0]
      if nxOut[i] > 1:
        dxOut[i] = xOut[i][1] - xOut[i][0]
      else:
        dxOut[i] = xNodal[i][1] - xNodal[i][0]
    return xOut, dimOut, nxOut, lxOut, dxOut, None

  def getRawData(dataFile):
    return pg.GData(dataFile).get_values()

  return getRawGrid, getRawData


@app.cell
def _controls(mo):
  # Compile / run controls
  sim_dir = mo.ui.text(
      value="/Users/mfrancis/Documents/gkeyll/code/gkeyll_v0/cbc_tst",
      label="Simulation directory",
      full_width=True,
  )
  input_file_ui = mo.ui.text(
      value="rt_gk_cbc_passive_3x2v_p1",
      label="Input file name",
      full_width=True,
  )
  sim_name_ui = mo.ui.text(
      value="rt_gk_cbc_passive_3x2v_p1",
      label="Simulation name",
      full_width=True,
  )
  gkylsoft_ui = mo.ui.text(
      value=str(__import__('pathlib').Path.home() / "gkylsoft"),
      label="gkylsoft path",
      full_width=True,
  )
  ly_fac = mo.ui.number(value=1, step=0.1, label="Ly factor")
  nx = mo.ui.number(value=32, step=1, label="Nx")
  ny = mo.ui.number(value=16, step=1, label="Ny")
  run_btn = mo.ui.run_button(label="Compile & Run")

  # Field-plot controls
  species = mo.ui.text(value="elc", label="Species")
  quantity = mo.ui.text(value="M0", label="Quantity")
  comp_ui = mo.ui.number(value=0, step=1, label="Component")
  boundary = mo.ui.dropdown(
      options={
          "Lower z": "lower",
          "Upper z": "upper"
      },
      value="Lower z",
      label="Boundary",
  )

  return sim_dir, input_file_ui, sim_name_ui, gkylsoft_ui, ly_fac, nx, ny, run_btn, species, quantity, comp_ui, boundary


@app.cell
def _layout(mo, sim_dir, input_file_ui, sim_name_ui, gkylsoft_ui, ly_fac, nx,
            ny, run_btn, species, quantity, comp_ui, boundary):
  mo.vstack([
      sim_dir,
      input_file_ui,
      sim_name_ui,
      gkylsoft_ui,
      mo.hstack([ly_fac, nx, ny, run_btn], justify="start"),
      mo.callout(
          mo.
          md("**Assumptions:** the input C file specifies the number of cells as `Nx` and `Ny`, "
             "and the domain size along y as `Ly` (in units of `rho_s`)."),
          kind="info",
      ),
      mo.md("---"),
      mo.md("**Field plot**"),
      mo.hstack([species, quantity, comp_ui, boundary], justify="start"),
  ])


@app.cell
def _compile_run(mo, re, subprocess, Path, sim_dir, input_file_ui, sim_name_ui,
                 ly_fac, nx, ny, run_btn):
  # Only execute when the run button is pressed.
  mo.stop(not run_btn.value, mo.md("Click **Compile & Run** to start."))

  _sdir = sim_dir.value.rstrip("/")
  _input_file = input_file_ui.value.strip()
  _sim_name = sim_name_ui.value.strip()

  # Find the C input file
  _c_path = Path(f"{_sdir}/{_input_file}.c")
  if not _c_path.exists():
    mo.stop(
        True,
        mo.callout(mo.md(f"**Error:** `{_c_path}` not found"), kind="danger"))

  # Warn if symlink
  if _c_path.is_symlink():
    mo.callout(
        mo.
        md(f" `{_c_path.name}` is a symlink -- edits will also modify the original file."
           ),
        kind="warn",
    )

  # Backup original on first run
  _orig = _c_path.with_suffix(".c.orig")
  if not _orig.exists():
    import shutil as _shutil
    _shutil.copy2(_c_path, _orig)

  # Edit parameters via regex
  _content = _c_path.read_text()

  _content = re.sub(
      r'(double Ly\s*=\s*)([\d.]+)(\*rho_s\s*;)',
      lambda m: f"{m.group(1)}{float(m.group(2)) * ly_fac.value}{m.group(3)}",
      _content)
  _content = re.sub(r'(int Nx\s*=\s*)\d+(\s*;)', rf'\g<1>{int(nx.value)}\g<2>',
                    _content)
  _content = re.sub(r'(int Ny\s*=\s*)\d+(\s*;)', rf'\g<1>{int(ny.value)}\g<2>',
                    _content)

  _c_path.write_text(_content)

  # Compile
  _make = subprocess.run(['make', _input_file],
                         cwd=_sdir,
                         capture_output=True,
                         text=True)

  if _make.returncode != 0:
    _msg = _make.stdout + "\n" + _make.stderr
    mo.stop(
        True,
        mo.callout(mo.md(f"**make failed:**\n```\n{_msg}\n```"), kind="danger"))

  # Run initialization
  _run = subprocess.run([f'./{_input_file}', '-s0'],
                        cwd=_sdir,
                        capture_output=True,
                        text=True)

  if _run.returncode != 0:
    _msg = _run.stdout + "\n" + _run.stderr
    mo.stop(
        True,
        mo.callout(mo.md(f"**Run failed:**\n```\n{_msg}\n```"), kind="danger"))

  result = (True, _sim_name, _sdir)
  return result,


@app.cell
def _donor_plot(mo, result, getRawGrid, getRawData, np, plt, patches, Line2D):
  # Wait for a successful compile+run.
  if result is None or not result[0]:
    mo.stop(True, mo.md("Run the simulation first."))

  _ok, _sim_name, _sdir = result

  # Load grid and shift from .gkyl output files
  _geo_file = f"{_sdir}/{_sim_name}-geo_corn_bmag.gkyl"
  _shift_file = f"{_sdir}/{_sim_name}-bc_zlower_twistshift.gkyl"

  _x_nodal, _, _geo_nx, _, _, _ = getRawGrid(_geo_file)

  _Nx = _geo_nx[0] - 1
  _Ny = _geo_nx[1] - 1
  _x_nodes = _x_nodal[0]
  _y_nodes = _x_nodal[1]

  _x_min, _x_max = _x_nodes[0], _x_nodes[-1]
  _y_min, _y_max = _y_nodes[0], _y_nodes[-1]
  _Lx, _Ly = _x_max - _x_min, _y_max - _y_min
  _dx, _dy = _Lx / _Nx, _Ly / _Ny

  _x_centers = 0.5 * (_x_nodes[:-1] + _x_nodes[1:])
  _dx_cells = np.diff(_x_nodes)

  _shift_data = getRawData(_shift_file)  # shape (Nx, 2)

  # Shift evaluation (p=1 DG)
  _sqrt2 = np.sqrt(2.0)
  _sqrt32 = np.sqrt(1.5)

  def _S_eval(x_arr):
    x_arr = np.asarray(x_arr)
    ix = np.clip(np.searchsorted(_x_nodes, x_arr, side='right') - 1, 0, _Nx - 1)
    xi = (x_arr - _x_centers[ix]) / (_dx_cells[ix] / 2.0)
    return _shift_data[ix, 0] / _sqrt2 + _sqrt32 * xi * _shift_data[ix, 1]

  _S_lo_cells = _shift_data[:, 0] / _sqrt2 - _sqrt32 * _shift_data[:, 1]
  _S_up_cells = _shift_data[:, 0] / _sqrt2 + _sqrt32 * _shift_data[:, 1]
  _dS_cells = _S_up_cells - _S_lo_cells

  def _wrap(v):
    return _y_min + np.mod(v - _y_min, _Ly)

  _y_tar_lo = _y_min
  _y_tar_up = _y_min + _dy

  # Figure
  _fig, _ax = plt.subplots(1, 1, figsize=(11, 7.5))
  _ax.set_aspect('equal')
  _ax.set_xlabel('x  (m)', fontsize=12)
  _ax.set_ylabel('y  (m)', fontsize=12)

  # Grid
  for _ix in range(_Nx):
    for _iy in range(_Ny):
      _ax.add_patch(
          patches.Rectangle((_x_nodes[_ix], _y_nodes[_iy]),
                            _dx_cells[_ix],
                            _dy,
                            lw=0.4,
                            edgecolor='#aaaaaa',
                            facecolor='white',
                            zorder=1))

  # Donor cells
  for _ix in range(_Nx):
    _x_lo = _x_nodes[_ix]
    _x_up = _x_nodes[_ix + 1]
    _xs = np.linspace(_x_lo, _x_up, 500)
    _donor_iys = set()
    for _frac in np.linspace(0.01, 0.99, 9):
      _y_probe = _y_tar_lo + _frac * _dy
      _yd = _wrap(_y_probe - _S_eval(_xs))
      _iys = np.clip(np.floor((_yd - _y_min) / _dy).astype(int), 0, _Ny - 1)
      _donor_iys.update(np.unique(_iys))
    for _iy in _donor_iys:
      _ax.add_patch(
          patches.Rectangle((_x_lo, _y_nodes[_iy]),
                            _dx_cells[_ix],
                            _dy,
                            alpha=0.38,
                            facecolor='royalblue',
                            edgecolor='none',
                            zorder=2))

  # Wrapped boundary lines
  _x_dense = np.linspace(_x_min, _x_max, 8000)
  _S_dense = _S_eval(_x_dense)

  def _plot_wrapped_line(y_raw, ls, lbl):
    y = _wrap(y_raw)
    jump = np.abs(np.diff(y)) > _Ly / 2
    mask = np.concatenate([[False], jump]) | np.concatenate([jump, [False]])
    _ax.plot(_x_dense,
             np.where(mask, np.nan, y),
             color='navy',
             lw=1.8,
             ls=ls,
             zorder=5,
             label=lbl)

  _plot_wrapped_line(_y_tar_lo - _S_dense, '-', r'$y_{\rm tar,lo} - S(x)$')
  _plot_wrapped_line(_y_tar_up - _S_dense, '--', r'$y_{\rm tar,up} - S(x)$')

  # Wrap points
  for _ix in range(_Nx):
    _x_lo = _x_nodes[_ix]
    _x_up = _x_nodes[_ix + 1]
    _S_lo = _S_lo_cells[_ix]
    _S_up = _S_up_cells[_ix]
    if _S_lo == _S_up:
      continue
    _n_lo = int(np.ceil(_S_lo / _Ly))
    _n_up = int(np.floor(_S_up / _Ly))
    for _n_wrap in range(_n_lo, _n_up + 1):
      _t = (_n_wrap * _Ly - _S_lo) / (_S_up - _S_lo)
      _x_wrap = _x_lo + _t * (_x_up - _x_lo)
      if _x_lo < _x_wrap < _x_up:
        _ax.axvline(_x_wrap,
                    color='limegreen',
                    lw=1.2,
                    ls=':',
                    zorder=4,
                    alpha=0.85)

  # ΔS/dy annotations
  for _ix, _dS in enumerate(_dS_cells):
    _color = 'red' if _dS >= _Ly else 'black'
    _ax.text(_x_centers[_ix],
             _y_min - 0.06 * _Ly,
             f'{_dS/_dy:.1f}',
             ha='center',
             va='top',
             fontsize=6,
             color=_color)
  _ax.text(_x_min - 0.004 * _Lx,
           _y_min - 0.06 * _Ly,
           r'$\Delta S/dy$:',
           ha='right',
           va='top',
           fontsize=7,
           color='black')

  # Legend
  _legend_elems = [
      patches.Patch(fc='royalblue', alpha=0.4, label='donor cells'),
      Line2D([0], [0], color='navy', lw=1.8, label=r'$y_{\rm tar,lo} - S(x)$'),
      Line2D([0], [0],
             color='navy',
             lw=1.8,
             ls='--',
             label=r'$y_{\rm tar,up} - S(x)$'),
      Line2D([0], [0],
             color='limegreen',
             lw=1.2,
             ls=':',
             label=r'wrap point: $S(x)=n\,L_y$'),
  ]
  _ax.legend(handles=_legend_elems,
             loc='upper center',
             ncols=2,
             fontsize=9,
             framealpha=0.92,
             frameon=True,
             bbox_to_anchor=(0.5, 1.0))

  _ax.set_xlim(_x_min - 0.01 * _Lx, _x_max + 0.01 * _Lx)
  _ax.set_ylim(_y_min - 0.14 * _Ly, _y_max + 0.10 * _Ly)
  _ax.set_title(
      f'twist-shift donor region for target $j=1$ (lowest y-cell)\n'
      fr'$N_x={_Nx},~N_y={_Ny},~L_x={_Lx:.3e}$ m$,~L_y={_Ly:.3e}$ m',
      fontsize=10,
  )

  plt.tight_layout()
  _out = mo.as_html(_fig)
  plt.close(_fig)
  _out


@app.cell
def _field_plot(mo, pg, np, plt, result, species, quantity, comp_ui, boundary):
  # Wait for a successful compile+run.
  if result is None or not result[0]:
    mo.stop(True, mo.md("Run the simulation first."))

  _ok, _sim_name, _sdir = result
  _bnd = boundary.value  # "lower" or "upper"
  _comp = int(comp_ui.value)

  _field_file = f"{_sdir}/{_sim_name}-{species.value}_{quantity.value}_0.gkyl"

  # Load data
  try:
    _pg_data = pg.GData(_field_file)
  except Exception as _e:
    mo.stop(
        True,
        mo.callout(mo.md(f"**Cannot load file:**\n`{_field_file}`\n\n{_e}"),
                   kind="danger"))

  # Determine poly_order and basis_type from file context
  _poly_order = _pg_data.ctx.get('poly_order', 1)
  _ctx_basis = _pg_data.ctx.get('basis_type', 'serendipity')
  _basis_type = 'ms' if _ctx_basis == 'serendipity' else _ctx_basis

  # Interpolate
  _pg_interp = pg.GInterpModal(_pg_data, _poly_order, _basis_type)
  _x_out, _data_out = _pg_interp.interpolate(_comp)
  # _x_out:   [x_nodal (Nx*p+1,), y_nodal (Ny*p+1,), z_nodal (Nz*p+1,)]
  # _data_out: (Nx*p, Ny*p, Nz*p, 1)  (with comp already selected → last dim = 1)

  # Slice at z boundary
  _iz = 0 if _bnd == "lower" else -1
  _field = _data_out[:, :, _iz, 0]  # shape (Nx_interp, Ny_interp)

  # Plot
  _fig, _ax = plt.subplots(figsize=(9, 5.5))
  _ax.set_aspect('equal')

  # pcolormesh with nodal x/y grids: data must be (Ny, Nx) = transposed
  _pcm = _ax.pcolormesh(_x_out[0],
                        _x_out[1],
                        _field.T,
                        shading='flat',
                        cmap='inferno')
  plt.colorbar(_pcm,
               ax=_ax,
               label=f"{species.value}_{quantity.value} [comp {_comp}]")

  _ax.set_xlabel('x  (m)', fontsize=12)
  _ax.set_ylabel('y  (m)', fontsize=12)
  _bnd_label = "lower" if _bnd == "lower" else "upper"
  _ax.set_title(
      f'species: {species.value},  quantity: {quantity.value},  '
      f'comp: {_comp},  z boundary: {_bnd_label}',
      fontsize=10,
  )

  plt.tight_layout()
  _out2 = mo.as_html(_fig)
  plt.close(_fig)
  _out2


@app.cell
def _ts_lib(mo, gkylsoft_ui, run_btn):
  mo.stop(not run_btn.value)
  import ctypes
  from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops

  _gkylsoft = gkylsoft_ui.value.rstrip("/")

  _dg = GkeyllDGops(_gkylsoft)
  _lc = _dg._lib  # libg0core CDLL handle; argtypes already set by GkeyllDGops

  _lib_gk = ctypes.CDLL(f"{_gkylsoft}/gkeyll/lib/libg0gyrokinetic.so")

  _c_vp = ctypes.c_void_p
  _c_i = ctypes.c_int
  _c_b = ctypes.c_bool

  # gkyl_sub_range_init(rng*, bigrng*, sublower*, subupper*)
  _lc.gkyl_sub_range_init.argtypes = [
      _c_vp, _c_vp, ctypes.POINTER(_c_i),
      ctypes.POINTER(_c_i)
  ]
  _lc.gkyl_sub_range_init.restype = None

  _lib_gk.gkyl_bc_twistshift_new.argtypes = [
      _c_i,
      _c_i,
      _c_i,
      _c_i,
      _c_i,
      _c_vp,
      ctypes.POINTER(_c_i),
      _c_vp,
      _c_vp,
      _c_vp,
      _c_vp,
      _c_vp,
      _c_i,
      _c_b,
  ]
  _lib_gk.gkyl_bc_twistshift_new.restype = _c_vp

  _lib_gk.gkyl_bc_twistshift_advance.argtypes = [_c_vp, _c_vp, _c_vp]
  _lib_gk.gkyl_bc_twistshift_advance.restype = None

  _lib_gk.gkyl_bc_twistshift_release.argtypes = [_c_vp]
  _lib_gk.gkyl_bc_twistshift_release.restype = None

  ts_lib = (_lc, _lib_gk)
  return ctypes, ts_lib


@app.cell
def _ts_plot(mo, ctypes, ts_lib, result, species, quantity, comp_ui, boundary,
             pg, np, plt):
  if result is None or not result[0]:
    mo.stop(True, mo.md("Run the simulation first."))

  _ok, _sim_name, _sdir = result
  _bnd = boundary.value
  _comp = int(comp_ui.value)
  _lc, _lib_gk = ts_lib

  # Load interior field
  _field_file = f"{_sdir}/{_sim_name}-{species.value}_{quantity.value}_0.gkyl"
  try:
    _pg_data = pg.GData(_field_file)
  except Exception as _e:
    mo.stop(
        True,
        mo.callout(mo.md(f"**Cannot load field:**\n`{_field_file}`\n\n{_e}"),
                   kind="danger"))

  _poly_order = int(_pg_data.ctx.get('poly_order', 1))
  _interior = np.ascontiguousarray(_pg_data.get_values(), dtype=np.float64)
  _Nx, _Ny, _Nz, _ncomp = _interior.shape

  # Extended buffer with 1 ghost cell in each direction, matching gkyl memory layout
  _ext = np.zeros((_Nx + 2, _Ny + 2, _Nz + 2, _ncomp), dtype=np.float64)
  _ext[1:-1, 1:-1, 1:-1, :] = _interior

  # Apply periodic BC along z before twist-shift: ghost ← opposite interior face
  _ext[:, :, 0, :] = _ext[:, :, -2, :]  # lower z ghost ← last interior z cell
  _ext[:, :, -1, :] = _ext[:, :, 1, :]  # upper z ghost ← first interior z cell

  # Load shift DG from file written by gyrokinetic_app_write_ts_shift
  _shift_file = f"{_sdir}/{_sim_name}-bc_z{_bnd}_twistshift.gkyl"
  try:
    _shift_pg = pg.GData(_shift_file)
  except Exception as _e:
    mo.stop(
        True,
        mo.callout(
            mo.md(f"**Cannot load shift file:**\n`{_shift_file}`\n\n{_e}"),
            kind="danger"))
  _shift_poly_order = int(_shift_pg.ctx.get('poly_order', _poly_order))
  _shift_vals = np.ascontiguousarray(_shift_pg.get_values(), dtype=np.float64)
  _Nx_shift = int(np.prod(_shift_vals.shape[:-1]))
  _shift_ncomp = _shift_vals.shape[-1]

  # Create gkyl objects (libg0core argtypes already set by GkeyllDGops)
  _xn = _pg_data.get_grid()
  _lower = (ctypes.c_double * 3)(_xn[0][0], _xn[1][0], _xn[2][0])
  _upper = (ctypes.c_double * 3)(_xn[0][-1], _xn[1][-1], _xn[2][-1])
  _cells = (ctypes.c_int * 3)(_Nx, _Ny, _Nz)
  _ghost = (ctypes.c_int * 3)(1, 1, 1)

  _basis_ptr = _lc.gkyl_cart_modal_serendip_new(ctypes.c_int(3),
                                                ctypes.c_int(_poly_order))
  _grid_ptr = _lc.gkyl_rect_grid_new(ctypes.c_int(3), _lower, _upper, _cells)

  # local_ext: full extended range [0,Nx+1]x[0,Ny+1]x[0,Nz+1] matching _ext buffer strides
  _lo_ext = (ctypes.c_int * 3)(0, 0, 0)
  _up_ext = (ctypes.c_int * 3)(_Nx + 1, _Ny + 1, _Nz + 1)
  _local_ext = _lc.gkyl_range_new(ctypes.c_int(3), _lo_ext, _up_ext)

  # bc_range: sub-range [1,Nx]x[1,Ny]x[0,Nz+1] -- x/y interior only, z includes ghosts.
  # Allocate via gkyl_range_new (opaque pointer), then let gkyl_sub_range_init overwrite
  # its fields in-place so it inherits _local_ext's linearizer (correct buffer strides).
  _lo_sub = (ctypes.c_int * 3)(1, 1, 0)
  _up_sub = (ctypes.c_int * 3)(_Nx, _Ny, _Nz + 1)
  _bc_range = _lc.gkyl_range_new(ctypes.c_int(3), _lo_sub, _up_sub)
  _lc.gkyl_sub_range_init(_bc_range, _local_ext, _lo_sub, _up_sub)
  _lc.gkyl_range_release(_local_ext)  # sub-range has copied the linearizer

  _f_arr = _lc.gkyl_array_new_from_buff(
      2, ctypes.c_size_t(_ncomp),
      ctypes.c_size_t((_Nx + 2) * (_Ny + 2) * (_Nz + 2)),
      _ext.ctypes.data_as(ctypes.c_void_p))

  _shift_arr = _lc.gkyl_array_new_from_buff(
      2, ctypes.c_size_t(_shift_ncomp), ctypes.c_size_t(_Nx_shift),
      _shift_vals.ctypes.data_as(ctypes.c_void_p))

  # Create twist-shift updater, apply in-place, then release everything
  _edge = 0 if _bnd == "lower" else 1
  _up = _lib_gk.gkyl_bc_twistshift_new(2, 1, 0, _edge, 3, _bc_range, _ghost,
                                       _basis_ptr, _grid_ptr, None, None,
                                       _shift_arr, _shift_poly_order, False)

  _lib_gk.gkyl_bc_twistshift_advance(_up, _f_arr, _f_arr)

  _lib_gk.gkyl_bc_twistshift_release(_up)
  _lc.gkyl_range_release(_bc_range)
  _lc.gkyl_rect_grid_release(_grid_ptr)
  _lc.gkyl_cart_modal_basis_release(_basis_ptr)
  _lc.gkyl_array_release(_f_arr)
  _lc.gkyl_array_release(_shift_arr)

  # Ghost cell is now filled in _ext; interpolate the DG coefficients
  _iz = 0 if _bnd == "lower" else -1
  _ghost_3d = _ext[1:-1, 1:-1, _iz, :].reshape(_Nx, _Ny, 1, _ncomp)

  _ghost_gdata = pg.GData()
  _ghost_gdata.ctx = dict(_pg_data.ctx)
  _ghost_gdata.push([_xn[0], _xn[1], _xn[2][0:2]], _ghost_3d)

  _ctx_basis = _pg_data.ctx.get('basis_type', 'serendipity')
  _basis_type = 'ms' if _ctx_basis == 'serendipity' else _ctx_basis
  _pg_interp = pg.GInterpModal(_ghost_gdata, _poly_order, _basis_type)
  _x_interp, _data_interp = _pg_interp.interpolate(_comp)
  _ghost_field = _data_interp[:, :, 0, 0]  # (Nx*p, Ny*p)

  _fig, _ax = plt.subplots(figsize=(9, 5.5))
  _ax.set_aspect('equal')
  _pcm = _ax.pcolormesh(_x_interp[0],
                        _x_interp[1],
                        _ghost_field.T,
                        shading='flat',
                        cmap='inferno')
  plt.colorbar(_pcm,
               ax=_ax,
               label=f"{species.value}_{quantity.value} ghost [comp {_comp}]")
  _ax.set_xlabel('x  (m)', fontsize=12)
  _ax.set_ylabel('y  (m)', fontsize=12)
  _bnd_label = "lower" if _bnd == "lower" else "upper"
  _ax.set_title(
      f'Ghost after twist-shift -- {species.value} {quantity.value},  '
      f'comp {_comp},  z {_bnd_label}',
      fontsize=10,
  )
  plt.tight_layout()
  _out3 = mo.as_html(_fig)
  plt.close(_fig)
  _out3
