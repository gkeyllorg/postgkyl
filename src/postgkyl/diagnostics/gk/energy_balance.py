"""Gyrokinetic energy-balance diagnostic.

Ported from ``src_bak/postgkyl/apps/gk_energy_balance.py``. Reads the
integrated time-trace files a gyrokinetic simulation writes (field/apar
energy rate of change, integrated Hamiltonian moments of ``df/dt``, of the
source(s), and of the boundary particle fluxes), sums them over species and
(for multiblock runs) blocks, and plots the energy-balance residual::

    E_err = S + S_BGK - bflux - (df/dt - dfield/dt [- dapar/dt])

with ``S`` the source and ``S_BGK`` the BGK source. Without those files, it
falls back to the energy balance Gkeyll writes when the app requests it
(``<name>-balance_energy.gkyl``), whose residual also contains the
background-charge term of adiabatic species::

    E_err = S + S_BGK - bflux - (df/dt - dfield/dt + dbg/dt)

Typer options become explicit keyword-only parameters; the old CLI's dataset
stack (``ctx.obj.data``) and ``verb_print`` echo are dropped -- the computed
traces come back as an :class:`EnergyBalanceTraces` alongside the Figure.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from . import utils

_DIRS = ("x", "y", "z")
_EDGES = ("lower", "upper")
_LINE_STYLES = ("-", "--", ":", "-.")
_XY_LABEL_FONT_SIZE = 17
_TITLE_FONT_SIZE = 17
_TICK_FONT_SIZE = 14
_LEGEND_FONT_SIZE = 14

# Hamiltonian moments store (M0, M1, H); energy balance uses H, index 2.
_MOMENTS = ("HamiltonianMoments", )
_ENERGY_MOMENT = 2


@dataclass(frozen=True)
class EnergyBalanceTraces:
  """Computed energy-balance time traces (all 1-D, aligned to ``time``).

  Attributes:
    time: Time stamps of the ``fdot`` trace after t=0, where the rates are
      defined.
    fdot: Rate of change of the Hamiltonian moment of the distribution
      function, summed over species and blocks.
    src: Rate of change from the sources, or ``None`` if no source file was
      found for any species/block (balance file: if it is zero throughout).
    bflux_tot: Rate of change from boundary particle fluxes, or ``None`` if
      none were found (balance file: if it is zero throughout).
    field_dot: Rate of change of the field energy.
    apar_dot: Rate of change of the vector-potential energy (electromagnetic
      simulations only), or ``None``.
    mom_err: The energy-balance residual (``None`` when ``relative_error``).
    mom_err_norm: The *relative* energy-balance residual (only set when
      ``relative_error=True``; ``None`` otherwise).
    bg_dot: Rate of change of the background-charge energy ``int rho_bg phi``
      of adiabatic species, or ``None``. Only Gkeyll's balance file has it,
      and it is ``None`` there if it is zero throughout.
    src_bgk: Rate of change from the BGK sources, or ``None`` if none was
      found (balance file: if it is zero throughout, or the file predates
      the ``src_bgk_mom`` column and includes it in ``src``).
  """

  time: np.ndarray
  fdot: np.ndarray
  src: np.ndarray | None
  bflux_tot: np.ndarray | None
  field_dot: np.ndarray
  apar_dot: np.ndarray | None
  mom_err: np.ndarray | None
  mom_err_norm: np.ndarray | None = None
  bg_dot: np.ndarray | None = None
  src_bgk: np.ndarray | None = None


def _accumulate(target: np.ndarray | None, addend) -> np.ndarray:
  """Sum ``addend`` into ``target`` (over species/blocks), copying on first
  use so the caller's array is never mutated in place."""
  addend = np.asarray(addend)
  return addend.copy() if target is None else target + addend


def energy_balance_error(fdot: np.ndarray,
                         src: np.ndarray,
                         bflux_tot: np.ndarray,
                         field_dot: np.ndarray,
                         apar_dot: np.ndarray | None = None) -> np.ndarray:
  """The energy-balance residual: ``S - bflux - (df/dt - dfield/dt [- dapar/dt])``.

  Pure array arithmetic -- the one formula every energy-balance trace
  (single- or multi-block, single- or multi-species) reduces to once the
  per-species/per-block sums are in hand.
  """
  fdot_terms = fdot - field_dot
  if apar_dot is not None:
    fdot_terms = fdot_terms - apar_dot
  return src - bflux_tot - fdot_terms


def _block_prefix(file_prefix: str, block_idx: int) -> str:
  return file_prefix.replace("*", str(block_idx))


def _moment_file(stem: str) -> str:
  return utils.integrated_moment_file(stem, _MOMENTS)


def _resolve(path: str,
             override: str | None,
             default: str,
             block_idx: int,
             species: str | None = None) -> str:
  """Resolve a file-family member's path: ``override`` (with ``*``
  substituted for the block index, then the species) if given, else the
  naming-convention ``default``."""
  if override is None:
    return default
  resolved = (path + override).replace("*", str(block_idx), 1)
  if species is not None:
    resolved = resolved.replace("*", species)
  return resolved


def _moment_file_traces(
    path: str,
    file_prefix: str,
    species: list[str],
    multib: str,
    relative_error: bool,
    field_dot_file: str | None,
    apar_dot_file: str | None,
    fdot_file: str | None,
    source_file: str | None,
    bflux_files: dict[str, str],
    f_file: str | None,
    field_file: str | None,
    apar_file: str | None,
    dt_file: str | None,
) -> EnergyBalanceTraces:
  """Energy-balance traces from the integrated-moment and field-energy files,
  summed over species and blocks."""
  probe = fdot_file or _moment_file(file_prefix + species[0] + "_fdot_")
  blocks = utils.get_block_indices(multib, probe)

  field_dot = apar_dot = fdot = src = src_bgk = bflux_tot = None
  has_apar_dot = has_src = has_src_bgk = has_bflux = False
  time_fdot = None

  for block_idx in blocks:
    block_prefix = _block_prefix(file_prefix, block_idx)

    fd_name = _resolve(path, field_dot_file,
                       block_prefix + "field_energy_dot.gkyl", block_idx)
    found, t, v, _ = utils.read_time_trace_if_present(fd_name)
    if not found:
      raise FileNotFoundError(f"Required file not found: {fd_name}")
    field_dot_pb = np.atleast_1d(v)  # A single sample is squeezed to 0-D.

    ad_name = _resolve(path, apar_dot_file,
                       block_prefix + "apar_energy_dot.gkyl", block_idx)
    has_apar_dot, t, v, _ = utils.read_time_trace_if_present(ad_name)
    if has_apar_dot:
      apar_dot_pb = np.atleast_1d(v)

    fdot_pb = src_pb = src_bgk_pb = bflux_tot_pb = None
    for sp in species:
      fdot_name = _resolve(path, fdot_file,
                           _moment_file(block_prefix + sp + "_fdot_"),
                           block_idx, sp)
      found, time_fdot, fdot_sp = utils.read_moment_if_present(
          fdot_name, _ENERGY_MOMENT)
      if not found:
        raise FileNotFoundError(f"Required file not found: {fdot_name}")

      src_name = _resolve(path, source_file,
                          _moment_file(block_prefix + sp + "_source_"),
                          block_idx, sp)
      found_s, _, v = utils.read_moment_if_present(src_name, _ENERGY_MOMENT)
      has_src = has_src or found_s
      src_sp = v if found_s else 0.0 * fdot_sp

      found_s, _, v = utils.read_moment_if_present(
          _moment_file(block_prefix + sp + "_source_bgk_"), _ENERGY_MOMENT)
      has_src_bgk = has_src_bgk or found_s
      src_bgk_sp = v if found_s else 0.0 * fdot_sp

      bflux_terms = []
      for d in _DIRS:
        for e in _EDGES:
          key = d + e
          bf_name = _resolve(
              path, bflux_files.get(key),
              _moment_file(block_prefix + sp + f"_bflux_{d}{e}_"), block_idx,
              sp)
          found_b, _, v = utils.read_moment_if_present(bf_name, _ENERGY_MOMENT)
          if found_b:
            has_bflux = True
            bflux_terms.append(v)
      bflux_sp = sum(bflux_terms) if bflux_terms else 0.0 * fdot_sp

      fdot_pb = _accumulate(fdot_pb, fdot_sp)
      src_pb = _accumulate(src_pb, src_sp)
      src_bgk_pb = _accumulate(src_bgk_pb, src_bgk_sp)
      bflux_tot_pb = _accumulate(bflux_tot_pb, bflux_sp)

    field_dot = _accumulate(field_dot, field_dot_pb)
    if has_apar_dot:
      apar_dot = _accumulate(apar_dot, apar_dot_pb)
    fdot = _accumulate(fdot, fdot_pb)
    src = _accumulate(src, src_pb)
    src_bgk = _accumulate(src_bgk, src_bgk_pb)
    bflux_tot = _accumulate(bflux_tot, bflux_tot_pb)

  # Rates are only defined once a step has been taken: skip t=0.
  time = time_fdot[1:]
  fdot, src, src_bgk = fdot[1:], src[1:], src_bgk[1:]
  bflux_tot = bflux_tot[1:]
  field_dot = field_dot[1:]
  if has_apar_dot:
    apar_dot = apar_dot[1:]

  if not relative_error:
    mom_err = energy_balance_error(fdot, src + src_bgk, bflux_tot, field_dot,
                                   apar_dot if has_apar_dot else None)
    mom_err_norm = None
  else:
    dt_name = _resolve(path, dt_file,
                       file_prefix.replace("_b*", "") + "dt.gkyl", 0)
    _, _, dt, _ = utils.read_time_trace_if_present(dt_name)

    field = apar = distf = None
    for block_idx in blocks:
      block_prefix = _block_prefix(file_prefix, block_idx)

      fld_name = _resolve(path, field_file, block_prefix + "field_energy.gkyl",
                          block_idx)
      has_field, t, v, _ = utils.read_time_trace_if_present(fld_name)
      field_pb = v if has_field else None

      ap_name = _resolve(path, apar_file, block_prefix + "apar_energy.gkyl",
                         block_idx)
      has_apar, t, v, _ = utils.read_time_trace_if_present(ap_name)
      apar_pb = v if has_apar else None

      distf_pb = None
      for sp in species:
        f_name = _resolve(path, f_file, _moment_file(block_prefix + sp + "_"),
                          block_idx, sp)
        _, _, v = utils.read_moment_if_present(f_name, _ENERGY_MOMENT)
        distf_pb = _accumulate(distf_pb, v)

      field = _accumulate(field, field_pb)
      if has_apar:
        apar = _accumulate(apar, apar_pb)
      distf = _accumulate(distf, distf_pb)

    mom_err = energy_balance_error(fdot, src + src_bgk, bflux_tot, field_dot,
                                   apar_dot if has_apar else None)
    denom = (distf - field - apar) if has_apar else (distf - field)
    mom_err_norm = mom_err * dt / denom[1:]
    mom_err = None

  return EnergyBalanceTraces(time=time,
                             fdot=fdot,
                             src=src if has_src else None,
                             bflux_tot=bflux_tot if has_bflux else None,
                             field_dot=field_dot,
                             apar_dot=apar_dot if has_apar_dot else None,
                             mom_err=mom_err,
                             mom_err_norm=mom_err_norm,
                             src_bgk=src_bgk if has_src_bgk else None)


def _balance_file_traces(file_names: list[str],
                         relative_error: bool) -> EnergyBalanceTraces:
  """Energy-balance traces from Gkeyll's balance files, one per block."""
  time, col = utils.read_balance_files(file_names)
  if relative_error and "mom_err_norm" not in col:
    raise ValueError(
        "The relative energy error of several blocks needs the moment files: "
        "each block's balance file normalizes by that block's energy.")
  # Rates are only defined once a step has been taken: skip t=0.
  col = {key: values[1:] for key, values in col.items()}
  # Older balance files have no src_bgk_mom: their src_mom includes it.
  src, src_bgk, bflux_tot, bg_dot = (v if v is not None and np.any(v) else None
                                     for v in (col["src_mom"],
                                               col.get("src_bgk_mom"),
                                               col["bflux_tot_mom"],
                                               col["bg_dot"]))
  return EnergyBalanceTraces(
      time=time[1:],
      fdot=col["mom_dot"],
      src=src,
      bflux_tot=bflux_tot,
      field_dot=col["field_dot"],
      apar_dot=None,
      mom_err=None if relative_error else col["mom_err"],
      mom_err_norm=col["mom_err_norm"] if relative_error else None,
      bg_dot=bg_dot,
      src_bgk=src_bgk)


def _plot(traces: EnergyBalanceTraces, relative_error: bool, logy: bool,
          absy: bool, xlabel: str, ylabel: str | None, title: str | None,
          indent_left: float, add_width: float) -> plt.Figure:
  """Plot every balance term and the residual, or the relative residual."""
  fig = plt.figure(figsize=(7.5, 4.5))
  ax = fig.add_axes([0.11 + indent_left, 0.15, 0.87 + add_width, 0.78])
  ax.plot([-1.0, 1.0], [0.0, 0.0], color="grey", linestyle=":", linewidth=1)

  absy_func = np.abs if absy else (lambda v: v)
  time = traces.time

  if not relative_error:
    legend_handles = []
    legend_strings = []

    if traces.src is not None:
      h, = ax.plot(time, absy_func(traces.src), linestyle=_LINE_STYLES[2])
      legend_handles.append(h)
      legend_strings.append(r"$\mathcal{S}$")
    if traces.src_bgk is not None:
      h, = ax.plot(time,
                   absy_func(traces.src_bgk),
                   linestyle=_LINE_STYLES[2],
                   marker="o",
                   markevery=8,
                   fillstyle="none")
      legend_handles.append(h)
      legend_strings.append(r"$\mathcal{S}_{\mathrm{BGK}}$")
    if traces.bflux_tot is not None:
      h, = ax.plot(time,
                   absy_func(-traces.bflux_tot),
                   linestyle=_LINE_STYLES[1])
      legend_handles.append(h)
      legend_strings.append(
          r"$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$"
      )
    h, = ax.plot(time,
                 absy_func(-traces.field_dot),
                 linestyle=":",
                 marker="+",
                 markevery=8)
    legend_handles.append(h)
    legend_strings.append(r"$-\dot{\phi}$")
    if traces.apar_dot is not None:
      h, = ax.plot(time,
                   absy_func(-traces.apar_dot),
                   linestyle=":",
                   marker="+",
                   markevery=8)
      legend_handles.append(h)
      legend_strings.append(r"$-\dot{A}_{\parallel}$")
    if traces.bg_dot is not None:
      h, = ax.plot(time,
                   absy_func(-traces.bg_dot),
                   linestyle=":",
                   marker="x",
                   markevery=8)
      legend_handles.append(h)
      legend_strings.append(
          r"$-\frac{\mathrm{d}}{\mathrm{d}t}\int\rho_{bg}\phi$")
    h, = ax.plot(time, absy_func(-traces.fdot), linestyle=_LINE_STYLES[0])
    legend_handles.append(h)
    legend_strings.append(r"$-\dot{f}$")
    h, = ax.plot(time, absy_func(traces.mom_err), linestyle=_LINE_STYLES[3])
    legend_handles.append(h)
    legend_strings.append(r"$E_{\dot{\mathcal{E}}}=$" +
                          utils.sum_label(legend_strings))

    ax.legend(legend_handles,
              legend_strings,
              fontsize=_LEGEND_FONT_SIZE,
              frameon=False)

    ylabel_string = ylabel or ""
    title_string = title or r"Energy balance"
  else:
    ax.plot(time, absy_func(traces.mom_err_norm))

    ylabel_string = ylabel or r"$E_{\dot{\mathcal{E}}}~\Delta t/\mathcal{E}$"
    title_string = title or r"Relative error in energy conservation"

  if logy:
    ax.set_yscale("log")
  if absy and ylabel_string:
    ylabel_string = r"|" + ylabel_string + r"|"

  ax.set_xlabel(xlabel, fontsize=_XY_LABEL_FONT_SIZE)
  ax.set_ylabel(ylabel_string, fontsize=_XY_LABEL_FONT_SIZE)
  ax.set_title(title_string, fontsize=_TITLE_FONT_SIZE)
  ax.set_xlim(time[0], time[-1])
  utils.set_tick_font_size(ax, _TICK_FONT_SIZE)
  return fig


def energy_balance(
    name: str,
    species: list[str] | None = None,
    *,
    path: str = "./",
    relative_error: bool = False,
    multib: str = "-10",
    field_dot_file: str | None = None,
    apar_dot_file: str | None = None,
    fdot_file: str | None = None,
    source_file: str | None = None,
    bflux_files: dict[str, str] | None = None,
    f_file: str | None = None,
    field_file: str | None = None,
    apar_file: str | None = None,
    dt_file: str | None = None,
    logy: bool = False,
    absy: bool = False,
    xlabel: str = "Time (s)",
    ylabel: str | None = None,
    title: str | None = None,
    indent_left: float = 0.0,
    add_width: float = 0.0,
    show: bool = False,
    saveas: str | None = None,
) -> tuple[plt.Figure, EnergyBalanceTraces]:
  """Plot (and compute) the energy balance of a gyrokinetic simulation.

  Reads, per species, the Hamiltonian moments
  ``<name>-<species>_<kind>integrated_HamiltonianMoments.gkyl`` (or the older
  ``<kind>integrated_moms.gkyl``): ``<kind>`` is ``fdot_`` (required) and,
  only if the run had sources or non-periodic boundaries, ``source_``,
  ``source_bgk_`` and ``bflux_<direction><side>_``. A
  ``<name>-field_energy_dot.gkyl`` file is required; ``<name>-apar_energy_dot
  .gkyl`` is read if present (electromagnetic simulations). If
  ``relative_error`` is requested, the moments of ``f`` (no ``<kind>``),
  ``field_energy.gkyl``, ``apar_energy.gkyl`` (if present) and
  ``<name>-dt.gkyl`` are also required.

  If no path override is given and an ``fdot_`` or ``field_energy_dot.gkyl``
  file is missing, the terms are instead read from
  ``<name>-balance_energy.gkyl``, which Gkeyll writes when the app requests
  the energy balance. That file already sums over every species (``species``
  is not used) and adds the background-charge term ``bg_dot`` of adiabatic
  species to the residual. Its relative error is only available for a
  single block. Older balance files have no ``src_bgk_mom`` column: their
  ``src_mom`` includes the BGK source. Without ``species``, only that file
  is read.

  The rates are only defined once a step has been taken, so t=0 is skipped.

  Args:
    name: Simulation name (also the file prefix).
    species: Species names to sum over; if omitted, only the balance file
      is read.
    path: Directory holding the simulation output.
    relative_error: Plot the relative error instead of every balance term.
    multib: ``"-10"`` (default) for a single block; ``"-1"`` to discover
      every block; otherwise a comma list or ``'start:stop[:step]'`` slice
      of block indices (see :func:`~postgkyl.diagnostics.gk.utils.
      get_block_indices`).
    field_dot_file: Explicit field-energy derivative path override.
    apar_dot_file: Explicit parallel-vector-potential energy derivative path.
    fdot_file: Explicit distribution derivative moments path override.
    source_file: Explicit source moments path override (not of the BGK
      source).
    bflux_files: Optional per-boundary path overrides, keyed by
      ``"<direction><side>"`` (e.g. ``"xlower"``); unlisted boundaries use
      the naming convention.
    f_file: Explicit integrated distribution moments path override.
    field_file: Explicit field-energy path override.
    apar_file: Explicit parallel-vector-potential energy path override.
    dt_file: Explicit time-step history path override. In path overrides,
      ``*`` stands for block index and, for per-species files, species name.
    logy: Log-scale the y axis.
    absy: Take the absolute value of every trace before plotting.
    xlabel: Horizontal-axis label.
    ylabel: Vertical-axis label; use the derived formula when ``None``.
    title: Figure title; use the balance description when ``None``.
    indent_left: Horizontal axes-position adjustment in figure units.
    add_width: Axes-width adjustment in figure units.
    show: Call ``plt.show()`` before returning.
    saveas: If given, save the figure to this path.

  Returns:
    ``(figure, traces)``.

  Raises:
    FileNotFoundError: if a required file family is missing.
    ValueError: if the relative error of several blocks is requested from
      balance files, a path override is given without ``species``, or no
      step was taken.
  """
  path = path.rstrip("/") + "/"
  file_prefix = f"{path}{name}-" if multib == "-10" else f"{path}{name}_b*-"

  overrides = (field_dot_file, apar_dot_file, fdot_file, source_file,
               bflux_files, f_file, field_file, apar_file, dt_file)
  if not species and any(overrides):
    raise ValueError("Path overrides select the moment files, which need the "
                     "species.")
  balance_name = file_prefix + "balance_energy.gkyl"
  moment_files = ([_moment_file(file_prefix + sp + "_fdot_")
                   for sp in species] +
                  [file_prefix + "field_energy_dot.gkyl"] if species else [])
  blocks = [] if any(overrides) else utils.balance_file_blocks(
      multib, balance_name, moment_files)

  if blocks:
    traces = _balance_file_traces(
        [balance_name.replace("*", str(b)) for b in blocks], relative_error)
  elif not species:
    raise FileNotFoundError(f"Required file not found: {balance_name}")
  else:
    traces = _moment_file_traces(path, file_prefix, species, multib,
                                 relative_error, field_dot_file, apar_dot_file,
                                 fdot_file, source_file, bflux_files or {},
                                 f_file, field_file, apar_file, dt_file)

  if traces.time.size == 0:
    raise ValueError("No time step was taken: the balance is only defined "
                     "after t=0.")
  fig = _plot(traces, relative_error, logy, absy, xlabel, ylabel, title,
              indent_left, add_width)
  if saveas:
    fig.savefig(saveas)
  if show:
    plt.show()
  return fig, traces
