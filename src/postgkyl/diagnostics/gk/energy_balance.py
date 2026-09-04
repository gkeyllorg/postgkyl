"""Gyrokinetic energy-balance diagnostic.

Ported from ``src_bak/postgkyl/apps/gk_energy_balance.py``. Reads the
integrated time-trace files a gyrokinetic simulation writes (field/apar
energy rate of change, integrated Hamiltonian moments of ``df/dt``, of the
source(s), and of the boundary particle fluxes), sums them over species and
(for multiblock runs) blocks, and plots the energy-balance residual::

    E_err = S - bflux - (df/dt - dfield/dt [- dapar/dt])

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

# Hamiltonian-moments files store (M0, M1, M2) per component; energy balance
# uses the M2 (Hamiltonian/energy) moment, index 2.
_ENERGY_MOMENT = 2


@dataclass(frozen=True)
class EnergyBalanceTraces:
  """Computed energy-balance time traces (all 1-D, aligned to ``time``).

  Attributes:
    time: Time stamps of the (dominant) ``fdot`` trace.
    fdot: Rate of change of the Hamiltonian moment of the distribution
      function, summed over species and blocks.
    src: Rate of change from sources, or ``None`` if no source file was
      found for any species/block.
    bflux_tot: Rate of change from boundary particle fluxes, or ``None`` if
      none were found.
    field_dot: Rate of change of the field energy.
    apar_dot: Rate of change of the vector-potential energy (electromagnetic
      simulations only), or ``None``.
    mom_err: The energy-balance residual (``None`` when ``relative_error``).
    mom_err_norm: The *relative* energy-balance residual (only set when
      ``relative_error=True``; ``None`` otherwise).
  """

  time: np.ndarray
  fdot: np.ndarray
  src: np.ndarray | None
  bflux_tot: np.ndarray | None
  field_dot: np.ndarray
  apar_dot: np.ndarray | None
  mom_err: np.ndarray | None
  mom_err_norm: np.ndarray | None = None
# end


def _accumulate(target: np.ndarray | None, addend) -> np.ndarray:
  """Sum ``addend`` into ``target`` (over species/blocks), copying on first
  use so the caller's array is never mutated in place."""
  addend = np.asarray(addend)
  return addend.copy() if target is None else target + addend
# end


def energy_balance_error(fdot: np.ndarray, src: np.ndarray, bflux_tot: np.ndarray,
    field_dot: np.ndarray, apar_dot: np.ndarray | None = None) -> np.ndarray:
  """The energy-balance residual: ``S - bflux - (df/dt - dfield/dt [- dapar/dt])``.

  Pure array arithmetic -- the one formula every energy-balance trace
  (single- or multi-block, single- or multi-species) reduces to once the
  per-species/per-block sums are in hand.
  """
  fdot_terms = fdot - field_dot
  if apar_dot is not None:
    fdot_terms = fdot_terms - apar_dot
  # end
  return src - bflux_tot - fdot_terms
# end


def _block_prefix(file_prefix: str, block_idx: int) -> str:
  return file_prefix.replace("*", str(block_idx))
# end


def _resolve(path: str, override: str | None, default: str,
    block_idx: int, species: str | None = None) -> str:
  """Resolve a file-family member's path: ``override`` (with ``*``
  substituted for the block index, then the species) if given, else the
  naming-convention ``default``."""
  if override is None:
    return default
  # end
  resolved = (path + override).replace("*", str(block_idx), 1)
  if species is not None:
    resolved = resolved.replace("*", species)
  # end
  return resolved
# end


def energy_balance(
    name: str,
    species: list[str],
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

  Requires, per species (named ``<name>-<species_name>``): an
  ``_fdot_integrated_moms.gkyl`` file, and (only if the run had sources or
  non-periodic boundaries) ``_source_integrated_moms.gkyl`` and
  ``_bflux_<direction><side>_integrated_HamiltonianMoments.gkyl`` files. A
  ``<name>-field_energy_dot.gkyl`` file is required; ``<name>-apar_energy_dot
  .gkyl`` is read if present (electromagnetic simulations). If
  ``relative_error`` is requested, the corresponding non-``_dot``
  (``_integrated_moms.gkyl``/``field_energy.gkyl``/``apar_energy.gkyl``) and
  ``<name>-dt.gkyl`` files are also required.

  Args:
    name: Simulation name (also the file prefix).
    species: Species names to sum over.
    path: Directory holding the simulation output.
    relative_error: Plot the relative error instead of every balance term.
    multib: ``"-10"`` (default) for a single block; ``"-1"`` to discover
      every block; otherwise a comma list or ``'start:stop[:step]'`` slice
      of block indices (see :func:`~postgkyl.diagnostics.gk.utils.
      get_block_indices`).
    field_dot_file: Explicit field-energy derivative path override.
    apar_dot_file: Explicit parallel-vector-potential energy derivative path.
    fdot_file: Explicit distribution derivative moments path override.
    source_file: Explicit source moments path override.
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
  """
  path = path.rstrip("/") + "/"
  bflux_files = bflux_files or {}

  file_prefix = f"{path}{name}-" if multib == "-10" else f"{path}{name}_b*-"
  probe = fdot_file or (file_prefix + species[0] + "_fdot_integrated_moms.gkyl")
  blocks = utils.get_block_indices(multib, probe)

  fig = plt.figure(figsize=(7.5, 4.5))
  ax = fig.add_axes([0.11 + indent_left, 0.15, 0.87 + add_width, 0.78])
  ax.plot([-1.0, 1.0], [0.0, 0.0], color="grey", linestyle=":", linewidth=1)

  absy_func = np.abs if absy else (lambda v: v)

  field_dot = apar_dot = fdot = src = bflux_tot = None
  has_apar_dot = has_src = has_bflux = False
  time_fdot = time_field_dot = time_apar_dot = time_bflux_tot = None

  for block_idx in blocks:
    block_prefix = _block_prefix(file_prefix, block_idx)

    fd_name = _resolve(path, field_dot_file,
        block_prefix + "field_energy_dot.gkyl", block_idx)
    found, t, v, _ = utils.read_time_trace_if_present(fd_name)
    if not found:
      raise FileNotFoundError(f"Required file not found: {fd_name}")
    # end
    time_field_dot, field_dot_pb = t, v

    ad_name = _resolve(path, apar_dot_file,
        block_prefix + "apar_energy_dot.gkyl", block_idx)
    has_apar_dot, t, v, _ = utils.read_time_trace_if_present(ad_name)
    if has_apar_dot:
      time_apar_dot, apar_dot_pb = t, v
    # end

    fdot_pb = src_pb = bflux_tot_pb = None
    for sp in species:
      fdot_name = _resolve(path, fdot_file,
          block_prefix + sp + "_fdot_integrated_moms.gkyl", block_idx, sp)
      found, t, v, _ = utils.read_time_trace_if_present(fdot_name)
      if not found:
        raise FileNotFoundError(f"Required file not found: {fdot_name}")
      # end
      time_fdot = t
      fdot_sp = v[:, _ENERGY_MOMENT]

      src_name = _resolve(path, source_file,
          block_prefix + sp + "_source_integrated_moms.gkyl", block_idx, sp)
      has_src, t, v, _ = utils.read_time_trace_if_present(src_name)
      if has_src:
        src_sp = v[:, _ENERGY_MOMENT]
      # end
      else:
        src_sp = 0.0 * fdot_sp
      # end

      bflux_terms = []
      for d in _DIRS:
        for e in _EDGES:
          key = d + e
          bf_name = _resolve(path, bflux_files.get(key),
              block_prefix + sp + f"_bflux_{d}{e}_integrated_HamiltonianMoments.gkyl",
              block_idx, sp)
          found_b, t, v, _ = utils.read_time_trace_if_present(bf_name)
          if found_b:
            has_bflux = True
            time_bflux_tot = t
            bflux_terms.append(v[:, _ENERGY_MOMENT])
          # end
        # end
      # end
      bflux_sp = sum(bflux_terms) if bflux_terms else 0.0 * fdot_sp

      fdot_pb = _accumulate(fdot_pb, fdot_sp)
      src_pb = _accumulate(src_pb, src_sp)
      bflux_tot_pb = _accumulate(bflux_tot_pb, bflux_sp)
    # end

    field_dot = _accumulate(field_dot, field_dot_pb)
    if has_apar_dot:
      apar_dot = _accumulate(apar_dot, apar_dot_pb)
    # end
    fdot = _accumulate(fdot, fdot_pb)
    src = _accumulate(src, src_pb)
    bflux_tot = _accumulate(bflux_tot, bflux_tot_pb)
  # end

  legend_handles = []
  legend_strings = []

  if not relative_error:
    src = src.copy()
    src[0] = 0.0  # No fdot/bflux contribution at t=0.

    mom_err = energy_balance_error(fdot, src, bflux_tot, field_dot,
        apar_dot if has_apar_dot else None)

    if has_src:
      h, = ax.plot(time_fdot, absy_func(src), linestyle=_LINE_STYLES[2])
      legend_handles.append(h)
      legend_strings.append(r"$\mathcal{S}$")
    # end
    if has_bflux:
      h, = ax.plot(time_bflux_tot, absy_func(-bflux_tot), linestyle=_LINE_STYLES[1])
      legend_handles.append(h)
      legend_strings.append(r"$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$")
    # end
    h, = ax.plot(time_field_dot, absy_func(-field_dot), linestyle=":",
        marker="+", markevery=8)
    legend_handles.append(h)
    legend_strings.append(r"$-\dot{\phi}$")
    if has_apar_dot:
      h, = ax.plot(time_apar_dot, absy_func(-apar_dot), linestyle=":",
          marker="+", markevery=8)
      legend_handles.append(h)
      legend_strings.append(r"$-\dot{A}_{\parallel}$")
    # end
    h, = ax.plot(time_fdot, absy_func(-fdot), linestyle=_LINE_STYLES[0])
    legend_handles.append(h)
    legend_strings.append(r"$-\dot{f}$")
    h, = ax.plot(time_fdot, absy_func(mom_err), linestyle=_LINE_STYLES[3])
    legend_handles.append(h)
    legend_strings.append(r"$E_{\dot{\mathcal{E}}}=$" + "".join(legend_strings))

    ax.legend(legend_handles, legend_strings, fontsize=_LEGEND_FONT_SIZE, frameon=False)

    ylabel_string = ylabel or ""
    title_string = title or r"Energy balance"
    mom_err_norm = None
  # end
  else:
    dt_name = _resolve(path, dt_file, file_prefix.replace("_b*", "") + "dt.gkyl", 0)
    _, time_dt, dt, _ = utils.read_time_trace_if_present(dt_name)

    field = apar = distf = None
    for block_idx in blocks:
      block_prefix = _block_prefix(file_prefix, block_idx)

      fld_name = _resolve(path, field_file, block_prefix + "field_energy.gkyl", block_idx)
      has_field, t, v, _ = utils.read_time_trace_if_present(fld_name)
      field_pb = v if has_field else None

      ap_name = _resolve(path, apar_file, block_prefix + "apar_energy.gkyl", block_idx)
      has_apar, t, v, _ = utils.read_time_trace_if_present(ap_name)
      apar_pb = v if has_apar else None

      distf_pb = None
      for sp in species:
        f_name = _resolve(path, f_file, block_prefix + sp + "_integrated_moms.gkyl",
            block_idx, sp)
        _, t, v, _ = utils.read_time_trace_if_present(f_name)
        distf_pb = _accumulate(distf_pb, v[:, _ENERGY_MOMENT])
      # end

      field = _accumulate(field, field_pb)
      if has_apar:
        apar = _accumulate(apar, apar_pb)
      # end
      distf = _accumulate(distf, distf_pb)
    # end

    field, field_dot = field[1:], field_dot[1:]
    if has_apar:
      apar, apar_dot = apar[1:], apar_dot[1:]
    # end
    fdot, src, bflux_tot, distf = fdot[1:], src[1:], bflux_tot[1:], distf[1:]

    mom_err = energy_balance_error(fdot, src, bflux_tot, field_dot,
        apar_dot if has_apar else None)
    denom = (distf - field - apar) if has_apar else (distf - field)
    mom_err_norm = mom_err * dt / denom

    ax.plot(time_dt, absy_func(mom_err_norm))

    ylabel_string = ylabel or r"$E_{\dot{\mathcal{E}}}~\Delta t/\mathcal{E}$"
    title_string = title or r"Relative error in energy conservation"
    mom_err = None
  # end

  if logy:
    ax.set_yscale("log")
  # end
  if absy and ylabel_string:
    ylabel_string = r"|" + ylabel_string + r"|"
  # end

  ax.set_xlabel(xlabel, fontsize=_XY_LABEL_FONT_SIZE)
  ax.set_ylabel(ylabel_string, fontsize=_XY_LABEL_FONT_SIZE)
  ax.set_title(title_string, fontsize=_TITLE_FONT_SIZE)
  ax.set_xlim(time_fdot[0], time_fdot[-1])
  utils.set_tick_font_size(ax, _TICK_FONT_SIZE)

  if saveas:
    fig.savefig(saveas)
  # end
  if show:
    plt.show()
  # end

  traces = EnergyBalanceTraces(
      time=time_fdot, fdot=fdot, src=src if has_src else None,
      bflux_tot=bflux_tot if has_bflux else None, field_dot=field_dot,
      apar_dot=apar_dot if has_apar_dot else None,
      mom_err=mom_err, mom_err_norm=mom_err_norm)
  return fig, traces
# end
