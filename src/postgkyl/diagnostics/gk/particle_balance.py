"""Gyrokinetic particle-balance diagnostic.

Ported from ``src_bak/postgkyl/apps/gk_particle_balance.py``. Same shape as
:mod:`postgkyl.diagnostics.gk.energy_balance`, but for a single
species and the M0 (density) moment, with no field/apar-energy terms::

    N_err = S - bflux - df/dt

Without the moment files, it falls back to the particle balance Gkeyll writes
when the species requests it (``<name>-<species>_balance_particle.gkyl``).
Without a species, it reads the balance summed over species that Gkeyll
writes when the app requests it (``<name>-balance_particle.gkyl``).
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

# Integrated moments holding the density, as their first component.
_MOMENTS = ("M0", "M0M1M2", "M0M1M2parM2perp", "HamiltonianMoments")
_DENSITY_MOMENT = 0


@dataclass(frozen=True)
class ParticleBalanceTraces:
  """Computed particle-balance time traces (all 1-D, aligned to ``time``).

  Attributes:
    time: Time stamps of the ``fdot`` trace after t=0, where the rates are
      defined.
    fdot: Rate of change of the M0 moment, summed over blocks.
    src: Rate of change from sources, or ``None`` if none was found
      (balance file: if it is zero throughout).
    bflux_tot: Rate of change from boundary particle fluxes, or ``None``
      (balance file: if it is zero throughout).
    mom_err: The particle-balance residual (``None`` when ``relative_error``).
    mom_err_norm: The *relative* residual (only set when
      ``relative_error=True``).
  """

  time: np.ndarray
  fdot: np.ndarray
  src: np.ndarray | None
  bflux_tot: np.ndarray | None
  mom_err: np.ndarray | None
  mom_err_norm: np.ndarray | None = None


def _accumulate(target: np.ndarray | None, addend) -> np.ndarray:
  """Sum ``addend`` into ``target`` (over blocks), copying on first use so
  the caller's array is never mutated in place."""
  addend = np.asarray(addend)
  return addend.copy() if target is None else target + addend


def particle_balance_error(fdot: np.ndarray, src: np.ndarray,
                           bflux_tot: np.ndarray) -> np.ndarray:
  """The particle-balance residual: ``S - bflux - df/dt``."""
  return src - bflux_tot - fdot


def _block_prefix(file_prefix: str, block_idx: int) -> str:
  return file_prefix.replace("*", str(block_idx))


def _moment_file(stem: str) -> str:
  return utils.integrated_moment_file(stem, _MOMENTS)


def _resolve(path: str, override: str | None, default: str,
             block_idx: int) -> str:
  """Resolve a file-family member's path: ``override`` (with ``*``
  substituted for the block index) if given, else the naming-convention
  ``default``."""
  if override is None:
    return default
  return (path + override).replace("*", str(block_idx))


def _moment_file_traces(
    path: str,
    file_prefix: str,
    species: str,
    multib: str,
    relative_error: bool,
    fdot_file: str | None,
    source_file: str | None,
    bflux_files: dict[str, str],
    f_file: str | None,
    dt_file: str | None,
) -> ParticleBalanceTraces:
  """Particle-balance traces from the integrated-moment files, summed over
  blocks."""
  probe = fdot_file or _moment_file(file_prefix + species + "_fdot_")
  blocks = utils.get_block_indices(multib, probe)

  fdot = src = bflux_tot = None
  has_src = has_bflux = False
  time_fdot = None

  for block_idx in blocks:
    block_prefix = _block_prefix(file_prefix, block_idx)

    fdot_name = _resolve(path, fdot_file,
                         _moment_file(block_prefix + species + "_fdot_"),
                         block_idx)
    found, time_fdot, fdot_pb = utils.read_moment_if_present(
        fdot_name, _DENSITY_MOMENT)
    if not found:
      raise FileNotFoundError(f"Required file not found: {fdot_name}")

    src_pb = 0.0 * fdot_pb
    for src_name in (_resolve(path, source_file,
                              _moment_file(block_prefix + species + "_source_"),
                              block_idx),
                     _moment_file(block_prefix + species + "_source_bgk_")):
      found_s, _, v = utils.read_moment_if_present(src_name, _DENSITY_MOMENT)
      if found_s:
        has_src = True
        src_pb = src_pb + v

    bflux_terms = []
    for d in _DIRS:
      for e in _EDGES:
        key = d + e
        bf_name = _resolve(
            path, bflux_files.get(key),
            _moment_file(block_prefix + species + f"_bflux_{d}{e}_"), block_idx)
        found_b, _, v = utils.read_moment_if_present(bf_name, _DENSITY_MOMENT)
        if found_b:
          has_bflux = True
          bflux_terms.append(v)
    bflux_pb = sum(bflux_terms) if bflux_terms else 0.0 * fdot_pb

    fdot = _accumulate(fdot, fdot_pb)
    src = _accumulate(src, src_pb)
    bflux_tot = _accumulate(bflux_tot, bflux_pb)

  # Rates are only defined once a step has been taken: skip t=0.
  time = time_fdot[1:]
  fdot, src, bflux_tot = fdot[1:], src[1:], bflux_tot[1:]
  mom_err = particle_balance_error(fdot, src, bflux_tot)
  mom_err_norm = None
  if relative_error:
    dt_name = _resolve(path, dt_file,
                       file_prefix.replace("_b*", "") + "dt.gkyl", 0)
    _, _, dt, _ = utils.read_time_trace_if_present(dt_name)

    distf = None
    for block_idx in blocks:
      block_prefix = _block_prefix(file_prefix, block_idx)
      f_name = _resolve(path, f_file,
                        _moment_file(block_prefix + species + "_"), block_idx)
      _, _, v = utils.read_moment_if_present(f_name, _DENSITY_MOMENT)
      distf = _accumulate(distf, v)

    mom_err_norm = mom_err * dt / distf[1:]
    mom_err = None

  return ParticleBalanceTraces(time=time,
                               fdot=fdot,
                               src=src if has_src else None,
                               bflux_tot=bflux_tot if has_bflux else None,
                               mom_err=mom_err,
                               mom_err_norm=mom_err_norm)


def _balance_file_traces(file_names: list[str],
                         relative_error: bool) -> ParticleBalanceTraces:
  """Particle-balance traces from Gkeyll's balance files, one per block."""
  time, col = utils.read_balance_files(file_names)
  if relative_error and "mom_err_norm" not in col:
    raise ValueError(
        "The relative particle error of several blocks needs the moment "
        "files: each block's balance file normalizes by that block's "
        "particle number.")
  # Rates are only defined once a step has been taken: skip t=0.
  col = {key: values[1:] for key, values in col.items()}
  src, bflux_tot = (v if np.any(v) else None
                    for v in (col["src_mom"], col["bflux_tot_mom"]))
  return ParticleBalanceTraces(
      time=time[1:],
      fdot=col["mom_dot"],
      src=src,
      bflux_tot=bflux_tot,
      mom_err=None if relative_error else col["mom_err"],
      mom_err_norm=col["mom_err_norm"] if relative_error else None)


def _plot(traces: ParticleBalanceTraces, relative_error: bool, logy: bool,
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
    if traces.bflux_tot is not None:
      h, = ax.plot(time,
                   absy_func(-traces.bflux_tot),
                   linestyle=_LINE_STYLES[1])
      legend_handles.append(h)
      legend_strings.append(
          r"$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$"
      )
    h, = ax.plot(time, absy_func(-traces.fdot), linestyle=_LINE_STYLES[0])
    legend_handles.append(h)
    legend_strings.append(r"$-\dot{f}$")
    h, = ax.plot(time, absy_func(traces.mom_err), linestyle=_LINE_STYLES[3])
    legend_handles.append(h)
    legend_strings.append(r"$E_{\dot{\mathcal{N}}}=$" + "".join(legend_strings))

    ax.legend(legend_handles,
              legend_strings,
              fontsize=_LEGEND_FONT_SIZE,
              frameon=False)

    ylabel_string = ylabel or ""
    title_string = title or r"Particle balance"
  else:
    ax.plot(time, absy_func(traces.mom_err_norm))

    ylabel_string = ylabel or r"$E_{\dot{\mathcal{N}}}~\Delta t/\mathcal{N}$"
    title_string = title or r"Relative error in particle conservation"

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


def particle_balance(
    name: str,
    species: str | None = None,
    *,
    path: str = "./",
    relative_error: bool = False,
    multib: str = "-10",
    fdot_file: str | None = None,
    source_file: str | None = None,
    bflux_files: dict[str, str] | None = None,
    f_file: str | None = None,
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
) -> tuple[plt.Figure, ParticleBalanceTraces]:
  """Plot (and compute) the particle balance of a species, or of all of them.

  Reads the integrated moments ``<name>-<species>_<kind>integrated_<moment>
  .gkyl`` of the first ``<moment>`` found among ``M0``, ``M0M1M2``,
  ``M0M1M2parM2perp`` and ``HamiltonianMoments`` (the density is their first
  component), or the older ``<kind>integrated_moms.gkyl``. ``<kind>`` is
  ``fdot_`` (required) and, only if the run had sources or non-periodic
  boundaries, ``source_``, ``source_bgk_`` and ``bflux_<direction><side>_``.
  If ``relative_error`` is requested, the moments of ``f`` (no ``<kind>``)
  and ``<name>-dt.gkyl`` are also required.

  If no path override is given and the ``fdot_`` file is missing, the terms
  are instead read from ``<name>-<species>_balance_particle.gkyl``, which
  Gkeyll writes when the species requests the particle balance. Its relative
  error is only available for a single block.

  Without ``species``, the balance summed over species is read from
  ``<name>-balance_particle.gkyl``, which Gkeyll writes when the app
  requests the particle balance.

  The rates are only defined once a step has been taken, so t=0 is skipped.

  Args:
    name: Simulation name (also the file prefix).
    species: Species name; if omitted, the sum over species.
    path: Directory holding the simulation output.
    relative_error: Plot the relative error instead of every balance term.
    multib: ``"-10"`` (default) for a single block; ``"-1"`` to discover
      every block; otherwise a comma list or ``'start:stop[:step]'`` slice
      of block indices.
    fdot_file: Explicit distribution derivative moments path override.
    source_file: Explicit source moments path override (not of the BGK
      source).
    bflux_files: Optional per-boundary path overrides, keyed by
      ``"<direction><side>"``; unlisted boundaries use the naming
      convention.
    f_file: Explicit integrated distribution moments path override.
    dt_file: Explicit time-step history path override. In path overrides,
      ``*`` stands for the block index.
    logy: Log-scale the y axis.
    absy: Take the absolute value of every trace before plotting.
    xlabel: Horizontal-axis label.
    ylabel: Vertical-axis label.
    title: Figure title.
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

  overrides = (fdot_file, source_file, bflux_files, f_file, dt_file)
  if species is None and any(overrides):
    raise ValueError("Path overrides select the moment files, which need a "
                     "species.")
  sp_prefix = file_prefix + (species + "_" if species is not None else "")
  balance_name = sp_prefix + "balance_particle.gkyl"
  blocks = [] if any(overrides) else utils.balance_file_blocks(
      multib, balance_name,
      [_moment_file(sp_prefix + "fdot_")] if species is not None else [])

  if blocks:
    traces = _balance_file_traces(
        [balance_name.replace("*", str(b)) for b in blocks], relative_error)
  elif species is None:
    raise FileNotFoundError(f"Required file not found: {balance_name}")
  else:
    traces = _moment_file_traces(path, file_prefix, species, multib,
                                 relative_error, fdot_file, source_file,
                                 bflux_files or {}, f_file, dt_file)

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
