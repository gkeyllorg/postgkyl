"""Gyrokinetic particle-balance diagnostic.

Ported from ``src_bak/postgkyl/apps/gk_particle_balance.py``. Same shape as
:mod:`postgkyl.diagnostics.gyrokinetics.energy_balance`, but for a single
species and the M0 (density) moment, with no field/apar-energy terms::

    N_err = S - bflux - df/dt
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

# Integrated-moments files store (M0, M1, M2, ...) per component; particle
# balance uses the M0 (density) moment, index 0.
_DENSITY_MOMENT = 0


@dataclass(frozen=True)
class ParticleBalanceTraces:
  """Computed particle-balance time traces (all 1-D, aligned to ``time``).

  Attributes:
    time: Time stamps of the ``fdot`` trace.
    fdot: Rate of change of the M0 moment, summed over blocks.
    src: Rate of change from sources, or ``None`` if none was found.
    bflux_tot: Rate of change from boundary particle fluxes, or ``None``.
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
# end


def _accumulate(target: np.ndarray | None, addend) -> np.ndarray:
  """Sum ``addend`` into ``target`` (over blocks), copying on first use so
  the caller's array is never mutated in place."""
  addend = np.asarray(addend)
  return addend.copy() if target is None else target + addend
# end


def particle_balance_error(fdot: np.ndarray, src: np.ndarray,
    bflux_tot: np.ndarray) -> np.ndarray:
  """The particle-balance residual: ``S - bflux - df/dt``."""
  return src - bflux_tot - fdot
# end


def _block_prefix(file_prefix: str, block_idx: int) -> str:
  return file_prefix.replace("*", str(block_idx))
# end


def _resolve(path: str, override: str | None, default: str,
    block_idx: int) -> str:
  """Resolve a file-family member's path: ``override`` (with ``*``
  substituted for the block index) if given, else the naming-convention
  ``default``."""
  if override is None:
    return default
  # end
  return (path + override).replace("*", str(block_idx))
# end


def gk_particle_balance(
    name: str,
    species: str,
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
  """Plot (and compute) the particle balance of a single species.

  Requires ``<name>-<species>_fdot_integrated_moms.gkyl``; and (only if the
  run had sources or non-periodic boundaries)
  ``<name>-<species>_source_integrated_moms.gkyl`` and
  ``<name>-<species>_bflux_<direction><side>_integrated_HamiltonianMoments
  .gkyl`` files. If ``relative_error`` is requested,
  ``<name>-<species>_integrated_moms.gkyl`` and ``<name>-dt.gkyl`` are also
  required.

  Args:
    name: Simulation name (also the file prefix).
    species: Species name.
    path: Directory holding the simulation output.
    relative_error: Plot the relative error instead of every balance term.
    multib: ``"-10"`` (default) for a single block; ``"-1"`` to discover
      every block; otherwise a comma list or ``'start:stop[:step]'`` slice
      of block indices.
    fdot_file: Explicit distribution derivative moments path override.
    source_file: Explicit source moments path override.
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
  """
  path = path.rstrip("/") + "/"
  bflux_files = bflux_files or {}

  file_prefix = f"{path}{name}-" if multib == "-10" else f"{path}{name}_b*-"
  probe = fdot_file or (file_prefix + species + "_fdot_integrated_moms.gkyl")
  blocks = utils.get_block_indices(multib, probe)

  fig = plt.figure(figsize=(7.5, 4.5))
  ax = fig.add_axes([0.11 + indent_left, 0.15, 0.87 + add_width, 0.78])
  ax.plot([-1.0, 1.0], [0.0, 0.0], color="grey", linestyle=":", linewidth=1)

  absy_func = np.abs if absy else (lambda v: v)

  fdot = src = bflux_tot = None
  has_src = has_bflux = False
  time_fdot = time_bflux_tot = None

  for block_idx in blocks:
    block_prefix = _block_prefix(file_prefix, block_idx)

    fdot_name = _resolve(path, fdot_file,
        block_prefix + species + "_fdot_integrated_moms.gkyl", block_idx)
    found, t, v, _ = utils.read_time_trace_if_present(fdot_name)
    if not found:
      raise FileNotFoundError(f"Required file not found: {fdot_name}")
    # end
    time_fdot = t
    fdot_pb = v[:, _DENSITY_MOMENT]

    src_name = _resolve(path, source_file,
        block_prefix + species + "_source_integrated_moms.gkyl", block_idx)
    has_src, t, v, _ = utils.read_time_trace_if_present(src_name)
    src_pb = v[:, _DENSITY_MOMENT] if has_src else 0.0 * fdot_pb

    bflux_terms = []
    for d in _DIRS:
      for e in _EDGES:
        key = d + e
        bf_name = _resolve(path, bflux_files.get(key),
            block_prefix + species + f"_bflux_{d}{e}_integrated_HamiltonianMoments.gkyl",
            block_idx)
        found_b, t, v, _ = utils.read_time_trace_if_present(bf_name)
        if found_b:
          has_bflux = True
          time_bflux_tot = t
          bflux_terms.append(v[:, _DENSITY_MOMENT])
        # end
      # end
    # end
    bflux_pb = sum(bflux_terms) if bflux_terms else 0.0 * fdot_pb

    fdot = _accumulate(fdot, fdot_pb)
    src = _accumulate(src, src_pb)
    bflux_tot = _accumulate(bflux_tot, bflux_pb)
  # end

  legend_handles = []
  legend_strings = []

  if not relative_error:
    src = src.copy()
    src[0] = 0.0  # No fdot/bflux contribution at t=0.

    mom_err = particle_balance_error(fdot, src, bflux_tot)

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
    h, = ax.plot(time_fdot, absy_func(-fdot), linestyle=_LINE_STYLES[0])
    legend_handles.append(h)
    legend_strings.append(r"$-\dot{f}$")
    h, = ax.plot(time_fdot, absy_func(mom_err), linestyle=_LINE_STYLES[3])
    legend_handles.append(h)
    legend_strings.append(r"$E_{\dot{\mathcal{N}}}=$" + "".join(legend_strings))

    ax.legend(legend_handles, legend_strings, fontsize=_LEGEND_FONT_SIZE, frameon=False)

    ylabel_string = ylabel or ""
    title_string = title or r"Particle balance"
    mom_err_norm = None
  # end
  else:
    dt_name = _resolve(path, dt_file, file_prefix.replace("_b*", "") + "dt.gkyl", 0)
    _, time_dt, dt, _ = utils.read_time_trace_if_present(dt_name)

    distf = None
    for block_idx in blocks:
      block_prefix = _block_prefix(file_prefix, block_idx)
      f_name = _resolve(path, f_file,
          block_prefix + species + "_integrated_moms.gkyl", block_idx)
      _, t, v, _ = utils.read_time_trace_if_present(f_name)
      distf = _accumulate(distf, v[:, _DENSITY_MOMENT])
    # end

    fdot, src, bflux_tot, distf = fdot[1:], src[1:], bflux_tot[1:], distf[1:]
    mom_err = particle_balance_error(fdot, src, bflux_tot)
    mom_err_norm = mom_err * dt / distf

    ax.plot(time_dt, absy_func(mom_err_norm))

    ylabel_string = ylabel or r"$E_{\dot{\mathcal{N}}}~\Delta t/\mathcal{N}$"
    title_string = title or r"Relative error in particle conservation"
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

  traces = ParticleBalanceTraces(
      time=time_fdot, fdot=fdot, src=src if has_src else None,
      bflux_tot=bflux_tot if has_bflux else None,
      mom_err=mom_err, mom_err_norm=mom_err_norm)
  return fig, traces
# end
