"""Loader for Gkeyll gyrokinetic distribution functions.

Reads the saved ``Jf`` (distribution times one or more Jacobians) together
with the velocity/configuration Jacobians, divides them out, and
interpolates onto a nodal grid, optionally applying velocity- and
position-space coordinate mappings.

Jf (phase-space) is weak-multiplied by jacobtot_inv (conf-space) via Gkeyll's
``gkyl_dg_mul_conf_phase_op_range`` staying gkyl-native and
already on the same (phase-space) grid as Jf, so a single ``interpolate()``
at the end suffices; no separate jacobtot_inv interpolation or manual
NumPy reshape/broadcast is needed. The division by jacobvel, in contrast,
happens on the *raw* modal coefficient arrays via plain NumPy division on
the ``.values`` views, not Gkeyll's weak-divide kernel: ``jacobvel`` carries
no DG basis metadata of its own and is stored piecewise-constant per cell (a
single component), so Gkeyll's ``weak_div`` (which requires both operands'
component count to be a multiple of a shared basis's ``num_basis``) cannot
take it as an operand at all. Scaling every one of the coefficients by that
one per-cell constant is nonetheless the exact quotient, and commutes freely
with the weak conf x phase multiply above (both are linear in Jf's
coefficients), so the two can happen in either order.

``resolve_frames``' range-discovery calls the shared
:mod:`postgkyl.diagnostics.discovery` helper instead of its own glob.
"""

from __future__ import annotations

from typing import Annotated

from postgkyl import operations
from postgkyl.cli_spec import CliType
from postgkyl.gdata import GData, GDataGroup, load

from .. import discovery

FrameSpec = int | str | list[int] | tuple[int, ...]


def resolve_frames(
    frame: FrameSpec,
    *,
    name: str,
    species: str,
    suffix: str = "",
    block_idx: int | None = None,
) -> list[int]:
  """Expand a frame specification into a concrete sorted list of frame indices.

  Args:
    frame: An ``int`` (single frame); a ``list``/``tuple`` of ints; a string
      with a single number (``"7"``) or comma-separated numbers
      (``"0,2,4"``); or a ``'start:stop[:step]'`` / ``':'`` range (range
      bounds default to the first/last frame discovered on disk).
    name: Simulation name prefix.
    species: Species name.
    suffix: Distribution-file suffix (see :func:`load_distf`).
    block_idx: Use block-specific files with a ``_b<idx>`` prefix.

  Returns:
    A sorted list of concrete frame indices.
  """
  if isinstance(frame, int):
    return [frame]
  if isinstance(frame, (list, tuple)):
    return [int(f) for f in frame]

  frame_spec = str(frame).strip()
  if "," in frame_spec:
    return [int(f.strip()) for f in frame_spec.split(",")]
  if ":" not in frame_spec:
    return [int(frame_spec)]

  prefix = f"{name}_b{block_idx}" if block_idx is not None else name
  frame_infix = f"{suffix}_" if suffix else ""
  stem = f"{prefix}-{species}_{frame_infix}"
  available = sorted(discovery.available_frames(stem))
  if not available:
    raise ValueError(
        f"No distribution frames found matching '{stem}<frame>.gkyl'.")
  parts = frame_spec.split(":")
  if len(parts) > 3:
    raise ValueError(
        f"Invalid frame range {frame_spec!r}; expected start:stop[:step].")
  lower = int(parts[0]) if parts[0] else available[0]
  upper = int(parts[1]) if parts[1] else available[-1] + 1
  step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
  if step <= 0:
    raise ValueError("Frame range step must be a positive integer.")
  resolved = [
      f for f in available if lower <= f < upper and (f - lower) % step == 0
  ]
  if not resolved:
    raise ValueError(
        f"Frame range {frame_spec!r} matches no files for '{stem}<frame>.gkyl'."
    )
  return resolved


def load_distf(
    name: str,
    species: str,
    frame: Annotated[FrameSpec, CliType(str)],
    *,
    tag: str = "f",
    suffix: str = "",
    use_c2p_vel: bool = False,
    use_mc2nu: bool = False,
    use_mapc2p: bool = False,
    block_idx: int | None = None,
    num_interp: int | None = None,
    jf_file: str | None = None,
    mapc2p_vel_file: str | None = None,
    jacobvel_file: str | None = None,
    mc2nu_file: str | None = None,
    mapc2p_file: str | None = None,
    jacobtot_inv_file: str | None = None,
) -> GData | GDataGroup:
  """Build real distribution functions from saved ``Jf`` data.

  A scalar frame returns one :class:`~postgkyl.gdata.gdata.GData`. A list,
  tuple, comma-separated string, or range returns a
  :class:`~postgkyl.gdata.gdatagroup.GDataGroup`, whose fluent operations
  broadcast over the loaded frames.

  Args:
    name: Simulation name prefix.
    species: Species name.
    frame: Frame index, comma-separated indices, or a
      ``start:stop[:step]`` range; ``:`` selects every available frame.
    tag: Tag for the resulting dataset.
    suffix: Use ``<name>-<species>_<suffix>_<frame>.gkyl`` as the input.
    use_c2p_vel: Convert velocity-space computational coordinates to
      physical ones using the ``mapc2p_vel`` mapping.
    use_mc2nu: Convert non-uniform computational coordinates to
      field-aligned ones.
    use_mapc2p: Convert position-space computational coordinates to
      Cartesian/cylindrical.
    block_idx: Use block-specific files with a ``_b<idx>`` prefix.
    num_interp: Interpolate onto a general mesh of the specified amount
      (default: ``poly_order + 1`` points per cell).
    jf_file: Explicit saved-distribution filename override.
    mapc2p_vel_file: Explicit velocity-coordinate mapping filename override.
    jacobvel_file: Explicit velocity-space Jacobian filename override.
    mc2nu_file: Explicit field-aligned coordinate mapping filename override.
    mapc2p_file: Explicit configuration-space mapping filename override.
    jacobtot_inv_file: Explicit inverse total-Jacobian filename override.

  Returns:
    One interpolated distribution function for a scalar frame, or a fluent
    group holding one distribution function per requested frame.
  """
  frames = resolve_frames(frame,
                          name=name,
                          species=species,
                          suffix=suffix,
                          block_idx=block_idx)
  datasets = [
      _load_distf_frame(name=name,
                        species=species,
                        frame=resolved_frame,
                        tag=tag,
                        suffix=suffix,
                        use_c2p_vel=use_c2p_vel,
                        use_mc2nu=use_mc2nu,
                        use_mapc2p=use_mapc2p,
                        block_idx=block_idx,
                        num_interp=num_interp,
                        jf_file=jf_file,
                        mapc2p_vel_file=mapc2p_vel_file,
                        jacobvel_file=jacobvel_file,
                        mc2nu_file=mc2nu_file,
                        mapc2p_file=mapc2p_file,
                        jacobtot_inv_file=jacobtot_inv_file)
      for resolved_frame in frames
  ]

  is_series = isinstance(frame,
                         (list, tuple)) or (isinstance(frame, str) and
                                            ("," in frame or ":" in frame))
  if not is_series:
    return datasets[0]
  for dataset, resolved_frame in zip(datasets, frames):
    dataset.set_label(str(resolved_frame))
  return GDataGroup(datasets)


def _load_distf_frame(
    name: str,
    species: str,
    frame: int,
    *,
    tag: str = "f",
    suffix: str = "",
    use_c2p_vel: bool = False,
    use_mc2nu: bool = False,
    use_mapc2p: bool = False,
    block_idx: int | None = None,
    num_interp: int | None = None,
    jf_file: str | None = None,
    mapc2p_vel_file: str | None = None,
    jacobvel_file: str | None = None,
    mc2nu_file: str | None = None,
    mapc2p_file: str | None = None,
    jacobtot_inv_file: str | None = None,
) -> GData:
  """Load and transform one resolved distribution-function frame."""
  prefix = f"{name}_b{block_idx}" if block_idx is not None else name
  frame_infix = f"{suffix}_" if suffix else ""

  if jf_file is None:
    jf_file = f"{prefix}-{species}_{frame_infix}{frame}.gkyl"
  if mapc2p_vel_file is None:
    mapc2p_vel_file = f"{prefix}-{species}_mapc2p_vel.gkyl"
  if jacobvel_file is None:
    jacobvel_file = f"{prefix}-{species}_jacobvel.gkyl"
  if mc2nu_file is None:
    mc2nu_file = f"{prefix}-geo_corn_mc2nu_pos_deflated.gkyl"
  if mapc2p_file is None:
    mapc2p_file = f"{prefix}-geo_corn_mapc2p_deflated.gkyl"
  if jacobtot_inv_file is None:
    jacobtot_inv_file = f"{prefix}-geo_int_jacobtot_inv.gkyl"

  jf_data = load(jf_file)
  # jacobvel is stored piecewise-constant per cell (see module docstring): one
  # coefficient per cell is exactly a poly_order=0 DG field, so this is real
  # metadata, not a guess -- it silences the load-time "missing basis"
  # warning honestly instead of leaving it to fire on every distf load.
  jacobvel_data = load(jacobvel_file, basis_type="serendipity", poly_order=0)
  jacobtot_inv_data = load(jacobtot_inv_file)

  weak_product = jf_data * jacobtot_inv_data
  f_coeffs = weak_product.values / jacobvel_data.values
  f_modal = weak_product._result(weak_product.grid, f_coeffs)
  # The composed distribution's true basis (gkhybrid, p1) is fixed by this
  # diagnostic's convention, not necessarily what jf_data's own file header
  # implies -- basis_type/poly_order/value_form are load-time-fixed
  # properties, so the override lands on ctx here rather than as an
  # interpolate() argument.
  f_modal.ctx.update(basis_type="gkhybrid", poly_order=1, value_form="modal")

  interpolated = f_modal.interpolate(num_interp=num_interp)
  out = interpolated._result(interpolated.grid, interpolated.values, tag=tag)

  # Coordinate maps run on the already-interpolated data via the shared map
  # verb. Velocity space (c2p_vel) deforms the trailing axes; configuration
  # space (mc2nu / mapc2p) deforms the leading ones.
  grid_type = []
  if use_c2p_vel:
    mc2p_vel = load(mapc2p_vel_file, poly_order=1, basis_type="serendipity")
    out = operations.map(out, mc2p_vel, space="vel")
    grid_type.append("c2p_vel")
  if use_mc2nu:
    out = operations.map(out, mc2nu_file, space="conf")
    grid_type.append("mc2nu")
  elif use_mapc2p:
    out = operations.map(out, mapc2p_file, space="conf")
    grid_type.append("mapc2p")
  if grid_type:
    out.ctx["grid_type"] = " + ".join(grid_type)
  return out
