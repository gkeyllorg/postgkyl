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

from postgkyl import operations
from postgkyl.gdata import load

from .. import discovery


def resolve_frames(
    frame: "int | str | list | tuple",
    *, name: str, species: str, suffix: str = "", block_idx: int | None = None,
) -> list:
  """Expand a frame specification into a concrete sorted list of frame indices.

  Args:
    frame: An ``int`` (single frame); a ``list``/``tuple`` of ints; a string
      with a single number (``"7"``) or comma-separated numbers
      (``"0,2,4"``); or a ``'start:stop[:step]'`` / ``':'`` range (range
      bounds default to the first/last frame discovered on disk).
    name: Simulation name prefix.
    species: Species name.
    suffix: Distribution-file suffix (see :func:`load_gk_distf`).
    block_idx: Use block-specific files with a ``_b<idx>`` prefix.

  Returns:
    A sorted list of concrete frame indices.
  """
  if isinstance(frame, int):
    return [frame]
  # end
  if isinstance(frame, (list, tuple)):
    return [int(f) for f in frame]
  # end

  frame_spec = str(frame).strip()
  if "," in frame_spec:
    return [int(f.strip()) for f in frame_spec.split(",")]
  # end
  if ":" not in frame_spec:
    return [int(frame_spec)]
  # end

  prefix = f"{name}_b{block_idx}" if block_idx is not None else name
  frame_infix = f"{suffix}_" if suffix else ""
  stem = f"{prefix}-{species}_{frame_infix}"
  available = sorted(discovery.available_frames(stem))
  parts = frame_spec.split(":")
  lower = int(parts[0]) if parts[0] else available[0]
  upper = int(parts[1]) if parts[1] else available[-1] + 1
  step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
  return [f for f in available if lower <= f < upper and (f - lower) % step == 0]
# end


def load_gk_distf(
    name: str, species: str, frame: int, *,
    tag: str = "f", suffix: str = "", use_c2p_vel: bool = False,
    use_mc2nu: bool = False, use_mapc2p: bool = False, block_idx: int | None = None,
    num_interp: int | None = None,
    jf_file: str | None = None,
    mapc2p_vel_file: str | None = None,
    jacobvel_file: str | None = None,
    mc2nu_file: str | None = None,
    mapc2p_file: str | None = None,
    jacobtot_inv_file: str | None = None,
) -> "GData":
  """Build a real distribution function from saved ``Jf`` data.

  Args:
    name: Simulation name prefix.
    species: Species name.
    frame: Frame index.
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
    A :class:`~postgkyl.gdata.gdata.GData` holding the interpolated
    distribution function.
  """
  prefix = f"{name}_b{block_idx}" if block_idx is not None else name
  frame_infix = f"{suffix}_" if suffix else ""

  if jf_file is None:
    jf_file = f"{prefix}-{species}_{frame_infix}{frame}.gkyl"
  # end
  if mapc2p_vel_file is None:
    mapc2p_vel_file = f"{prefix}-{species}_mapc2p_vel.gkyl"
  # end
  if jacobvel_file is None:
    jacobvel_file = f"{prefix}-{species}_jacobvel.gkyl"
  # end
  if mc2nu_file is None:
    mc2nu_file = f"{prefix}-geo_corn_mc2nu_pos_deflated.gkyl"
  # end
  if mapc2p_file is None:
    mapc2p_file = f"{prefix}-geo_corn_mapc2p_deflated.gkyl"
  # end
  if jacobtot_inv_file is None:
    jacobtot_inv_file = f"{prefix}-geo_int_jacobtot_inv.gkyl"
  # end

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
  # end
  if use_mc2nu:
    out = operations.map(out, mc2nu_file, space="conf")
    grid_type.append("mc2nu")
  # end
  elif use_mapc2p:
    out = operations.map(out, mapc2p_file, space="conf")
    grid_type.append("mapc2p")
  # end
  if grid_type:
    out.ctx["grid_type"] = " + ".join(grid_type)
  # end
  return out
# end
