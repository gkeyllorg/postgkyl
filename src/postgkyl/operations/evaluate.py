"""The ``evaluate`` verb -- evaluate RPN math expressions over datasets.

The numeric operators live in :mod:`postgkyl.numerics.ev_ops` (pure
``(grid, values)`` functions, keyed by token in ``numerics.ev_cmds``); this
module is the stack machine that drives them and the glue that resolves
``f``/``fN`` tokens against an explicit list of datasets.

Expressions use Reverse Polish Notation, e.g. ``"f0 f1 +"`` adds two datasets
and ``"f 2 *"`` doubles one. Data tokens are:

- ``f`` / ``fN``  -- the ``N``-th provided dataset (``f`` == ``f0``),
- ``fN[c]``       -- component ``c`` of that dataset (slices like ``0:3`` work),
- ``fN.key``      -- the scalar ``ctx[key]`` of that dataset.

Literals supply numbers, component ranges, and axis selectors. Arithmetic on
modal data uses the canonical DG dispatcher. Integration and gradients use the
same operations as the Python API. Other point-value consumers first unpack
DG locations and fields; unsupported stencils raise before touching storage.
"""

from __future__ import annotations

from typing import Annotated
from postgkyl.cli_spec import CliArgument

import operator
import re
from copy import deepcopy

import numpy as np

from postgkyl import dg
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.gdatastate import materialize_point_values
from postgkyl.gdatastate.layout import dg_layout
from postgkyl.numerics import ev_cmds
from postgkyl.numerics.calculus import parse_axis
from postgkyl.operations.arithmetic import binary, require_compatible_operands
from postgkyl.operations.differentiate import differentiate
from postgkyl.operations.integrate import integrate
from postgkyl.operations.select import select

# Use the same arithmetic dispatcher as GData's operators.
_MODAL_BINARY_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "pow": operator.pow,
}
_MODAL_UNARY_POWERS = {"sq": 2, "sqrt": 0.5}

# RPN tokens that are exact, shape-preserving pointwise math on nodal/quad
# point values (elementwise, no cross-cell/cross-node access, no reduction)
# -- safe to compute on the raw view and wrap back into a native array.
# Everything else in numerics.ev_cmds (dot, avg, max, min, mean, len, grad,
# grad2, int, div, curl) is a reduction or a finite-difference derivative
# and must leave the native domain instead.
_POINTWISE_TOKENS = frozenset({
    "+",
    "-",
    "*",
    "/",
    "pow",
    "sq",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "abs",
    "log",
    "log10",
    "exp",
    "max2",
    "min2",
})

# f, f0, f12 ... with optional [comp] selection and optional .ctxkey suffix.
_DATA_TOKEN = re.compile(r"^f(\d*)(?:\[([^\]]*)\])?(?:\.(\w+))?$")


def _rep_of(ctx: dict) -> str:
  return ctx.get("value_form", "modal")


def _compare(a, b) -> bool:
  """Value equality for nested metadata, independent of object identity."""
  if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
    return np.array_equal(a, b)
  if isinstance(a, dict) and isinstance(b, dict):
    return a.keys() == b.keys() and all(_compare(a[k], b[k]) for k in a)
  if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
    return len(a) == len(b) and all(_compare(x, y) for x, y in zip(a, b))
  return a == b


def _merged_context(contexts):
  """Retain agreeing metadata; keep source provenance separate from structure.

  A combined value has no single source-file header. Retain provenance only
  for a unary operation; never compare nested headers as live field metadata.
  """
  fields = [ctx for ctx in contexts if ctx]
  result, conflicts = {}, set()
  for ctx in fields:
    for key, value in ctx.items():
      if key == "_load_metadata" and len(fields) > 1:
        continue
      if key in result and not _compare(result[key], value):
        conflicts.add(key)
      else:
        result[key] = value
  return deepcopy({k: v for k, v in result.items() if k not in conflicts})


def _axis_literal(value):
  if isinstance(value, np.ndarray) and value.ndim == 0:
    value = int(value)
  return None if isinstance(value, str) and value == "all" else value


def _semantic_kernel(token, grids, values, contexts):
  """Route physical reductions and gradients through their canonical verbs."""
  if token not in {"int", "avg", "grad", "grad2", "len"}:
    return None
  index = 0 if token == "grad" else 1
  if not grids[index]:
    raise ValueError(f"evaluate: '{token}' requires a dataset with a grid")
  data = GDataState(ctx=contexts[index]).push(grids[index], values[index])
  axis = None if token == "grad" else _axis_literal(values[0])
  if token in {"int", "avg"}:
    result = integrate(data, axis=axis)
    if not isinstance(result, GDataState):
      result = data._result([],
                            np.atleast_1d(result),
                            interpolated=True,
                            value_form=None,
                            mapped_axes={})
    if token == "avg":
      axes = parse_axis(axis, data.num_dims)
      if any(np.ndim(data.grid[a]) != 1 for a in axes):
        raise ValueError("evaluate avg requires separable coordinates")
      volume = np.prod([data.grid[a][-1] - data.grid[a][0] for a in axes])
      if volume <= 0:
        raise ValueError("evaluate avg requires a positive domain volume")
      result = binary(operator.truediv, result, volume)
    return result
  if token == "len":
    coord = data.grid[int(axis)]
    if coord.ndim != 1:
      raise ValueError("evaluate len requires a separable coordinate axis")
    return data._result([],
                        np.atleast_1d(coord[-1] - coord[0]),
                        interpolated=True,
                        value_form=None,
                        mapped_axes={})
  if token == "grad":
    return differentiate(data)
  axes = parse_axis(axis, data.num_dims)
  derivatives = [differentiate(data, direction=a) for a in axes]
  if not derivatives:
    raise ValueError("evaluate grad2 needs at least one axis")
  result = derivatives[0]
  if len(derivatives) > 1:
    combined = np.concatenate([d.values for d in derivatives], axis=-1)
    if result.backend == "gkyl":
      combined = dg.rep.wrap(combined)
    result = result._result(result.grid, combined)
  return result


def _modal_view(value, ctx: dict):
  """Read-only NumPy view for point-value operations; other values pass through."""
  if dg.modal.is_native(value):
    return value.view(ctx.get("cells"))
  return value


def _as_scalar(value):
  """A Python float if ``value`` is scalar-shaped, else None."""
  if isinstance(value, (int, float, np.integer, np.floating)):
    return float(value)
  if isinstance(value, np.ndarray) and value.ndim == 0:
    return float(value)
  return None


def _modal_kernel(token: str, tmp_grid, tmp_values, tmp_ctx):
  """Lower RPN modal arithmetic to the canonical dataset arithmetic dispatcher.

  Wrap stack entries in state without copying their native buffers. This keeps
  basis/grid validation and DG kernel selection owned by arithmetic.binary.
  """
  if not any(dg.modal.is_native(v) for v in tmp_values) and not any(
      grid and ctx.get("basis_type") and not ctx.get("interpolated")
      and _rep_of(ctx) == "modal" for grid, ctx in zip(tmp_grid, tmp_ctx)):
    return None

  if token not in _MODAL_BINARY_OPS and token not in _MODAL_UNARY_POWERS:
    raise ValueError(
        f"evaluate: '{token}' is not defined for modal data; use .apply(...) "
        "for a projected pointwise function or explicitly convert with "
        ".represent(to='quad')/.interpolate() for point-value operations.")

  operands = []
  for grid, value, ctx in zip(tmp_grid, tmp_values, tmp_ctx):
    if grid:
      operands.append(GDataState(ctx=ctx).push(grid, value))
    else:
      scalar = _as_scalar(value)
      if scalar is None:
        raise ValueError("cannot mix native modal data with a plain array")
      operands.append(scalar)

  if token in _MODAL_UNARY_POWERS:
    result = binary(operator.pow, operands[0], _MODAL_UNARY_POWERS[token])
  else:
    # The top of the RPN stack is the right operand.
    result = binary(_MODAL_BINARY_OPS[token], operands[1], operands[0])
  return result


def _native_kernel(token: str, tmp_grid, tmp_values, tmp_ctx, func):
  """Dispatch canonical operations, then pointwise or unpacked array math."""
  semantic = _semantic_kernel(token, tmp_grid, tmp_values, tmp_ctx)
  if semantic is not None:
    return semantic
  is_native = [dg.modal.is_native(v) for v in tmp_values]
  reps = {
      _rep_of(c)
      for grid, c, native in zip(tmp_grid, tmp_ctx, is_native)
      if native or (grid and c.get("basis_type") and not c.get("interpolated"))
  }
  if reps == {"modal"}:
    return _modal_kernel(token, tmp_grid, tmp_values, tmp_ctx)

  # Empty/absent grids belong to scalars, literals, or terminal reductions.
  # Wrap field stack entries without copying buffers so all entry points use
  # the same compatibility rules, including after component selection.
  fields = [
      GDataState(ctx=ctx).push(grid, value)
      for grid, value, ctx in zip(tmp_grid, tmp_values, tmp_ctx) if grid
  ]
  for other in fields[1:]:
    require_compatible_operands(fields[0], other)

  if token not in _POINTWISE_TOKENS:
    for index, (grid, value,
                ctx) in enumerate(zip(tmp_grid, tmp_values, tmp_ctx)):
      if not grid:
        continue
      field = GDataState(ctx=ctx).push(grid, value)
      if dg_layout(field) is not None and token in {"div", "curl"}:
        raise ValueError(
            f"evaluate '{token}' needs an unpacked point grid; call "
            ".interpolate() explicitly before a finite-difference stencil")
      shadow = materialize_point_values(field)
      tmp_grid[index], tmp_values[index], tmp_ctx[index] = (shadow.grid,
                                                            shadow.values,
                                                            shadow.ctx)
    return None

  if not any(is_native) or token not in _POINTWISE_TOKENS:
    return None

  view_values = [_modal_view(v, c) for v, c in zip(tmp_values, tmp_ctx)]
  out_grid, out_values = func(tmp_grid, view_values)
  return out_grid, [dg.rep.wrap(v) for v in out_values]


def apply_operator(grid_stack, value_stack, ctx_stack, token: str) -> bool:
  """Reduce the RPN stacks in place by applying ``token`` if it is an operator.

  Each stack entry is a list of "sets" (grids/values/ctx dicts); an operator
  pops ``num_in`` entries, applies its pure function from
  :data:`postgkyl.numerics.ev_cmds` over every set (broadcasting shorter
  inputs), and pushes ``num_out`` results. Modal arithmetic retains its result
  state metadata; other operators merge inputs' ctx, dropping conflicts.

  Args:
    grid_stack, value_stack, ctx_stack: the parallel RPN stacks, mutated in
      place.
    token: the candidate operator token (e.g. ``'+'``, ``'sqrt'``, ``'int'``).

  Returns:
    True if ``token`` was a known operator and the stacks were reduced;
    False if ``token`` is not an operator (the stacks are untouched).

  Raises:
    ValueError: if the operator's function raises while evaluating.
  """
  if token not in ev_cmds:
    return False
  num_in = ev_cmds[token]["num_in"]
  num_out = ev_cmds[token]["num_out"]
  func = ev_cmds[token]["func"]

  in_grid, in_values, in_ctx, num_sets = [], [], [], []
  for _ in range(num_in):
    in_grid.append(grid_stack.pop())
    in_values.append(value_stack.pop())
    in_ctx.append(ctx_stack.pop())
    num_sets.append(len(in_values[-1]))
  for _ in range(num_out):
    grid_stack.append([])
    value_stack.append([])
    ctx_stack.append([])

  for set_idx in range(max(num_sets)):
    tmp_grid, tmp_values, tmp_ctx = [], [], []
    for i in range(num_in):
      tmp_grid.append(in_grid[i][min(set_idx, num_sets[i] - 1)])
      tmp_values.append(in_values[i][min(set_idx, num_sets[i] - 1)])
      tmp_ctx.append(in_ctx[i][min(set_idx, num_sets[i] - 1)])
    try:
      native_out = _native_kernel(token, tmp_grid, tmp_values, tmp_ctx, func)
      if isinstance(native_out, GDataState):
        value = native_out.native if native_out.backend == "gkyl" else native_out.values
        out_grid, out_values = [native_out.grid], [value]
      elif native_out is not None:
        out_grid, out_values = native_out
      else:
        view_values = [_modal_view(v, c) for v, c in zip(tmp_values, tmp_ctx)]
        out_grid, out_values = func(tmp_grid, view_values)
    except Exception as err:
      raise ValueError(str(err)) from err

    # Modal arithmetic owns the output layout and basis, including products
    # between configuration-space and phase-space fields.
    if isinstance(native_out, GDataState):
      out_ctx = dict(native_out.ctx)
    else:
      out_ctx = _merged_context(tmp_ctx)

    for i in range(num_out):
      grid_stack[-num_out + i].append(out_grid[i])
      value_stack[-num_out + i].append(out_values[i])
      this_ctx = dict(out_ctx)
      ctx_stack[-num_out + i].append(this_ctx)
  return True


def _push_token(token: str, datasets, grid_stack, value_stack,
                ctx_stack) -> bool:
  """Push a single non-operator ``token`` (data reference or literal).

  Returns False only if the token cannot be interpreted at all.
  """
  match = _DATA_TOKEN.match(token)
  if match:
    idx = int(match.group(1)) if match.group(1) else 0
    comp = match.group(2)
    ctx_key = match.group(3)
    dat = datasets[idx]
    if ctx_key is not None:
      if ctx_key not in dat.ctx:
        raise ValueError(
            f"evaluate: unknown ctx key '{ctx_key}' on dataset f{idx}")
      grid, values = None, np.array(dat.ctx[ctx_key])
    elif comp is None:
      grid = dat.grid
      values = dat.native if dat.backend == "gkyl" else dat.values
    else:
      dat = select(dat, comp=comp)
      grid = dat.grid
      values = dat.native if dat.backend == "gkyl" else dat.values
    grid_stack.append([grid])
    value_stack.append([values])
    ctx_stack.append([dat.ctx if ctx_key is None else {}])
    return True

  # Numeric / axis literal fallback (mirrors the CLI token parser).
  if "(" in token or "[" in token:
    value_stack.append([eval(token)])  # noqa: S307 -- trusted expression source
  elif ":" in token or "," in token:
    value_stack.append([str(token)])
  else:
    try:
      value_stack.append([np.array(float(token))])
    except ValueError:
      return False
  grid_stack.append([None])
  ctx_stack.append([{}])
  return True


def available_operators() -> list[str]:
  """The RPN operator tokens ``evaluate`` recognizes (e.g. ``'+'``, ``'sqrt'``)."""
  return sorted(ev_cmds)


def evaluate(chain: Annotated[str, CliArgument()],
             *datasets: "GDataState",
             tag: str | None = None,
             label: str | None = None) -> "GDataState":
  """Evaluate an RPN expression over an explicit list of datasets.

  ``f``/``fN`` tokens in ``chain`` refer to ``datasets[N]`` (``f`` == ``f0``);
  see the module docstring for the token grammar. The result is built via
  ``datasets[0]._result(...)`` (so it stays the caller's concrete dataset
  class) and holds the single value left on top of the stack.

  Args:
    chain: the RPN expression, e.g. ``"f0 f1 +"`` or ``"f sq 2 *"``.
    *datasets: the datasets referenced positionally by the ``f``/``fN``
      tokens. At least one is required (it anchors the result's class).
    tag: optional tag for the returned dataset (defaults to ``'default'``).
    label: optional label for the returned dataset (defaults to ``chain``).

  Returns:
    A dataset holding the evaluated grid/values and the merged ctx.

  Raises:
    ValueError: if ``datasets`` is empty, the expression is empty, a token
      is unrecognized, or an operator fails.
  """
  if not datasets:
    raise ValueError("evaluate: at least one dataset is required.")

  grid_stack, value_stack, ctx_stack = [], [], []
  for token in filter(None, chain.split(" ")):
    if apply_operator(grid_stack, value_stack, ctx_stack, token):
      continue
    if not _push_token(token, datasets, grid_stack, value_stack, ctx_stack):
      raise ValueError(
          f"evaluate: token '{token}' is neither data nor an operator")

  if not value_stack:
    raise ValueError("evaluate: expression produced no result")

  final_grid = grid_stack[-1][0]
  final_values = value_stack[-1][0]
  final_ctx = deepcopy(ctx_stack[-1][0])
  out_grid = final_grid if final_grid is not None else datasets[0].grid
  result = datasets[0]._result(out_grid,
                               final_values,
                               tag=(tag or "default"),
                               label=(label if label is not None else chain))
  # Keep stack metadata, replacing only facts derived from the output buffers
  # and grid. Native cell layout comes from the stack, not the flat array or
  # datasets[0] (which can be the conf operand of a conf * phase product).
  derived = {"num_comps", "lower", "upper"}
  if result.backend == "numpy":
    derived.add("cells")
  kept = {k: result.ctx[k] for k in derived if k in result.ctx}
  result.ctx = final_ctx
  result.ctx.update(kept)
  return result
