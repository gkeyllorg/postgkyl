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

Anything else is parsed as a numeric/axis literal (a float, a ``"0,1"`` /
``"0:3"`` axis spec, or a Python literal in brackets/parens). Every operator
in ``numerics.ev_cmds`` is a plain array function -- none needed a
``NotImplementedError`` GData-only placeholder (see the numerics module
docstring), so there is nothing left to resolve here.

A data token referencing native (gkyl-backed) data is kept native, not
forced through ``select()``'s point-value guard, regardless of
value_form -- see ``_native_kernel``:

- **modal** (DG coefficients): ``+ - * / pow sq sqrt`` use the same
  arithmetic dispatcher as GData operators, including weak multiplication,
  division, and projected fractional powers. Unsupported operations and
  invalid operands raise; coefficients are never treated as point values.
- **nodal/quad** (point values): every operator in ``_POINTWISE_TOKENS``
  (``+ - * / pow sq sqrt sin cos tan abs log log10 exp max2 min2
  scale_comp scale_zi_axis``) is exact regardless of packing, so it is
  computed with plain NumPy on the view and the result is wrapped back into
  a native array -- computed on the view, wrapped back native, staying
  in-value_form, mirroring ``operations.arithmetic``'s ufunc dispatch.
  Anything else (``dot``, ``avg``, ``max``, ``min``, ``mean``, ``len``,
  ``grad``, ``grad2``, ``int``, ``div``, ``curl``) is a genuine reduction or
  finite-difference derivative -- not a per-point transform -- so it leaves
  the native domain for plain NumPy math on the raw view, same as before;
  ``apply_operator`` then strips the now-stale ``value_form`` tag and
  marks the result ``interpolated`` (mirroring ``.interpolate()``) so
  ``info()`` doesn't keep claiming a value_form the data no longer has.
"""

from __future__ import annotations

import operator
import re

import numpy as np

from postgkyl import dg
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.numerics import ev_cmds
from postgkyl.operations.arithmetic import binary, require_compatible_operands
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
    "scale_comp",
    "scale_zi_axis",
})

# f, f0, f12 ... with optional [comp] selection and optional .ctxkey suffix.
_DATA_TOKEN = re.compile(r"^f(\d*)(?:\[([^\]]*)\])?(?:\.(\w+))?$")


def _rep_of(ctx: dict) -> str:
  return ctx.get("value_form", "modal")


def _compare(a, b) -> bool:
  """Equality that also handles NumPy arrays (used when merging ctx dicts)."""
  if isinstance(a, np.ndarray):
    return np.array_equal(a, b)
  return a == b


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
  if not any(dg.modal.is_native(v) for v in tmp_values):
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
  """Dispatch a native (gkyl-backed) operand to the value_form-correct math.

  Returns the arithmetic result state for modal data, ``(out_grid,
  out_values)`` with native arrays for point-value transforms, or ``None``
  when the caller should run the NumPy function on point values.

  - Every native operand modal: delegates to :func:`_modal_kernel` for
    DG arithmetic; unsupported operations raise.
  - Every native operand the *same* nodal/quad value_form, and ``token``
    in :data:`_POINTWISE_TOKENS`: exact NumPy math on the raw view, wrapped
    back native -- mirrors ``operations.arithmetic``'s "compute on the view,
    wrap back native, stay in-value_form" pointwise dispatch.
  - Point-value dataset pairs use arithmetic's shared compatibility check
    before their grids or DG metadata can be discarded by array operations.
  - Any other token (reductions, finite-difference derivatives): returns
    ``None`` so the caller's plain-NumPy path runs -- the result then
    genuinely leaves the native/value_form domain.
  """
  is_native = [dg.modal.is_native(v) for v in tmp_values]
  reps = {
      _rep_of(c)
      for v, c, native in zip(tmp_values, tmp_ctx, is_native) if native
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
        out_grid, out_values = [native_out.grid], [native_out.native]
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
      # Merge ctx of all inputs; drop keys that disagree between inputs.
      out_ctx: dict = {}
      remove_list = []
      for i in range(num_in):
        for key in tmp_ctx[i]:
          if key in out_ctx and _compare(tmp_ctx[i][key], out_ctx[key]):
            pass  # already copied and matches; nothing to do
          elif key in out_ctx:
            remove_list.append(key)  # discrepancy; mark for removal
          else:
            out_ctx[key] = tmp_ctx[i][key]
      for key in dict.fromkeys(remove_list):
        out_ctx.pop(key)

    # A native nodal/quad operand whose result did *not* come back wrapped
    # native (a genuine reduction/derivative, per _native_kernel) has left
    # the per-point field domain: the merged ctx's 'value_form' is now
    # stale (it still names a value_form this output no longer has), so
    # drop it and mark the result the same way .interpolate() does -- no
    # longer gkyl-native -- rather than let info() keep describing it as a
    # value_form it left behind.
    was_native_nonmodal = any(
        dg.modal.is_native(v) and _rep_of(c) != "modal"
        for v, c in zip(tmp_values, tmp_ctx))

    for i in range(num_out):
      grid_stack[-num_out + i].append(out_grid[i])
      value_stack[-num_out + i].append(out_values[i])
      this_ctx = dict(out_ctx)
      if was_native_nonmodal and not dg.modal.is_native(out_values[i]):
        this_ctx.pop("value_form", None)
        this_ctx["interpolated"] = True
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
    elif comp is None and dat.backend == "gkyl":
      # Keep native data on the stack (rather than forcing it through
      # select()'s point-value guard), regardless of value_form: RPN
      # math routes through Gkeyll's own weak kernels for modal data, or
      # exact NumPy math wrapped back native for nodal/quad point values,
      # when the operator supports it; unsupported modal operations raise
      # -- see _native_kernel.
      grid, values = dat.grid, dat.native
    else:
      # select() carries the shared operability guard (raw modal coefficients
      # refuse; nodal/quad value_forms, already point values, pass) for
      # a comp-sliced modal token (still genuinely unsafe -- slicing raw DG
      # coefficients by component can mix basis functions) and every
      # already-point-value token. select() itself now keeps a gkyl-backed
      # nodal/quad result native, so keep pushing the native array here too
      # (not its plain-view .values) so it stays eligible for _native_kernel.
      selected = select(dat, comp=comp)
      grid = selected.grid
      values = selected.native if selected.backend == "gkyl" else selected.values
    grid_stack.append([grid])
    value_stack.append([values])
    ctx_stack.append([dat.ctx])
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


def evaluate(chain: str,
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
  final_ctx = dict(ctx_stack[-1][0])
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
