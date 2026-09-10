---
name: postgkyl-data
description: Preserve DG representation and backend semantics when changing loading, arithmetic, conversions, integration, or terminal data consumers.
---

# Preserve data meaning

Storage (`backend`: `gkyl` or `numpy`) and representation
(`ctx["value_form"]`: `modal`, `nodal`, or `quad`) are separate facts. Do not
add an `is_modal` flag or infer point-value capabilities from storage alone.

- Modal coefficients use Gkeyll DG operations: weak multiply/divide, coefficient
  linear combinations, scalar scaling/mean shifts, repeated weak multiplication
  for integer powers, and native integration. Ufuncs, array conversion, selection,
  and plotting must not treat coefficients as field values.
- Nodal and quadrature values are fields at points. Pointwise NumPy operations
  are allowed; native results wrap back into native storage in the same value_form.
  Plot true point locations; non-tensor node sets use explicit `.to_quad()`.
- Reject mixed backends or mixed value_forms when combining datasets.
- `.interpolate()` is the one-way bridge to a new, by-value NumPy field array.
  Interpolation matrices come from Gkeyll basis functions, applied per cell;
  nodal input first uses the exact nodal-to-modal transform.
- Only explicit `.to_modal()`, `.to_nodal()`, and `.to_quad()` perform representation
  conversions. `.apply(fn, num_quad=...)` spells modal → quad → fn → projection,
  equivalent to `fn(d.to_quad()).to_modal()`. Respect quadrature exactness limits.
- Full modal integration is terminal and native; partial integration uses native
  averaging with physical-volume scaling and returns lower-dimensional modal data.
  `.average()` and `.eval_at_coord_proj()` likewise stay modal/native and compose.
  `.local_poly()` builds a discontinuity-preserving plotting mesh.

Resolve `basis_type`, `poly_order`, and `value_form` once from file metadata or
explicit load options. Downstream verbs read ctx and raise if required metadata
is missing; never add basis/order override parameters to them.

For spatial data with unresolved basis metadata, loading warns and defaults to
serendipity p0 nodal data. When value_form is defaulted, express the grid as cell
centers, matching one point per cell. Dynvectors have no spatial DG basis and
are exempt. The readers' narrower case—known basis but no value_form tag—assumes
modal silently; keep these two defaults distinct.

Route results through `_result` and terminal native point-value consumers through
`gdatastate.materialize_point_values`. Preserve native ownership and read-only
coefficient views; use the existing representation tests to verify semantics.
