---
name: testing
description: Write precise, independent analytic tests for Postgkyl mathematics, data semantics, and public behavior.
---

# Testing

Scientific tests establish agreement with an independently known answer.
Coverage identifies untested paths; exercising a branch or agreeing with a
second production path does not establish numerical correctness.

## Analytic references and data

- Reuse data in `tests/test_data/`. When a new analytic function is needed,
  first add its reproducible data generation to `tests/generate_test_data.py`.
  `tests/conftest.py` generates these fixtures before collection in
  `tests/test_data/generated/`. Keep generated binaries ignored; do not commit
  large data files.
- Record the function, domain, representation, component order, and generation
  method. Distinguish exact DG coefficients, projected smooth functions, cell
  averages, and point samples. A p0 cell-center sample of a cosine is not its
  exact cell average or the continuous cosine between sample locations.
- Prefer closed-form modal coefficients or independent polynomial/quadrature
  calculations. The generator's `polynomial_factors` defines separable fields;
  `_generate_polynomial_fields` uses closed-form Legendre integrals without
  Postgkyl or Gkeyll basis evaluation. Keep the generator self-contained because
  the documentation distributes it as a standalone script. Sharing fixture
  definitions is fine; sharing the production computation being tested is not.
- Derive expected values from the analytic function, its derivatives,
  antiderivatives, or physical laws. Do not call the production helper to
  construct its own expected result. Round trips, API/CLI parity, and
  agreement between two implementations supplement an analytic check.
- Random coefficients remain useful for algebraic properties, ownership,
  shape handling, and invalid inputs. They are not an analytic oracle and
  need not be positive or describe a physical distribution.

## Precision and stress cases

- Check complete arrays and all components, with unequal axis lengths,
  offsets, and cell widths where relevant. Exercise boundaries, mixed modes,
  nonuniform grids, and discontinuities when the operation supports them.
- Use roundoff-level tolerances for exactly representable polynomials and
  exact quadrature. State why a tolerance is appropriate; set both `rtol`
  and `atol` intentionally, especially for zeros and small physical values.
- For approximate methods, test a derived error bound or convergence order.
  For example, midpoint integration of x^2 underestimates the integral by
  L*h^2/12. Do not describe sampled integration as exact or increase the mesh
  size until a loose assertion happens to pass.
- Respect [DG semantics](../data/SKILL.md). Weak multiplication expects an L2
  projection; weak division solves projected equations. Neither generally
  equals pointwise arithmetic. Local DG derivatives exclude inter-cell jumps.
  Do not require exact recovery of modes outside the retained basis or a
  capability the public API explicitly does not support.
- Keep contract, error-path, and rendering tests where they check meaningful
  behavior; those tests do not all need a scientific analytic reference.

## Verification and failures

Run focused tests, then the broader suite and formatting checks from the
[development skill](../development/SKILL.md). Native tests use the existing
`needs_gkeyll` skip condition for a missing compiled library.

When asked to write validation tests and defer fixes, verify the reference and
the documented contract, leave genuine failures visible, continue with other
tests, and report the failing test IDs and discrepancies. Do not loosen
tolerances, copy observed output into expectations, add skips/xfails, or change
production behavior just to make the validation suite pass.

If a valid test case aborts or hangs in native code, isolate that operation in
a subprocess with a timeout, assert successful exit, and still check its
analytic result. A crash must fail the test without terminating the suite.

Examples are real-world workflows that read data and make plots for the
documentation; they supplement focused numerical tests.
