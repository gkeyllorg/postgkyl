# Native physics operations

A `vlasov` (or higher) build provides
`postgkyl.gpython.vlasov.coulomb_log` and `spitzer_coll_freq`. The GData
operations live in `postgkyl.diagnostics.vm.collisions` and infer basis and
grid metadata. `spitzer_coll_freq` takes Maxwellian moment fields packed as
`[n, u..., T/m]`, a collocated magnetic-field magnitude, and explicit physical
constants in consistent units. The native Coulomb-log expression is
species-ordered; the Spitzer updater averages the two species orders.
For modal Coulomb-log inputs, pointwise evaluation is projected back into the
original basis. Spitzer uses cell averages for the Coulomb log and quadrature
for the remaining density/thermal-speed factor.

A `gyrokinetic` (or higher) build also provides
`postgkyl.diagnostics.gk.mom_gyrokinetic(distribution, bmag, mass=..., moment="M0")`
and the native-array counterpart `postgkyl.gpython.gyrokinetic.mom_gyrokinetic`.
The GData operation infers configuration dimensions from `bmag` and returns
configuration-space modal data. Supported phase bases are p1 gkhybrid and p2
serendipity in 1x1v, 1x2v, and 2x2v, plus p1 3x2v. Uniform physical velocity
grids are required; mapped velocity coordinates need an explicit map binding
and are not supported by this interface.

Moment names are `M0`, `M1`, `M2`, `M2par`, `M2perp`, `M3par`, `M3perp`,
`ThreeMoments`, `FourMoments`, and `HamiltonianMoments`. Combined outputs pack
modal fields in the named order: `(M0,M1,M2)`, `(M0,M1,M2par,M2perp)`, or
`(M0,mass*M1,charge*phi*M0+mass*M2/2)`. In 1v, `FourMoments` contains three
fields and individual perpendicular moments are unavailable.
`HamiltonianMoments` requires `phi`. In 2v the native measure is
`(2*pi/mass) dv_parallel dmu`, with no extra multiplication by B. Any spatial
Jacobian already in the stored distribution remains in the moment; divide it
out explicitly when the physical quantity requires that normalization.

These functions also generate the CLI commands `vm_collisions_coulomb_log`,
`vm_collisions_spitzer_coll_freq`, and `gk_mom_gyrokinetic`; use `pgkyl <command> --help` for
dataset-reference options. A core-only installation imports normally and
reports the required build option when an optional binding is called.

The runnable example uses the generated analytic test data:

```{literalinclude} _inputs/examples/scripts/native_physics.py
:language: python
```
