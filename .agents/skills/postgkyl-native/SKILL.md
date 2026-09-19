---
name: postgkyl-native
description: Change or troubleshoot the Gkeyll native bridge, basis matrices, readers, memory ownership, or native builds in Postgkyl.
---

# Keep the foreign boundary compiled

`gpython/` is the only doorway to Gkeyll. No ctypes declarations, native struct
layouts, or foreign signatures in Python or other layers. `gpython.available()`
is the single capability switch.

The shim lives in `gkeyll/core/zero/{gkyl_gpython.h,gpython.c}` and builds into
Gkeyll's `libg0core.so`. Struct access, by-value basis conventions, and function
pointer dispatch belong there, checked against that tree's headers.
`gpython/csrc/_gpythonmodule.c` wraps only the shim's opaque handles, scalars,
and buffers. `_lib.py` imports the extension and checks `GPYTHON_API_VERSION`.

Build against the pinned clone using `scripts/build_gkeyll.sh` and
`scripts/build_gpython.sh`. Bundle libg0core beside the extension and retain
relative `$ORIGIN`/`@loader_path` linking. Preserve generated build provenance;
Gkeyll is a build-time clone and need not exist at runtime.

GkylArray capsules own and release arrays. Zero-copy construction pins its NumPy
buffer; view base chains pin the capsule so views outlive their dataset. Never
return unowned C memory. Build interpolation and representation matrices by
evaluating Gkeyll's own basis through the shim, not duplicated basis formulas.

`dg/` orchestrates kernels; `io/` dispatches readers. Prefer GkylCReader for native
field reads; retain the Python reader fallback for unavailable native libraries,
partial loads, and dynvectors. Keep NumPy installed before building so the
extension uses the runtime ABI; use `pip install --no-build-isolation -e '.[test]'`.

Verify handshake, memory lifetime, basis/interpolation, and modal algebra using
the relevant existing tests. Report native test skips explicitly when no compiled
library is available; skips do not establish native correctness.
