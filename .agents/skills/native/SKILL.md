---
name: native
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

## Local Gkeyll development

Make native kernel and shim changes directly in the linked `gkeyll/` checkout
under the Postgkyl workspace. This producer tree owns the C implementation even
though Postgkyl ignores it. Do not substitute Postgkyl-managed patch files,
build-time patch application, or Python workarounds for changes that belong
there. Do not commit changes in either repository; leave the diffs for the user.
Do not change the dependency pin merely to capture local development work.

Inspect `git status` in both repositories before editing, preserve existing
changes, and report the native diff separately from the Postgkyl diff. Follow
Gkeyll's own instructions and formatting configuration for its source files.

For a local producer build, configure if needed and build in place:

```bash
cd gkeyll
./configure --use-lapack-lite=yes --app=core  # when configuration is needed
make core -j2
cd ..
sh scripts/build_gpython.sh
```

`scripts/build_gkeyll.sh` is the clean, pinned installation path: it checks out
`scripts/gkeyll-revision` and refuses tracked modifications. Do not use it to
build ongoing native edits or clear those edits to satisfy it. The distinction
between release installation and local producer development is intentional.

Bundle libg0core beside the extension and retain relative `$ORIGIN`/`@loader_path`
linking. Preserve generated build provenance; Gkeyll is needed at build time,
not at runtime.

## Data ownership and verification

GkylArray capsules own and release arrays. Zero-copy construction pins its NumPy
buffer; view base chains pin the capsule so views outlive their dataset. Never
return unowned C memory. Build interpolation and representation matrices by
evaluating Gkeyll's own basis through the shim, not duplicated basis formulas.

`dg/` orchestrates kernels; `io/` dispatches readers. Prefer GkylCReader for native
field reads; retain the Python reader fallback for unavailable native libraries,
partial loads, and dynvectors. Keep NumPy installed before building so the
extension uses the runtime ABI. For clean installations, use
`pip install --no-build-isolation -e '.[test]'`; rebuild local native edits
with the in-place workflow above.

Verify handshake, memory lifetime, basis/interpolation, and modal algebra using
the relevant existing tests. Report native test skips explicitly when no compiled
library is available; skips do not establish native correctness.
